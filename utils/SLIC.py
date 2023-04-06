import numpy as np 
from math import sqrt
from PIL import Image
import cv2
import scipy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel_h, sobel_v
import torch
import pickle

N_CLUSTERS = 1400
M = 15 #weight in distance function in clustering algorithm
OPENING_KERNEL = 10
SLIC_ROUNDS = 10
SAVE_CLUSTER_IMAGE = False

class Cluster():
    index = 0
    def __init__(self, x:int, y:int, intensity:float) -> None:
        self.x = x
        self.y = y
        self.intensity = intensity
        self.pixels = {}
        self.index = Cluster.index
        Cluster.index += 1
        self.neighbours = []#indexes of adjacent clusters

    def update_cluster(self, max_x):
        x = 0
        y = 0
        intensity = 0
        for pixel, i in self.pixels.items():
            x += pixel%max_x
            y += pixel//max_x
            intensity += i
        self.x = int(x/len(self.pixels))
        self.y = int(y/len(self.pixels))
        self.intensity = int(intensity/len(self.pixels))

    def mask(self, image_width:int, image_height:int):
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for pixel in self.pixels.keys():
            x = pixel%image_width
            y = pixel//image_width
            mask[y][x] = 1
        return np.uint8(mask)
    
    def features(self, gradient, image):
        return self.general_cluster_features(gradient, image, self.mask(gradient.shape[1], gradient.shape[0]))



    @staticmethod
    def general_cluster_features(gradient, image, mask):
        intensity_values = image[mask>0]
        """
        Intersity properties
        """

        mean = np.mean(intensity_values)
        variance = np.var(intensity_values)
        skewness = scipy.stats.skew(intensity_values)
        TenBinHistogram, bin_edges = np.histogram(intensity_values, bins = 10, density=False)
        """
        Texture properties
        """

        distances = [1] # distance between pixels
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 256

        glcm = graycomatrix(np.uint8(image*mask), distances, angles, levels=levels, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        energy = graycoprops(glcm, 'energy').flatten()

        eps = np.finfo(float).eps  # machine epsilon to avoid log(0)
        entropy = -np.sum(glcm * np.log2(glcm + eps))

        

        """
        Gradient propperties

        It is eficient to compute gradient of the entire image and then extract the values for each cluster
        """


        magnitude_gradient = np.sqrt(gradient[:,:,0]**2 + gradient[:,:,1]**2)
        orientation_gradient = np.arctan2(gradient[:,:,1], gradient[:,:,0])

        grad_mag_sp = magnitude_gradient[mask>0]
        grad_orient_sp = orientation_gradient[mask>0]

        orientation_histogram, _ = np.histogram(grad_orient_sp, bins=11, range=(-np.pi, np.pi), density=True)

        magnitude_histogram, _ = np.histogram(grad_mag_sp, bins=11, range=(0, 255), density=True)

        return [mean]#, variance, contrast, correlation, energy, entropy, magnitude_histogram, orientation_histogram]


class SLIC():
    def __init__(self, K:int) -> None:
        self.K = K

        self.clusters = []
    
        
    def init_clusters(self):
        index = (int(i * self.S) for i in range(self.K))
        for i in index:
            x = i % self.image.shape[1]
            y = self.S * (i//self.image.shape[1])
            i = self.image[y][x]
            self.clusters.append(Cluster(x, y, i))
        
    def __call__(self, image_name, image_or_pkl:bool):
        self.make_superpixels(image_name, name_or_pil = True)
        if image_or_pkl:
            self.save_bin(image_name)
        else:
            self.create_image()
            self.save_image(image_name)

    @staticmethod
    def load_image(image_name:str):
        img_pil = Image.open(image_name)
        try:
            return np.array(img_pil.getdata(),dtype = np.uint8).reshape(img_pil.size[1], img_pil.size[0]) #Only if the image is gray
        except ValueError:
            return np.array(img_pil.getdata()).reshape(img_pil.size[1], img_pil.size[0], 3)[:,:,0]

    def make_superpixels(self, image, name_or_pil:bool):
        if name_or_pil:
            self.image = self.load_image(image)
        else:
            self.image = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)[:,:,0]

        self.N = self.image.shape[0] * self.image.shape[1]
        self.S = int(sqrt(self.N/self.K))

        self.init_clusters()
        self.distances = np.ones_like(self.image) * np.inf
        self.labels = np.ones_like(self.image, dtype=int) * -1

        for epoch in range(SLIC_ROUNDS):
            print(epoch)
            self.update_clusters()

    def set_neighbours(self):
        """
        Each cluster gets assigned its neighbours - their indexes in a single list
        """

        for cluster in self.clusters:
            mask = cluster.mask(self.labels.shape[1], self.labels.shape[0])
            kernel = np.ones((2,2),np.uint8)
            gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            neighbours_pixels = self.labels * gradient
            cluster.neighbours = np.unique(neighbours_pixels)

    def distance(self, cluster:Cluster, x:int, y:int):
        m = M
        dc = (self.image[y][x] - cluster.intensity)**2
        ds = (cluster.x - x)**2 + (cluster.y - y)**2
        return sqrt(dc + (ds*m**2)/self.S**2)

    def pixel_update(self, cluster:Cluster, x:int, y:int):
        distance = self.distance(cluster, x, y)
        if distance < self.distances[y][x]:
            self.distances[y][x] = distance

            if self.labels[y][x] != -1:
                label = self.labels[y][x]
                try:
                    former_cluster = self.clusters[label]
                except IndexError as err:
                    # print(label)
                    raise err
                former_cluster.pixels.pop(y*self.image.shape[1] + x)


            self.labels[y][x] = cluster.index
            cluster.pixels[y*self.image.shape[1] + x] = self.image[y][x]


    def update_clusters(self):
        for cluster in self.clusters:
            center_x = cluster.x
            center_y = cluster.y

            for y in range(center_y-self.S, center_y + self.S):
                if y<0:
                    continue
                if y >= self.image.shape[0]:
                    break
    
                for x in range(center_x-self.S, center_x + self.S):
                    if x<0:
                        continue
                    if x >= self.image.shape[1]:
                        break
                    
                    self.pixel_update(cluster, x, y)
        for cluster in self.clusters:
            cluster.update_cluster(max_x = self.image.shape[1])

    def create_image(self):
        for x in range(self.image.shape[1]):
            for y in range(self.image.shape[0]):
                cluster_id = self.labels[y][x]
                if cluster_id > -1:
                    self.labels[y][x] = self.clusters[cluster_id].intensity
                else:
                    self.labels[y][x] = self.image[y][x]

    def save_image(self, image_name):
        image = Image.fromarray(np.uint8(self.labels))
        image.save(F"{image_name[:-4]}_clusters.jpg")

    def save_bin(self, image_name):
        with open(F"{image_name[:-4]}.pkl", "wb") as f:
            pickle.dump(self, f)

    def make_gradient(self):
        grad_x = sobel_h(self.image)
        grad_y = sobel_v(self.image)
        grad_x = np.expand_dims(grad_x, 2)
        grad_y = np.expand_dims(grad_y, 2)
        return np.concatenate((grad_x, grad_y), axis = 2)



class RegionManager:
    def __init__(self, segmentation_model, slic: SLIC):
        """
        model: segmentation model for pseudolabels prediction
        slic: Just initialized slic, not with already done superpixels, so it can be done inside this class
        """
        self.model = segmentation_model
        self.slic = slic
        
        """ 
        This will be list of lists of clusters
        In this list are superpixels which are from more than a half covered in the pseudo label - see self.fit_pseudolabel_to_superpixels
        """
        self.pseudo_label_superpixels = []

        """
        Tensor of masks (one for each seperable object)
        See self.get_masks
        """
        self.og_masks = None
        self.masks_in_progress = None #will be  1D array with length of number of objects with ones where we still refine
        

    def get_masks(self, pil_image:Image, debug:Image = None, semantic:bool = True) -> np.ndarray:
        """
        It is possible that there are several objects which we want to segment
        However each could have different properties based on intesity etc.
        Thus it is uncessary to split objects into several masks

        This methods uses cv2's floodfill to get each object

        Several methods are defined in this method to make it more readable

        method: get_start_point - floodfill needs a start point from which it starts its algorythm
        method: add_mask        - handles adding mask into the final tensor as cv's mask has different shape
        method: split_objects   - handles the main loop, creating mask from floodfill and then adding it to tensor

        input: image_name       - file to load
        input: semantic         - if true, not each objects has unique color -> we have to separate them

        return : mask           - np.ndar
        """

        def torch_load(pil_image:Image)->torch.Tensor:
            img = torch.Tensor(pil_image.getdata()).reshape(3, pil_image.size[1], pil_image.size[0])/255
            # img = img.reshape((1,*img.size()))
            return img[0,:,:]

        def get_start_point(image):
            point = np.argmax(image)
            return point%image.shape[1], point//image.shape[1]


        def add_mask(masks:np.ndarray, mask:np.ndarray)->np.ndarray:
            """
            Handles concatenating an instance mask into the tensor of instace masks
            """
            mask = mask[1:-1,1:-1]
            mask = mask - np.sum(masks, axis = 2)
            mask = np.reshape(mask, (*mask.shape, 1))
            
            masks = np.concatenate((masks, mask), axis = 2)
            return masks
        
        def split_objects(source: np.ndarray) -> np.ndarray:
            """
            Take in binary mask of semantic segmentation and creates instance segmentation mask

            Variables here might be little bit confusing. cv2 uses image and mask. The image is where 
            we are look
            """
            masks = np.zeros_like(source)
            masks = np.reshape(masks, (*masks.shape, 1))
            mask = None
            while np.sum(source) > 0:
                start_point = get_start_point(source)

                output = cv2.floodFill(source, mask, start_point, newVal=0)
                mask = output[2].copy()
                masks = add_mask(masks, mask)
            for i in range(masks.shape[2]):
                masks[:,:,i] *= i
            self.masks_in_progress = np.ones(i)
            return np.sum(masks, axis=2)
        
        image = torch_load(pil_image)
        if self.model is not None:
            mask = self.model.predict(image)    #predict returns numpy array as it easier to transfer to PIL
        elif debug is not None:
            mask = np.array(debug.getdata(), dtype=np.uint8).reshape(debug.size[1], debug.size[0])
        else:
            raise TypeError("Get masks should have gotten debug argument or model for prediction should not be None")
        
        if np.sum(mask) == 0:
            self.masks = None

        if semantic:
            self.masks = split_objects(mask)
        else:
            self.masks = mask


    def fit_pseudolabel_to_superpixels(self):
        for i in range(len(self.masks_in_progress)):

            temp_labels = self.slic.labels.copy()
            temp_labels *= (self.masks == i + 1) #it can not start from zero as zero is label for black background
            lst = []
            for cluster in self.slic.clusters:
                fitting_pixels = np.sum(temp_labels == cluster.index)
                if fitting_pixels/len(cluster.pixels) >= 0.5 and fitting_pixels/len(cluster.pixels) <= 1:
                    lst.append(cluster)

            self.pseudo_label_superpixels.append(lst)

    def get_neighbours(self) -> list:
        """
        Firts it calculates neighbours for each superpixel assigned to generated pseudolabel
        Then for each object's mask (if more objects was in generated mask) made out of superpixels
        finds the neighbours to this mask 
        """
        self.slic.set_neighbours() #each cluster has list with ids of its neighbours

        main_list = []
        # for mask_list in self.pseudo_label_superpixels: #mask list is list of clusters assigned to this object's mask
        for index in range(len(self.masks_in_progress)):
            if not self.masks_in_progress[index]:
                main_list.append(None)
                continue
            mask_list = self.pseudo_label_superpixels[index]
            neighbours_list = []
            mask_list_ids = []
            for cluster in mask_list:
                neighbours_list.extend(cluster.neighbours) #cluster.neighbours has ids of clusters
                mask_list_ids.append(cluster.index)
            neighbours_list = np.unique(np.array(neighbours_list))
            neighbours_list = neighbours_list[~np.isin(neighbours_list, mask_list_ids)]
            neighbours_list = [self.slic.clusters[index] for index in neighbours_list]
            main_list.append(neighbours_list)
        return main_list
    
    def content_similarity(self, features1, features2, sigma =1) -> np.ndarray:
        mu = lambda x: np.exp(0.5 * (x/sigma)**2)
        S = []
        # print(features1)
        # print(features2)
        for index, (feature1, feature2), in enumerate(zip(features1, features2)):
            if isinstance(feature1, (list, tuple)):
                # print(feature1)
                feature1 = np.array(feature1)
                feature2 = np.array(feature2)

            S.append(mu(np.sum((feature1-feature2)**2/(feature1 + feature2 + 1e-5))))

        S = np.array(S)
        # print(S)
        ret = 0.25 * (np.max(S) + np.min(S) + np.nanmean(S) + np.var(S))
        # print(ret)
        return ret
    
    def border_similarity(self, cluster:Cluster, features:list, psuedo_label_id: int):
        """
        input: pseudo_label     - which pseudolabel list of clusters should be used
        """
        S = []
        superpixel_list = self.pseudo_label_superpixels[psuedo_label_id]
        superpixel_list_indexes = [superpixel.index for superpixel in superpixel_list]
        for neigbour_index in cluster.neighbours:
            if neigbour_index in superpixel_list_indexes:
                neigbour_features = self.slic.clusters[neigbour_index].features(self.gradient, self.slic.image)
                S.append(self.content_similarity(neigbour_features, features))
            else:
                continue
        return np.mean(np.array(S))
    
    def argmax_similarity(self, pseudo_label:list, pseudo_label_id:int, neighbours:list, Wb:float = 1, Wc:float = 1):
        """
        input: pseudo_label     list of clusters which a pseudolabel consists of
        input: neighbours       list of clusters which are next to those of pseudolabel

        """
        # print("Start of argmax func")
        similarity_list = []
        pseudolabel_mask = np.expand_dims(np.zeros(self.slic.labels.shape), axis = 2)
        for superpixel in pseudo_label:
            superpixel_mask = superpixel.mask(pseudolabel_mask.shape[1], pseudolabel_mask.shape[0])
            # try:
            pseudolabel_mask = np.concatenate((pseudolabel_mask, np.expand_dims(superpixel_mask, axis = 2)), axis = 2)
            # except ValueError as err:
            #     print(superpixel_mask.shape, superpixel_mask.shape)
        pseudolabel_mask = np.sum(pseudolabel_mask, axis = 2)
        # intensity_values = np.uint8(self.slic.image * pseudolabel_mask)

        # print("extracting features")
        pseudolabel_features = Cluster.general_cluster_features(self.gradient, np.uint8(self.slic.image), pseudolabel_mask)
        
        # print("calculating similarity")
        for neighbour in neighbours:
            neighbour_mask =neighbour.mask(pseudolabel_mask.shape[1], pseudolabel_mask.shape[0])
            neighbour_features = neighbour.general_cluster_features(self.gradient, self.slic.image, neighbour_mask)
            content_similarity = self.content_similarity(pseudolabel_features, neighbour_features)
            border_similarity = self.border_similarity(neighbour, neighbour_features, pseudo_label_id)
            Sim = Wc*content_similarity + Wb * border_similarity
            similarity_list.append((Sim, neighbour_features, neighbour))
        similarity_list.sort(key = lambda x: x[0], reverse = False)
        for _,features, neighbour in similarity_list:
            if self.is_available(neighbour, features, pseudo_label_id):
                return neighbour
        else:
            return None
             
    def is_available(self, cluster:Cluster, features:list, pseudo_label_id:int):
        for pseudo_label in self.pseudo_label_superpixels:
            if cluster in pseudo_label:
                return False
            
        best_sim = 0
        best_neighbour = None
        for neighbour_index in cluster.neighbours:
            neighbour = self.slic.clusters[neighbour_index]
            neighbour_features = neighbour.features(self.gradient, self.slic.image)
            similarity = self.content_similarity(features, neighbour_features)
            
            if similarity > best_sim:
                best_sim = similarity
                best_neighbour = neighbour

        if best_neighbour in self.pseudo_label_superpixels[pseudo_label_id]:
            return True
        else:
            return False

    def refine(self):
        neighbours_list = self.get_neighbours() #list of lists of clusters
        for pseudo_label_index in range(len(self.masks_in_progress)):
            print(F"Refining pseudolabel {pseudo_label_index}")
            if not self.masks_in_progress[pseudo_label_index]:
                continue
            new_cluster = self.argmax_similarity(self.pseudo_label_superpixels[pseudo_label_index],
                                                pseudo_label_index, 
                                                neighbours_list[pseudo_label_index])
            if new_cluster is None:
                self.masks_in_progress[pseudo_label_index] = 0
                continue
            self.pseudo_label_superpixels[pseudo_label_index].append(new_cluster)

    def save(self, image_name):
        masks = np.zeros_like(self.gradient[:,:,0])
        if self.masks is None:
            masks.save(F"{image_name[:-4]}_refined.png")
            return
        
        for pseudolabel in self.pseudo_label_superpixels:
            for superpixel in pseudolabel:
                masks += superpixel.mask(self.gradient.shape[1], self.gradient.shape[0])
        mask = cv2.morphologyEx(masks, cv2.MORPH_OPEN, np.ones((OPENING_KERNEL,OPENING_KERNEL)))
        output = Image.fromarray(np.uint8(mask * 255))
        output.save(F"{image_name[:-4]}_refined.png")

        




    def complex_refinement(self, image_name:str):
        #TODO move the image load here
        pil_image = Image.open(image_name)
        print("Image loaded")
        self.slic.make_superpixels(pil_image)
        print("Superpixel made")
        self.slic.save()
        self.get_masks(pil_image, DEBUG) #sets self.og_masks and returns if there is any segmentation
        if self.masks is None:
            self.save(image_name)
            return

        print("Mask separated and saved")
        self.fit_pseudolabel_to_superpixels()
        print("Pseudolabel fitted to superpixels")
        self.gradient = self.slic.make_gradient()
        print("gradient")
        
        while np.sum(self.masks_in_progress) > 0:
            print("refining")
            self.refine()

        self.save(image_name)
            

def refined_pseudolabels(model, images_folder, masks_folder):
    pass



if __name__ == "__main__":
    DEBUG = Image.open("4.gif")
    slic = SLIC(N_CLUSTERS)
    slic("4.png", False)

    # model = None #TODO load model
    # manager = RegionManager(model, slic)
    # #TODO loop to generate 
    # manager.complex_refinement("4.png")