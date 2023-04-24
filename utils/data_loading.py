import logging
import torch
from torchvision import transforms
from os import path, listdir
from PIL import Image
from pathlib import Path
from .confidence_aware_model_funcs import adjust_fft_amplitude

"""
To change parameters of specific transforms, go to func AUGMENTATION PIPELINE
"""
###########################################################################
#                                                                         #
#                Transformation classes for augmentation                  #
#                                                                         #
###########################################################################
"""
Toechvisions Random apply is not satisfactory for the purpose of these classes. 
We have to handle augmenting segmentation mask in a proper way. Thus the basic class to inherit
from is MyRandomTransform. It defines __call__ which checks for probability and decides if 
forward function of specific transofrm should be applied. All specific transforms define only 
__init__ and forward method
"""

class MyRandomTransform(torch.nn.Module):
    """
    Creates a class to inherit from
    Children classes then defines only forward method which is called from __call__ method
    with given probability p
    """
    def __init__(self, p:float):
        super().__init__()
        assert isinstance(p, (float, int))
        assert p >= 0
        if p > 1:
            assert p / 100 <= 1
            self.p = p/100
        else:
            self.p = p

    def forward(self, tensor):
        raise NotImplementedError

    def __call__(self, tensor):
        if torch.rand(1) <= self.p:
            return self.forward(tensor)
        else:
            return tensor
###########################################################################

class MyRandomElastic(MyRandomTransform):
    def __init__(self, p:float, alpha:tuple):
        super().__init__(p)
        try:
            assert alpha[1] - alpha[0] >= 0
            assert alpha[0] >= 0
            self.alpha = tuple(map(float,alpha))
        except (IndexError, TypeError):
            assert isinstance(alpha, (float, int))
            self.alpha = (float(alpha), float(alpha))

    def forward(self, tensor):
        alpha = self.alpha[0] + torch.rand(1)*(self.alpha[1] - self.alpha[0])
        transform = transforms.ElasticTransform(float(alpha))
        tensor = transform(tensor)
        tensor[-1,:,:] = tensor[-1,:,:] > 0
        return tensor

class MyRandomGaussianNoise(MyRandomTransform):
    """
    Adds gaussian noise to a picture (mu is mean and sigma standard deviation), nothing to the mask. 
    """
    def __init__(self, p, sigma:float, mu = 0):
        super().__init__(p)

        if isinstance(sigma, (int, float)):
            self.sigma = (abs(sigma), abs(sigma))
        elif isinstance(sigma, (list, tuple)):
            assert sigma[0] <= sigma[1] and sigma[0] >= 0
            self.sigma = sigma
        assert isinstance(mu, (int, float))
        self.mu = float(mu)

    def forward(self, tensor):
        sigma = self.sigma[0] + (self.sigma[1]-self.sigma[0])*torch.rand(1)
        noise = torch.normal(self.mu,float(sigma), (1, 847, 1068))
        if tensor.size()[0] >1:
            tensor[:-1, :, :] += noise #leaves the mask out
            return tensor
        else:
            new_tensor = tensor + noise
            return new_tensor

        

class MyRandomAffine(torch.nn.Module):
    """
    Applies translation and rotation, each with its own given probability p
    """
    def __init__(self, translation_p: float, rotation_p:float, 
                    translation_range:tuple, max_rotation):
        assert isinstance(translation_p, (int, float))
        assert isinstance(rotation_p, (float, int))
        self.rotation_p = rotation_p
        self.translation_p = translation_p
        self.translation_range = translation_range
        self.max_rotation = max_rotation

    def forward(self, tensor):
        degrees = self.max_rotation if torch.rand(1) <= self.rotation_p else 0
        translation_range = self.translation_range if torch.rand(1) <= self.translation_p else None
        transform = transforms.RandomAffine(degrees, translation_range)
        return transform(tensor)

    def __call__(self, tensor):
        return self.forward(tensor)

class MyRandomGammaCorrection(MyRandomTransform):
    """
    Applies Gamma correction with probability p
    Gamma is chosen randomly from given range
    """
    def __init__(self, gamma_range: tuple, p:float):
        super().__init__(p)
        assert gamma_range[1] >= gamma_range[0]
        assert gamma_range[0] > 0
        self.low_gamma = gamma_range[0]
        self.high_gamma = gamma_range[1]

    def forward(self, tensor):
        gamma = self.low_gamma + torch.rand(1)*(self.high_gamma - self.low_gamma)
        tensor = tensor ** gamma
        return tensor

class MyRandomGaussianBlur(MyRandomTransform):
    """
    Applies Gaussian blur with probability p
    Chooses kernel at random from given range
    """
    def __init__(self, p:float, kernel_range: tuple, sigma: tuple = (1.0, 2.0)):
        super().__init__(p)
        assert kernel_range[0] < kernel_range[1]
        self.low_kernel = kernel_range[0]
        self.high_kernel = kernel_range[1]
        
        self.sigma = sigma


    def forward(self, tensor):
        kernel = torch.randint(self.low_kernel, self.high_kernel, (1,))[0]
        kernel = kernel + 1 if kernel % 2 == 0 else kernel
        if len(tensor.size()) > 2 and tensor.size()[0] >1 :
            tensor[:-1,:,:] = transforms.GaussianBlur((kernel, kernel), self.sigma)(tensor[:-1,:,:])
        else:
            tensor = transforms.GaussianBlur((kernel, kernel), self.sigma)(tensor)
        return tensor
    
class MyRandomCrop(MyRandomTransform):
    """
    Crops the original image with random size and at random location and then rezise it 
    to original shape. All with probability p.
    """
    def __init__(self, p:float, scale:tuple):
        super().__init__(p)
        try:
            assert scale[1]-scale[0] >= 0
            assert scale[0] >= 0
            assert scale[1] <= 1
            self.scale = scale
        except TypeError:
            assert isinstance(scale, float)
            assert scale >= 0 and scale <= 1
            self.scale = (scale, 1.0)
    
    def forward(self, tensor):
        tensor = transforms.RandomResizedCrop((847, 1068),scale = self.scale)(tensor)
        tensor[-1,:,:] = tensor[-1,:,:] > 0 #Makes mask binary again if the bilinear interpolation breaks it
        return tensor


###########################################################################
#                                                                         #
#                Custom dataset with augmentation pipeline                #
#                                                                         #
###########################################################################

class ImageDataset(torch.utils.data.Dataset):
    """
    Only for loading images
    It is a standalone class for generating pseudolabels
    """

    def __init__(self, images_dir:str):
        self.labeled_images_dir = images_dir
        self.images_path = images_dir

        self.labeled_ids = [path.splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.labeled_ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.labeled_ids)} examples')

    def __len__(self):
        return len(self.labeled_ids)
    
    def get_image_name(self, name):
        img_file = list(self.images_path.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        return img_file[0]
        
    @staticmethod
    def read_image(name, grayscale = True):
        """
        If grayscale is true, image with only one channnel is returned, no matter how many channels it has
        f grayscale is false, grayscale images are scale to have 3 channel, multichannel images remain unchanged 
        e.g. colors and even alpha channel remains
        """
        image_pil = Image.open(name)
        image = torch.Tensor(image_pil.getdata())
        if len(image.size()) > 1:
            image = image.reshape(image_pil.size[1], image_pil.size[0],image.size()[1])/255
            image = image.permute((2, 0, 1))
            if grayscale:
                image = image[:1, :, :]
        else:
            image = image.reshape(image_pil.size[1], image_pil.size[0])/255
            image = image.unsqueeze(0)
            if not grayscale:
                image = torch.vstack((image, image, image))

        # image = image.unsqueeze(0)
        return image
    
    def get_image(self, name):
        # name = self.get_image_name(name)
        name = F'{name}.png'
        name = path.join(self.images_path, name)
        image = self.read_image(name)
        return image
    
    def __getitem__(self, idx):
        name = self.labeled_ids[idx]
        image = self.get_image(name)
        return image.float().contiguous()



class BaseDataset(ImageDataset):
    """
    Implementing basic data loading of masks and puts masks and images toghther
    Contains the augmentation pipeline
    """
    def __init__(self, labeled_images_dir:str, masks_dir:str, enable_augment:bool = False):
        super().__init__(labeled_images_dir)

        self.masks_dir = masks_dir
        self.masks_path = masks_dir
        self.enable_augment = enable_augment

    def __str__(self):
        return 'Basic'


    def get_mask(self, name):
        # mask_file = list(self.masks_path.glob(name + '.*'))

        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        name = path.join(self.masks_path, F'{name}.png')
        mask_pil = Image.open(name)
        mask = torch.Tensor(mask_pil.getdata()).reshape(1,mask_pil.size[1], mask_pil.size[0])
        return mask//255

    def augmentation_pipeline(self, img, mask):
        if self.enable_augment:
            
            tensor = torch.vstack((img, mask))
            pipeline = transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                #MyRandomGammaCorrection((0.6, 1.4), 0.3),
                #MyRandomGaussianBlur((7,31), 0.3),
                MyRandomCrop(p = 0.3, scale = 0.5),
                MyRandomAffine(translation_p=0.4, rotation_p=0.4, translation_range=(0.2, 0.2), max_rotation=20),
                MyRandomElastic(p = 0.2, alpha = (50,150))
            ])
            augmented_tensor = pipeline(tensor)
            return augmented_tensor[:-1,:,:], augmented_tensor[-1,:,:]
        else:
            return img, mask
        
    def process_image_mask(self, image, mask, name):
        assert image.size() == mask.size(), \
            f'Image and mask {name} should be the same size, but are {image.size()} and {mask.size()}'
        
        image, mask = self.augmentation_pipeline(image, mask)
        mask = torch.squeeze(mask)
        clas = (torch.sum(mask) > 0).to(torch.uint8)


        return {
            'image': image.float().contiguous(),
            'mask': mask.long().contiguous(),
            'class': clas.long().contiguous()
        }
    
    def get_item(self, name):
        image = self.get_image(name)
        mask = self.get_mask(name)
        return self.process_image_mask(image, mask, name)

    def __getitem__(self, idx):
        name = self.labeled_ids[idx]
        return self.get_item(name)
        

class NaivePseudoLablesDataset(BaseDataset):
    def __init__(self, labeled_images_dir: str, masks_dir: str, unlabeled_images_dir:str, pseudolabel_dir:str, enable_augment: bool = False):
        super().__init__(labeled_images_dir, masks_dir, enable_augment)

        self.unlabeled_images_dir = unlabeled_images_dir
        self.pseudolabel_dir = pseudolabel_dir
        self.unlabeled_ids = [path.splitext(file)[0] for file in listdir(unlabeled_images_dir) if not file.startswith('.')]

        if not self.unlabeled_ids:
            raise RuntimeError(f'No input file found in {unlabeled_images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.unlabeled_ids)} examples')

    def __str__(self):
        return 'Naive pseudolabels'
    def __len__(self):
        return len(self.labeled_ids) + len(self.unlabeled_ids)
    
    def get_name(self, idx):
        if idx >= len(self.labeled_ids):
            name = self.unlabeled_ids[idx - len(self.labeled_ids)]
            self.images_path = self.unlabeled_images_dir
            self.masks_path = self.pseudolabel_dir
        else:
            name = self.labeled_ids[idx]
            self.images_path = self.labeled_images_dir
            self.masks_path = self.masks_dir
        return name

    
    def __getitem__(self, idx):
        name = self.get_name(idx)
        return self.get_item(name)
    

class FourierTransformsPseudolabels(NaivePseudoLablesDataset):
    def get_image(self, name):
        image1 = super().get_image(name)
        index = int(torch.rand(1) * self.__len__())

        name2 = self.get_name(index)
        image2 = super().get_image(name2)

        mask = torch.zeros_like(image2)
        mask[23:823,64:1024] = 1

        return adjust_fft_amplitude(image1, image2, mask)
    
    def __str__(self):
        return 'Fourier Transform pseudolabels'
















class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir:str, masks_dir:str, enable_augment:bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.enable_augment = enable_augment

        self.ids = [path.splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    def augmentation_pipeline(self, img, mask):
        if self.enable_augment:
            tensor = torch.vstack((img, mask))
            pipeline = transforms.Compose([
                #transforms.RandomVerticalFlip(),
                #transforms.RandomHorizontalFlip(),
                #MyRandomGammaCorrection((0.6, 1.4), 0.3),
                #MyRandomGaussianBlur((7,31), 0.3),
                MyRandomAffine(translation_p=0, rotation_p=0.4, translation_range=(0.2, 0.2), max_rotation=20)
            ])
            augmented_tensor = pipeline(tensor)
            return augmented_tensor[:-1,:,:], augmented_tensor[-1,:,:]
        else:
            return img, mask

        

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        img_pil = Image.open(img_file[0])
        img = torch.Tensor(img_pil.getdata()).reshape(img_pil.size[1], img_pil.size[0])/255
        #img = img.float()[0,:,:]/255
        img = img.reshape((1,*img.size()))
        mask_pil = Image.open(mask_file[0])
        mask = torch.Tensor(mask_pil.getdata()).reshape(1,mask_pil.size[1], mask_pil.size[0])

        assert img_pil.size == mask_pil.size, \
            f'Image and mask {name} should be the same size, but are {img_pil.size} and {mask_pil.size}'

        img, mask = self.augmentation_pipeline(img, mask)
        mask = torch.squeeze(mask)

        return {
            'image': img.float().contiguous(),
            'mask': mask.long().contiguous()
        }
    

