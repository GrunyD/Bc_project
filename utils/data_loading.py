import logging
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from os import path, listdir
from PIL import Image
from pathlib import Path
# from confidence_aware_model_funcs import adjust_fft_amplitude
import math
import numpy as np

IMAGE_WIDTH = 1068
IMAGE_HEIGHT = 847

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
    def __init__(self, p:float, scale:tuple, whole_image_scale:float = 1):
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
        self.transform = transforms.RandomResizedCrop((int(whole_image_scale*IMAGE_HEIGHT), int(whole_image_scale*IMAGE_WIDTH)), scale = self.scale, antialias = True)
    
    def forward(self, tensor):
        # tensor = transforms.RandomResizedCrop((int(IMAGE_HEIGHT*scale), 1068),scale = self.scale)(tensor)
        tensor = self.transform(tensor)
        tensor[-1,:,:] = tensor[-1,:,:] > 0 #Makes mask binary again if the bilinear interpolation breaks it
        return tensor

###########################################################################
#                                                                         #
#  Custom functions implementing backward pass for consistency learning   #
#                                                                         #
###########################################################################
"""
In consistency learning we want model to make consistent predictions, now matter if the 
image is flipped, rotated, translated or deformed in other way
However, we have to implement the backward pass to ensure that optimizer is optimizing the 
right weights.
Torchvision tensor function do not have backward pass and we have to crate one

See PyTorch Docs:
    Example: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

The forward mathod input has to be torch.Tensor so the autograd thinks we need gradient. I tried
passing it as tuple (with additional arguments) and the backward method was never called.
So correct implemetation is 
def forward(ctx, input:torch.Tensor, *args):
    ...

The backward method has to return as many outputs as there were inputs (ctx excluded). But 
could return None if the gradient is not needed.
"""
class ConsistencyHFlipFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return F.hflip(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return F.hflip(grad_output)
    
class ConsistencyHFlip(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ConsistencyHFlipFunction.apply(x)
    
class ConsistencyVFlipFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return F.vflip(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return F.vflip(grad_output)
    
class ConsistencyVFlip(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ConsistencyVFlipFunction.apply(x)
    
class ConsistencyRotationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, angle):
        ctx.angle = angle
        return F.affine(input, float(angle), translate = (0,0), scale = 1, shear = (0,0))
    
    @staticmethod
    def backward(ctx, grad_output):
        angle = ctx.angle
        return F.affine(grad_output, float(-1 * angle), translate = (0,0), scale = 1, shear = (0,0)), None

class ConsistencyRotation(torch.nn.Module):
    def __init__(self, angle_range):
        super().__init__()
        self.angle = torch.rand(1)*angle_range*2 - angle_range

    def forward(self, x):
        return ConsistencyRotationFunction.apply(x, float(self.angle))


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

    def __init__(self, images_dir:str, scale:float = 1):
        self.labeled_images_dir = images_dir
        self.images_path = images_dir
        self.scale = scale

        self.labeled_ids = [path.splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        # self.labeled_ids = [idx for idx in self.labeled_ids if int(idx) <= 521]
        if not self.labeled_ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.labeled_ids)} examples')

    def __len__(self):
        return len(self.labeled_ids)
    
    def get_image_name(self, name):
        img_file = list(self.images_path.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        return img_file[0]
        
    def read_image(self, name, grayscale = True):
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
        if self.scale !=  1:
            image = F.resize(image, [int(image.size()[-2]*self.scale), int(image.size()[-1]*self.scale)], antialias = True)

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
    def __init__(self, labeled_images_dir:str, masks_dir:str, scale:float = 1, enable_augment:bool = True, enable_fourier_augment:bool = False, **kwargs):
        super().__init__(labeled_images_dir, scale)

        self.masks_dir = masks_dir
        self.masks_path = masks_dir
        self.enable_augment = enable_augment
        self.fourier_augment = enable_fourier_augment

    def __str__(self):
        return 'Base set'


    def get_mask(self, name):
        # mask_file = list(self.masks_path.glob(name + '.*'))

        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        name = path.join(self.masks_path, F'{name}.png')

        mask_pil = Image.open(name)
        mask = torch.Tensor(mask_pil.getdata()).reshape(1,mask_pil.size[1], mask_pil.size[0])
        if self.scale != 1:
            mask = F.resize(mask, [int(mask.size()[-2]*self.scale), int(mask.size()[-1]*self.scale)], antialias = True)

        return (mask > 0).to(torch.int64)

    def augmentation_pipeline(self, img:torch.Tensor, mask:torch.Tensor):
        if self.enable_augment:
            pipeline = transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                MyRandomCrop(p = 0.3, scale = 0.5, whole_image_scale=self.scale),
                MyRandomAffine(translation_p=0.3, rotation_p=0.3, translation_range=(0.2, 0.2), max_rotation=20),
                MyRandomElastic(p = 0.2, alpha = (50,150))
            ])
            tensor = torch.vstack((img, mask))

            augmented_tensor = pipeline(tensor)
            return augmented_tensor[:-1,:,:], augmented_tensor[-1,:,:]
            # return augmented_tensor
        else:
            return img, mask
        
    def process_image_mask(self, image, mask, name):
        if mask is None:
            mask = torch.zeros_like(image)
            clas = torch.tensor(-1)
        else:
            assert image.size()[-2:] == mask.size()[-2:], \
                f'Image and mask {name} should be the same size, but are {image.size()} and {mask.size()}'
            clas = (torch.sum(mask) > 0).to(torch.uint8)
        
         
        image, mask = self.augmentation_pipeline(image, mask)
        mask = torch.squeeze(mask).to(torch.uint8)
        

        
        return {
            'image': image.float().contiguous(),
            'mask': mask.long().contiguous(),
            'class': clas.long().contiguous()
        }

    @staticmethod
    def adjust_fft_amplitude(main_image:torch.Tensor, adjusting_image:torch.Tensor, mask:torch.Tensor, alpha:float = 0.5)->torch.Tensor:
        """
        If A is amplitude specter of main_image and A' is amplitude specter of adjusting_image, then the formula given in the paper
        (Enhancing Pseudo Label Quality for Semi-Supervised Domain-Generalized Medical Image Segmentation, 2022)
        is following:

        A_new = A*(1-alpha)*(1-mask) + A'* alpha * mask

        Notice that only amplitude is adjusted, not the phase, thus the structure of image will not be changed
        """
        amplitude = lambda ff: torch.sqrt(ff.real**2 + ff.imag**2)
        
        ff_main_image = torch.fft.fftshift(torch.fft.fft2(main_image))
        ff_adjusting_image = torch.fft.fftshift(torch.fft.fft2(adjusting_image, s = (main_image.size(-2), main_image.size(-1))))

        new_amplitude_multiplier = amplitude(ff_adjusting_image)/amplitude(ff_main_image)

        assert alpha > 0 and alpha < 1, "Alpha has to be bigger than 0"
        # assert beta > 0, "Alpha has to be bigger than 0"
        ff_adjusted_image = ff_main_image * (1-mask)  + ff_main_image  * mask * new_amplitude_multiplier
        return amplitude(torch.fft.ifft2(torch.fft.ifftshift(ff_adjusted_image)))
    
    def get_image(self, name):
        image = super().get_image(name)
        if not self.fourier_augment:
            return image
        torch.manual_seed(42)
        second_image = super().get_image(self.get_name(int(torch.rand(1)*len(self))))
        # print(name)
        mask = torch.zeros_like(image)
        safe_frquencies = 0.48
        height_limit = int(image.size(-2)*safe_frquencies)
        # height_limit = 1
        lenght_limit = int(image.size(-1)*safe_frquencies)
        mask[:,height_limit:-height_limit, lenght_limit:-lenght_limit] = 1
        # print(torch.sum(mask))
        mask = mask.to(torch.uint8)
        return self.adjust_fft_amplitude(image, second_image, mask)

    
    def get_item(self, name, mask_exists = True):
        mask = self.get_mask(name) if mask_exists else None
        image = self.get_image(name)
        return self.process_image_mask(image, mask, name)

    def get_name(self, idx):
        """
        It is defined here so get image can call it when using fourier transform augmentation
        In more next datasets the get_name method gets overwritten so get_image can get image even from 
        unlabeled images
        """
        return self.labeled_ids[idx]

    def __getitem__(self, idx):
        name = self.get_name(idx)
        item = self.get_item(name)
        item['name'] = name
        return item

class ConsistencyDataset(BaseDataset):
    def __init__(self, labeled_images_dir:str, masks_dir:str, scale:float = 1, perturb_prob:float|None = 0.7, **kwargs):
        """
        perturb_prob: Probability of adding perturbation, If None the whole Perturbation is ignored and it acts like BaseDataset
                        The option to suppres the pipeline is here for pseudolabels dataset. We might want to use only pseudolabels
                        and maybe pseudolabels with consistency learning. Inheriting from multiple classes could be too complicated
                        maintain, thus normal pseudolabels will be achieved by suppressing consistency
        """
        super().__init__(labeled_images_dir, masks_dir,scale, **kwargs)
        self.perturb_prob = perturb_prob

    def __str__(self):
        return "Consistency supervised set"
    
    @staticmethod
    def collate_pipeline(batch):
        """
        Pipeline Sequential can not be collated into batch by default collate
        When this dataset is passed to dataloader, this function has to be 
        passed as collate_fn keyword argument
        """
        collate = torch.utils.data.default_collate
        
        if batch[0].get('pipeline') is None:
            return collate(batch)
        
        pipelines = []
        for d in batch:
            pipelines.append(d['pipeline'])
            del d['pipeline']

        batch = collate(batch)
        batch['pipeline'] = pipelines
        return batch
                
    def get_transforms(self, max_rotation_range = 20):
        """
        Each transform is added with the same probability and all of them change their parameters
        everytime new pipeline is created
        """
        pipeline = []
        potential_transforms = [ConsistencyHFlip(),
                                ConsistencyVFlip(),
                                ConsistencyRotation(max_rotation_range)
                                ]
        for transform in potential_transforms:
            if torch.rand(1) <= self.perturb_prob:
                pipeline.append(transform)

        return torch.nn.Sequential(*pipeline)
    
    def get_perturbated_image_and_pipeline(self, image):
        pipeline = self.get_transforms()
        with torch.no_grad():
            perturbated_image = pipeline(image)
        return dict(perturbated_image = perturbated_image, pipeline = pipeline)
    
    def get_item(self, name, mask_exists = True):
        """
        Rewrite get_item function of BaseDataset
        This class does not have its own __getitem__, BaseDataset's is called, which get the name of the file and then calls get_item
        This function gets called. It calls BaseDatasets get_item and adds perturbated_image and pipeline into the dict
        """
        return_dict = super().get_item(name, mask_exists)
        if self.perturb_prob is not None:
            return_dict.update(self.get_perturbated_image_and_pipeline(return_dict['image']))
        return return_dict

###########################################################################
#                                                                         #
#                       Semisupervised Datasets                           #
#                                                                         #
###########################################################################


class SemiSupervisedDataset(ConsistencyDataset): #Should not be used as stand-alone -> only for inheritance
    def __init__(self, labeled_images_dir:str, masks_dir:str, unlabeled_images_dir:str, **kwargs):
        super().__init__(labeled_images_dir, masks_dir, **kwargs)


        self.unlabeled_images_dir = unlabeled_images_dir
        self.unlabeled_ids = [path.splitext(file)[0] for file in listdir(unlabeled_images_dir) if not file.startswith('.')][:len(self.labeled_ids)]

        if not self.unlabeled_ids:
            raise RuntimeError(f'No input file found in {unlabeled_images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.unlabeled_ids)} examples')

    def __len__(self):
        return len(self.labeled_ids) + len(self.unlabeled_ids)
    
    def __str__(self):
        return "Semisupervised"

class SemiSupervisedConsistencyDataset(SemiSupervisedDataset):
    def __str__(self):
        return F"{super().__str__()} Consistency"
    
    def get_name(self, idx):
        if idx >= len(self.labeled_ids):
            self.images_path = self.unlabeled_images_dir
            name = self.unlabeled_ids[idx - len(self.labeled_ids)]
        else:
            self.images_path = self.labeled_images_dir
            name = self.labeled_ids[idx]
        return name


    def __getitem__(self, idx):
        name = self.get_name(idx)
        if idx >= len(self.labeled_ids):
            return self.get_item(name, mask_exists = False)
            
        else:
            return self.get_item(name)
            
class PseudoLablesDataset(SemiSupervisedDataset):
    def __init__(self, labeled_images_dir: str, masks_dir: str, unlabeled_images_dir:str, pseudolabels_dir:str, perturb_prob = None, **kwargs):
        super().__init__(labeled_images_dir, masks_dir, unlabeled_images_dir, perturb_prob = perturb_prob, **kwargs)
        self.pseudolabel_dir = pseudolabels_dir
        
    def __str__(self):
        return F'Pseudolabels{"with consistency" if self.perturb_prob else ""}'
    
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

class FourierCotrainDataset(SemiSupervisedDataset):
    def __init__(self, labeled_images_dir: str, masks_dir: str, unlabeled_images_dir: str, **kwargs):
        super().__init__(labeled_images_dir, masks_dir, unlabeled_images_dir, **kwargs)
        self.enable_augment = False
        self.fourier_augment = True

    def __str__(self):
        return "Fourier cotraining"

    def get_name(self, idx):
        if idx >= len(self.labeled_ids):
            self.images_path = self.unlabeled_images_dir
            name = self.unlabeled_ids[idx - len(self.labeled_ids)]
        else:
            self.images_path = self.labeled_images_dir
            name = self.labeled_ids[idx]
        return name

    def get_item_with_fourier(self, name):
        fourier_image = self.get_image(name)
        self.fourier_augment = False
        normal_image = self.get_image(name)
        # mask = torch.zeros_like(image).to(torch.uint8)
        # image = torch.vstack((image, second_image))
        # image, mask = self.augmentation_pipeline(image, mask)
        # image, fourier_image = torch.unsqueeze(image[0], dim = 0), torch.unsqueeze(image[1], dim = 0)
        self.fourier_augment = True


        return {"image": normal_image, "fourier_image": fourier_image, "class": -1}
        

    def __getitem__(self, idx):
        name = self.get_name(idx)
        if idx >= len(self.labeled_ids):
            return self.get_item_with_fourier(name)
        else:
            return self.get_item(name)

class DANDataset(SemiSupervisedDataset):
    def __init__(self, labeled_images_dir: str, masks_dir: str, unlabeled_images_dir: str, **kwargs):
        super().__init__(labeled_images_dir, masks_dir, unlabeled_images_dir, **kwargs)

    def __getitem__(self, idx):
        return_dict = dict()
        if idx >= len(self.labeled_ids):
            name = self.unlabeled_ids[idx - len(self.labeled_ids)]
            self.images_path = self.unlabeled_images_dir
            return_dict['class'] = 0

        else:
            name = self.labeled_ids[idx]
            self.images_path = self.labeled_images_dir
            return_dict['class'] = 0
        return_dict['image'] = self.get_image(name)
        return return_dict
    
###########################################################################
#                                                                         #
#                           Testing this file                             #
#                                                                         #
###########################################################################

if __name__ == '__main__':
    pass


    """
    The following code tests naive consistency training dataset
    It gets image, perturbated image and pipeline. 
    Perturbated image and image should be different. After you put image through pipeline,
    they should be the same
    The number of choosing image is arbitrary and you can choose any number you want
    """
    data = '/home.stud/grundda2/.local/data'
    images = path.join(data, 'test_images')
    unlabeled_images = path.join(data, 'val_images')
    masks = path.join(data, 'masks')

    dataset = ConsistencyDataset(images, masks, enable_augment = True, perturb_prob=1)
    d = dataset[88]
    print("They are different:", torch.sum(torch.abs(d["perturbated_image"] - d['image'])))
    print("Perturbated to being same: ",torch.sum(torch.abs(d["perturbated_image"] - d["pipeline"](d['image']))))
    a = torch.rand(1, requires_grad = True)
    loss = torch.sum(torch.abs(a* d["perturbated_image"] - d["pipeline"](a*d['image'])))
    loss.backward()
    # with torch.no_grad():
    #     a = torch.sum(d['image'] - )
