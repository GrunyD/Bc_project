import logging
import torch
from torchvision import transforms
from os import path, listdir
from PIL import Image
from pathlib import Path

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


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir:str, masks_dir:str, training_set):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.training_set = training_set

        self.ids = [path.splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    def augmentation_pipeline(self, img, mask):
        if self.training_set:
            tensor = torch.vstack((img, mask))
            pipeline = transforms.Compose([
                #MyRandomElastic(p = 0.2, alpha = (50,150))
                #transforms.RandomVerticalFlip(),
                #transforms.RandomHorizontalFlip(),
               # MyRandomGammaCorrection((0.6, 1.4), 0.3),
               # MyRandomGaussianBlur(0.3,(7,31)),
                MyRandomAffine(translation_p=0.4, rotation_p=0, translation_range=(0.2, 0.2), max_rotation=20)
                #MyRandomCrop(p = 0.3, scale = 0.5),
                #MyRandomGaussianNoise(p = 0.3, sigma = (0.01, 0.1))
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
