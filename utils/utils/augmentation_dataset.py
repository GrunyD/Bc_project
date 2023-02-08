import logging
import torch
from torchvision import transforms
from os import path, listdir
from PIL import Image
from pathlib import Path


###########################################################################
#                                                                         #
#                Transformation classes for augmentation                  #
#                                                                         #
###########################################################################

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

class MyRandomGammaCorrection(torch.nn.Module):
    """
    Applies Gamma correction with probability p
    Gamma is chosen randomly from given range
    """
    def __init__(self, gamma_range: tuple, p:float):
        super().__init__()
        assert gamma_range[1] > gamma_range[0]
        assert gamma_range[0] > 0
        self.low_gamma = gamma_range[0]
        self.high_gamma = gamma_range[1]
        assert isinstance(p, (int, float))
        self.p = p

    def forward(self, tensor):
        if torch.rand(1) <= self.p:
            gamma = self.low_gamma + torch.rand(1)*(self.high_gamma - self.low_gamma)
            tensor = tensor ** gamma

        return tensor

    def __call__(self, tensor):
        return self.forward(tensor)

class MyRandomGaussianBlur(torch.nn.Module):
    """
    Applies Gaussian blur with probability p
    Chooses kernel at random from given range
    """
    def __init__(self, kernel_range: tuple, p:float, sigma: tuple = (1.0, 2.0)):
        super().__init__()
        assert kernel_range[0] < kernel_range[1]
        self.low_kernel = kernel_range[0]
        self.high_kernel = kernel_range[1]
        
        self.sigma = sigma

        assert isinstance(p, (float, int))
        self.p = p

    def forward(self, tensor):
        if torch.rand(1) <= self.p:
            kernel = torch.randint(self.low_kernel, self.high_kernel, (1,))[0]
            kernel = kernel + 1 if kernel % 2 == 0 else kernel
            tensor[:-1,:,:] = transforms.GaussianBlur((kernel, kernel), self.sigma)(tensor[:-1,:,:])
        return tensor

    def __call__(self, tensor):
        return self.forward(tensor)


###########################################################################
#                                                                         #
#                Custom dataset with augmentation pipeline                #
#                                                                         #
###########################################################################


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir:str, masks_dir:str):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.training_set = True

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
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                MyRandomGammaCorrection((0.6, 1.4), 0.3),
                MyRandomGaussianBlur((7,31), 0.3),
                MyRandomAffine(translation_p=0.5, rotation_p=0.3, translation_range=(0.1, 0.1), max_rotation=10)
            ])
            augmented_tensor = pipeline(tensor)
        return augmented_tensor[:-1,:,:], augmented_tensor[-1,:,:]

        

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        img_pil = Image.open(img_file[0])
        img = torch.Tensor(img_pil.getdata()).reshape(img_pil.size[1], img_pil.size[0],3).permute([2,0,1])
        img = img.float()[0,:,:]/255
        img = img.reshape((1,*img.size()))
        mask_pil = Image.open(mask_file[0])
        mask = torch.Tensor(mask_pil.getdata()).reshape(1,mask_pil.size[1], mask_pil.size[0])

        assert img_pil.size == mask_pil.size, \
            f'Image and mask {name} should be the same size, but are {img_pil.size} and {mask_pil.size}'

        img, mask = self.augmentation_pipeline(img, mask)

        return {
            'image': img.float().contiguous(),
            'mask': mask.long().contiguous()
        }