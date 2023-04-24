"""
This script is meant to calculate the mean and standard deviation of all 
given images. It was written for grayscale images of this bachelor project as
all images did not have rectangle shape and had to be padded. But depending 
on the source the padding was different for each image.

Several type of paddings but all of them are in corners -> we use floodfill by cv2 to filter them out
"""
#0.4593777512924429
#std0.05667971439080055
from tqdm import tqdm
import cv2
import os
import numpy as np
from PIL import Image

DIRS = ['/home.stud/grundda2/.local/data/images/']

n = 0
for dir in DIRS:
    n += len(os.listdir(dir))



def calculate(Stat:str, mean:float = 0):
    total_pixels = 0
    pixel_values = 0

    if mean == 0:
        func = lambda x:np.sum(x)
    else:
        func = lambda x: np.sum((x - mean)**2)

    with tqdm(total = n, desc=F'Calculating {Stat}', unit = 'img') as outerpbar:
        for dir in DIRS:
            dir_name = dir.split('/')[-1]
            with tqdm(total = len(os.listdir(dir)), desc= F'Calculating {Stat} {dir_name}', unit='img') as innerpbar:
                for filename in os.listdir(dir):
                    filename = os.path.join(dir, filename)
                    # print(filename)
                    try:
                        pil_image = Image.open(filename)
                        image = np.array(pil_image.getdata(), dtype=np.float32)
                        if np.max(image) >= 1:
                            image = image/255
                        
                        channels = image.size //(pil_image.size[1]*pil_image.size[0])
                        image = np.reshape(image, (pil_image.size[1], pil_image.size[0], channels))

                        
                            
                    except IOError as err:
                        pass

                    mask = None
                    # for point in ((0,0), (0, image.shape[1]-1), (image.shape[0]-1, 0), (image.shape[0]-1, image.shape[1]-1)):
                    #     if image[point[0], point[1], 0] == 255:
                    #         continue
                    point = (0,0)
                    _, _, mask, _ = cv2.floodFill(image, mask, point, newVal=1)

                    mask = mask[1:-1,1:-1] == 0
                    pixel_values += func(image[mask])#np.sum(image[mask])
                    total_pixels += np.sum(mask)


                    outerpbar.update(1)
                    innerpbar.update(1)

    return pixel_values/total_pixels



mean = calculate("mean")
print(F"Mean calculated: {mean}")
std = calculate("std", mean)
print(F"Std calculated: {std}")

