import os
import argparse
from tqdm import tqdm

PATH_TO_DATA = "/home.stud/grundda2/.local/data"
PATH_TO_VAL = os.path.join(PATH_TO_DATA, 'val_images')
PATH_TO_IMAGES = os.path.join(PATH_TO_DATA, 'images')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--dataset', '-d', metavar='E', type=int, default=1, help='Which validation dataset should be used. Rest will be used as training data.')
    return parser.parse_args()

def new_val_set(number:int):
    assert number >=1 and number <=9, F"Number of dataset has to be between 1 and 9 (both included), your number is {number}"
    
    for filename in tqdm(os.listdir(PATH_TO_VAL)):
        os.rename(os.path.join(PATH_TO_VAL, filename), os.path.join(PATH_TO_IMAGES, filename))

    with open(F'validation_set{number}.txt', "r") as f:
        for filename in tqdm(f.readlines()):
            filename = filename[:-1]
            os.rename(os.path.join(PATH_TO_IMAGES, filename), os.path.join(PATH_TO_VAL, filename))
            
number = get_args().dataset
print(number)
new_val_set(number)