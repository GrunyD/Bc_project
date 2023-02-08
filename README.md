# Bc_project
This repo is representing the code used in bachelor project. Topic of the project is to segment dental restorations in bitewing x-ray images. However it could be used to segment any images.
As an input it takes grayscale images and segmentation masks(binary mask).
It is based on U-net architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## How to run
python3 my_train.py

All values to set are in the top of the file. You have to link paths to training data, validation data and masks. After hours of training it saves the model in the same folder as the my_train.py file

## Predict
python3 predict.py "image_name"
or 
python3 predict.py 
