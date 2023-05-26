# Bachelor project
This repo is representing the code used in bachelor project. Topic of the project is to segment dental restorations in bitewing x-ray images. However it could be used to segment any images.
It is based on U-net architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## Structure
Whole project was implemented using PyTorch
### unet
Contains files which implements layers and necessary for U-Net and ensambles it togthether. It also implements U-Net ++.

### utils

#### Data loading
This file implements a lot of functions to do proper data loading for different learning methods. Imageset only loads images to iterate over them without corresponding masks (comes handy for pseudolabel generation).
Base dataset implements loading images with corresponding mask. It also implements data augmentation. Those are implemented at the top of the file.
Consistency dataset implements dataloading for consistency learning. With image and mask it also returns perturbated image and the given pipeline which
is later used for calculating loss function. The pipeline components are implemented above datasets and they take care of backpropagation (which is needed for consistency learning in order for gradient to be computed correctly)
If perturb prob is None, it is Basic dataset. (Yes they could be mergerd into one)
Pseudolabels, COtrain and semisup Consistency dataset work with unlabeled data.
DANDataset is was not tested. It was prepared for adversarial learning.

#### Loss function
Implements loss functions for different deep learning tasks. It takes in the desired weights to balance them properly.

#### Normalization 
Is not used during training. It is meant for computing mean and std before training for normlazitaion of the picture.

### Data
For anyone following this work and trying to compare the results. These were indexes of files used in cross validation. During Stage 2 (described in my work) I used validation_set1 to evaluate. The rest was used for cross validation.

Yolo labels is zip file containing around 1000 text files for yolo training as it requires. 

complete annotation is notebook created to put annotations toghether. In this task we used CVAT to annotate images. However there were several stages of datasets. Thus in cvat they are 

### Yolo handling
Only contains file which evaluates yolo trained model. For anyone to come to have easier job.

## Requirements and istallation
#### TQDM
You do not have to install it but is is a nice tool to see that your code is running and how fast. 

#### Wandb
Tool for mapping your runs and hyperparameters. Can only recommend.

I used python 3.10.4 downloaded form python.org

#### Pytorch

## How to run
python3 my_train.py

All values to set are in the top of the file. You have to link paths to training data, validation data and masks. Set all paths correctly. Wandb is nice and free tool for mapping your runs. I recommend using it. This file rely on wandb to be defined as it saves model according to wandb run name.

## Predict
python3 predict.py "absolute path to image"
or 
python3 predict.py "absolute path to folder"

Creates another image/folder right next to image/folder with png segmentation masks.

## Evaluation
Watch out for the difference between evaluation.py and evaluate.py
Evaluation.py is meant to evaluate saved model on diven validation or test dataset.

Evaluate.py is for evaluation during training
(Yes it should be one file)

With evaluation.py you can create grids to compare models and it computes all kind of metrics. The file is a little bit messy at this time.


## Contact
If you are cvut student following this work, do not hesitate to contact me on telegram (@grunysek) for tips and tricks about this code, whole work or how to begin and what to look out for.
