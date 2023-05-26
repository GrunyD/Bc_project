import argparse
from pathlib import Path
import torch
import wandb
from tqdm import tqdm
import numpy as np
from os import path
import copy

#My libraries

from evaluate import evaluate
from unet import Unet_model
import unet
# from utils.loss_function import ClassSegLoss
# from utils.data_loading import ImageDataset, NaivePseudoLablesDataset, BaseDataset, FourierTransformsPseudolabels
from utils import loss_function, data_loading
from PIL import Image

global run 


path_to_data = Path('/home.stud/grundda2/.local/data/')
get_path = lambda dir: Path(path.join(path_to_data, dir))

DIRS = dict(
    labeled_images = get_path('images/'),
    masks = get_path('masks/'),
    unlabeled_images = get_path('unlabeled_images/'),
    # pseudolabels = get_path('pseudolabels/'),
    pseudolabels = get_path('pseudolabels2'),
    val_images = get_path('val_images'),
    models = Path('/datagrid/personal/grundda/models'),
    val_unlabeled_images = get_path('val_unlabeled_images')
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 50


#MODEL ARGS
MODEL_TYPE = Unet_model.UNet
PRETRAINED_MODEL =None#'royal-durian-472.pth'#'resilient-sponge-470.pth'#"gentle-oath-445.pth"#['genial-aardvark-367.pth','dainty-dragon-432.pth']
N_CHANNELS = 1 #How many channels the images have
N_CLASSES = 2 #How many classes are we predicting (including background)
MEAN = 0.4593777512924429   #Expected value of training Dataset
STD = 0.23807501840974526   #Standard deviation of training Dataset
DEPTH = 4       # How many downsampling layers are used
BASE_KERNEL = 64
CONV_DOWNSAMPLE = False # If True, net uses double strided convolution to downsample instead of maxpool
BILINEAR = True

#DATASET ARGS
ENABLE_AUGMENT = False #If True, augmentation is turned on - edit in utils/data_loading.py -> BaseDataset
SEMISUPERVISED = None # 'Pseudolabels', 'Consistency','Confidence' None
FOURIER_TRANSFORM = False #If True, augmentation via adjusting magnitude of fourier spectrum is applied
CONSISTENCY = False# If True, consistency loss function is applied and dataloader gets perturbated images
                    # See utils utils/data_loading.py -> ConsistencyDataset
PERTURBATED_PROBABILITY = 0.8 # Goes with consistency training
BATCH_SIZE = 3
SCALE = 1    #Scaling images to fit to memory of GPU

#LOSS
SUPERVISED_LOSS_FUNCTION = loss_function.ClassSegLoss(x_weight=1, dice_weight=1, class_weight=0, iou_weight=0)
CONSISTENCY_LOSS_FUNCTION = loss_function.ConsistencyLoss(difference_weight=5, dice_weight=1, iou_weight=0)
CONFIDENCE_LOSS_FUNCTION = loss_function.ConfidenceAwareLoss(weight = 3)

# OPTIMIZER ARGSWANDB OFFLINE

LEARNIG_RATE = 1e-5
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999

# SCHEDULER
PATIENCE = 6

def define():
#################################################
#   Defining net 
    net = MODEL_TYPE(n_channels=N_CHANNELS, n_classes=N_CLASSES,mean = MEAN, std = STD, depth = DEPTH, conv_downsample = CONV_DOWNSAMPLE, bilinear=BILINEAR)
    net.to(device= DEVICE)
    if PRETRAINED_MODEL is not None:
        if isinstance(PRETRAINED_MODEL, (tuple, list)):
            net.unet1.load_state_dict(torch.load(path.join(DIRS['models'],PRETRAINED_MODEL[0]), map_location=DEVICE))
            net.unet2.load_state_dict(torch.load(path.join(DIRS['models'],PRETRAINED_MODEL[1]), map_location=DEVICE))
        else:
            net.load_state_dict(torch.load(path.join(DIRS['models'],PRETRAINED_MODEL), map_location=DEVICE))
    # second_net = Unet_model.UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES,mean = MEAN, std = STD, depth = DEPTH, conv_downsample = CONV_DOWNSAMPLE)
    # second_net.to(device= DEVICE)
    # second_net.load_state_dict(torch.load(path.join(DIRS['models'],'genial-aardvark-367.pth'), map_location=DEVICE))
    # net.encoder = second_net.encoder
#################################################
#   Defining Data Loaders
    val_set = data_loading.BaseDataset(DIRS['val_images'], DIRS['masks'], enable_augment=False)
    kwargs = dict(
                labeled_images_dir = DIRS['labeled_images'],
                masks_dir = DIRS['masks'],
                eneable_augment = ENABLE_AUGMENT,
                enable_fourier_augment = FOURIER_TRANSFORM,
                perturb_prob = PERTURBATED_PROBABILITY if CONSISTENCY else None,
                scale = SCALE,
                base_kernel = BASE_KERNEL)
    

    if SEMISUPERVISED is None:
        #Consistency dataset is inheriting from Base dataset, if None is passed as perturb_prob, it becomes Basedataset
        # trainset = data_loading.ConsistencyDataset(DIRS["labeled_images"], DIRS['masks'], enable_augment=ENABLE_AUGMENT,enable_fourier_augment=FOURIER_TRANSFORM, perturb_prob=perturb_prob)
        trainset = data_loading.ConsistencyDataset(**kwargs)
        # trainset = data_loading.BaseDataset(**kwargs)
    elif SEMISUPERVISED == 'Adversial':
        kwargs.update({'unlabeled_images_dir': DIRS['unlabeled_images']})
        trainset = data_loading.DANDataset(**kwargs)
        if MODEL_TYPE == Unet_model.Classificator:
           kwargs['labeled_images_dir'] = DIRS['val_images']
           kwargs['unlabeled_images_dir'] = DIRS['val_unlabeled_images']
           val_set = data_loading.DANDataset(**kwargs)

    elif SEMISUPERVISED == 'Consistency':
        kwargs['unlabeled_images_dir'] = DIRS['unlabeled_images']
        kwargs['perturb_prob'] = PERTURBATED_PROBABILITY
        trainset = data_loading.SemiSupervisedConsistencyDataset(**kwargs)
        
    elif SEMISUPERVISED == 'Pseudolabels':
        #Based on the index of pseudolabels given, different dataset is chosen and used
        assert PRETRAINED_MODEL is not None, "You can not train on pseudolabels without pretrained model"
        kwargs.update({'unlabeled_images_dir': DIRS['unlabeled_images'], 'pseudolabels_dir':DIRS['pseudolabels']})
        trainset = data_loading.PseudoLablesDataset(**kwargs)
        generate_pseudolabels(net)

    elif SEMISUPERVISED == 'Confidence':
        kwargs['unlabeled_images_dir'] = DIRS['unlabeled_images']
        trainset = data_loading.FourierCotrainDataset(**kwargs)
    else:
        raise ValueError(F'SEMISUPERVISED value has to be either "Confidence", "Consistency", "Pseudolabels" or None, but yours is {SEMISUPERVISED}')

    

    collate = trainset.collate_pipeline if CONSISTENCY else torch.utils.data.default_collate
    batch_size = 1 if CONSISTENCY else BATCH_SIZE
    
    train_loader = torch.utils.data.DataLoader(trainset, shuffle = True, collate_fn = collate, batch_size = batch_size, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, num_workers = 8)

#################################################
#   Defining Wandb
    
    experiment = wandb.init(project='bc_project', entity="gruny", reinit = True)#Reinit = True is for multiple runs in one script
    experiment.config.update(dict(epochs=EPOCHS, 
                                    batch_size=BATCH_SIZE, 
                                    architecture = str(net),
                                    dataset = str(trainset),
                                    semisupervised = str(False  if SEMISUPERVISED is None else True),

                                    ))
    experiment.define_metric("DSC", summary ="max")
    experiment.define_metric("DSC true positive", summary ="max")
    experiment.define_metric("Classification", summary ="max")



    return net, train_loader, val_loader, experiment


def generate_pseudolabels(net:Unet_model.Model):
    unlabeled_images_set = data_loading.ImageDataset(DIRS['unlabeled_images'])
    images_loader = torch.utils.data.DataLoader(unlabeled_images_set, shuffle = False, num_workers=8)
    pseudolabels = DIRS["pseudolabels"]
    net.eval()
    with tqdm(total=len(unlabeled_images_set), unit='img') as pbar:
        for index, image in enumerate(images_loader):
            name = unlabeled_images_set.labeled_ids[index] #It is called labeled ids inside the Dataset
            image = image.to(device = DEVICE)
            prediction = net.predict(image).get('segmentation')
            assert prediction is not None, "The model does not return segmentation"
            prediction = prediction[1].detach().cpu().numpy()
            image = Image.fromarray(np.uint8(prediction)*255)
            image.save(F'{path.join(pseudolabels, name)}.png')
            pbar.update(1)
    print('Pseudolabels generated')
    net.train()


def train(net, train_loader,  val_loader, experiment):
    n_train = len(train_loader.dataset)
    
    #######################################
    #
    optimizer = torch.optim.AdamW(net.parameters(), lr = LEARNIG_RATE, weight_decay=WEIGHT_DECAY, betas = (BETA1, BETA2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100, eta_min=1e-7)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    global_step = 0

    print(n_train)
    #######################################
    #Training loop
    best_dict = dict()
    best_dice = 0.0
    last_improvement = 0
    for epoch in range(1, EPOCHS+1):
        net.train()

        if epoch%20 == 0 and SEMISUPERVISED == 'Pseudolabels':
            generate_pseudolabels(net) #net.eval and net.train is inside the func


        with tqdm(total=n_train, desc=f'Epoch {epoch}/{EPOCHS}', unit='img') as pbar:
            for batch in train_loader:
                clas = batch['class']
                
                
                    
                images = batch['image']
                    
                

                assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                

                # if clas ==1:
                images = images.to(device=DEVICE, dtype=torch.float32)
                
                with torch.cuda.amp.autocast(enabled=True):

                    prediction = net(images)
                    loss = 0

                    
                    if SEMISUPERVISED == 'Confidence':
                        
                        o_pred1, o_pred2 = prediction
                        true_classes = batch.get('class')
                        fourier_image = batch.get('fourier_image')

                        if fourier_image is None:
                            true_masks = batch.get('mask')
                            true_masks = true_masks.to(device=DEVICE, dtype=torch.long)
                            true_classes = true_classes.to(device =DEVICE, dtype = torch.long)
                            loss = loss + SUPERVISED_LOSS_FUNCTION(o_pred1, true_masks, true_classes)
                            loss = loss + SUPERVISED_LOSS_FUNCTION(o_pred2, true_masks, true_classes)
                            # print("Supervised loss:", loss)
                        else:
                            fourier_image = batch.get('fourier_image')
                            fourier_image = fourier_image.to(device = DEVICE, dtype =torch.float32)
                            f_pred1, f_pred2 = net(fourier_image)
                            loss = loss + CONFIDENCE_LOSS_FUNCTION(o_pred1, f_pred1, o_pred2, f_pred2)
                            print("Confidence loss: ", loss)  
                    elif SEMISUPERVISED == 'Adversarial':
                        loss = loss + SUPERVISED_LOSS_FUNCTION(prediction)     
                    else:
                        if CONSISTENCY:  
                            pipelines = batch.get('pipeline') 
                            per_image = batch['perturbated_image']
                            per_image = per_image.to(device = DEVICE, dtype=torch.float32)
                            per_prediction = net(per_image)  
                            loss = loss + CONSISTENCY_LOSS_FUNCTION(prediction, per_prediction, pipelines)

                        true_classes = batch.get('class')
                        # if true_classes != -1:
                        true_masks = batch.get('mask')
                        true_masks = true_masks.to(device=DEVICE, dtype=torch.long)
                        true_classes = true_classes.to(device =DEVICE, dtype = torch.long)
                        loss = loss + SUPERVISED_LOSS_FUNCTION(prediction, true_masks, true_classes)

                    



                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()


                #############################
                # Visible bar and wandb
                pbar.update(BATCH_SIZE)
                global_step += BATCH_SIZE
            
                    
                #############################
                # Evaluation round
                
        with torch.no_grad():


            val_score = evaluate(net, val_loader, DEVICE)
            print(F"DSC: {val_score[0]}\nDSC true positive: {val_score[1]}")

            # if prediction.get('segmentation') is not None:
            experiment.log({
                        # 'learning rate': optimizer.param_groups[0]['lr'],
                        'DSC': val_score[0],
                        'DSC true positive': val_score[1],
                        'epoch': epoch,
                    })
            train_loader.dataset.enable_augment = False
            train_score = evaluate(net, train_loader, DEVICE)
            train_loader.dataset.enable_augment = True
            print(F"train DSC: {train_score[0]}\ntrain DSC true positive: {train_score[1]}")
            experiment.log({
                "train DSC": train_score[0],
                "train DSC true positive": train_score[1],
                'epoch':epoch
            })
            if val_score[1] > best_dice:
                best_dice = val_score[1]
                best_dict = copy.deepcopy(net.state_dict())
                                
                            # if prediction.get('classification') is not None:
                            #     experiment.log({'Classification': val_score[2],'epoch':epoch})
            
        scheduler.step()
            

        
    run = dict(net = best_dict, name = experiment.name)
    experiment.finish()   
    return run    



def main():
    net, train_loader, val_loader, experiment = define()
    run = train(net, train_loader, val_loader, experiment)
    name= str(net)
    if run is not None:
        name = run.get('name')
    
    torch.save(run["net"], F"/datagrid/personal/grundda/models/{name}.pth")
    torch.cuda.empty_cache()


if __name__ == '__main__': 
    main()

