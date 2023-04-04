import argparse
from pathlib import Path
import torch
import wandb
from tqdm import tqdm
from os import path

#My libraries
from utils.augmentation_dataset import AugmentedDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet.Unet_model import UNet1, UNet0, UNet2, UNetPP


# VERSION = "6.8"
path_to_data = Path('/datagrid/personal/grundda')
dir_images = path.join(path_to_data,Path('data/images1channel/'))
dir_masks = path.join(path_to_data, Path('data/masks/'))
dir_val_images = path.join(path_to_data,Path('data/val_images1channel/'))
# model_name = F"MODEL_6.8_.pth"

EPOCHS = 70
BATCH_SIZE = 2
N_CHANNELS = 1
N_CLASSES = 2


# OPTIMIZER ARGS
LEARNIG_RATE = 1e-5
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999

# SCHEDULER
PATIENCE = 6



def train(net, learning_rate):
    
    #######################################
    #Datasets

    #Traning dataset : Val dataset = 9:1
    train_dataset = AugmentedDataset(dir_images, dir_masks, training_set=True)
    val_dataset = AugmentedDataset(dir_val_images, dir_masks, training_set=False)
    n_train = len(train_dataset)

    #train_fraction_dataset_len = int(n_train * val_percent)
    #rest_of_dataset_len = n_train - train_fraction_dataset_len
    #train_dataset, rest_of_dataset = torch.utils.data.random_split(whole_train_dataset, [train_fraction_dataset_len, rest_of_dataset_len], generator = torch.Generator().manual_seed(0))

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, drop_last=True, batch_size = BATCH_SIZE)


    #######################################
    #
    optimizer = torch.optim.AdamW(net.parameters(), lr = learning_rate, weight_decay=WEIGHT_DECAY, betas = (BETA1, BETA2))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=PATIENCE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,120)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0

    ######################################
    #Wandb info
    experiment = wandb.init(project='bc_project', entity="gruny", reinit = True)#Reinit = True is for multiple runs in one script
    experiment.config.update(dict(epochs=EPOCHS, 
                                    batch_size=BATCH_SIZE, 
                                    # learning_rate_init=learning_rate,
                                    architecture = str(net), 
                                    optimizer=optimizer,
                                    # scheduler = scheduler,
                                    # criterion = criterion,
                                    weight_decay = WEIGHT_DECAY,
                                    momentum = MOMENTUM,
                                    augment = "Translation",
                                    #patience = PATIENCE,
                                    # version = VERSION
                                    #volume = val_percent
                                    ))
    experiment.define_metric("DSC", summary ="max")
    #######################################
    #Training loop
    best_dict = dict()
    best_dice = 0.0
    last_improvement = 0
    for epoch in range(1, EPOCHS+1):
        net.train()
        epoch_loss = 0
        #val_score_list = []
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{EPOCHS}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                images = images.to(device=DEVICE, dtype=torch.float32)
                true_masks = true_masks.to(device=DEVICE, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=True):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) + 3* dice_loss(torch.argmax(masks_pred, dim = 1).float(), true_masks.float())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()


                #############################
                # Visible bar and wandb

                pbar.update(images.shape[0])
                global_step += 1
                # epoch_loss += loss.item()
                #experiment.log({
                #        'train loss': loss.item(),
                #       'epoch': epoch
                #  })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                    
                #############################
                # Evaluation round
                division_step = (n_train // (10 * BATCH_SIZE))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(net, val_loader, DEVICE)
                        #train_dataset.training_set = False #Turns off the augmentation
                        #training_score = evaluate(net, train_loader, DEVICE)
                        #train_dataset.training_set = True #Turns the augmentation back on
                        #val_score_list.append(val_score)
                    # print(val_score_init)
                        experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'DSC': val_score,
                                #'training_Dice': training_score,
                                'epoch': epoch,
                            })
                        if val_score > best_dice:
                            best_dict = net.state_dict().copy()
                            best_dice = val_score
                            last_improvement = epoch
                        #else:
                           # if epoch-last_improvement >= 25 and epoch >= 40:
                            #    this_run = dict(net = best_dict, name = experiment.name, score = best_dice)
                             #   experiment.finish()
                              #  print(best_dice)
                               # return this_run
            scheduler.step()


    this_run = dict(net = best_dict, name = experiment.name, score = best_dice)
    experiment.finish()   
    print(best_dice)
    return this_run    

if __name__ == '__main__': 
    for i in range(3):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet1(n_channels=N_CHANNELS, n_classes=N_CLASSES,mean = 0.4506735083369092, std = 0.23919170057270236)#, depth= 8, base_kernel_num = 32)
        net.to(device= DEVICE)
        try:
            run = train(net=net, learning_rate = 1e-5)
        except Exception as err:
            print(err)
        try:
            torch.save(run["net"], F"/datagrid/personal/grundda/models/{run['name']}_{run['score']:3f}.pth")
        except Exception as err:
            print(err)
        torch.cuda.empty_cache()

