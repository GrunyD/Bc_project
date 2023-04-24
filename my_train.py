import argparse
from pathlib import Path
import torch
import wandb
from tqdm import tqdm
from os import path

#My libraries
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet.Unet_model import  Model, UNet, Classification_UNet
import unet
from utils.data_loading import ImageDataset, NaivePseudoLablesDataset, BaseDataset, FourierTransformsPseudolabels
from PIL import Image


# VERSION = "6.8"
path_to_data = Path('/datagrid/personal/grundda/data/')
get_path = lambda dir: path.join(path_to_data, Path(dir))

DIRS = dict(
    labeled_images = get_path('images1channel/'),
    masks = get_path('masks/'),
    unlabeled_images = get_path('unlabeled_images/'),
    pseudolabels = get_path('pseudolabels/'),
    val_images = get_path('vals_images1channel/')
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 50
BATCH_SIZE = 2

#MODEL ARGS
MODEL_TYPE = UNet
ENABLE_AUGMENT = True
PRETRAINED_MODEL = None
N_CHANNELS = 1
N_CLASSES = 2
MEAN = 0.4593777512924429
STD = 0.23807501840974526
DEPTH = 5
PSEUDO_LABELS = None
NAME = 'NewUNET'


# OPTIMIZER ARGS
LEARNIG_RATE = 1e-5
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999

# SCHEDULER
PATIENCE = 6

def generate_pseudolabels(net:Model):
    unlabeled_images_set = ImageDataset(DIRS['unlabeled_images'])
    images_loader = torch.utils.data.DataLoader(unlabeled_images_set, shuffle = False)
    pseudolabels = DIRS["pseudolabels"]

    for index, image in enumerate(images_loader):
        name = unlabeled_images_set.labeled_ids[index]
        prediction = net.predict(image)
        image = Image.fromarray(prediction)
        image.save(F'{path.join(pseudolabels, name)}.png')


def train(net, trainset,  val_set, experiment):
    train_loader = torch.utils.data.DataLoader(trainset, shuffle = True, batch_size =BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_set)
    n_train = len(trainset)
    
    #######################################
    #
    optimizer = torch.optim.AdamW(net.parameters(), lr = LEARNIG_RATE, weight_decay=WEIGHT_DECAY, betas = (BETA1, BETA2))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=PATIENCE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,120)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0


    #######################################
    #Training loop
    best_dict = dict()
    best_dice = 0.0
    last_improvement = 0
    for epoch in range(1, EPOCHS+1):
        net.train()

        # if epoch%20 == 0 and isinstance(trainset, NaivePseudoLablesDataset):
        #     generate_pseudolabels(net)


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
                # pbar.set_postfix(**{'loss (batch)': loss.item()})

                    
                #############################
                # Evaluation round
                division_step = (n_train // (10 * BATCH_SIZE))
                if division_step > 0:
                    if global_step % division_step == 0:
                        with torch.no_grad():
                            val_score = evaluate(net, val_loader, DEVICE)
                            print(val_score)
                            experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'DSC': val_score,
                                    'epoch': epoch,
                                })
                            if val_score > best_dice:
                                best_dict = net.state_dict().copy()
                                best_dice = val_score
                                last_improvement = epoch
            scheduler.step()


    this_run = dict(net = best_dict, name = experiment.name, score = best_dice)
    experiment.finish()   
    print(best_dice)
    return this_run    

def define(model_type:type, pretrained_net:str = None, enable_augment:bool = True, pseudo_labels = None):
#################################################
#   Defining net 
    net = model_type(n_channels=N_CHANNELS, n_classes=N_CLASSES,mean = MEAN, std = STD, depth = DEPTH)
    net.to(device= DEVICE)
    if pretrained_net is not None:
        net.load_state_dict(torch.load(pretrained_net, map_location=DEVICE))

#################################################
#   Defining Data Loaders
    if pseudo_labels is None:
        trainset = BaseDataset(DIRS["labeled_images"], DIRS['masks'], enable_augment=enable_augment)
    else:
        #Based on the index of pseudolabels given, different dataset is chosen and used
        classes = [NaivePseudoLablesDataset, FourierTransformsPseudolabels]
        chosen_class = classes[pseudo_labels]

        trainset = chosen_class(
            DIRS['labeled_images'],
            DIRS['masks'],
            DIRS['unlabeled_images'],
            DIRS['pseudolabels'],
            enable_augment= enable_augment
        )
    val_set = BaseDataset(DIRS['val_images'], DIRS['masks'], enable_augment=False)

#################################################
#   Defining Wandb
    experiment = wandb.init(project='bc_project', entity="gruny", reinit = True)#Reinit = True is for multiple runs in one script
    experiment.config.update(dict(epochs=EPOCHS, 
                                    batch_size=BATCH_SIZE, 
                                    architecture = str(net),
                                    dataset = str(trainset),
                                    semisupervised = str(False  if PSEUDO_LABELS is None else True),

                                    ))
    experiment.define_metric("DSC", summary ="max")



    return net, trainset, val_set, experiment


def main():
    net, trainset, valset, experiment = define(model_type = MODEL_TYPE, 
                                            pretrained_model = PRETRAINED_MODEL,
                                            enable_augment=ENABLE_AUGMENT, 
                                            pseudo_labels=PSEUDO_LABELS)
    
    run = train(net, trainset, valset, experiment)
    name = run['name'] if run['name'] is not None else NAME
    torch.save(run["net"], F"/datagrid/personal/grundda/models/{name}_{run['score']:3f}.pth")
    torch.cuda.empty_cache()


if __name__ == '__main__': 
    # for i in range(3):
    #     main()
    main()

