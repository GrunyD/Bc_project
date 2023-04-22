import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

from torchvision import transforms
import numpy as np
#from utils.data_loading import BasicDataset
from PIL import Image
import os



def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


####################################################################
#
#       Dice score on testing images
#
####################################################################

def get_predicted_mask(filename, net, scale_factor, out_threshold, device):
    full_image = Image.open(filename)

    img = torch.from_numpy(BasicDataset.preprocess(full_image, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((full_image.size[1], full_image.size[0])),
        transforms.ToTensor()
        ])

    full_mask = tf(probs.cpu()).squeeze()
    return (full_mask > out_threshold).numpy()

def evaluate_model(path_to_test_images, path_to_true_masks, net, scale_factor,device, out_threshold):
    net.eval()

    dice = 0
    for image in os.listdir(path_to_test_images):
        pred_mask = get_predicted_mask(os.path.join(path_to_test_images,image), net, scale_factor, out_threshold, device)
        pred_mask = np.argmax(pred_mask,0)
        
        im_frame = Image.open(os.path.join(path_to_true_masks, F"{image[:-4]}.gif"))
        true_mask = np.array(im_frame.getdata()).reshape((pred_mask.shape))
        dice += dice_coeff(torch.from_numpy(pred_mask), torch.from_numpy(true_mask), reduce_batch_first=False)

    return dice/len(os.listdir(path_to_test_images))
