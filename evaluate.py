import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

from torchvision import transforms
import numpy as np
#from utils.data_loading import BasicDataset
from PIL import Image
import os
from unet import UNet
from utils import data_loading



def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score_all = 0
    dice_score_true_positive = 0
    classification = 0
    num_true_positive = 0
    # poz_neg = dict(TP = 0, TN = 0, FP = 0, FN = 0)


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, true_class = batch['image'], batch['mask'], batch['class']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        true_class = true_class.to(device, dtype = torch.long)
        # if true_class ==-1:
        #     continue

        with torch.no_grad():
            # predict the mask
            prediction = net(image)
            mask_pred = prediction.get('segmentation')
            class_pred = prediction.get('classification')
            dice_score = None
            # convert to one-hot format
            # if net.n_classes == 1:
            #     if mask_pred is not None:
            #         mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                
            #     # compute the Dice score
            #         dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            #     if class_pred is not None:
            #         class_pred = (F.sigmoid(class_pred)>0.5).float()
            #     else:
            #         class_pred = 0
            
            if mask_pred is not None:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            
            if class_pred is not None:
                class_pred = int(torch.argmax(class_pred))
            else:
                if mask_pred is not None:
                    class_pred = int(torch.sum(mask_pred) > 0)
                else:
                    class_pred = 0
        # key = F"{'T' if class_pred == true_class else 'F'}{'P' if class_pred == 1 else 'N'}"
        # poz_neg[key] += 1
        classification += int(class_pred == true_class[0])
        if dice_score is not None:
            dice_score_all += dice_score

            if true_class[0] == 1:
                num_true_positive += 1
                dice_score_true_positive += dice_score
            


    # return_dict = dict(
    #     total_dice = dice_score_all/num_val_batches,
    #     true_positive_dice = dice_score_true_positive/num_true_positive,
    #     accuracy = classification/num_val_batches
    #     precision = ,
    #     recall = 
    # )
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score_all / num_val_batches, dice_score_true_positive/(num_true_positive+ 1e-9), classification/num_val_batches

if __name__ == "__main__":
    MEAN = 0.4593777512924429   #Expected value of training Dataset
    STD = 0.23807501840974526   #Standard deviation of training Dataset
    DEPTH = 5   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(1, 2, MEAN, STD, DEPTH)
    net.load_state_dict(torch.load('/datagrid/personal/grundda/models/genial-aardvark-367.pth', map_location=device))
    net.to(device = device)

    dataset = data_loading.BaseDataset('/home.stud/grundda2/.local/data/test_images', '/home.stud/grundda2/.local/data/masks', enable_augment=False)
    loader = torch.utils.data.DataLoader(dataset)
    print(evaluate(net, loader, device))

####################################################################
#
#       Dice score on testing images
#
####################################################################

# def get_predicted_mask(filename, net, scale_factor, out_threshold, device):
#     full_image = Image.open(filename)

#     img = torch.from_numpy(BasicDataset.preprocess(full_image, scale_factor, is_mask=False))
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)
#     with torch.no_grad():
#         output = net(img)
#         probs = torch.sigmoid(output)[0]

#         tf = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((full_image.size[1], full_image.size[0])),
#         transforms.ToTensor()
#         ])

#     full_mask = tf(probs.cpu()).squeeze()
#     return (full_mask > out_threshold).numpy()

# def evaluate_model(path_to_test_images, path_to_true_masks, net, scale_factor,device, out_threshold):
#     net.eval()

#     dice = 0
#     for image in os.listdir(path_to_test_images):
#         pred_mask = get_predicted_mask(os.path.join(path_to_test_images,image), net, scale_factor, out_threshold, device)
#         pred_mask = np.argmax(pred_mask,0)
        
#         im_frame = Image.open(os.path.join(path_to_true_masks, F"{image[:-4]}.gif"))
#         true_mask = np.array(im_frame.getdata()).reshape((pred_mask.shape))
#         dice += dice_coeff(torch.from_numpy(pred_mask), torch.from_numpy(true_mask), reduce_batch_first=False)

#     return dice/len(os.listdir(path_to_test_images))
