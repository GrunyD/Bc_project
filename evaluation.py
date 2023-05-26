import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from os import mkdir, path,listdir
from pathlib import Path
import json
from unet import Unet_model
import numpy as np
from utils.data_loading import BaseDataset, ImageDataset
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from pycocotools.coco import COCO

ALPHA = 0.2
LIGHT_GREY = 236
PADDING_WIDTH = 10
UPPER_ROW_PADDING = 280
LOWER_ROW_PADDING = 300
PICTURE_WIDTH = 1068
PICTURE_HEIGHT = 847

DATA = '/home.stud/grundda2/.local/data/'
MASKS = path.join(DATA, 'masks/')
VAL_IMAGES = path.join(DATA, 'val_images')
TEST_IMAGES = path.join(DATA, 'test_images')

def gif_read(path):
    im_frame = Image.open(path)
    return torch.ByteTensor(im_frame.getdata()).reshape(im_frame.size[1], im_frame.size[0],)

def to_image(mask):
    mask1 = 255*mask
    if len(mask.shape) == 2:
        mask2 = mask1.broadcast_to([3, *mask.shape]) 
    return mask2

def _TP_FP_FN(true_mask, prediction):
    true_mask = true_mask.to(torch.uint8)
    prediction = prediction.to(torch.uint8)
    TP = true_mask * prediction
    FP = prediction - TP
    FN = true_mask - TP
    return TP, FP, FN

def comparison_mask(true_mask, prediction):
    TP, FP, FN = _TP_FP_FN(true_mask, prediction)
    FP *= 2
    FN *= 3
    return TP + FP + FN

def comparison_picture(true_mask, prediction):

    TP, FP, FN = _TP_FP_FN(true_mask, prediction)
    zer = torch.zeros((3,TP.size(-2),TP.size(-1)), dtype=torch.uint8)
    zer[0] = FP
    zer[1] = TP
    zer[2] = FN
    return zer*200

def dice_score(true_mask, prediction):
    # assert true_mask.size()[-2:] == prediction.size()[-2:], F"{true_mask.size()}, {prediction.size()[-2:]}"
    # epsilon = 1e-6
    # true_mask = F.one_hot(torch.squeeze(true_mask)).permute(2, 0, 1)
    # inter = torch.sum(true_mask[1:,...]*prediction[1:,...])
    # sets_sum = torch.sum(prediction[1:,...]) + torch.sum(true_mask[1:,...])
    # return (2 * inter + epsilon) / (sets_sum + epsilon)
    return (2*torch.sum(true_mask*prediction) + 1e-7)/(torch.sum(true_mask) + torch.sum(prediction) + 1e-7)

def IOU(true_mask, prediction):
    assert true_mask.size()[-2:] == prediction.size()[-2:]
    epsilon = 1e-6
    true_mask = F.one_hot(torch.squeeze(true_mask)).permute(2, 0, 1)
    inter = torch.sum(true_mask[1:,...]*prediction[1:,...])
    union = torch.sum((prediction[1:,...] + true_mask[1:,...]) > 0)
    return (inter + epsilon) / (union + epsilon)


def image_with_segmentation(image, mask, color):
    # print(image.size())
    # print(mask.size())
    # if image.ndim != 3:
    #     # image = torch.expand_dims(image, 0)
    
    # print(image.device)
    # print(mask.device)
    image =image.to(torch.device('cpu'))
    mask = mask.to(torch.device('cpu'))
    return torchvision.utils.draw_segmentation_masks(image, mask > 0, alpha = ALPHA, colors = color)

# def image_with_predicted_mask(image, predicted_mask):
#     return torchvision.utils.draw_segmentation_masks(image, predicted_mask,alpha = ALPHA, colors = "cyan")

# def image_with_comparison_mask(image, true_mask, prediction):
#     comparison_mask = comparison_mask(true_mask, prediction)
#     return torchvision.utils.draw_segmentation_masks(image, comparison_mask,alpha = ALPHA, colors = ["green", "red", "blue"])

def add_padding(picture):
    picture = picture.permute([1,2,0])
    row = torch.ones([PADDING_WIDTH, 1068, 3]) *LIGHT_GREY
    col = torch.ones([847+2*PADDING_WIDTH, PADDING_WIDTH//2, 3]) * LIGHT_GREY
    picture = torch.vstack([row, picture, row])
    picture = torch.hstack([col, picture, col])
    return picture

def grid_shape(grid_padding, cols, rows):
    img_rows = 847
    img_cols = 1068
    return img_rows*rows + (rows+1)*grid_padding, img_cols*cols + (cols+1)*grid_padding

def my_grid(image, true_mask, prediction, kunt_prediction, grid_padding, padding):
    col_padding = padding

    
    if kunt_prediction is None:
        cols_num = 2
        grid_list = [image, 
                    image_with_segmentation(image, prediction, "cyan"),
                    image_with_segmentation(image, true_mask, "red"),
                    comparison_picture(true_mask, prediction),
                    ]
                    
    else:
        cols_num = 3
        grid_list = [image, 
                    image_with_segmentation(image, prediction, "cyan"),
                    image_with_segmentation(image, kunt_prediction,"yellow"),
                    image_with_segmentation(image, true_mask, "red"),
                    comparison_picture(true_mask, prediction),
                    comparison_picture(true_mask, kunt_prediction)
                    ]

    grid_size = grid_shape(grid_padding, cols_num, 2)

    grid =  torchvision.utils.make_grid(grid_list, padding=grid_padding, pad_value=236, nrow=cols_num).permute([1,2,0])
    
    cols = 255*torch.ones((grid_size[0] + LOWER_ROW_PADDING + UPPER_ROW_PADDING, col_padding, 3), dtype=torch.uint8)
    row_upper = 255*torch.ones((UPPER_ROW_PADDING, grid_size[1], 3), dtype=torch.uint8)
    row_lower = 255*torch.ones((LOWER_ROW_PADDING, grid_size[1], 3), dtype=torch.uint8)
    
    grid = torch.vstack((row_upper, grid, row_lower))
    grid = torch.hstack((cols, grid, cols))
    return grid.permute([2,0,1])

def get_image_ids(image_names, annotation_file):
    with open(annotation_file, "r") as read_file:
        data = json.load(read_file)
    images = data["images"]
    ret = {}
    for dic in images:
        if int(dic["file_name"][:-4]) in image_names:
            ret[dic["file_name"][:-4]] = dic["id"]
    return ret

    
def evaluation_picture(image, true_mask, prediction, kunt_prediction, path_to_export, file_name="", dice = '',picture_id="", model = ""):
    
    grid_padding = 30
    padding = 50
    black = (0,0,0)
    image = torchvision.transforms.ToPILImage()(my_grid(image, true_mask, prediction, kunt_prediction, grid_padding, padding))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("LEMONMILK-Light.otf", 60)

    # left_start = grid_padding + padding
    # upper_start = 50
    # text_gap = 100
    
    # HEADER
    #draw.text((left_start, upper_start), F"Model: {model}", (0,0,0), font = font)
    # draw.text((left_start, upper_start + text_gap), F"Dice: {dice:.3f}", (0,0,0), font = font)
    # draw.text((1500, upper_start), F"File name: {file_name}", (0,0,0), font = font)
    #draw.text((1500, upper_start + text_gap), F"Picture id: {picture_id}", (0,0,0), font = font)
    
    # LEGEND
    font_size = 60
    font = ImageFont.truetype("LEMONMILK-Light.otf", font_size)

    # dice = dice_score(true_mask, F.one_hot(torch.squeeze(prediction)).permute(2,0,1))
    rows = [UPPER_ROW_PADDING-font_size-10, UPPER_ROW_PADDING + 3*grid_padding + 2*PICTURE_HEIGHT]
    
    if kunt_prediction is not None:
        text_list = ["X-ray image (input)", "Grundfest model prediction","Kunt model prediction",
                    "Ground truth annotation","Prediction evaluation", "Predicition evaluation"]
        kunt_dice = dice_score(true_mask, kunt_prediction)
        cols = [grid_padding + padding, padding + 2*grid_padding + PICTURE_WIDTH,padding + 3*grid_padding + 2*PICTURE_WIDTH]

    else:
        text_list = ["X-ray image (input)", "Pseudolabels",
                    "Ground truth annotation","Evaluation"]
        cols = [grid_padding + padding, padding + 2*grid_padding + PICTURE_WIDTH]




    for indexr, row in enumerate(rows):
        for indexc , col in enumerate(cols):
            draw.text((col, row), text_list[indexr*len(cols) + indexc], black, font = font)

    draw.text((cols[0], rows[0] - 90), F"File name: {file_name}", (0,0,0), font = font)
    draw.text((cols[1], rows[0] - 90), F"Dice: {dice:.3f}", (0,0,0), font = font)
    if kunt_prediction is not None:
        draw.text((cols[2], rows[0] - 90), F"Dice: {kunt_dice:.3f}", (0,0,0), font = font)
    


    radius = 2
    rec_width = 40
    gap = 80
    y = rows[1] + gap
    x = cols[1] + gap
    draw.rounded_rectangle((cols[1], y, cols[1]+rec_width, y + rec_width), radius = radius, fill = (0, 255, 0))
    draw.text((x, y), "True positive", black, font = font)
    y += gap
    draw.rounded_rectangle((cols[1], y, cols[1]+rec_width, y + rec_width), radius = radius, fill = (255, 0, 0))
    draw.text((x, y), "False positive", black, font = font)
    y += gap
    draw.rounded_rectangle((cols[1], y, cols[1]+rec_width, y + rec_width), radius = radius, fill = (0, 0, 255))
    draw.text((x, y), "False negative", black, font = font)

    image.save(path.join(path_to_export,file_name))
    return dice

def predict(path_to_image, model, out_threshold):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_img = Image.open(path_to_image)
    # img = torch.from_numpy(BasicDataset.preprocess(full_img, 1., is_mask=False))
    img = ImageDataset.read_image(path_to_image)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = model(img)

    tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

    probs = F.softmax(output, dim=1)[0]
    full_mask = tf(probs.cpu()).squeeze()
    mask =  (full_mask > out_threshold).numpy()
    return torch.from_numpy(np.array(np.argmax(mask,0), dtype = np.uint8))


def morph_grid_search(model, path_to_images, path_to_masks):
    for image_name in listdir(path_to_images):
        path_to_image = path.join(path_to_images, F"{image_name}.png")
        true_mask = gif_read(path.join(path_to_masks, F"{image_name}.gif"))
        prediction = predict(path_to_image, model, 0.5)

        size = 60
        grid = np.zeros((size, size))
        for opening_kernel in range(1, size+1):
            for closing_kernel in range(1,size+1):
                op = Unet_model.Opening_and_Closing(opening_kernel, closing_kernel)
                new_grid = np.zeros(size, size)
                new_grid[opening_kernel][closing_kernel] = dice_score(true_mask, op(prediction))
        grid = grid + new_grid


    grid = grid/len(listdir(path_to_images))
    # print(grid)

def get_detected_segmentations(coco, name, prediction):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().to(torch.device('cpu')).numpy()
    elif isinstance(prediction, np.ndarray):
        pass
    else:
        raise TypeError("prediction has to be either numpy array or torch tensor")
    
    prediction = np.squeeze(prediction)
    
    for image in coco.dataset['images']:
        if image['file_name'][:-4]==name:
            image_id = image['id']
            break
    rests = 0
    found_rests = 0
    for ann in coco.dataset['annotations']:
        if ann['image_id']==image_id:
            rest = coco.annToMask(ann)
            rest_size = np.sum(rest)
            if np.sum(rest - prediction > 0)/rest_size <0.5:
                found_rests += 1
            rests += 1
    

def get_seg_from_annot(coco, name):
    for image in coco.dataset['images']:
        if image['file_name'][:-4]==name:
            image_id = image['id']
            break

    mask = np.zeros((847,1068))
    for ann in coco.dataset['annotations']:
        if ann['image_id']==image_id:
            mask += coco.annToMask(ann)

    mask = torch.tensor(mask>0)
    # mask = F.one_hot(mask.to(torch.long),2)
    # return {'segmentation':mask.permute(2,0,1)}
    return {'segmentation':mask}
    



def positives_negatives(prediction:int, true_class:int, pos_neg:dict):
    """
    Gets dict with True positive, True negative, False positive and False negative values and updates it
    """
    # print(prediction)
    # print(true_class)
    # print(prediction == true_class)
    key = F"{'T' if prediction == true_class else 'F'}{'P' if prediction else 'N'}"
    # print(key)
    pos_neg[key] += 1
    return pos_neg
    
def eval_model(net, images, masks, segmentation = True, threshold = 0.9, eval_pictures = False):
    """
    inputs:
        net:            torch.Module trained model - should output dict with 'segmentation' and posibly 'classification' keys
        images:         dir where are images for validation
        masks:          dir with ground truth masks, it supposes that they have the same name as images but are in different folder
        segmentation:   Some models could be only classifiers, then we have to pass this as False

    returns: 
            dict with following keys and values

        CLASSIFICATION
        accuracy:            percentage of how many images were classified correctly
        precision:           how many images of returned ones are actually returned correctly
        recall:              how many images of those which should be returned were returned
    
        if SEGMENTATION
        total_dice:         dice score calculated as mean over dice scores of each image
        overall_dice: dice score calculated as sum of all intersections of all images divied by sum of all masks areas
        true_positive_dice:   given we have good classification method, we validate how good segmentation is only on images   
                                    where actully is something to segment

        total_IOU:                Intersection over union as mean of this metric calculated for each image
        overall_IOU         IOU as sum of all intersection over sum of all unions
        true_positive_IOU   IOU as mean of IOU calculated for each image where is something to segment

        
    """
    dice_histogram = []
    total_IOU = 0
    total_dice = 0

    true_positive_IOU = 0
    true_positive_dice = 0
    true_positive_count = 0

    all_TP_pixels = 0
    all_FP_pixels = 0
    all_FN_pixels = 0

    pos_neg = dict(
        TP = 0,
        TN = 0,
        FP = 0,
        FN = 0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    val_set = BaseDataset(images, masks, enable_augment=False)
    loader = torch.utils.data.DataLoader(val_set, batch_size = 1,num_workers=8)
    if isinstance(net,torch.nn.Module):
        net = net.to(device = device)
        net.eval()

    for batch in tqdm(loader, total=len(val_set), unit='image'):

        

        if isinstance(net,torch.nn.Module):
            image = batch['image'].to(device = device)
            mask = batch['mask'].to(device = device)
            clas = batch['class'].to(device = device)
            with torch.no_grad():
                prob = net(image)
            seg_prob = F.softmax(prob['segmentation'],dim=1)
            predicted_mask = (torch.squeeze(seg_prob)[1,:,:] > threshold).to(torch.int64)
            # predicted_mask = F.one_hot(predicted_mask,2).permute(2,0,1)
        else:
            image = batch['image']
            mask = batch['mask']
            clas = batch['class']
            # print(clas)
            prob = get_seg_from_annot(net, batch['name'][0])
            predicted_mask = prob['segmentation'].to(torch.int)

        # if prob.get('segmentation') is not None:
        predicted_class = clas if batch['name'][0] != '632' else 0

        if predicted_class == 0:
            predicted_mask = torch.zeros((847, 1068)).to(device = device)
        

        inter = (mask*predicted_mask).to(torch.int)
        false_positive_sum = torch.sum((predicted_mask - inter) > 0)
        false_negative_sum = torch.sum((mask - inter) > 0)
        true_positive_sum  = torch.sum(inter)

        dice = (2*true_positive_sum + 1e-8)/(2*true_positive_sum + false_negative_sum + false_positive_sum + 1e-8)
        iou = (true_positive_sum+ 1e-8)/(true_positive_sum + false_positive_sum + false_negative_sum + 1e-8)
        
        
        total_dice += dice
        total_IOU += iou

        all_TP_pixels += true_positive_sum
        all_FP_pixels += false_positive_sum
        all_FN_pixels += false_negative_sum

        if clas:
            dice_histogram.append(dice.item())
            true_positive_dice += dice
            true_positive_IOU += iou
            true_positive_count += 1

        if eval_pictures:
            image = (image*255).to(torch.device('cpu'),torch.uint8)
            image = torch.broadcast_to(image[0], (3, image.size(2), image.size(3)))
            mask = (mask).to(torch.device('cpu'))
            predicted_mask = (predicted_mask).to(torch.device('cpu'))
            evaluation_picture(image, mask, predicted_mask, None, "/home.stud/grundda2/bc_project/Bc_project/evaluationpseudo/", file_name=F"{batch['name'][0]}.png",dice=dice)
    
        # class_prob = prob.get('classification')
        # if  class_prob is not None:
        #     class_prob = torch.squeeze(F.softmax(class_prob,dim=1).detach())
        #     predicted_class = int(class_prob[1] > threshold)
        # else:
        #     # print(predicted_mask.size())
            
        #     predicted_class = int(torch.sum(predicted_mask > threshold)>0)

        # # predicted_class = prediction.get('classification')
        # pos_neg = positives_negatives(predicted_class, clas, pos_neg)
        # # print("jou")

            # pbar.update(1)
    eps = 1e-8
    return_dict = dict(
        class_precision = (eps + pos_neg['TP'])/(pos_neg['TP'] + pos_neg['FP'] + eps),
        class_recall = (eps + pos_neg['TP'])/(pos_neg['TP'] + pos_neg['FN'] + eps),
        class_accuracy = (eps + pos_neg['TP'] + pos_neg['TN'])/(eps + pos_neg['TP'] + pos_neg['TN'] + pos_neg['FP'] + pos_neg['FN']),
    )
    if segmentation:
        return_dict.update(dict(
            total_dice = total_dice/len(val_set),
            overall_dice = 2*all_TP_pixels/(all_TP_pixels*2 + all_FN_pixels + all_FP_pixels),
            true_positive_dice = true_positive_dice/true_positive_count,

            total_IOU = total_IOU/len(val_set),
            overall_IOU = all_TP_pixels/(all_TP_pixels + all_FN_pixels + all_FP_pixels),
            true_positive_IOU = true_positive_IOU/true_positive_count,

            pixel_precision = all_TP_pixels/(all_TP_pixels + all_FP_pixels),
            pixel_recall = all_TP_pixels/(all_TP_pixels + all_FN_pixels),

            dice_histogram = dice_histogram


        ))
    return return_dict

def precision_recall_curve(precision_recall_dict):
    fig1, ax1= plt.subplots()
    fig2, ax2 = plt.subplots()
    for key in precision_recall_dict:
        arr = np.array(precision_recall_dict[key])
        threshold = arr[:, 0]
        accuracy = arr[:, 1]
        precision = arr[:, 2]
        recall = arr[:, 3]
        
        label_name = key
        ax1.plot(recall, precision, label = label_name)
        ax2.plot(threshold, accuracy, label = label_name)

    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    

    ax2.set_title('Correctly classified images')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Confidence threshold')
    

    ax1.legend(loc = 'lower left')
    ax2.legend(loc = 'lower left')

    fig1.savefig('Precision_recall_curve_U.png')
    fig2.savefig('Accuracy_U.png')

def precision_recall_on_threshold(precision_recall_dict):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for key in precision_recall_dict:
        arr = np.array(precision_recall_dict[key])
        threshold = arr[:, 0]
        rate = arr[:, 1]
        precision = arr[:, 2]
        recall = arr[:, 3]
        
        label_name = key
        ax1.plot(threshold, recall, label = label_name)
        ax2.plot(threshold, precision, label = label_name)

    ax1.set_title('Recall with respect to confidence threshold')
    ax1.set_xlabel('Confidence threshold')
    ax1.set_ylabel('Recall')

    ax2.set_title('Precision with respect to confidence threshold')
    ax2.set_xlabel('Confidence threshold')
    ax2.set_ylabel('Precision')

    ax1.legend(loc = 'lower left')
    ax2.legend(loc = 'upper left')

    fig1.savefig('Recall_wr_threshold_U.png')
    fig2.savefig('Precision_wr_threshold_U.png')

    

                
            
def get_PR_result(nets, images_dir, masks_dir):
    thresholds = [i/100 for i in range(50,96,2)]
    r = dict()
    for net, key in zip(nets, ['Unet based classificator', 'Unet with classification branch', 'Unet']):
        accuracy = []
        precision = []
        recall = []
        thrs = []
        for threshold in tqdm(thresholds,total= len(thresholds), unit='threshold'):
            result = eval_model(net, images_dir, masks_dir,segmentation=False, threshold=threshold)
            accuracy.append(result['accuracy'])
            precision.append(result['precision'])
            recall.append(result['recall'])
            thrs.append(threshold)

        r[key] = np.array([thrs, accuracy, precision, recall]).T
        # result = dict(result=result)
    with open("result.pkl", "wb") as f:
        pickle.dump(r, f)
    precision_recall_curve(result)
    precision_recall_on_threshold(result)

    
def mai_graphs():
    torch.cuda.empty_cache()
    # model_name = Path("genial-aardvark-367.pth")UNet
    # model_name = Path("holographic-federation-408.pth")Classification Unet
    # model_name = Path('hokey-trooper-410.pth')
    path_to_data = Path('/home.stud/grundda2/.local/data')
    images_dir = path.join(path_to_data, 'test_images')
    masks_dir = path.join(path_to_data, 'masks')
    
    
    # for model, model_name in [(Unet_model.Classificator,Path('hokey-trooper-410.pth')),(Unet_model.Classification_UNet,Path("holographic-federation-408.pth")),(Unet_model.UNet,Path("faithful-night-435.pth"))]:
   
   
   
    model = Unet_model.UNet
    # model_name = Path("worldly-gorge-499.pth") #Pseudolabels
    model_name = Path("misty-water-497.pth") #Consistency
    # model_name = Path('rose-violet-489.pth') #Normal
    net = model(n_channels = 1, n_classes=2,mean = 0.4593777512924429, std = 0.23807501840974526, depth = 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    model_path = path.join('/datagrid/personal/grundda/models', model_name)
    net.load_state_dict(torch.load(model_path, map_location = device))

    # net = COCO('/home.stud/grundda2/bc_project/Bc_project/Hanievaluation/Haniannotation.json')
    

        
    result = eval_model(net, images_dir, masks_dir, True, eval_pictures=False)
    print(result)
    # result=eval_model(net, images_dir, masks_dir)
    # get_PR_result(nets[2], images_dir, masks_dir)
    results = []
    # for threshold in range(50,96):
    #     result = eval_model(nets[2], images_dir, masks_dir, True, eval_pictures=False,threshold=threshold/100)
    #     results.append(result)
    # import pickle
    # with open("threshold_unet.pkl","wb") as f:
    #     pickle.dump(results,f)
    # for threshold, result in enumerate(results, start=50):
    #     print(threshold)
    #     print("TP DSC: ",result['true_positive_dice'])
    #     print("DSC: ",result['total_dice'])
        #for key, value in result.items():
            #print(key, value)
    torch.cuda.empty_cache()


# def main():
#     path_home = Path("/mnt/home.stud/grundda2/bc_project")
#     path_to_data = path.join(path_home, "data")
    
#     with open(path.join(path_to_data, "val_indices.txt"), "r") as f:
#         images_names = list(map(int, f.readlines()))
#     name = "Kunt"
#     evaluate_pictures_without_net(name, path_to_data, images_names)
def my_grid2(image, true_mask, predictions,grid_padding, padding):
    col_padding = padding
    colors = ['cyan', 'yellow', 'green','pink']
    grid_list = [image,]
    for index,pred in enumerate(predictions):
        grid_list.append(image_with_segmentation(image, pred, colors[index]))
    grid_list.append(image_with_segmentation(image, true_mask, "red"))
    for pred in predictions:
        grid_list.append(comparison_picture(true_mask, pred))

    cols_num = len(predictions) + 1
    grid_size = grid_shape(grid_padding, cols_num, 2)

    grid =  torchvision.utils.make_grid(grid_list, padding=grid_padding, pad_value=236, nrow=cols_num).permute([1,2,0])
    
    cols = 255*torch.ones((grid_size[0] + LOWER_ROW_PADDING + UPPER_ROW_PADDING, col_padding, 3), dtype=torch.uint8)
    row_upper = 255*torch.ones((UPPER_ROW_PADDING, grid_size[1], 3), dtype=torch.uint8)
    row_lower = 255*torch.ones((LOWER_ROW_PADDING, grid_size[1], 3), dtype=torch.uint8)
    
    grid = torch.vstack((row_upper, grid, row_lower))
    grid = torch.hstack((cols, grid, cols))
    return grid.permute([2,0,1])

def evaluation_picture2(image, true_mask, predictions, names, path_to_export, file_name=""):
    print(names)
    grid_padding = 30
    padding = 50
    black = (0,0,0)
    image = torchvision.transforms.ToPILImage()(my_grid2(image, true_mask, predictions, grid_padding, padding))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("LEMONMILK-Light.otf", 60)

    # LEGEND
    font_size = 60
    font = ImageFont.truetype("LEMONMILK-Light.otf", font_size)

    # dice = dice_score(true_mask, F.one_hot(torch.squeeze(prediction)).permute(2,0,1))
    rows = [UPPER_ROW_PADDING-font_size-10, UPPER_ROW_PADDING + 3*grid_padding + 2*PICTURE_HEIGHT]
    
    text_list = ["X-ray image",]
    for name in names:
        text_list.append(name)
    text_list.append("Ground truth")
    for name in names:
        text_list.append("Evaluation")
    dice = [dice_score(true_mask, prediction) for prediction in predictions]
    cols = [index*PICTURE_WIDTH + (index+1)*grid_padding + padding for index in range(len(predictions)+1)]

    print(names)

    print(text_list)
    for indexr, row in enumerate(rows):
        for indexc , col in enumerate(cols):
            draw.text((col, row), text_list[indexr*len(cols) + indexc], black, font = font)

    draw.text((cols[0], rows[0] - 90), F"File name: {file_name}", (0,0,0), font = font)
    for index in range(len(predictions)):
        draw.text((cols[index+1], rows[0] - 90), F"Dice: {dice[index]:.3f}", (0,0,0), font = font)

    radius = 2
    rec_width = 40
    gap = 80
    y = rows[1] + gap
    x = cols[1] + gap
    draw.rounded_rectangle((cols[1], y, cols[1]+rec_width, y + rec_width), radius = radius, fill = (0, 255, 0))
    draw.text((x, y), "True positive", black, font = font)
    y += gap
    draw.rounded_rectangle((cols[1], y, cols[1]+rec_width, y + rec_width), radius = radius, fill = (255, 0, 0))
    draw.text((x, y), "False positive", black, font = font)
    y += gap
    draw.rounded_rectangle((cols[1], y, cols[1]+rec_width, y + rec_width), radius = radius, fill = (0, 0, 255))
    draw.text((x, y), "False negative", black, font = font)

    image.save(path.join(path_to_export,file_name))

def get_image(image_path, file_name):
    image = torchvision.io.read_image(path.join(image_path, file_name), torchvision.io.ImageReadMode.RGB)
    return image

def get_mask(image_path, file_name):
    image = torchvision.io.read_image(path.join(image_path, file_name))
    return image>0

def get_eval_pictures(images, masks, paths, names, export_path):
    for image_name in listdir(images):
        image = get_image(images, image_name)
        mask = get_mask(masks, image_name)
        predictions = []
        prediction_names = []
        for index, p in enumerate(paths):
            if path.isfile(path.join(p, image_name)):
                predictions.append(get_mask(p, image_name))
                prediction_names.append(names[index])
            else:
                print(path.join(p, image_name))

        

        evaluation_picture2(image, mask, predictions, prediction_names, export_path, image_name)

        

        

        


if __name__ == "__main__":
    # eval_model()
    # mai_graphs()
    images = "/home.stud/grundda2/.local/data/images"
    masks = "/home.stud/grundda2/.local/data/masks"
    paths = ["/home.stud/grundda2/bc_project/Bc_project/evaluationmodel/", "/home.stud/grundda2/bc_project/Bc_project/adelevaluation","/home.stud/grundda2/bc_project/Bc_project/Hanievaluation"]
    names = ["Model prediction", "Student A", "Student B"]
    get_eval_pictures(images, masks, paths, names, "/home.stud/grundda2/bc_project/Bc_project/eval3")
    # main()
    # with open("result.pkl", "rb") as f:
    #     result = pickle.load(f)
    # print(result)
    # precision_recall_curve(result)
    # precision_recall_on_threshold(result)
    
    
        
