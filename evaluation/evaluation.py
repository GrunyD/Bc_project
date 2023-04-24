import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from os import mkdir, path,listdir
from pathlib import Path
import json
from unet.Unet_model import UNet1, Opening_and_Closing
import numpy as np
from utils.data_loading import BaseDataset, ImageDataset
import torch.nn.functional as F

ALPHA = 0.2
LIGHT_GREY = 236
PADDING_WIDTH = 10
UPPER_ROW_PADDING = 280
LOWER_ROW_PADDING = 300
PICTURE_WIDTH = 1068
PICTURE_HEIGHT = 847

def gif_read(path):
    im_frame = Image.open(path)
    return torch.ByteTensor(im_frame.getdata()).reshape(im_frame.size[1], im_frame.size[0],)

def to_image(mask):
    mask1 = 255*mask
    if len(mask.shape) == 2:
        mask2 = mask1.broadcast_to([3, *mask.shape]) 
    return mask2

def _TP_FP_FN(true_mask, prediction):
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
    zer = torch.zeros((3,*TP.shape), dtype=torch.uint8)
    zer[0] = FP
    zer[1] = TP
    zer[2] = FN
    return zer*200

def dice_score(true_mask, prediction):
    # print(true_mask.size())
    # print(prediction.size())
    assert true_mask.size() == prediction.size()
    epsilon = 1e-6
    inter = torch.sum(true_mask*prediction)
    sets_sum = torch.sum(prediction) + torch.sum(true_mask)
    return (2 * inter + epsilon) / (sets_sum + epsilon)

def inter_over_union(true_mask, prediction):
    epsilon = 1e-6
    inter = torch.sum(true_mask * prediction)
    sets_sum = torch.sum(true_mask) + torch.sum(prediction) - inter
    return (inter + epsilon)/(sets_sum + epsilon)

def image_with_segmentation(image, mask, color):
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

    
def evaluation_picture(image, true_mask, prediction, kunt_prediction, path_to_export, file_name="", picture_id="", model = ""):
    
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

    dice = dice_score(true_mask, prediction)
    rows = [UPPER_ROW_PADDING-font_size-10, UPPER_ROW_PADDING + 3*grid_padding + 2*PICTURE_HEIGHT]
    
    if kunt_prediction is not None:
        text_list = ["X-ray image (input)", "Grundfest model prediction","Kunt model prediction",
                    "Ground truth annotation","Prediction evaluation", "Predicition evaluation"]
        kunt_dice = dice_score(true_mask, kunt_prediction)
        cols = [grid_padding + padding, padding + 2*grid_padding + PICTURE_WIDTH,padding + 3*grid_padding + 2*PICTURE_WIDTH]

    else:
        text_list = ["X-ray image (input)", "Grundfest model prediction",
                    "Ground truth annotation","Prediction evaluation"]
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


def evaluate_pictures(model, model_name, path_to_data, image_names, comparison:bool):
    path_to_export = path.join(path_to_data, F"evaluation/{model_name}_eval")
    path_to_images = path.join(path_to_data, "data/val_images1channel")
    path_to_masks = path.join(path_to_data, "data/masks")
    path_to_kunt_masks = path.join(path_to_data, "data/kunt_masks")
    ids = get_image_ids(image_names, path.join(path_to_data, "instances_default.json"))
    try:
        mkdir(path_to_export)
    except FileExistsError:
        pass
    scores = []
    # for image_name in image_names:
    for image_name in range(1, 522):
        image_id = ids[str(image_name)]
        try:
            path_to_image = path.join(path_to_images, F"{image_name}.png")
            image = torchvision.io.read_image(path_to_image)
        except RuntimeError:
            path_to_image = path.join(path_to_data, F"data/images1channel/{image_name}.png")
            image = torchvision.io.read_image(path_to_image)

        true_mask = gif_read(path.join(path_to_masks, F"{image_name}.gif"))
        prediction = predict(path_to_image, model, 0.5)
        kunt_prediction = gif_read(path.join(path_to_kunt_masks, F"{image_name}.gif")) if comparison else None
        score = evaluation_picture(torch.vstack((image,image, image)), true_mask, prediction, kunt_prediction, path_to_export, image_name, image_id, model_name)
        #score = dice_score(true_mask, prediction)
        score = inter_over_union(true_mask, prediction)
        scores.append(score)
    return np.mean(np.array(scores))


# def evaluate_pictures_without_net(name, path_to_data, image_names):
#     path_to_export = path.join(path_to_data, F"evaluation/{name}_eval")
#     path_to_images = path.join(path_to_data, "val_images1channel")
#     path_to_masks = path.join(path_to_data, "masks")
#     path_to_predicted_masks = path.join(path_to_data, "kunt_masks")
#     ids = get_image_ids(image_names, path.join(path_to_data, "instances_default.json"))
#     try:
#         mkdir(path_to_export)
#     except FileExistsError:
#         pass
#     scores = []
#     for image_name in image_names:
#         image_id = ids[str(image_name)]
#         path_to_image = path.join(path_to_images, F"{image_name}.png")
#         image = torchvision.io.read_image(path_to_image)
#         true_mask = gif_read(path.join(path_to_masks, F"{image_name}.gif"))
#         prediction = predict(path_to_image, model, 0.3)
#         #prediction = gif_read(path.join(path_to_predicted_masks, F"{image_name}.gif"))
#         score = evaluation_picture(torch.vstack((image,image, image)), true_mask, prediction, path_to_export, image_name, image_id, name)
#         scores.append(score)
#     return np.mean(np.array(scores))

def morph_grid_search(model, path_to_images, path_to_masks):
    for image_name in listdir(path_to_images):
        path_to_image = path.join(path_to_images, F"{image_name}.png")
        true_mask = gif_read(path.join(path_to_masks, F"{image_name}.gif"))
        prediction = predict(path_to_image, model, 0.5)

        size = 60
        grid = np.zeros((size, size))
        for opening_kernel in range(1, size+1):
            for closing_kernel in range(1,size+1):
                op = Opening_and_Closing(opening_kernel, closing_kernel)
                new_grid = np.zeros(size, size)
                new_grid[opening_kernel][closing_kernel] = dice_score(true_mask, op(prediction))
        grid = grid + new_grid


    grid = grid/len(listdir(path_to_images))
    print(grid)


def eval_model(net, val_images, masks, device, model_name):
    dataset = BaseDataset(val_images, masks, enable_augment= False)
    loader = torch.utils.data.DataLoader(dataset)
    dice= 0
    index= 0
    # for item in loader:
    #     image = item['image'].to(device)
    #     mask= torch.squeeze(item['mask'])
    #     output = torch.from_numpy(net.predict(image))
    #     output = output.to(torch.uint8)
    #     print(output.dtype)
    #     image = torch.squeeze((image*255).to(torch.uint8))
    #     print(image.size())
    #     score = dice_score(mask, output)
    #     evaluation_picture(torch.vstack((image,image, image)), mask, output, None, '/datagrid/personal/grundda/eval_new', file_name=F'{index}.png')

    #     dice.append(score)
    #     print(score)
    #     index += 1

    for filename in listdir(val_images):
        print(filename)
        path_to_image = path.join(val_images, filename)
        prediction = predict(path_to_image, net, 0.5)
        # print(torch.max(prediction), prediction.dtype)
        image = torchvision.io.read_image(path_to_image)
        # print(torch.max(image), image.dtype)
        mask = (torch.squeeze(torchvision.io.read_image(path.join(masks, filename)))>0).to(torch.uint8)
        # print(torch.max(mask), mask.dtype)
        # evaluation_picture(torch.vstack((image,image, image)), mask, prediction, None, '/home.stud/grundda2/bc_project/data/eval', file_name=filename)
        score = dice_score(mask, prediction)
       # print(score)
        dice+= score

    # dice= np.array(dice)
    # print(np.mean(dice))
    print(model_name,val_images, dice/len(listdir(val_images)))
    



# def main():
#     path_home = Path("/mnt/home.stud/grundda2/bc_project")
#     path_to_data = path.join(path_home, "data")
    
#     with open(path.join(path_to_data, "val_indices.txt"), "r") as f:
#         images_names = list(map(int, f.readlines()))
#     name = "Kunt"
#     evaluate_pictures_without_net(name, path_to_data, images_names)


if __name__ == "__main__":
    path_home = Path("/mnt/home.stud/grundda2/bc_project")
    path_to_data = path.join("/datagrid/personal/grundda")

    model_name = "upbeat-eon-226_0.851379.pth"
    val_images= path.join(path_to_data, 'data/test_images')
    masks= path.join(path_to_data, 'data/masks')

    net = UNet1(n_channels = 1, n_classes=2,mean = 0.4506735083369092, std = 0.23919170057270236)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    adres = ['/datagrid/personal/grundda/data/val_images','/datagrid/personal/grundda/data/test_images']
    for model_name in listdir('/datagrid/personal/grundda/models/'):
        try:
            net.load_state_dict(torch.load(path.join(path_to_data, Path(F"models/{model_name}")), map_location=device))
        except Exception as err:
            print(err)
            continue
        for adre in adres:
        
#  with open(path.join(path_to_data, "val_indices.txt"), "r") as f:
#      images_names = list(map(int, f.readlines()))
    #images_names = [i for i in range(1,522)]
    #  print(evaluate_pictures(net, "Comparison", path_to_data, images_names, comparison = False))
        
            eval_model(net, adre, masks, device, model_name)
     #morph_grid_search(net, path_to_images = path.join(path_to_data, "val_images1channel"), path_to_masks=path.join(path_to_data, "masks"))
