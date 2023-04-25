import cv2 as cv
import numpy as np 
from PIL import Image
import os
import torchvision
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

ONLY_CLASSIFICATION = False
IMAGE_CLASS_INFO = False
EVALUATION_IMAGES = False

PATH_TO_IMAGES = '/home.stud/grundda2/.local/data/val_images'
PATH_TO_MASKS = '/home.stud/grundda2/.local/data/masks'
COMPARISON_IMAGES = False

CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_COLORS = [(0, 255, 0), (200, 255, 0), (255, 255, 0), (255, 200, 0), (255, 0, 0)]

DARKNET = '/home.stud/grundda2/bc_project/darknet/'
WEIGHTS = os.path.join(DARKNET,'backup')
CFG = os.path.join(DARKNET, 'cfg/yolov3-voc.cfg')

#MEAN = 0.459377751294429 * 255
#STD = 0.05667971439080055 * 255 
# Actual values does not work as yolo was training without them
MEAN = 0
STD = 255
"""
cv.imread returns image with values between 0 and 255
We can not adjust the return np.array because blob from image takes in
uint8 (or I suppose so because otherwise it throws error)
Thus the mean has to be in scale of 255 and then we move it to the 
normalized scale by dividing it by 255

The scale given to the blobfromimage function is 1/std -> it multiplies the values
with the scale argument
"""
def get_blob(path_to_image:str):
    image = cv.imread(path_to_image)
    return cv.dnn.blobFromImage(image, 1/STD, (416,416), MEAN), image.shape

def get_true_class(filename):
    path_to_file = os.path.join(PATH_TO_MASKS, filename)
    image = np.array(Image.open(path_to_file).getdata())
    return int(any(image > 0))

def boxes_result(outs, shape, confidence_threshold = None):
    confidences = []
    boxes = []
    # class_ids = []
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * shape[1])
                center_y = int(detection[1] * shape[0])
                w = int(detection[2] * shape[1])
                h = int(detection[3] * shape[0])
                x = center_x - w / 2
                y = center_y - h / 2
                # class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, confidences # , class_ids

def classification_result(outs, confidence_threshold = None):
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                return 1
    else:
        return 0
    
def positives_negatives(prediction:int, true_class:int, pos_neg:dict):
    key = F"{str(prediction == true_class)}_{'positive' if prediction else 'negative'}"
    pos_neg[key] += 1
    return pos_neg

    
def get_result(outs, shape, confidence_threshold):
    if ONLY_CLASSIFICATION:
        return classification_result(outs, confidence_threshold)
    else:
        return boxes_result(outs, shape, confidence_threshold)


def eval_model(net, confidence_threshold = None):
    classification_score = 0
    count = 0
    pos_neg = dict(
    True_positive = 0,
    True_negative = 0,
    False_positive = 0,
    False_negative = 0,
    )
    with tqdm(total = len(os.listdir(PATH_TO_IMAGES)), unit = 'img') as pbar:
        for image_name in os.listdir(PATH_TO_IMAGES):
            pbar.update(1)
            count += 1
            path_to_image = os.path.join(PATH_TO_IMAGES, image_name)
            blob, shape = get_blob(path_to_image)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())

            prediction = get_result(outs, shape, confidence_threshold)
            true_class = get_true_class(image_name)

            if not ONLY_CLASSIFICATION:
                boxes, confidences =prediction[0],prediction[1]
                prediction = int(len(boxes) > 0)

            classification_score += int(prediction == true_class)
            pos_neg = positives_negatives(prediction, true_class, pos_neg)
            if IMAGE_CLASS_INFO:
                print(F"{bcolors.OKGREEN if prediction == true_class else bcolors.FAIL}{image_name}:\n  True clas: {true_class}\n  Prediction: {prediction}{bcolors.ENDC}")

                if not ONLY_CLASSIFICATION:
                    if prediction:
                        confidences =  np.array(confidences)
                        lowest_conf = np.min(confidences)
                        highest_conf = np.max(confidences)
                        print(F"{bcolors.WARNING if highest_conf - CONFIDENCE_THRESHOLD < 0.05 else ''}  Highest confidence: {highest_conf}{bcolors.ENDC}")
                        print(F"{bcolors.WARNING if lowest_conf - CONFIDENCE_THRESHOLD < 0.05 else ''}  Lowest confidence: {lowest_conf}{bcolors.ENDC}")
            if EVALUATION_IMAGES:
                segmask_path = os.path.join(PATH_TO_MASKS, image_name)
                draw_bounding_boxes(path_to_image, boxes, confidences, segmask_path= segmask_path, comparison = COMPARISON_IMAGES)

    classification_rate = classification_score/count
    print(F"Correctly classified: {classification_rate}")

    precision = pos_neg['True_positive']/(pos_neg['True_positive'] + pos_neg['False_positive'] + 1e-7)
    print(F"Precision: {precision}")

    recall = pos_neg['True_positive']/(pos_neg['True_positive'] + pos_neg['False_negative'] + 1e-7)
    print(F"Recall: {recall}")

    return classification_rate, precision, recall

    
def draw_bounding_boxes(image_path:str, boxes:list, confidences:list, segmask_path:str = None, comparison = False):
    """
    inputs:
        image_path:         string with absolute path to image
        boxes:              list of lists of (x, y, w, h) - x & y are top left corner
        confidences:        list of floats of how confident this prediction is
        segmask:            string with absolute path to ground truth
        comparison:         if true, original image without mask and bboxes is added for clean comparison

    returns:
        image:              Tensor with boxes and segmentation mask, should be put into torchvision.utils.save_image
                        or
                            List of tensors, which also can be directly put into save_image
        
    """
    def load_image(image_path):
        image = torchvision.io.read_image(image_path)
        if image.size()[0] == 1:
            image = torch.broadcast_to(image, (3, image.size()[1], image.size()[2]))
        return image

    boxes = torch.tensor(boxes)
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]
    confidences = torch.tensor(confidences)
    
    index = ((1-confidences)*10).to(int)
    colors = [CONFIDENCE_COLORS[i] for i in index]

    image = load_image(image_path)

    if segmask_path is not None:
        mask = torchvision.io.read_image(segmask_path) > 0
        image = torchvision.utils.draw_segmentation_masks(image, mask, colors = 'cyan', alpha = 0.2)

    
    image = torchvision.utils.draw_bounding_boxes(image, boxes, colors = colors, width = 2)
    image = image/255


    if comparison:
        og_image = load_image(image_path)/255
        return [og_image, image]
    else:
        return image

def eval_trained_weights():
    number_of_weights = len(os.listdir(WEIGHTS))
    result = {}
    thresholds = [i/100 for i in range(50, 96, 2)]
    with tqdm(total = number_of_weights, unit='model') as pbar:
        for weights in os.listdir(WEIGHTS):
            net = cv.dnn.readNet(os.path.join(WEIGHTS,weights), CFG)
            desc = (list(weights.split('_')))[-1]
            result[weights] = []
            pbar.update(1)
            with tqdm(total = len(thresholds), unit='Confidence threshold', desc= desc) as pbar2:
                for confidence_threshold in thresholds:
                    pbar2.update(1)
                    classification_rate, precision, recall = eval_model(net, confidence_threshold)
                    
                    result[weights].append([confidence_threshold, classification_rate, precision, recall])
    return result

def precision_recall_curve(precision_recall_dict):
    fig1, ax1= plt.subplots()
    fig2, ax2 = plt.subplots()
    for key in precision_recall_dict:
        arr = np.array(precision_recall_dict[key])
        threshold = arr[:, 0]
        rate = arr[:, 1]
        precision = arr[:, 2]
        recall = arr[:, 3]

        ax1.plot(recall, precision, label = key)
        ax2.plot(threshold, rate, label = key)

    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')

    ax2.set_title('Correctly classified images')
    ax2.set_ylabel('Fraction of correctly classified images')
    ax2.set_xlabel('Confidence threshold')

    ax1.legend(loc = 'lower left')
    ax2.legend(loc = 'lower left')

    fig1.savefig('Precision_recall_curve.png')
    fig2.savefig('Classification score.png')

def main():
    result = eval_trained_weights()
    with open('precision.pkl', 'wb') as f:
        pickle.dump(result, f)
    precision_recall_curve(result)





if __name__ == "__main__":
    main()
