from .dice_score import dice_loss
from torch import nn
import torch

class SegmentationLoss():
    def __init__(self, x_weight = 1, dice_weight = 1):
        self.x_entropy = nn.CrossEntropyLoss()
        self.x_weight = x_weight
        self.dice_weight = dice_weight

    def x_loss(self, prediction, target):
        return self.x_weight * self.x_entropy(prediction, target) 

    def dice_loss(self, prediction, target):
        predicted_segmentation = torch.argmax(prediction, dim = 1).float()
        target = target.float()
        return self.dice_weight * dice_loss(predicted_segmentation, target)
    
    def seg_loss(self, prediction, target):
        return self.x_loss(prediction, target) + self.dice_loss(prediction, target)

    def __call__(self, prediction, target):
        return self.seg_loss(prediction, target)

class ClassSegLoss(SegmentationLoss):
    def __init__(self, x_weight = 1, dice_weight = 1, class_weight = 1):
        super().__init__(x_weight, dice_weight)
        self.class_weight = class_weight

    def class_loss(self, prediction, target):
        if prediction is None:
            return 0
        return self.class_weight * self.x_entropy(prediction, target)

    def class_with_seg_loss(self, prediction, segmentation_target, classification_target):
        return self.seg_loss(prediction['segmentation'], segmentation_target) + self.class_loss(prediction.get('classification'), classification_target)

    def __call__(self, prediction, segmentation_target, classification_target):
        return self.class_with_seg_loss(prediction, segmentation_target, classification_target)


