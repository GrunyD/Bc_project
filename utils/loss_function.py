from torch import nn
import torch

"""
This file is probably not complete and needs some fixes if working with higher batch number. 

From each loss function you can choose only some of the functions by setting the weight to zero
"""

    
def norm(prediction):
    if prediction.size(1) == 1:
        return nn.functional.sigmoid(prediction)
    else:
        return nn.functional.softmax(prediction, dim = 1)
    
def dice_coef(tensor1, tensor2):
    return (2*torch.sum(tensor1*tensor2) + 1e-7)/(torch.sum(tensor1) + torch.sum(tensor2) + 1e-7)

def iou(tensor1, tensor2):
    return (torch.sum(tensor1*tensor2) + 1e-7)/(torch.sum(tensor1) + torch.sum((1-tensor2)*tensor1) + 1e-7)

class ClassSegLoss(nn.CrossEntropyLoss):
    """
    This is supposed to be supervised loss function. Targets are torch.long segmentation masks or classification of images (only positive/negative)
    """
    def __init__(self, x_weight = 1, dice_weight = 1, class_weight = 1, iou_weight = 1):
        super().__init__()
        self.x_weight = x_weight
        self.dice_weight = dice_weight
        self.class_weight = class_weight
        self.iou_weight = iou_weight

    def class_loss(self, prediction, target):
        if prediction is None:
            return 0
        return super().forward(prediction, target)
    
    def dice_loss(self, prediction, target):
        if prediction is None:
            return 0
        target = nn.functional.one_hot(target, num_classes = prediction.size(1)).permute((0,3,1,2))
        return 1 - dice_coef(norm(prediction)[:,1:,...], target[:,1:,...])
    
    def x_loss(self, prediction, target):
        if prediction is None:
            return 0
        return super().forward(prediction, target)
    
    def iou_loss(self, prediction, target):
        if prediction is None:
            return 0
        target = nn.functional.one_hot(target, num_classes = prediction.size()[1]).permute((0,3,1,2))
        IOU = iou(norm(prediction)[:,1:,...], target[:,1:,...])
        return 1 - IOU
    

    def forward(self, prediction:dict, segmentation_target:torch.Tensor, classification_target:torch.Tensor = 0):
        """
        INPUTS: 
            Prediction: Dict with predictions, so far just segmentation or classification
  
        In case of consistency semisupervised learning, the model i working with unlabeled images without
        pseudolabeles. Thus they do not have any kind of mask. However I needed to pass some mask into the dictionary
        in order to not throw an error. Thus those with class -1 are marked as unlabaled images and the loss should be 0.

        Now only counts with batchnumber 1
        """
        # if classification_target.item() == -1:
        #     return 0
        
        segmentation = prediction.get('segmentation')
    
        x_loss = self.x_weight * self.x_loss(segmentation, segmentation_target)

        d_loss = self.dice_weight * self.dice_loss(segmentation, segmentation_target)
        c_loss = self.class_weight * self.class_loss(prediction.get('classification'), classification_target)
        iou_loss = self.iou_weight * self.iou_loss(segmentation, segmentation_target)
        return x_loss + d_loss + c_loss + iou_loss

class ConsistencyLoss(nn.modules.loss._Loss):
    def __init__(self, difference_weight:float = 1, dice_weight:float = 1, iou_weight:float = 1):
        super().__init__()
        self.difference_weight = difference_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    def forward(self, prediction, per_prediction, pipelines):
        prediction = prediction.get('segmentation')
        per_prediction = per_prediction.get('segmentation')


        prediction = pipelines[0](prediction) 
        """
        There should be for loop because with higher batch number we have more pipelines in
        the pipelines list. However for some reason the for loop damages backward propagation.
        Since I can not have higher batch number without huge downscaling of the image I will leave this
        like this.
        """
        prediction = norm(prediction)
        per_prediction = norm(per_prediction)
        dif_loss = self.difference_weight * self.absolute_difference_loss(prediction, per_prediction)
        dic_loss = self.dice_weight * self.dice_loss(prediction, per_prediction)
        iou_loss = self.iou_weight * self.iou_loss(prediction, per_prediction)
        return dif_loss + dic_loss + iou_loss
    
    def absolute_difference_loss(self, prediction, per_prediction):
        area = prediction.size()[-1] * prediction.size()[-2]
        loss = torch.sum(torch.abs(prediction - per_prediction))/area
        return loss
    
    def dice_loss(self, prediction, per_prediction):
        dice = dice_coef(prediction[:,1:], per_prediction[:,1:])
        return 1 - dice

    def iou_loss(self, prediction, per_prediction, gamma = 6):
        """
        Hard Intersection over Union is not differentiable. Thus this implements 'soft IOU' which does not work with 
        strict bool values, however, puts the values pretty close
        """
        
        prediction = norm(prediction**gamma)
        per_prediction = norm(per_prediction**gamma)
        pre_sum = torch.sum(prediction)
        per_pre_sum = torch.sum(per_prediction)
        if pre_sum > per_pre_sum:
            IOU = iou(per_prediction[:,1:], prediction[:,1:])
        else:
            IOU = iou(prediction[:,1:], per_prediction[:,1:])
        return 1 - IOU
        
class ConfidenceAwareLoss(nn.modules.loss._Loss):
    def __init__(self, weight = 3):
        super().__init__()
        self.weight = weight

    def forward(self, O_prediction1, F_prediction1, O_prediction2, F_prediction2):
        def KL_divergence(O_prediction, F_prediction):
            kl_distance = nn.KLDivLoss(reduction='none')
            sm = torch.nn.Softmax(dim=1)
            log_sm = torch.nn.LogSoftmax(dim=1)
            print(torch.max(O_prediction), torch.min(O_prediction))
            variance = torch.sum(kl_distance(log_sm(O_prediction), sm(F_prediction)), dim=1)
            # return torch.nanmean(F_prediction * torch.log(F_prediction/O_prediction))
            return variance

        def unsupervised_loss(prediction, ground_truth, variance):
            """
            Variance tells us how close those two predictions were.
            If they were too different, the variance is large thus encreasing 
            the whole loss but decresing the part from cross entropy as the ground 
            truth is not accurate
            """
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(prediction, ground_truth)
            return torch.mean(torch.exp(-variance) * loss)  + torch.mean(variance)
        
        
        O_prediction1 = O_prediction1['segmentation']
        F_prediction1 = F_prediction1['segmentation']
        O_prediction2 = O_prediction2['segmentation']
        F_prediction2 = F_prediction2['segmentation']

        variance1 = KL_divergence(O_prediction1, F_prediction1)
        variance2 = KL_divergence(O_prediction2, F_prediction2)
        print(variance1)
        print(variance2)

        combined_prediction1 = (O_prediction1 + F_prediction1)/2
        combined_prediction2 = (O_prediction2 + F_prediction2)/2

        # dim = 1 should be over classes - dim 0 is batch number and then are dimensions of inputs
        seg_map1 = torch.argmax(combined_prediction1, dim = 1).detach()
        seg_map2 = torch.argmax(combined_prediction2, dim = 1).detach()

        loss1 = unsupervised_loss(combined_prediction1, seg_map2, variance2)
        loss2 = unsupervised_loss(combined_prediction2, seg_map1, variance1)

        return self.weight * (loss1 + loss2)

    
if __name__ == "__main__":
    loss = ConfidenceAwareLoss(1)
    torch.manual_seed(40)
    a = norm(torch.rand(1,2,800,1000))
    b = norm(torch.rand(1,2,800,1000))
    c = norm(torch.rand(1,2,800,1000))
    d = c#norm(torch.rand(1,2,800,1000))
    l = loss(a,b, c, d)
    print(l)

