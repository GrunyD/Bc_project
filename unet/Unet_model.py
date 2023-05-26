import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from . import segalg

#########################################################################
#                                                                       #
#                           Unet parts                                  #
#                                                                       #
#########################################################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_downsample = False):
        super().__init__()
        # if conv_downsample:
        #     self.down = nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1, stride=2),
        #         DoubleConv(in_channels, out_channels)
        #     )
            
        # else:
        #     self.down = nn.Sequential(
        #         nn.MaxPool2d(2),
        #         DoubleConv(in_channels, out_channels)
        #     )
        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear == False:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        elif bilinear == True:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels*2, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
#########################################################################
#                           Other parts                                 #
#########################################################################

class UpPP(nn.Module): #For UNet++
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x_lower:torch.Tensor, x_list: list):
        x_lower = self.up(x_lower)
        diffY = x_list[0].size()[2] - x_lower.size()[2]
        diffX = x_list[0].size()[3] - x_lower.size()[3]

        x_lower = F.pad(x_lower, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_lower, *x_list], dim = 1)
        return self.conv(x)

class Norm(nn.Module): #Normalization layer
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x-self.mean)/self.std
    
#
# Morphological post processing layers
#

class Morphological_layer(nn.Module):
    def __init__(self, kernel_size:int|tuple ):
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            self.erode_kernel = np.ones((kernel_size[0], kernel_size[0]))
            self.dilate_kernel = np.ones((kernel_size[1], kernel_size[1]))
        else:
            self.erode_kernel = np.ones((int(kernel_size), int(kernel_size)))
            self.dilate_kernel = self.erode_kernel

class Opening(Morphological_layer):
    def __init__(self, kernel_size:int|tuple):
        super().__init__(kernel_size)

    def forward(self, x):
        erosion =  cv2.erode(x, self.erode_kernel, iterations = 1)
        dilatation = cv2.dilate(erosion, self.dilate_kernel, iterations = 1)
        return dilatation

class Closing(Morphological_layer):
    def __init__(self, kernel_size:int|tuple):
        super().__init__(kernel_size)

    def forward(self, x):
        dilatation = cv2.dilate(x, self.dilate_kernel, iterations = 1)
        erosion =  cv2.erode(dilatation, self.erode_kernel, iterations = 1)
        return erosion

class Opening_and_Closing(nn.Module):
    def __init__(self, opening_kernel_size, closing_kernel_size) -> None:
        super().__init__()
        self.opening = Opening(opening_kernel_size)
        self.closing = Closing(closing_kernel_size)
    
    def forward(self, x):
        return torch.from_numpy(self.closing(self.opening(x)))

#########################################################################
#                                                                       #
#                            Ensembling                                 #
#                                                                       #
#########################################################################

class Encoder(torch.nn.Module):
    def __init__(self, n_channels, mean, std, depth, base_kernel = 64, conv_downsample = False, **kwargs):
        super().__init__()
        self.depth = depth

        self.norm = Norm(mean, std)

        encoder = [DoubleConv(n_channels, base_kernel)]
        for i in range(depth):
            inc = base_kernel*(2**i)
            outc = base_kernel*(2**(i+1))

            if kwargs['bilinear'] and i == depth-1:
                outc = outc//2
            encoder.append(Down(inc, outc, conv_downsample=conv_downsample))

        self.encoder = nn.ModuleList(encoder)

    def forward(self, x):
        x = self.norm(x)
        skip_cons = [x]
        for layer in self.encoder:
            skip_cons.append(layer(skip_cons[-1]))

        return skip_cons
    
class Decoder(torch.nn.Module):
    def __init__(self, n_classes, depth, base_kernel = 64, bilinear = False, **kwargs):
        super().__init__()
        decoder = []
        factor = 2 if bilinear else 1
        for i in range(depth):
            inc = base_kernel*(2**(depth-i))//factor
            outc = int(base_kernel*(2**(depth - i - 1))//factor)
            
            decoder.append(Up(inc, outc, bilinear))

        self.decoder = nn.ModuleList(decoder)

        self.out_conv = OutConv(base_kernel//(1+int(bilinear)), n_classes)

    def forward(self, skip_cons):
        x = skip_cons[-1]
        for index, layer in enumerate(self.decoder):
            x = layer(x, skip_cons[-2-index])

        logits = self.out_conv(x)
        return logits

class Classifier(torch.nn.Module):
    def __init__(self,n_classes, depth, base_kernel = 64, conv_downsample = False, bilinear = False):
        super().__init__()
        decoder = []
        for i in range(depth):
            inc = base_kernel*(2**(depth-i))
            outc = int(base_kernel*(2**(depth - i - 1)))
            
            decoder.append(DoubleConv(inc, outc))

        self.decoder = nn.ModuleList(decoder)
        image_width = 1068//2**depth + int(conv_downsample) #If double strided convolution is used instead of maxpooling, then we have to add 1
        image_height = 847//2**depth + int(conv_downsample)
        self.linear_decide = nn.Linear(image_height*image_width*base_kernel, n_classes)


    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_decide(x)
        return x
     
#########################################################################
#                                                                       #
#                         Unet based models                             #
#                                                                       #
#########################################################################

class Model(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

    def predict(self, image:torch.Tensor, classification_threshold = 0.5, **kwargs) -> np.ndarray:
        """
        Input: image        - already prepeared torch tensor for model processing
        
        Output: dict    
            segmentation:   One Hot encoded for easy multiclass dice calculation
            classification: Int

        """
        
        self.eval()
        with torch.no_grad():
            output = self.forward(image)

        return_dict = dict()
        seg_pred = output.get('segmentation')
        cls_pred = output.get('classification')

        if seg_pred is not None:
            probs = nn.functional.softmax(seg_pred, dim=1)[0] # To get rid of batch dimension
            #softmax perserves dimensions -> get rid of batch dimension
            #argmax will leave the channel dimension -> index 0 to get rid of channels
            #one hot adds dimension for each class including background
            seg_pred = nn.functional.one_hot(torch.argmax(probs, dim = 0), self.n_classes)
            seg_pred = seg_pred.permute(2, 0, 1)
            return_dict.update({'segmentation': seg_pred})

        if cls_pred is not None:
            cls_pred = int(torch.nn.functional.softmax(cls_pred)[1] >= classification_threshold)
        else:
            assert seg_pred is not None, "At least one of the duo segmnetation and classification has to be not None"
            cls_pred = int(torch.sum(probs[1:,...] >= classification_threshold) > 0)
        return_dict.update({'classification': cls_pred})
        return return_dict
        
    def __str__(self):
        return self.__class__.__name__

class UNet(Model):
    def __init__(self, n_channels, n_classes, mean, std, depth, bilinear = False, conv_downsample = False, base_kernel = 64, **kwargs):
        super().__init__(n_channels, n_classes)
        self.encoder = Encoder(n_channels, mean, std, depth, base_kernel, conv_downsample, bilinear = bilinear)
        self.decoder = Decoder(n_classes, depth, base_kernel, bilinear)
        self.depth = depth

    def forward(self, x, **kwargs):
        skip_cons = self.encoder(x)
        return {"segmentation": self.decoder(skip_cons)}
    
    def __str__(self):
        return F"{super().__str__()} depth:{self.depth}"
        
class Classification_UNet(UNet):
    def __init__(self, n_channels, n_classes, mean, std, depth, bilinear = False, base_kernel = 64, conv_downsample = False, **kwargs):
        super().__init__(n_channels, n_classes, mean, std, depth, bilinear,conv_downsample, base_kernel,  **kwargs)

        
        self.classifier = Classifier(n_classes, depth, base_kernel, conv_downsample)

    def forward(self, x, **kwargs):
        skip_cons = self.encoder(x)

        ret = {"segmentation": self.decoder(skip_cons),
              "classification": self.classifier(skip_cons[-1])}

        return ret
    
class Classificator(Model):
    def __init__(self, n_channels, n_classes, mean, std, depth, base_kernel = 64, conv_downsample = False, **kwargs):
        super().__init__(n_channels, n_classes)
        self.encoder = Encoder(n_channels, mean, std, depth, base_kernel, conv_downsample)

        self.decide = Classifier(n_classes, depth, base_kernel, conv_downsample)

    def forward(self, x, **kwargs):
        skip_connections = self.encoder(x)
        return {"classification":self.decide(skip_connections[-1])}

class CoTrainer(Model):
    def __init__(self, n_channels, n_classes, mean, std, depth, base_kernel = 64, **kwargs):
        super().__init__(n_channels, n_classes, **kwargs)
        self.unet1 = UNet(n_channels, n_classes, mean, std, depth, base_kernel = base_kernel)
        self.unet2 = UNet(n_channels, n_classes, mean, std, depth, base_kernel = base_kernel)

    def forward(self, x,**kwargs):
        x1 = self.unet1(x)
        x2 = self.unet2(x)
        if self.training:
            return x1, x2
        else:
            return dict( segmentation = 
                        (nn.functional.softmax(x1['segmentation'], dim = 1) + nn.functional.softmax(x2['segmentation'], dim = 1))/2)


class UNetPP(nn.Module):
    def __init__(self, n_channels, n_classes, mean, std, depth, base_kernel_num):
        super().__init__()
        self.depth = depth
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = Norm(mean, std)
        encoder = [DoubleConv(n_channels, base_kernel_num),]

        for i in range(1, depth):
            encoder.append(Down(base_kernel_num*2**(i-1), base_kernel_num*2**i))
        
        self.encoder = nn.ModuleList(encoder)
        skip_conns_and_decoder = [[] for _ in range(depth-1)]
        for i in range(depth-1):
            for j in range(1,depth -i):
                #One node gets j+1 outputs of convs -> (j+1)*channels
                skip_conns_and_decoder[i].append(UpPP((base_kernel_num*2**i)*(j+1), base_kernel_num*2**i))
        self.skip_conns_and_decoder = nn.ModuleList(nn.ModuleList(skip_conns_and_decoder[i]) for i in range(len(skip_conns_and_decoder)))
        """
        [
        [X, X, X, X],
        [X, X, X],
        [X, X],
        [X]
        ]
        Creates this grid of layers according to UNet++ paper
        """

        self.outConv = OutConv(base_kernel_num, n_classes)
    def __str__(self) -> str:
        return "U-Net++"
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        encoded_x = [x,]
        for X in self.encoder:
            encoded_x.append(X(encoded_x[-1]))
            

        outputs = [[i,] for i in encoded_x[1:]]
        for j in range(self.depth-1):
            for i in range(self.depth-1):
                if len(self.skip_conns_and_decoder[i]) > j:
                    X = self.skip_conns_and_decoder[i][j]
                    outputs[i].append(X(outputs[i+1][j], outputs[i]))

        return self.outConv(outputs[0][-1])


class ScribleSegmentation(nn.Module):
    def __init__(self, K=2000, lam=0.1, exponent = 2, sigma = 64):
        super().__init__()
        self.K = K
        self.lam = lam
        self.exponent = exponent
        self.sigma = sigma

    def forward(self, image, prediction):
        image = torch.squeeze(image)
        image = image.detach().cpu().numpy()
        if np.max(image) <= 1:
            image = image*255
        prediction = prediction.detach().cpu().numpy()
        prediction = segalg.get_segmentation(image, prediction, self.K, self.lam, self.exponent, self.sigma)
        # a = torch.from_numpy(np.uint8(prediction)).to(torch.int64)
    
        return torch.nn.functional.one_hot(torch.from_numpy(np.uint8(prediction)).to(torch.int64)).permute(2,0,1).to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
