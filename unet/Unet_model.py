import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

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
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

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
        self.up = nn.ConvTranspose2d(out_channels//2, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_lower:torch.Tensor, x_list: list):
        x_lower = self.up(x_lower)
        diffY = x_list[0].size()[2] - x_lower.size()[2]
        diffX = x_list[0].size()[3] - x_lower.size()[3]

        x_lower = F.pad(x_lower, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_lower, *x_list])
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
#                           Unet models                                 #
#                                                                       #
#########################################################################

##  0.4506735083369092 - mean
##  0.057212669622861305 - std**2

class UNet0(nn.Module):
    def __init__(self, n_channels, n_classes, mean, std):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.norm = Norm(mean, std)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
    
    def __str__(self):
        return "U-Net0"



class UNet1(UNet0):
    """
    One more Maxpooling layer -> 1 level deeper
    """
    def __init__(self, n_channels, n_classes, mean, std):
        super().__init__(n_channels, n_classes, mean, std)
        self.down5 = Down(1024, 2048)
        self.up5 = Up(2048, 512)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up5(x6, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
    
    def __str__(self):
        return "U-Net1"

class UNet2(UNet1):
    def __init__(self, n_channels, n_classes, mean, std):
        super().__init__(n_channels, n_classes, mean, std)
        self.down6 = Down(2048, 4096)
        self.up6 = Up(4096, 2048)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up6(x7, x6)
        x = self.up5(x, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits

    def __str__(self):
        return "U-Net2"

class UNetPP(nn.Module):
    def __init__(self, n_channels, n_classes, mean, std, depth, base_kernel_num):
        super().__init__()
        self.depth = depth
        
        self.norm = Norm(mean, std)
        self.encoder = [DoubleConv(n_channels, base_kernel_num)]

        for i in range(1, depth):
            self.encoder.append(Down(base_kernel_num*2**(i-1), base_kernel_num*2**i))

        self.skip_conns_and_decoder = [[] for _ in range(depth-1)]
        for i in range(depth-1):
            for j in range(1,depth -i):
                #One node gets j+1 outputs of convs -> (j+1)*channels
                self.skip_conns_and_decoder[i].append(UpPP((base_kernel_num*2**i)*(j+1), base_kernel_num*2**i))
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

        outputs = [[encoded_x[i],] for i in encoded_x[1:]]
        for j in range(self.depth-1):
            for i in range(self.depth-1):
                if len(self.skip_conns_and_decoder[i]) > j:
                    X = self.skip_conns_and_decoder[i][j]
                    outputs[i].append(X(outputs[i+1][j], outputs[i]))

        return self.outConv(outputs[0][-1])