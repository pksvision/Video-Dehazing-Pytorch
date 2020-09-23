import argparse
import os

import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from utils.options.options import opt


class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))


class CC_Module(nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()   

        layer1_1 = Conv2D_pxp(1, 32, 3)
        layer1_2 = Conv2D_pxp(1, 32, 5)
        layer1_3 = Conv2D_pxp(1, 32, 7)

        layer2_1 = Conv2D_pxp(96, 32, 3)
        layer2_2 = Conv2D_pxp(96, 32, 5)
        layer2_3 = Conv2D_pxp(96, 32, 7)

        layer3_1 = Conv2D_pxp(96, 1, 3)
        layer3_2 = Conv2D_pxp(96, 1, 5)
        layer3_3 = Conv2D_pxp(96, 1, 7)

        self.d_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=32)
        self.d_relu1 = nn.ReLU()

        self.d_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=3)
        self.d_relu1 = nn.ReLU()


    def forward(self, input):


        return 
