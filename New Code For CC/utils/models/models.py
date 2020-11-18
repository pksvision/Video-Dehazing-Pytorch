import argparse
import os

import numpy as np

#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from utils.options.options import opt


class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k,s,p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))


class CC_Module(nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()   

        self.layer1_1 = Conv2D_pxp(1, 32, 3,1,1)
        self.layer1_2 = Conv2D_pxp(1, 32, 5,1,2)
        self.layer1_3 = Conv2D_pxp(1, 32, 7,1,3)

        self.layer2_1 = Conv2D_pxp(96, 32, 3,1,1)
        self.layer2_2 = Conv2D_pxp(96, 32, 5,1,2)
        self.layer2_3 = Conv2D_pxp(96, 32, 7,1,3)

        self.layer3_1 = Conv2D_pxp(96, 1, 3,1,1)
        self.layer3_2 = Conv2D_pxp(96, 1, 5,1,2)
        self.layer3_3 = Conv2D_pxp(96, 1, 7,1,3)

        self.d_conv1 = nn.ConvTranspose2d(in_channels=193, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=32)
        self.d_relu1 = nn.ReLU()

        self.d_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = nn.BatchNorm2d(num_features=3)
        self.d_relu2 = nn.ReLU()


    def forward(self, input):
        #n*3*400*400
        # n=input.shape[0]
        # input_1=(input[:,0,:,:]).view(n,1,input.shape[2],input.shape[2]) #n*1*400*400
        # input_2=(input[:,1,:,:]).view(n,1,400,400) #n*1*400*400
        # input_3=(input[:,2,:,:]).view(n,1,400,400) #n*1*400*400

        input_1 = torch.unsqueeze(input[:,0,:,:], dim=1)
        input_2 = torch.unsqueeze(input[:,1,:,:], dim=1)
        input_3 = torch.unsqueeze(input[:,2,:,:], dim=1)
        
        #layer 1
        l1_1=self.layer1_1(input_1) #n*32*400*400
        l1_2=self.layer1_2(input_2) #n*32*400*400
        l1_3=self.layer1_3(input_3) #n*32*400*400
        
        #Input to layer 2- n*96*400*400
        input_2=torch.cat((l1_1,l1_2),1)
        input_2=torch.cat((input_2,l1_3),1)
        
        #layer 2
        l2_1=self.layer2_1(input_2) #n*32*400*400
        l2_2=self.layer2_2(input_2) #n*32*400*400
        l2_3=self.layer2_3(input_2) #n*32*400*400
        
        #Input to layer 3- n*96*400*400
        input_3=torch.cat((l2_1,l2_2),1)
        input_3=torch.cat((input_3,l2_3),1)
        
        #layer 3
        l3_1=self.layer3_1(input_3) #n*1*400*400
        l3_2=self.layer3_2(input_3) #n*1*400*400
        l3_3=self.layer3_3(input_3) #n*1*400*400
        
        #input to decoder unit
        temp_d1=torch.add(input_1,l3_1)#n*1*400*400
        temp_d2=torch.add(input_2,l3_2)#n*1*400*400
        temp_d3=torch.add(input_3,l3_3)#n*1*400*400
        input_d1=torch.cat((temp_d1,temp_d2),1)
        input_d1=torch.cat((input_d1,temp_d3),1)#n*3*400*400
        
        #decoder
        output_d1=self.d_relu1(self.d_bn1(self.d_conv1(input_d1)))#n*3*400*400
        final_output=self.d_relu2(self.d_bn2(self.d_conv2(output_d1)))#n*3*400*400
        
        return final_output
