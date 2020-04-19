#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:57:51 2020

@author: vikasran
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    expansion = 1

    def __init__(self, in_planes=3, planes=64, stride=1):
        super(Net, self).__init__()
       
        # ==================This is Prep Blosck============================
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 32x32x64
        self.bn1 = nn.BatchNorm2d(planes)

        # =================This is LAYER - 1 ======================== 
        in_planes= 64
        planes = 128
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 32x32x128
        self.maxpool1 = nn.MaxPool2d(2,2) # 16x16x128
        self.bn2 = nn.BatchNorm2d(planes) #16x16x128

        # ================Residue Block of layer - 1 =================
        in_planes = 128
        planes = 128
        self.Rconv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 16x16x128
        self.Rbn1 = nn.BatchNorm2d(planes)

        self.Rconv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 16x16x128
        self.Rbn2 = nn.BatchNorm2d(planes)

        # ==================This is LAYER - 2 =========================

        in_planes= 128
        planes = 256
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 16x16x256
        self.maxpool2 = nn.MaxPool2d(2,2) # 8x8x256
        self.bn3 = nn.BatchNorm2d(planes) # 8x8x256

        # ==================This is LAYER - 3 =========================

        in_planes= 256
        planes = 512
        self.conv4 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 8x8x512
        self.maxpool3 = nn.MaxPool2d(2,2) # 4x4x512
        self.bn4 = nn.BatchNorm2d(planes) # 4x4x512

        # ================Residue Block of layer - 3=================
        in_planes = 512
        planes = 512
        self.Rconv3 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 4x4x512
        self.Rbn3 = nn.BatchNorm2d(planes)

        self.Rconv4 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 4x4x512
        self.Rbn4 = nn.BatchNorm2d(planes)

        # ================= Max-Pooling Block ======================

        self.maxpool4 = nn.MaxPool2d(4,2)

        print("planes: %d"%planes)
        self.fc = nn.Linear(planes, 10)




        
        

        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.maxpool1(self.conv2(out))))

        direct_1 = out 

        out = F.relu(self.Rbn1(self.Rconv1(out)))
        out = F.relu(self.Rbn2(self.Rconv2(out)))



        out += direct_1#x3

        out = F.relu(self.bn3(self.maxpool2(self.conv3(out)))) # x4

        out = F.relu(self.bn4(self.maxpool3(self.conv4(out)))) # x5

        direct_2 = out

        out = F.relu(self.Rbn3(self.Rconv3(out)))
        out = F.relu(self.Rbn4(self.Rconv4(out)))

        out += direct_2 # x7

        out = self.maxpool4(out)
        print(type(out))
        
        print("Value of out.shape is:{}".format(out.shape))
       
        print("Value of out.size(0)is:{}".format(out.size(0)))
        print("Value of out.size(1)is:{}".format(out.size(1)))
        print("Value of out.size(2)is:{}".format(out.size(2)))

        print("Value of out.size(3)is:{}".format(out.size(3)))
        out = out.view(out.size(0), -1)
        print("Value of out.shape is:{}".format(out.shape))
        out = self.fc(out)
        return out



        #return F.log_softmax(out)



#model = model.cuda() if gpu else model.cpu()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(3, 32, 32))