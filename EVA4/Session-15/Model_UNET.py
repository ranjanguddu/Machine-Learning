import torchvision
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class Resnet_v1(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Resnet_v1, self).__init__()
        self.conv1_k3 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=False)

        self.conv3_k3 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.conv5_k1 = nn.Conv2d(input_channel,output_channel, kernel_size=1, stride=2, padding=0, bias=False)


    def forward(self, x):
        same = self.conv5_k1(x)
        out = self.conv1_k3(x)
        out = self.conv2_k1(out)
        out = self.bn1(out)
        out = self.relu(out)
        #out = nn.ReLU(nn.BatchNorm3d(self.conv2_k1(self.conv1_k3(x))), inplace=False)

        out = self.conv3_k3(out)
        out = self.conv4_k1(out)
        out = self.bn2(out)

        out += same

        out = self.relu(out)

        return out

class Resnet_v2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Resnet_v2, self).__init__()
        self.conv1_k3 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=False)

        self.conv3_k3 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.conv5_k3 = nn.ConvTranspose2d(input_channel,output_channel, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self,x):
        same = self.conv5_k3(x)
        out = self.conv1_k3(same)
        out = self.conv2_k1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3_k3(out)
        out = self.conv4_k1(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out

class UNET_Model(nn.Module):
    def __init__(self, n_class):
        super(UNET_Model, self).__init__()
        self.initconv = nn.Sequential(
            nn.Conv2d(6,64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rb1 = Resnet_v1(64,128)
        self.rb2 = Resnet_v1(128,256)
        self.rb3 = Resnet_v1(256, 512)
        
        self.rb4 = Resnet_v2(512, 256)
        self.rb5 = Resnet_v2(256, 128)
        self.rb6 = Resnet_v2(128, 64)
        self.rb7 = Resnet_v2(64, 64)
        self.rb8 = Resnet_v2(64, 32)

        self.lastconv = nn.Sequential(nn.Conv2d(32,n_class,kernel_size=1, stride=1, padding=0, bias=False))

        self.rb4d = Resnet_v2(512, 256)
        self.rb5d = Resnet_v2(256, 128)
        self.rb6d = Resnet_v2(128, 64)
        self.rb7d = Resnet_v2(64, 64)
        self.rb8d = Resnet_v2(64, 32)

        self.lastconvD = nn.Sequential(nn.Conv2d(32,n_class,kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, data):

        out0 = self.initconv(data)
        out1 = self.maxpool(out0)
        out2 = self.rb1(out1)
        out3 = self.rb2(out2)
        out4 = self.rb3(out3)

        out4 = nn.functional.interpolate(out4, scale_factor=2, mode='bilinear')

        out5 = self.rb4(out4)
        out5 += out3

        out5 = nn.functional.interpolate(out5, scale_factor=2, mode='bilinear')

        out6 = self.rb5(out5)
        out6 += out2

        out6 = nn.functional.interpolate(out6, scale_factor=2, mode='bilinear')

        out7 = self.rb6(out6)
        out7 += out1

        out7 = nn.functional.interpolate(out7, scale_factor=2, mode='bilinear')

        out8 = self.rb7(out7)
        out8 += out0

        out9 = nn.functional.interpolate(out8, scale_factor=2, mode='bilinear')
        out9 = self.rb8(out9)

        mask_out = self.lastconv(out9)

        ############  for Depth prediction ####################

        out5D = self.rb4d(out4)
        out5D = nn.functional.interpolate(out5D, scale_factor=2, mode='bilinear') 

        out6D = self.rb5d(out5D)
        out6D += out2

        out6D = nn.functional.interpolate(out6D, scale_factor=2, mode='bilinear')
        out7D = self.rb6d(out6D)

        out7D += out1

        out7D = nn.functional.interpolate(out7D, scale_factor=2, mode='bilinear')

        out8D = self.rb7d(out7D)
        out8D  += out0

        out9D = nn.functional.interpolate(out8D, scale_factor=2, mode='bilinear')
        out9D = self.rb8d(out9D)

        depth_out = self.lastconvD(out9D)
        
        return mask_out, depth_out