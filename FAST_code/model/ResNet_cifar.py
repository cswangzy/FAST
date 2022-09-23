import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import os
import copy

# # 分类数目
# num_class = 100
# # 各层数目
# resnet18_params = [2, 2, 2, 2]
# resnet34_params = [3, 4, 6, 3]
# resnet50_params = [3, 4, 6, 3]
# resnet101_params = [3, 4, 23, 3]
# resnet152_params = [3, 8, 36, 3]


# # 定义Conv1层
# def Conv1(in_planes, places, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
#         nn.BatchNorm2d(places),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )


# # 浅层的残差结构
# class BasicBlock(nn.Module):
#     def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 1):
#         super(BasicBlock,self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling

#         # torch.Size([1, 64, 56, 56]), stride = 1
#         # torch.Size([1, 128, 28, 28]), stride = 2
#         # torch.Size([1, 256, 14, 14]), stride = 2
#         # torch.Size([1, 512, 7, 7]), stride = 2
#         self.basicblock = nn.Sequential(
#             nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(places),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(places * self.expansion),
#         )

#         # torch.Size([1, 64, 56, 56])
#         # torch.Size([1, 128, 28, 28])
#         # torch.Size([1, 256, 14, 14])
#         # torch.Size([1, 512, 7, 7])
#         # 每个大模块的第一个残差结构需要改变步长
#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(places*self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # 实线分支
#         residual = x
#         out = self.basicblock(x)

#         # 虚线分支
#         if self.downsampling:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
#         return out


# # 深层的残差结构
# class Bottleneck(nn.Module):

#     # 注意:默认 downsampling=False
#     def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
#         super(Bottleneck,self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling

#         self.bottleneck = nn.Sequential(
#             # torch.Size([1, 64, 56, 56])，stride=1
#             # torch.Size([1, 128, 56, 56])，stride=1
#             # torch.Size([1, 256, 28, 28]), stride=1
#             # torch.Size([1, 512, 14, 14]), stride=1
#             nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
#             nn.BatchNorm2d(places),
#             nn.ReLU(inplace=True),
#             # torch.Size([1, 64, 56, 56])，stride=1
#             # torch.Size([1, 128, 28, 28]), stride=2
#             # torch.Size([1, 256, 14, 14]), stride=2
#             # torch.Size([1, 512, 7, 7]), stride=2
#             nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(places),
#             nn.ReLU(inplace=True),
#             # torch.Size([1, 256, 56, 56])，stride=1
#             # torch.Size([1, 512, 28, 28]), stride=1
#             # torch.Size([1, 1024, 14, 14]), stride=1
#             # torch.Size([1, 2048, 7, 7]), stride=1
#             nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(places * self.expansion),
#         )

#         # torch.Size([1, 256, 56, 56])
#         # torch.Size([1, 512, 28, 28])
#         # torch.Size([1, 1024, 14, 14])
#         # torch.Size([1, 2048, 7, 7])
#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(places*self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # 实线分支
#         residual = x
#         out = self.bottleneck(x)

#         # 虚线分支
#         if self.downsampling:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     def __init__(self,blocks, blockkinds, num_classes=num_class):
#         super(ResNet,self).__init__()

#         self.blockkinds = blockkinds
#         self.conv1 = Conv1(in_planes = 3, places= 64)

#         # 对应浅层网络结构
#         if self.blockkinds == BasicBlock:
#             self.expansion = 1
#             # 64 -> 64
#             self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
#             # 64 -> 128
#             self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
#             # 128 -> 256
#             self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
#             # 256 -> 512
#             self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

#             self.fc = nn.Linear(512, num_classes)

#         # 对应深层网络结构
#         if self.blockkinds == Bottleneck:
#             self.expansion = 4
#             # 64 -> 64
#             self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
#             # 256 -> 128
#             self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
#             # 512 -> 256
#             self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
#             # 1024 -> 512
#             self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

#             self.fc = nn.Linear(2048, num_classes)

#         self.avgpool = nn.AvgPool2d(2, stride=1)

#         # 初始化网络结构
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # 采用了何凯明的初始化方法
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def make_layer(self, in_places, places, block, stride):

#         layers = []

#         # torch.Size([1, 64, 56, 56])  -> torch.Size([1, 256, 56, 56])， stride=1 故w，h不变
#         # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 512, 28, 28])， stride=2 故w，h变
#         # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 1024, 14, 14])，stride=2 故w，h变
#         # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])， stride=2 故w，h变
#         # 此步需要通过虚线分支，downsampling=True
#         layers.append(self.blockkinds(in_places, places, stride, downsampling =True))

#         # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 256, 56, 56])
#         # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 512, 28, 28])
#         # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 1024, 14, 14])
#         # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 7, 7])
#         # print("places*self.expansion:", places*self.expansion)
#         # print("block:", block)
#         # 此步需要通过实线分支，downsampling=False， 每个大模块的第一个残差结构需要改变步长
#         for i in range(1, block):
#             layers.append(self.blockkinds(places*self.expansion, places))

#         return nn.Sequential(*layers)


#     def forward(self, x):

#         # conv1层
#         x = self.conv1(x)   # torch.Size([1, 64, 56, 56])

#         # conv2_x层
#         x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
#         # conv3_x层
#         x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
#         # conv4_x层
#         x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
#         # conv5_x层
#         x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])

#         x = self.avgpool(x) # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
#         x = x.view(x.size(0), -1)   # torch.Size([1, 2048]) / torch.Size([1, 512])
#         x = self.fc(x)      # torch.Size([1, 5])

#         return x

# def ResNet18():
#     return ResNet(resnet18_params, BasicBlock)

# def ResNet34():
#     return ResNet(resnet34_params, BasicBlock)

# def ResNet50():
#     return ResNet(resnet50_params, Bottleneck)

# def ResNet101():
#     return ResNet(resnet101_params, Bottleneck)

# def ResNet152():
#     return ResNet(resnet152_params, Bottleneck)


# def ResNet_cifar(lr):
#     model = ResNet18()
#     optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
#     return model, optimizer


#定义ResBlk类，包含两层卷积层及shortcut计算。
class ResBlk(nn.Module):
    '''
    resnet block
    '''

    def __init__(self,ch_in,ch_out,stride=1):
        '''

        :param ch_in:
        :param ch_out:
        :param stride:
        :return:
        '''
        super(ResBlk,self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,\
                stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,\
                stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        #shortcut计算，当输入输出通道不相等时，通过1*1的卷积核，转化为相等的通道数。
        self.shortcut = nn.Sequential()
        if ch_out != ch_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    #前项传播运算
    def forward(self,x):
        '''

        :param x:
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #shortcut
        #extra module:[b,ch_in,h,w] => [b,ch_out, h,w]
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#ResNet18结构框架
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64)
        )

        #followed blocks
        #重复的layer，分别为2个resblk.
        self.blk1 = self.make_layer(64,64,2,stride=1)
        self.blk2 = self.make_layer(64,128,2,stride=1)
        self.blk3 = self.make_layer(128,256,2,stride=1)
        self.blk4 = self.make_layer(256,512,2,stride=1)

        self.outlayer = nn.Linear(512*1*1,100)


    def make_layer(self,ch_in,ch_out,block_num,stride=1):
        '''
        #构建layer，包含多个ResBlk
        :param ch_in:
        :param ch_out:
        :param block_num:为每个blk的个数
        :param stride:
        :return:
        '''
        layers = []
        layers.append(ResBlk(ch_in,ch_out,stride))

        for i in range(1,block_num):
            layers.append(ResBlk(ch_out,ch_out))

        return nn.Sequential(*layers)

    def forward(self,x):
        '''

        :param x:
        :return:
        '''
        x = self.pre(x)

        #[b,64,h,w] => [b,1024,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        #[b,512,h,w] => [b,512,1,1]
        x = F.adaptive_avg_pool2d(x,[1,1])
        #print('after pool:', x.shape)
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x



def ResNet_cifar(lr):
    model = ResNet18()
    optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
    return model, optimizer