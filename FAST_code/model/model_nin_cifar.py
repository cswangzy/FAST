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

import torch
import torch.nn as nn
import torch.nn.functional as F

#实际上是NIN

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
          #  nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
'''
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class layer0(nn.Module):
    def __init__(self):
        super(layer0, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer1(nn.Module):
    def __init__(self):
        super(layer1, self).__init__()
        self.inchannel = 64
        self.resnet = self.make_layer(ResidualBlock, 64,  2, stride=1)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.resnet(x)
        return x


class layer2(nn.Module):
    def __init__(self):
        super(layer2, self).__init__()
        self.inchannel = 64
        self.resnet = self.make_layer(ResidualBlock, 128, 2, stride=2)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.resnet(x)
        return x

class layer3(nn.Module):
    def __init__(self):
        super(layer3, self).__init__()
        self.inchannel = 128
        self.resnet = self.make_layer(ResidualBlock, 256, 2, stride=2)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.resnet(x)
        return x

class layer4(nn.Module):
    def __init__(self):
        super(layer4, self).__init__()
        self.inchannel = 256
        self.resnet = self.make_layer(ResidualBlock, 512, 1, stride=2)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.resnet(x)
        return x

class layer5(nn.Module):
    def __init__(self):
        super(layer5, self).__init__()
        self.inchannel = 256
        self.resnet = self.make_layer(ResidualBlock, 512, 1, stride=2)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.resnet(x)
        return x

class layer6(nn.Module):
    def __init__(self):
        super(layer6,self).__init__()
        self.pool = nn.AvgPool2d(4,4) 
    def forward(self,x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

class layer7(nn.Module):
    def __init__(self):
        super(layer7,self).__init__()
        self.fc = nn.Linear(512, 10)
    def forward(self,x):
        x = self.fc(x)
      #  return x
        return F.log_softmax(x, dim=1)

def construct_resnet(partition_way, lr):
    models=[]
    optimizers=[]
    for i in range(0,len(partition_way)):
        if i==0:
            if partition_way[i] == 0:
                model = layer0()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==1:
            if partition_way[i] == 0:
                model = layer1()
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==2:
            if partition_way[i] == 0:
                model = layer2()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==3:
            if partition_way[i] == 0:
                model = layer3()
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==4:
            if partition_way[i] == 0:
                model = layer4()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==5:
            if partition_way[i] == 0:
                model = layer4()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)

        if i==6:
            if partition_way[i] == 0:
                model = layer5()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==7:
            if partition_way[i] == 0:
                model = layer6()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
    return models, optimizers

'''
'''
class NiN(nn.Module):
    def __init__(self, num_classes):
        super(NiN, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )
'''
class layer0(nn.Module):
    def __init__(self):
        super(layer0, self).__init__()
        self.conv = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False)
        self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer1(nn.Module):
    def __init__(self):
        super(layer1, self).__init__()
        self.conv = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0,bias=False)
        self.BN1 = nn.BatchNorm2d(160)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer2(nn.Module):
    def __init__(self):
        super(layer2, self).__init__()
        self.conv = nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0,bias=False)
        self.BN1 = nn.BatchNorm2d(96)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer3(nn.Module):
    def __init__(self):
        super(layer3,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(0.5)
    def forward(self,x):
        x = self.pool(x)
        x =self.drop(x)
        return x



class layer4(nn.Module):
    def __init__(self):
        super(layer4, self).__init__()
        self.conv = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2,bias =False)
        self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer5(nn.Module):
    def __init__(self):
        super(layer5, self).__init__()
        self.conv = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0,bias = False)
        self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer6(nn.Module):
    def __init__(self):
        super(layer6, self).__init__()
        self.conv = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0,bias=False)
        self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer7(nn.Module):
    def __init__(self):
        super(layer7,self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(0.5)
    def forward(self,x):
        x = self.pool(x)
        x =self.drop(x)
        return x



class layer8(nn.Module):
    def __init__(self):
        super(layer8, self).__init__()
        self.conv = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1,bias=False)
        self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer9(nn.Module):
    def __init__(self):
        super(layer9, self).__init__()
        self.conv = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0,bias = False)
        self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer10(nn.Module):
    def __init__(self):
        super(layer10, self).__init__()
        self.conv = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0,bias=False)
       # self.BN1 = nn.BatchNorm2d(10)
    def forward(self, x):
        x = self.conv(x)
      #  x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer11(nn.Module):
    def __init__(self):
        super(layer11,self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.drop = nn.Dropout(0.5)
    def forward(self,x):
        x = self.pool(x)
        x =self.drop(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)


def construct_nin_cifar(partition_way, lr):
    models=[]
    optimizers=[]
    for i in range(0,len(partition_way)):
        if i==0:
            if partition_way[i] == 0:
                model = layer0()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==1:
            if partition_way[i] == 0:
                model = layer1()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==2:
            if partition_way[i] == 0:
                model = layer2()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==3:
            if partition_way[i] == 0:
                model = layer3()
             #   optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==4:
            if partition_way[i] == 0:
                model = layer4()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
            models.append(model)
            optimizers.append(optimizer)
        if i==5:
            if partition_way[i] == 0:
                model = layer5()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==6:
            if partition_way[i] == 0:
                model = layer6()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==7:
            if partition_way[i] == 0:
                model = layer7()
              #  optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==8:
            if partition_way[i] == 0:
                model = layer8()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==9:
            if partition_way[i] == 0:
                model = layer9()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==10:
            if partition_way[i] == 0:
                model = layer10()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==11:
            if partition_way[i] == 0:
                model = layer11()
           #     optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
    return models, optimizers