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



#the defination of VGG16, including 22 layer

'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
 
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x #F.softmax(x, dim=1)
'''
class layer0(nn.Module):
    def __init__(self):
        super(layer0, self).__init__()
        self.conv = nn.Conv2d(1, 6, 5,1,2)
      #  self.BN1 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
     #   x = self.BN1(x)
        return x

class layer1(nn.Module):
    def __init__(self):
        super(layer1,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer2(nn.Module):
    def __init__(self):
        super(layer2, self).__init__()
        self.conv = nn.Conv2d(6, 16, 5)
     #   self.BN1 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
      #  x = self.BN1(x)
        return x

class layer3(nn.Module):
    def __init__(self):
        super(layer3,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        x = self.pool(x)
        x = x.view(x.size()[0],-1)
        return x



class layer4(nn.Module):
    def __init__(self):
        super(layer4,self).__init__()
        self.fc = nn.Linear(16*5*5, 120)
     #   self.drop = nn.Dropout(0.1)
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
      #  x = self.drop(x)
        return x

class layer5(nn.Module):
    def __init__(self):
        super(layer5,self).__init__()
        self.fc = nn.Linear(120, 84)
    #    self.drop = nn.Dropout(0.1)
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
      #  x = self.drop(x)
        return x

class layer6(nn.Module):
    def __init__(self):
        super(layer6,self).__init__()
        self.fc = nn.Linear(84, 62)
    def forward(self,x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def construct_lenet_emnist(partition_way, lr):
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
                optimizer = None
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
    return models, optimizers