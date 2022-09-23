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
models = [
    nn.Sequential(    #layer 1
        nn.Conv2d(3,64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 2
        nn.Conv2d(64,64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2,stride=2),
    ),
    nn.Sequential(    #layer 3
        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 4
        nn.Conv2d(128,128,kernel_size=3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2,stride=2),
    ),
    nn.Sequential(    #layer 5
        nn.Conv2d(128,256,kernel_size=3,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 6
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 7
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2,stride=2),
    ),
    nn.Sequential(    #layer 8
        nn.Conv2d(256,512,kernel_size=3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 9
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 10
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2,stride=2),
    ),
    nn.Sequential(    #layer 11
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 12
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
    ),
    nn.Sequential(    #layer 13
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=1, stride=1),
    ),
    nn.Sequential(    #layer 14
          nn.Linear(512,4096),
          nn.ReLU(True),
          nn.Dropout(),
    ),
    nn.Sequential(    #layer 15
          nn.Linear(4096, 4096),
          nn.ReLU(True),
          nn.Dropout(),
    ),
    nn.Sequential(    #layer 16
          nn.Linear(4096,10),
    )
]
'''
class layer0(nn.Module):
    def __init__(self):
        super(layer0, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer1(nn.Module):
    def __init__(self):
        super(layer1, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer2(nn.Module):
    def __init__(self):
        super(layer2,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer3(nn.Module):
    def __init__(self):
        super(layer3, self).__init__()
        self.conv = nn.Conv2d(64, 128,kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(128)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer4(nn.Module):
    def __init__(self):
        super(layer4, self).__init__()
        self.conv = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(128)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer5(nn.Module):
    def __init__(self):
        super(layer5,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer6(nn.Module):
    def __init__(self):
        super(layer6, self).__init__()
        self.conv = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(256)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer7(nn.Module):
    def __init__(self):
        super(layer7, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(256)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer8(nn.Module):
    def __init__(self):
        super(layer8, self).__init__()
        self.conv = nn.Conv2d(256, 256,kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(256)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer9(nn.Module):
    def __init__(self):
        super(layer9,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer10(nn.Module):
    def __init__(self):
        super(layer10, self).__init__()
        self.conv = nn.Conv2d(256, 512, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer11(nn.Module):
    def __init__(self):
        super(layer11, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer12(nn.Module):
    def __init__(self):
        super(layer12, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer13(nn.Module):
    def __init__(self):
        super(layer13,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer14(nn.Module):
    def __init__(self):
        super(layer14, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer15(nn.Module):
    def __init__(self):
        super(layer15, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer16(nn.Module):
    def __init__(self):
        super(layer16, self).__init__()
        self.conv = nn.Conv2d(512, 512,kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        return x

class layer17(nn.Module):
    def __init__(self):
        super(layer17,self).__init__()
        self.pool1 = nn.MaxPool2d(1,1)
        self.pool2 = nn.AvgPool2d(1,1)
    def forward(self,x):
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view(-1,512)
        return x

class layer18(nn.Module):
    def __init__(self):
        super(layer18,self).__init__()
        self.fc = nn.Linear(512, 1024)
        self.drop = nn.Dropout()
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.drop(x)
        return x

class layer19(nn.Module):
    def __init__(self):
        super(layer19,self).__init__()
        self.fc = nn.Linear(1024, 1024)
        self.drop = nn.Dropout()
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.drop(x)
        return x

class layer20(nn.Module):
    def __init__(self):
        super(layer20,self).__init__()
        self.fc = nn.Linear(1024, 62)
    def forward(self,x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def construct_VGG_emnist(partition_way, lr):
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
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==2:
            if partition_way[i] == 0:
                model = layer2()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==3:
            if partition_way[i] == 0:
                model = layer3()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
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
            models.append(model)
            optimizers.append(optimizer)
        if i==5:
            if partition_way[i] == 0:
                model = layer5()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==6:
            if partition_way[i] == 0:
                model = layer6()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==7:
            if partition_way[i] == 0:
                model = layer7()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==8:
            if partition_way[i] == 0:
                model = layer8()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==9:
            if partition_way[i] == 0:
                model = layer9()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==10:
            if partition_way[i] == 0:
                model = layer10()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==11:
            if partition_way[i] == 0:
                model = layer11()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==12:
            if partition_way[i] == 0:
                model = layer12()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==13:
            if partition_way[i] == 0:
                model = layer13()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==14:
            if partition_way[i] == 0:
                model = layer14()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==15:
            if partition_way[i] == 0:
                model = layer15()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==16:
            if partition_way[i] == 0:
                model = layer16()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==17:
            if partition_way[i] == 0:
                model = layer17()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==18:
            if partition_way[i] == 0:
                model = layer18()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==19:
            if partition_way[i] == 0:
                model = layer19()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==20:
            if partition_way[i] == 0:
                model = layer20()
                optimizer = optim.SGD(params = model.parameters(), lr = lr)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
    return models, optimizers