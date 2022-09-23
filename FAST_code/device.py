#!/usr/bin/env python
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
import socket
import time
import struct
import argparse
from util.utils import send_msg, recv_msg, time_printer, time_count
import copy
from torch.autograd import Variable
from model.AlexNet_Cifar import AlexNet_cifar
from model.LeNet_Emnist import Lenet_Emnist
from model.ResNet_cifar import ResNet_cifar
from util.utils import printer
import math
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=3, metavar='N',
                        help='number of working devices (default: 3)')
parser.add_argument('--node_num', type=int, default=0, metavar='N',
                        help='device index (default: 1)')
parser.add_argument('--use_gpu', type=int, default=0, metavar='N',
                        help=' ip port')
parser.add_argument('--model_type', type=str, default='LeNet', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--use_gpu_id', type=int, default=0, metavar='N',
                        help=' ip port')  
args = parser.parse_args()

if args.use_gpu_id == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.use_gpu_id == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
elif args.use_gpu_id == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
elif args.use_gpu_id == 3:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
elif args.use_gpu_id == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
elif args.use_gpu_id == 5:
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
elif args.use_gpu_id == 6:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
elif args.use_gpu_id == 7:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    
# if args.use_gpu == 0:
#     print('use gpu')
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
# else:
#     torch.set_default_tensor_type(torch.FloatTensor)

device_gpu = torch.device("cuda" if args.use_gpu == 0 else "cpu")


device_num = args.device_num
node_num = args.node_num
class_num = 0
batchsize = 64
lr = 0.001
sock_ps = socket.socket()
sock_ps.connect(('localhost', 50011))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_ps,msg)


print('---------------------------------------------------------------------------')


if args.dataset_type == 'cifar100':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    trainset = datasets.ImageFolder('/data/zywang/Dataset/cifar-100-python/device_30_label_quantity_skew_50/device'+str(node_num)+'/', transform = transform)
  #  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    class_num = 100
    batchsize = 32
    lr = 0.001

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
   # trainset = datasets.CIFAR10('/data/zywang/Dataset/cifar10', download=True, train=True, transform=transform)
    trainset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/device_30_label_quantity_skew_3/device'+str(node_num)+'/', transform = transform)
    #trainset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/train', transform = transform)
 #   trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    class_num = 10
    batchsize = 64
    lr = 0.01

elif args.dataset_type == 'emnist':
    transform = transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
          ])
    trainset = datasets.ImageFolder('/data/zywang/Dataset/emnist/device_30_label_quantity_skew_20/device'+str(node_num)+'/', transform = transform)
 #   trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    class_num = 62
    batchsize = 1024
    lr = 0.001

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([  transforms.Resize((144,144)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    #trainset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE100/25_device_train/device'+str(node_num)+'/', transform = transform)
    trainset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE100/train', transform = transform)
  #  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    class_num = 100
    batchsize = 64

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Resize((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
   # trainset = datasets.ImageFolder('/data/zywang/Dataset/image_coopfl/train', transform = transform)
    trainset = datasets.ImageFolder('/data/zywang/PartImagenet/train', transform = transform)
   # trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)




criterion = nn.CrossEntropyLoss()
local_update = 1
Datasize = len(trainset)
print("dataset is with " + str(Datasize) + " data samples and " + str(class_num)+ " classes")
class_size_pre = np.zeros(class_num, dtype = np.int)
class_size_after = np.zeros(class_num, dtype = np.int)
for i in trainset:
    class_size_pre[i[1]] += 1

def data_sampling(sampling_rate, trainset):
    for i in range(class_num):
        class_size_after[i] = class_size_pre[i] * sampling_rate[i]
    trainsetnew = []
    print(class_size_pre,class_size_after)
    
    if class_size_pre.all() == class_size_after.all():
        trainsetnew = trainset
    else:
        for i in range(class_num):
            a = random.randint(sum(class_size_pre[0:i]),sum(class_size_pre[0:i+1]))
            for j in range(class_size_after[i]):
                if a + j >= sum(class_size_pre[0:i+1]): 
                    trainsetnew.append(trainset[a+j-class_size_pre[i]])
                else:
                    trainsetnew.append(trainset[a+j])
    print(len(trainsetnew))
    return trainsetnew





def local_train(tau, sampling_rate):
    global model
    global optimizer
    global trainset
    global La, epi, model_divergence
    time_1 = time.time()
    for ep in range(tau):
        count = 0
        trainsetnew = data_sampling(sampling_rate, trainset)
        trainloader = torch.utils.data.DataLoader(trainsetnew, batch_size=batchsize, shuffle=True)

        print("strat training")
        for images, labels in trainloader:
            model.train()
            count+=1
            if count % 10 == 0:
                print("batch_training "+str(count))
#forward
            images, labels = images.to(device_gpu), labels.to(device_gpu) 
            y = model(images)
            loss = criterion(y, labels)  
            if count%50 == 0:
                print("trianing loss", loss) 
 #梯度初始化为zero
            optimizer.zero_grad()       
#backward   
            loss.backward()
#更新参数    
            optimizer.step()      
 #parater_estimate           
            if  count == 1 and ep == tau-1:
                mu_loss = []
                mu_w0 = []
                mu_w1 = []
                mu_grad0 = []
                mu_grad1 = []
                count_w = 0
                mu_loss.append(loss)
                for para in model.named_parameters():
                #    print(para,para[1].data)
                    mu_w0.append(copy.deepcopy(para[1].data))
                    if para == None:
                        mu_grad0.append(copy.deepcopy(para[1].data))
                    else:
                        mu_grad0.append(copy.deepcopy(para[1].grad.data))
            if  count == 2 and ep == tau-1:
                mu_loss.append(loss)
                for para in model.named_parameters():
                    mu_w1.append(copy.deepcopy(para[1].data))
                    if para == None:
                        mu_grad1.append(copy.deepcopy(para[1].data))
                    else:
                        mu_grad1.append(copy.deepcopy(para[1].grad.data))
                La, epi, model_divergence = parameter_estimate(mu_loss[0], mu_loss[1], mu_w0, mu_w1, mu_grad0, mu_grad1)
    return time.time()-time_1,La, epi, model_divergence, math.sqrt(len(trainset)/len(trainsetnew)),len(trainsetnew)

def parameter_estimate(loss0, loss1, w0, w1, w0_grad, w1_grad):
    new_w0 = np.array(w0[0].cpu())
    new_w1 = np.array(w1[0].cpu())
    new_w0_grad = np.array(w0_grad[0].cpu())
    new_w1_grad = np.array(w1_grad[0].cpu())
    for i in range(1,len(w0)):
        w0[i] = np.array(w0[i].cpu())
        new_w0 = np.append(new_w0,w0[i])
    for i in range(1,len(w1)):
        w1[i] = np.array(w1[i].cpu())
        new_w1 = np.append(new_w1,w1[i])
    for i in range(1,len(w0_grad)):
        w0_grad[i] = np.array(w0_grad[i].cpu())
        new_w0_grad = np.append(new_w0_grad,w0_grad[i])
    for i in range(1,len(w1_grad)):
        w1_grad[i] = np.array(w1_grad[i].cpu())
        new_w1_grad = np.append(new_w1_grad,w1_grad[i])
   # w0_grad = np.array(w0_grad)
    La = np.linalg.norm(new_w0_grad-new_w1_grad)/np.linalg.norm(new_w0-new_w1)
    epi = np.linalg.norm(new_w0_grad)
    model_divergence = np.linalg.norm(new_w0_grad)/La
    print("L=" + str(La)+ " ;epi=" +str(epi) +" ;model_divergence="+str(model_divergence))
    return La, epi, model_divergence


t_cp = 0 #compute
t_cm = 0 #communication
La = 0.02
epi = 1
model_divergence = 0

if args.model_type == "AlexNet":
    model, optimizer = AlexNet_cifar(lr)
elif args.model_type == "LeNet":
    model, optimizer = Lenet_Emnist(lr)
elif args.model_type == "ResNet":
    model, optimizer = ResNet_cifar(lr)


count_div = -1
div = 10
remember = 0
while True:
    
    count_div = count_div + 1
    msg = recv_msg(sock_ps,'SERVER_TO_CLIENT')
    print("received")
    model = copy.deepcopy(msg[1])
    optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
    tau = msg[2]
    sampling_rate = msg[3]
    if sum(sampling_rate) == 0:
        print("unselected")
        t_cp = 0
    else:
        print(tau, sampling_rate)
        data_size = 0
        t_cp, La, epi, model_divergence, data_rate, data_size = local_train(tau, sampling_rate)
        if count_div % div == 0:
        #     if args.model_type == "AlexNet":
        #         remember = random.uniform(0.8,2.2)
        #     elif args.model_type == "LeNet":
        #         remember = random.uniform(2.5,5.5)
        #         remember = float(remember)/10
        #     elif args.model_type == "ResNet":
        #         remember = random.uniform(1.8,4.2)
            if args.model_type == "AlexNet":
                remember = random.uniform(0.8,5.2)
            elif args.model_type == "LeNet":
                remember = random.uniform(1.5,7.5)
                remember = float(remember)/10
            elif args.model_type == "ResNet":
                remember = random.uniform(1.8,8.5)

        t_cp = float(remember*data_size*tau)/100 

    
    msg = ['CLIENT_TO_SERVER', model, node_num,t_cm, time.time(),t_cp, La, epi, model_divergence, data_rate]
    send_msg(sock_ps, msg) 
   


