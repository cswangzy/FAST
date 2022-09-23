#!/usr/bin/env python
import socket
import time
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import argparse
import torchvision
import torch.nn.functional as F
import random
import os
import socket
import threading
import time
import struct
from util.utils import send_msg, recv_msg, time_printer,add_model, scale_model, printer_model, time_duration
import copy
from torch.autograd import Variable
from model.AlexNet_Cifar import AlexNet_cifar
from model.LeNet_Emnist import Lenet_Emnist
from model.ResNet_cifar import ResNet_cifar
from util.utils import printer
from FAST.util.alg import fast, fedavg, csfedavg, dsfedavg, randfast
#from alg import layer_selection_generation, heals_algorithm, hierfavg_algorithm,hfl_noniid,hfel,Heals_random
import math
import numpy.ma as ma
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=1, metavar='N',
                        help='number of working devices ')
parser.add_argument('--model_type', type=str, default='LeNet', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='emnist', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--class_num', type=int, default=10, metavar='N',
                        help='number of class ')
parser.add_argument('--alg_type', type=int, default='0', metavar='N',  
                        help=' ip port')  
args = parser.parse_args()

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
device_gpu = torch.device("cuda" if True else "cpu")
   
lr = 0.01
device_num = args.device_num
class_num = 0
epoch_max = 500
acc_count = []
criterion = nn.CrossEntropyLoss()

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 50011))
#listening_sock.bind(('172.16.50.10', 50010))


device_sock_all = [None]*device_num
#connect to device
for i in range(device_num):
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    msg = recv_msg(client_sock)
    print('Got connection from node '+ str(msg[1]))
    print(client_sock)
    device_sock_all[msg[1]] = client_sock


#get the information about edge and device



if args.dataset_type == 'cifar100':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/cifar-100-python/test_cifar100', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    class_num = 100

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/cifar10/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    class_num = 10

elif args.dataset_type == 'emnist':
    transform = transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
          ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/emnist/emnist_train/byclass_test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)
    class_num = 62

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([
                            transforms.CenterCrop(144),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE100/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    class_num = 100

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Scale((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


def test(model, dataloader, dataset_name,epoch, total_delay):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            x=data.to(device_gpu)
            y = model(x)
            loss += criterion(y, target)
            pred = y.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
   # print(correct, len(dataloader.dataset))
    printer("Epoch {} Duration {}s Testing loss: {}  Accuracy :{}".format(epoch,total_delay,loss/len(dataloader),float(correct) / len(dataloader.dataset)))





#the algorithm stops when accuauracy of changed less than 2% in 10 epochs 
def train_stop():
    if len(acc_count)<11:
        return False
    max_acc = max(acc_count[len(acc_count)-10:len(acc_count)])
    min_acc = min(acc_count[len(acc_count)-10:len(acc_count)])
    if max_acc-min_acc <=0.0002:
        return True
    else:
        return False



rec_models = []
flow_count=0
if args.model_type == "AlexNet":
    model, optimizer = AlexNet_cifar(lr)
elif args.model_type == "LeNet":
    model, optimizer = Lenet_Emnist(lr)
elif args.model_type == "ResNet":
    model, optimizer = ResNet_cifar(lr)



model_size = 0
for para in model.parameters():
    model_size+=sys.getsizeof(para.storage())/(1024*1024/8)
printer("model size " +str(model_size)+"Mb")

def add_model_one(dst_model, src_model):
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    param1.data + dict_params2[name1].data)
    return dst_model

def minus_model_one(dst_model, src_model):
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    -param1.data + dict_params2[name1].data)
    return dst_model

def scale_model_one(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def model_aggregation(rec_models, device_num):
    for i in range(1, len(rec_models)):
        rec_models[0] = copy.deepcopy(add_model_one(rec_models[0], rec_models[i]))
    rec_models[0] = copy.deepcopy(scale_model_one(rec_models[0],1.0/len(rec_models)))
    return rec_models[0]



#fast(tcm, tcp, La, epi, model_divergence, device_num, lr, epoch, tau, bound,data_rate, class_num,frac, sampling_index)
agents = []
tcm = np.zeros(device_num, dtype = np.float)
tcp = np.zeros(device_num, dtype = np.float)
La = 0
model_divergence = 0
model_div = np.zeros(device_num, dtype=np.float)
epi = 0
tau = 1
bound1 = 100000
data_rate = 0
sampling_rate = np.ones((device_num,class_num), dtype=np.float)
frac = np.linspace(0, 1, 20)
sampling_index = np.ones((device_num,class_num), dtype=np.int)
for i in range(device_num):
    for j in range(class_num):
        sampling_index[i][j] = len(frac) - 1


def send_msg_to_device(sock_adr, msg):
    send_msg(sock_adr, msg)

def rev_msg_device(sock,sampling_rate):
    global rec_models
    global tcm, tcp, La, epi, model_divergence, data_rate, model_div, model_size, bandwidth
    msg = recv_msg(sock,"CLIENT_TO_SERVER")
    node_num = msg[2]
    if sum(sampling_rate[node_num]) > 0:
        rec_models.append(msg[1])
    tcm[node_num] = model_size/bandwidth[node_num]
    tcp[node_num] = msg[5]
    La = La + msg[6]/device_num
    epi = epi + msg[7]/device_num
    model_divergence = model_divergence + msg[8]/device_num
    model_div[node_num] = msg[8]
    data_rate = data_rate + msg[9]

bandwidth = np.zeros(device_num, dtype = np.int)
total_delay = 0
start_time  = time.time()
for epoch in range(0, epoch_max):

    if epoch % 10 == 0:
        for i in range(device_num):
           # bandwidth[i] = random.randint(5,10)
           bandwidth[i] = random.randint(1,10)+random.random()

    bound = bound1-total_delay
    
    if args.alg_type == 0:
        agents, sampling_rate, tau, sampling_index = fast(tcm, tcp, La, epi, model_divergence, device_num, lr, epoch, tau, bound,data_rate, class_num,frac, sampling_index,agents,sampling_rate)
        tcm = np.zeros(device_num, dtype = np.float)
        tcp = np.zeros(device_num, dtype = np.float)
        La = 0
        model_divergence = 0
        epi = 0
        data_rate = 0
    elif args.alg_type == 1:
        sampling_rate, tau = fedavg(device_num,class_num)
    elif args.alg_type == 2:
        sampling_rate, tau = csfedavg(model_div, model_divergence, device_num,class_num, tau)
        model_divergence = 0
        model_div = np.zeros(device_num, dtype=np.float)
    elif args.alg_type == 3:
        agents, sampling_rate, tau, sampling_index = dsfedavg(tcm, tcp, La, epi, model_divergence, device_num, lr, epoch, tau, bound,data_rate, class_num,frac, sampling_index,agents,sampling_rate)
        tcm = np.zeros(device_num, dtype = np.float)
        tcp = np.zeros(device_num, dtype = np.float)
        La = 0
        model_divergence = 0
        epi = 0
        data_rate = 0
    elif args.alg_type == 4:
        sampling_rate, tau = randfast(device_num,class_num)
        
    
    print("tau and sampling rate are "  + str(sampling_rate)+ str(tau))
#model distribution
    for i in range(device_num):
        msg = ['SERVER_TO_CLIENT', model, tau, sampling_rate[i], time.time()]
        send_device_msg = threading.Thread(target=send_msg_to_device, args=(device_sock_all[i], msg))
        send_device_msg.start()

    test(model, testloader, "Test set", epoch, total_delay)
    flow_count += model_size*device_num
    printer("flow_count "+str(2*flow_count)+'Mb')
    
#model aggregation  
    rev_msg_d = []
    for i in range(device_num):
        print("rec models")
        rev_msg_d.append(threading.Thread(target = rev_msg_device, args = (device_sock_all[i],sampling_rate)))
        rev_msg_d[i].start()
    for i in range(device_num):
        rev_msg_d[i].join()

    model = copy.deepcopy(model_aggregation(rec_models, device_num))
    max_1 = 0
    print(tcm,tcp)
    for i in range(device_num):
        if max_1 < tcp[i]+tcm[i]:
            max_1 = tcp[i]+tcm[i]
    total_delay += max_1

    # for model in models:
    #     for para in model.parameters():
    #         print(para)
    rec_models.clear()




    if train_stop():
        break

print("The traing process is over")



