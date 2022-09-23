import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy
import torchvision
import torch.nn.functional as F
import random
import numpy as np
hook = sy.TorchHook(torch)
import itertools
import fl_model_central_update_divergence_gpu_sh as fl_model
import math


data_tramission=[]

def time_printer(start_time, end_time, model,i,forward,location):
    durasec = int(start_time - end_time)
    duramsec = int((start_time - end_time - int(start_time - end_time)) * 1000)
    if forward==1:
        print("Forward, Layer:{}-{} location:{} output type:{} size:{:.2f}MB,runtime:{}s{}ms".format(i-1,i,location,model.shape,
            sys.getsizeof(model.copy().get().storage()) / (1024 * 1024), durasec, duramsec))
    else:
        print("Backward Layer:{}-{} location:{} output type:{} size:{:.2f}MB,runtime:{}s{}ms".format(i,i-1, location,model.shape, sys.getsizeof(
            model.copy().get().storage()) / (1024 * 1024), durasec, duramsec))


def model_partition(device, edge, partition_way,models):
    layer=0
    for model in models:
        if partition_way[layer] == 0:
            model.send(device)
          #  print("hello")
        else:
            model.send(edge)
          #  print("world")
        layer += 1



def all_partition(L):
    partition = list(itertools.product('01', repeat=L))
    a=[]
    for i in range(len(partition)):
        a.append([])
        for j in range(len(partition[i])):
            a[i].append(int(partition[i][j]))
    return a


def delay_count(B,C_device,C_edge,data_tramission,partition_way,forward_compute):
    delay = 0
    for i in range(len(partition_way)):    #compute delay
        if partition_way[i]==0:
            delay += float(forward_compute[i])/C_device
        else:
            delay += float(forward_compute[i]) / C_edge
    for i in range(len(partition_way)-1,-1,-1):
        if partition_way[i] == 0:  # compute delay
            delay += 2*float(forward_compute[i]) / C_device
        else:
            delay += 2*float(forward_compute[i]) / C_edge
    for i in range(len(partition_way)):  #tramission delay
        if i==0 and partition_way[i]!=0:
            delay += data_tramission[0]/B
        elif partition_way[i]!=partition_way[i-1]:
            delay+=2*data_tramission[i]/B
    return delay


class SplitNN(torch.nn.Module):
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        self.outputs = [None] * len(self.models)
        self.inputs = [None] * len(self.models)
        super().__init__()

    def forward(self, x):
        self.inputs[0] = x
      #  print("Input type:{} size:{:.2f}MB".format(x.shape, sys.getsizeof(x.copy().get().storage()) / (1024 * 1024)))
        start_time = time.time()
        self.outputs[0] = self.models[0](self.inputs[0])
        data_tramission.append(float(sys.getsizeof(self.inputs[0].copy().get().storage()) / (1024 * 1024) * 8))
        end_time = time.time()
      #  time_printer(start_time, end_time, self.outputs[0], 1, 1, self.models[0].location)
        data_tramission.append(float(sys.getsizeof(self.outputs[0].copy().get().storage()) / (1024 * 1024) * 8))
        for i in range(1, len(self.models)):
            self.inputs[i] = self.outputs[i - 1].detach().requires_grad_()
        #    print("Layer{} input type:{} size:{:.2f}MB".format(i + 1, self.inputs[i].shape,
                                                          #     sys.getsizeof(self.inputs[i].copy().get().storage()) / (
                                                                         #  1024 * 1024)))
            if self.outputs[i - 1].location != self.models[i].location:
                self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()
            start_time = time.time()
            self.outputs[i] = self.models[i](self.inputs[i])
            if i == 12:
                self.outputs[i] = self.outputs[i].view(-1, 512)
            end_time = time.time()
          #  time_printer(start_time, end_time, self.outputs[i], i + 1, 1, self.models[i].location)
            data_tramission.append(float(sys.getsizeof(self.outputs[i].copy().get().storage()) / (1024 * 1024) * 8))

        #  print(self.outputs[-1].get())
        self.outputs[-1] = self.outputs[-1].view(self.outputs[-1].shape[0], -1)
        return F.log_softmax(self.outputs[-1], dim=1)

    #  return self.outputs[-1]

    def backward(self, start_time, end_time):
        for i in range(len(self.models) - 2, -1, -1):
            grad_in = self.inputs[i + 1].grad.copy()
            if self.outputs[i].location != self.inputs[i + 1].location:
                grad_in = grad_in.move(self.outputs[i].location)
           # time_printer(start_time, end_time, grad_in, i + 2, 0, self.inputs[i + 1].location)
            start_time = time.time()
            self.outputs[i].backward(grad_in)
            end_time = time.time()
          #  if i == 0:
              # time_printer(start_time, end_time, self.inputs[0], i + 1, 0, self.inputs[i + 1].location)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    @property
    def location(self):
        return self.models[0].location if self.models and len(self.models) else None


def train_partition(x, target, splitNN):
    #1) Zero our grads
    splitNN.zero_grads()
    #2) Make a prediction
    pred = splitNN.forward(x)
    #3) Figure out how much we missed by
    criterion = nn.NLLLoss()
    loss = criterion(pred, target)
    #4) Backprop the loss on the end layer
    start_time = time.time()
    loss.backward()
    end_time= time.time()
    #5) Feed Gradients backward through the nework
    splitNN.backward(start_time,end_time)
    #6) Change the weights
   # start_time = time.time()
    splitNN.step()
   # end_time = time.time()
   # print(int(1000*(end_time-start_time)))
    return loss



def train_batch(batch_len, n):
    a = []
    clu = []
    for i in range(batch_len):
        a.append(i)
    random.shuffle(a)
    count = 0
    for i in range(n):
        clu.append(a[i])
    return clu