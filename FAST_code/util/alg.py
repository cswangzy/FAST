from pyexpat import model
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
import math
from util.utils import minus_model, scale_model, add_model


"""
    This class contains some MAB's algorithms.

    @author Alison Carrera

"""

class EpsilonGreedy(object):
    def __init__(self, n_arms=20,  eps=20, gamma=0.2):
        """
            EpsilonGreedy constructor.

            :param n_arms: Number of arms which this instance need to perform.
        """
        self.data_list = list()
        self.n_arms = n_arms
        self.number_of_selections = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        self.action = np.linspace(0.1, 1, self.n_arms)

        self.gamma = gamma
        self.eps = eps

    def select(self):
     #   print("reward", np.round(self.rewards, 4))
        p_arms = np.exp(self.rewards * self.eps) / np.sum(np.exp(self.rewards * self.eps))
     #   print("p of arms ", np.round(p_arms, 2))
        try:
            chosen_arm = np.random.choice(list(range(self.n_arms)), p=p_arms)
        except:
            chosen_arm = random.randint(0, self.n_arms-1)

     #   print("chosen arm", chosen_arm, self.action[chosen_arm])

        return chosen_arm

    def add_data(self, arm, rwd):
        self.data_list.append((arm, rwd))
        
        self.number_of_selections = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)

        for a, r in self.data_list[-50:]:
            self.reward(a, r)

    def reward(self, arm, rwd):
        """
            This method gives a reward for a given arm.

            :param chosen_arm: Value returned from select().
        """
        self.number_of_selections[arm] += 1
        if self.number_of_selections[arm] == 1:
            self.rewards[arm] = rwd
        else:
            self.rewards[arm] += self.gamma * (rwd - self.rewards[arm])


        

def data_sampling(class_num, device_num, rward, agents,epoch, sampling_rate, frac, sampling_index):
    
    if epoch == 0:
        for i in range(device_num):
            agents.append([])
            for j in range(class_num):
                sampling_rate[i][j] = 1
                agents[i].append(EpsilonGreedy())
        
    for i in range(device_num):
        for j in range(class_num):
            agents[i][j].add_data(sampling_index[i][j],rward)
            sampling_rate[i][j] = frac[agents[i][j].select()]
            sampling_index[i][j] = agents[i][j].select()
            
    return agents, sampling_rate, sampling_index
            


def fedavg(device_num,class_num):
    sampling_rate = np.ones((device_num,class_num), dtype=np.float)
    tau = 10
    return sampling_rate, tau


def randfast(device_num, class_num):
    sampling_rate = np.ones((device_num,class_num), dtype=np.float)
    for i in range(device_num):
        for j in range(class_num):
            sampling_rate[i][j] = random.random()        
    tau = random.randint(1,21)
    return sampling_rate, tau
