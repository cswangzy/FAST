import os, random, shutil
import numpy as np
import time
import numpy as np
import pickle, struct, socket, math
from torchvision import datasets, transforms
import numpy as np
import pickle, struct, socket, math
import torch
import sys
import time
import torchvision
import random
import numpy as np
import math
import copy


def moveFile(fileDir,filenumber,rate):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
      #  rate= random.random()   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        print(rate)
     #   if rate>1:
         #   rate = random.random()
        rate = 0.05
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print(sample)
        for name in sample:
                shutil.copy(fileDir+name, tarDir+name)
        return

def non_iid_partition_quantity_imbalance(ratio, class_num, worker_num):
    distribution = np.zeros((worker_num, class_num))
    for i in range(worker_num):
        label = random.sample(range(0,class_num),ratio)
        for j in label:
            distribution[i][j] = 1
    partition_sizes = np.zeros((class_num, worker_num), dtype = np.float)
    for i in range(worker_num):
        for j in range(class_num):
            if distribution[i][j] == 1:
                partition_sizes[j][i] = 1.0/sum(distribution[:,j])
    print(distribution, partition_sizes)
 #   time.sleep(10)
    return partition_sizes

def non_iid_quantity_skew(vm_num, data_len, a, b, class_num): #a是均值，b是error
    np.random.seed(3)
    while True:
          c = np.random.normal(a,b,vm_num)
          flag=1
          for i in range(len(c)):
              if c[i]<0:
                  flag=0
          if flag==1:
              break
    #print(c)
    count = 0
    for i in range(len(c)-1):
        #print(c[i])
        c[i]=int(c[i])
        count+=c[i]
        #print(c[i])
    c[vm_num-1]=data_len - count
    while c[vm_num-1]<0 :
        for i in range(vm_num-1):
            c[i] = c[i]-1
        c[vm_num-1] += vm_num-1
    c = c.astype(int)
    
    d = []
    for i in range(len(c)):
        d.append(float(c[i])/sum(c))
    print(sum(d))
    partition_sizes = np.zeros((class_num, vm_num), dtype = np.float)
    for i in range(vm_num):
        for j in range(class_num):
            partition_sizes[j][i] = d[i]
    return partition_sizes

work_rememer = []

def non_iid_partition_distribution_imalance(ratio, class_num, worker_num=10):
    global work_rememer
    partition_sizes = np.ones((class_num, worker_num)) * ((1 - ratio) / (worker_num-1))
    for i in range(class_num):
        worker_idx = random.randint(0,worker_num-1)
        work_rememer.append(worker_idx)
        partition_sizes[i][worker_idx] = ratio
    print(partition_sizes)
 #   time.sleep(10)
    return partition_sizes

def partition_data(class_num, worker_num=10):
    partition_sizes = np.ones((class_num, worker_num)) * (1.0 / worker_num)
    return partition_sizes

path = "/data/zywang/Dataset/emnist/emnist_train/byclass_train"  # 指定需要读取文件的目录
files = os.listdir(path)  # 采用listdir来读取所有文件
files.sort()  # 排序

transform = transforms.Compose([
                        transforms.Resize(28),
                        #transforms.CenterCrop(227),
                        transforms.Grayscale(1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
        ])
testset = datasets.ImageFolder(path, transform = transform)

#rate = guass(10, 13000, 1300, 500)
#rate = rate()
vm_num = 30
class_num = 62
#rate = partition_data(class_num, vm_num)
#rate = non_iid_quantity_skew(vm_num, len(testset), len(testset)/vm_num, 500, class_num)
#rate = non_iid_partition_distribution_imalance(0.7, class_num, vm_num)
rate = non_iid_partition_quantity_imbalance(35, class_num, vm_num)
print("rate"+str(rate))
print(rate)
# fid = "/data/zywang/Dataset/IMAGE100/non_iid_07.txt"
# with open(fid,'a') as fid:
#     content = str(work_rememer).rstrip('\n') + '\n'
#     fid.write(content)
#     fid.flush()

count = 0
root = '/data/zywang/Dataset/emnist/device_30_label_quantity_skew_35'
if os.path.exists(root)==False:
    os.mkdir(root)


for i in range(vm_num):
    for j in range(class_num):
        tarDir = root + '/device'+str(i)+'/'    #移动到新的文件夹路径
        if os.path.exists(tarDir)==False:
            os.mkdir(tarDir)
        tarDir = root + '/device' + str(i) + '/' + str(j) + '/'
        if os.path.exists(tarDir)==False:
            os.mkdir(tarDir)

for file_ in files:  # 循环读取每个文件名
    fileDir = path+ '/' +file_+ '/'  # 源图片文件夹路径
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    print("filenumber"+str(filenumber))

    for index,file in enumerate(pathDir):
        print(index,":",file)    
        for i in range(vm_num) :
            if filenumber*sum(rate[count][0:i]) <= index and index < filenumber*sum(rate[count][0:i+1]):
               # if filenumber*sum(rate[count][0:i]) == index:
                tarDir = root + '/device'+str(i)+'/'    #移动到新的文件夹路径
                if os.path.exists(tarDir)==False:
                    os.mkdir(tarDir)
                tarDir = root + '/device' + str(i) + '/' + file_ + '/'
                if os.path.exists(tarDir)==False:
                    os.mkdir(tarDir)
                source_file = fileDir + '/' + file
                print("source_file:",source_file)
                shutil.copy(source_file,tarDir)

    for i in range(vm_num):
        if rate[count][i] == 0:
            tarDir = root + '/device'+str(i)+'/'    #移动到新的文件夹路径
            if os.path.exists(tarDir)==False:
                os.mkdir(tarDir)
            tarDir = root + '/device' + str(i) + '/' + file_ + '/'
            if os.path.exists(tarDir)==False:
                os.mkdir(tarDir)
            source_file = fileDir + '/' + pathDir[0]
            print("source_file:",source_file)
            shutil.copy(source_file,tarDir)               

         

    count+=1