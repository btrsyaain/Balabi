import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime
import json
import time
import random
from poison_train_defense import CIFAR10
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def generate_vec(model,dataloader,unit,bottom_series):
    vecs = []
    for i, (x_in, y_in,_) in enumerate(dataloader):
        if unit != 1:
            # 依照參與者數分割資料集，由 y 軸切割
            x_part = x_in.split([int(x_in.size()[2] * unit), x_in.size()[2] - int(x_in.size()[2] * unit)], dim=2)
            pred = model(x_part[bottom_series]).detach().cpu().numpy()
        else:
            pred = model(x_in).detach().cpu().numpy()

        vecs.append(pred)
    vecs = np.array(vecs)
    shape = vecs.shape
    '''
    測試維度影響防禦效能
    #4096
    vecs = vecs.reshape((shape[0] * shape[1], shape[2]*shape[3]*shape[4]))
    '''
    vecs = vecs.reshape((shape[0] , -1))
    return vecs

def get_data_by_class(testset, target_class):
    data = []
    label = []
    for idx in range(len(testset)):
        x,y = testset[idx]
        if y==target_class:
            data.append({'id':idx,'data':x})
            label.append(y)
    return data,label

def generate_all_clean_vecs(class_num,model,testset,unit,bottom_series=0):
    all_clean_vecs = []

    transform_for_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    for label in range(class_num):
        target_data, target_label = get_data_by_class(testset, label)
        target_set = CIFAR10(target_data,target_label,transform=transform_for_test)
        # target_set = testset[label]
        targetloader = torch.utils.data.DataLoader(target_set, batch_size=1000, shuffle=True)
        vecs = generate_vec(model, targetloader,unit,bottom_series)
        all_clean_vecs.append(vecs)
    return np.array(all_clean_vecs)

def generate_target_clean_vecs(model,testset,unit,bottom_series=0):
    target_set = testset
    targetloader = torch.utils.data.DataLoader(target_set, batch_size=1000, shuffle=True)
    target_clean_vecs = generate_vec(model, targetloader, unit, bottom_series)
    return target_clean_vecs
