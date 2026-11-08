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
from data import CIFAR10
from Attack import _split_for_parties
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

def generate_target_clean_vecs(model, stealset, args, attacker_list):
    target_set = stealset
    targetloader = torch.utils.data.DataLoader(target_set, batch_size=500, shuffle=True)
    attacker_clean_vec_dict = generate_vec(model, targetloader, args, attacker_list)
    return attacker_clean_vec_dict

def generate_vec(model, dataloader, args, attacker_list):
    num_party = args.P
    attacker_clean_vec_dict = {}
    with torch.inference_mode(mode=True):
        model.eval()
        for a in attacker_list:
            vecs = []
            for i, (x_in, y_in,_) in enumerate(dataloader):
                x_in = x_in.to(args.device)
                # x_list = torch.split(x_in, int(x_in.size(2) / num_party), dim=2)
                # remainder = x_in.size(2) % num_party
                # if remainder > 0:
                #     # 將最後一部份補充進 x_list
                #     x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                #     x_list = x_list[:-2]
                x_list = _split_for_parties(x_in, args.P, args.D)
                embedding = model.bottommodels[a](x_list[a]).detach().cpu().numpy()

                vecs.append(embedding)
            # vecs = np.array(vecs)
            # shape = vecs.shape
            # vecs = vecs.reshape((shape[1] , -1))
            vecs = np.concatenate(vecs, axis=0)
            attacker_clean_vec_dict[a] = vecs
    return attacker_clean_vec_dict
