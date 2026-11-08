import data
import time
import os
import math
import itertools
from functools import reduce
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cupy as cp
from cal_centers import train_mean_shift

class NeedMoreEpochs(Exception):
    pass
class TotalNumberWrong(Exception):
    pass

def get_start_defense_epoch(args):
    if args.DM == "VFLIP":
        s_d_e = args.epochs-20
    if args.DM == "SHAP_MAE":
        s_d_e = args.epochs-20
    if s_d_e <= 0:
        raise NeedMoreEpochs("Please make sure the number of epochs more than 20")
    return s_d_e

def get_shapley_samples(trn_x, trn_y, num_sample, label):
    targets = []
    t = label
    print(f"正在抽樣 label 為 {t} 的 sample")
    for idx in range(len(trn_y)):
        if trn_y[idx] == t:
            targets.append(trn_x[idx]['id'])

    sample_id = random.sample(targets, num_sample)
    sample_x = []
    label = []
    for idx in sample_id:
        sample_x.append(trn_x[idx])
        label.append(trn_y[idx])
    shapley_set = data.CIFAR10(sample_x,label)
    shapley_id = torch.tensor(sample_id)
    return shapley_set, shapley_id

@torch.no_grad()
def do_shapley(model, vec, y_in, args):
    # 每個類別分檔
    # 創建嵌套字典
    shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}  # 外層字典
    returned_shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}
    class_count_into_for_number_list = [0 for _ in range(args.num_class)]
    count_number_list = [0 for _ in range(args.num_class)]
    vec_base = vec.detach().clone()  # 確保 vec_base 不被修改
    num_combinations = 2**args.P
    
    for i in range(args.num_class):
        # print(f"現在正在計算 label 為 {i} shapley value")
        with torch.inference_mode(mode=True):
            model.eval()
            for p in range(args.P+1):
                combinations = []
                combinations.extend(list(itertools.combinations(list(range(args.P)),p)))
                for combination in combinations:
                    # print(combination)
                    vec_clone = vec_base.clone()
                    #將元素以 args.bottomModel_output_dim 個為一單位
                    indices = (torch.arange(args.bottomModel_output_dim * args.P) // args.bottomModel_output_dim)
                    combination_tensor = torch.tensor(combination, dtype=indices.dtype, device=indices.device)
                    # 使用 isin 判斷 mask
                    mask = torch.isin(indices, combination_tensor).view(1, -1)  # 將 mask 轉為 2D
                    mask = mask.expand(vec.size(0), -1)  # 匹配 vec 的形狀
                    vec_clone[mask] = 0
                    
                    label_mask = (y_in == i)
                    pred = model.top(vec_clone[label_mask])
                    softmax = nn.Softmax(dim=1)
                    pred_softmax = softmax(pred)
                    log_pred_softmax = pred_softmax[:,i]
                    
                    count_number_list[i]+= log_pred_softmax.shape[0]
                    # print(f"這 batch 新增{log_pred_softmax.shape[0]}筆資料")
                    # print(f"第{i}類別在第{into_shapley_count}個 batch 目前有{count_number_list[i]}筆資料")
                    
                    # time.sleep(1)
                    
                    # print(len(shapley_outer_dict[i][p]))
                    
                    if class_count_into_for_number_list[i]==0: #初始化
                        shapley_inner_key = list(range(args.P))+["Prob"]  # 內層字典的鍵
                        shapley_outer_dict[i] = {key: [1] for key in shapley_inner_key if key != "Prob"}  # 先初始化整個字典
                        
                        # 確保 "Prob" 欄位有 `num_combinations` 行的 Tensor
                        shapley_outer_dict[i]["Prob"] = [torch.empty(0, device=vec.device)] * num_combinations

                        pred_base = model.top(vec_base[label_mask])
                        pred_softmax_base = softmax(pred_base)
                        log_pred_softmax_base = pred_softmax_base[:, i]
                        shapley_outer_dict[i]["Prob"][0] = log_pred_softmax_base  # 設定初始值
                        
                    else:
                        if len(shapley_outer_dict[i][0]) < num_combinations: #隨便取一個來抓現在有多少紀錄
                            for participant in range(args.P):
                                if participant in combination:
                                    shapley_outer_dict[i][participant].append(0)
                                else:
                                    shapley_outer_dict[i][participant].append(1)
                                    
                            row_number= class_count_into_for_number_list[i]%num_combinations
                            
                            shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                            )
                            
                            # print(f"在dict內新增{log_pred_softmax.cpu().shape[0]}筆資料")
                            # print("新增後dict內有",{len(shapley_outer_dict[i]["Prob"][0][0])},"筆資料")
                            
                            # shapley_outer_dict[i]["Prob"].extend([[log_pred_softmax.reshape(log_pred_softmax.shape[0],-1).cpu()]])
                        else:
                            # print(f"情況3現在dict內有{len(shapley_outer_dict[i][0])}個row")
                            # print(f"欄位滿了，所以用插入的")
                            for m in range(len(shapley_outer_dict[i][0])):
                                count = 0
                                for part in range(args.P):
                                    if (part in combination) != (shapley_outer_dict[i][part][m]):
                                        # print(f"判定結果為 {part in combination} 和 {shapley_outer_dict[i][part][m]}")
                                        count+=1
                                if count == args.P:
                                    # print("新增前有",{len(shapley_outer_dict[i]["Prob"][0])},"筆資料")
                                    # print(f"在dict內新增{log_pred_softmax.cpu().shape[0]}筆資料")
                                    row_number= class_count_into_for_number_list[i]%num_combinations
                                    # shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                    #     shapley_outer_dict[i]["Prob"][row_number],log_pred_softmax,dim=0)
                                    '''
                                    測試當全部參與者都沒有放入向量時，預測機率設定為0，是否能夠提升
                                    shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                        (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                                    )
                                    '''
                                    zero_tensor = torch.zeros_like(log_pred_softmax)
                            
                                    shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                        (shapley_outer_dict[i]["Prob"][row_number], zero_tensor), dim=0
                                    )
                                    
                                    # shapley_outer_dict[i]["Prob"][m].extend([log_pred_softmax.cpu()])
                                    # print("新增後dict內有",{len(shapley_outer_dict[i]["Prob"][0])},"筆資料")
                                    # shapley_outer_dict[i]["Prob"][m].extend([log_pred_softmax.reshape(log_pred_softmax.shape[0],-1).cpu()])
                                # print(f"count:{count}")
                                # time.sleep(1)
                    # print(shapley_outer_dict[i]["Prob"])
                    # for c in range(num_combinations):
                    #     print("新增後dict內第",c,"row有",shapley_outer_dict[i]["Prob"][c].shape,"筆資料,加入與否為:",end=" ")
                    # print(f"應該要有{count_number_list[i]}筆資料")
                    class_count_into_for_number_list[i]+=1
                    # time.sleep(1)
        
        returned_shapley_inner_key = list(range(args.P))
        returned_shapley_outer_dict[i] = {key: torch.empty(0, device=vec.device) for key in returned_shapley_inner_key}
        for m in range(len(shapley_outer_dict[i][0])):
            shapley_outer_dict[i]["Prob"][m] = shapley_outer_dict[i]["Prob"][m].cpu()
        shapley_dataframe = pd.DataFrame.from_dict(shapley_outer_dict[i])
        # print(shapley_dataframe)
        list_each_participant_shapley_in_this_batch = calculate_each_participant_shap_in_one_batch(shapley_dataframe,args)
        for p in range(args.P):
            # print(returned_shapley_outer_dict[i][p])
            # print(list_each_participant_shapley_in_this_batch[p])
            returned_shapley_outer_dict[i][p] = torch.cat(
                (returned_shapley_outer_dict[i][p], list_each_participant_shapley_in_this_batch[p].to(args.device)), dim=0)
            # print(f"回傳後蒐集完的 dict 為:{returned_shapley_outer_dict[i][p]}")
        
    model.train()
    return returned_shapley_outer_dict

def do_every_class_for_every_sample_shapley(model, vec, args):
    '''
    測試時專用
    input : vec
    output : 這一 batch 每個 sample 在每個 class 的貢獻度
    '''
    # 每個類別分檔
    # 創建嵌套字典
    shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}  # 外層字典
    returned_shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}
    class_count_into_for_number_list = [0 for _ in range(args.num_class)]
    count_number_list = [0 for _ in range(args.num_class)]
    vec_base = vec.detach().clone()  # 確保 vec_base 不被修改
    num_combinations = 2**args.P
    
    for i in range(args.num_class):
        # print(f"現在正在計算 label 為 {i} shapley value")
        with torch.inference_mode(mode=True):
            model.eval()
            for p in range(args.P+1):
                combinations = []
                combinations.extend(list(itertools.combinations(list(range(args.P)),p)))
                for combination in combinations:
                    # print(combination)
                    vec_clone = vec_base.clone()
                    #將元素以 args.bottomModel_output_dim 個為一單位
                    indices = (torch.arange(args.bottomModel_output_dim * args.P) // args.bottomModel_output_dim)
                    combination_tensor = torch.tensor(combination, dtype=indices.dtype, device=indices.device)
                    # 使用 isin 判斷 mask
                    mask = torch.isin(indices, combination_tensor).view(1, -1)  # 將 mask 轉為 2D
                    mask = mask.expand(vec.size(0), -1)  # 匹配 vec 的形狀
                    vec_clone[mask] = 0
                    
                    # label_mask = (y_in == i)
                    pred = model.top(vec_clone)
                    softmax = nn.Softmax(dim=1)
                    pred_softmax = softmax(pred)
                    log_pred_softmax = pred_softmax[:,i]
                    
                    count_number_list[i]+= log_pred_softmax.shape[0]
                    # print(f"這 batch 新增{log_pred_softmax.shape[0]}筆資料")
                    # print(f"第{i}類別在第{into_shapley_count}個 batch 目前有{count_number_list[i]}筆資料")
                    
                    # time.sleep(1)
                    
                    # print(len(shapley_outer_dict[i][p]))
                    
                    if class_count_into_for_number_list[i]==0: #初始化
                        shapley_inner_key = list(range(args.P))+["Prob"]  # 內層字典的鍵
                        shapley_outer_dict[i] = {key: [1] for key in shapley_inner_key if key != "Prob"}  # 先初始化整個字典
                        
                        # 確保 "Prob" 欄位有 `num_combinations` 行的 Tensor
                        shapley_outer_dict[i]["Prob"] = [torch.empty(0, device=vec.device)] * num_combinations

                        pred_base = model.top(vec_base)
                        pred_softmax_base = softmax(pred_base)
                        log_pred_softmax_base = pred_softmax_base[:, i]
                        shapley_outer_dict[i]["Prob"][0] = log_pred_softmax_base  # 設定初始值
                        
                    else:
                        if len(shapley_outer_dict[i][0]) < num_combinations: #隨便取一個來抓現在有多少紀錄
                            for participant in range(args.P):
                                if participant in combination:
                                    shapley_outer_dict[i][participant].append(0)
                                else:
                                    shapley_outer_dict[i][participant].append(1)
                                    
                            row_number= class_count_into_for_number_list[i]%num_combinations
                            shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                            )
                            # print(f"在dict內新增{log_pred_softmax.cpu().shape[0]}筆資料")
                            # print("新增後dict內有",{len(shapley_outer_dict[i]["Prob"][0][0])},"筆資料")
                            
                            # shapley_outer_dict[i]["Prob"].extend([[log_pred_softmax.reshape(log_pred_softmax.shape[0],-1).cpu()]])
                        else:
                            # print(f"情況3現在dict內有{len(shapley_outer_dict[i][0])}個row")
                            # print(f"欄位滿了，所以用插入的")
                            for m in range(len(shapley_outer_dict[i][0])):
                                count = 0
                                for part in range(args.P):
                                    if (part in combination) != (shapley_outer_dict[i][part][m]):
                                        # print(f"判定結果為 {part in combination} 和 {shapley_outer_dict[i][part][m]}")
                                        count+=1
                                if count == args.P:
                                    # print("新增前有",{len(shapley_outer_dict[i]["Prob"][0])},"筆資料")
                                    # print(f"在dict內新增{log_pred_softmax.cpu().shape[0]}筆資料")
                                    row_number= class_count_into_for_number_list[i]%num_combinations
                                    # shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                    #     shapley_outer_dict[i]["Prob"][row_number],log_pred_softmax,dim=0)
                                    shapley_outer_dict[i]["Prob"][row_number] = torch.cat(
                                        (shapley_outer_dict[i]["Prob"][row_number], log_pred_softmax), dim=0
                        )
                                    # shapley_outer_dict[i]["Prob"][m].extend([log_pred_softmax.cpu()])
                                    # print("新增後dict內有",{len(shapley_outer_dict[i]["Prob"][0])},"筆資料")
                                    # shapley_outer_dict[i]["Prob"][m].extend([log_pred_softmax.reshape(log_pred_softmax.shape[0],-1).cpu()])
                                # print(f"count:{count}")
                                # time.sleep(1)
                    # print(shapley_outer_dict[i]["Prob"])
                    # for c in range(num_combinations):
                    #     print("新增後dict內第",c,"row有",shapley_outer_dict[i]["Prob"][c].shape,"筆資料,加入與否為:",end=" ")
                    # print(f"應該要有{count_number_list[i]}筆資料")
                    class_count_into_for_number_list[i]+=1
                    # time.sleep(1)
        
        returned_shapley_inner_key = list(range(args.P))
        returned_shapley_outer_dict[i] = {key: torch.empty(0, device=vec.device) for key in returned_shapley_inner_key}
        for m in range(len(shapley_outer_dict[i][0])):
            shapley_outer_dict[i]["Prob"][m] = shapley_outer_dict[i]["Prob"][m].cpu()
        shapley_dataframe = pd.DataFrame.from_dict(shapley_outer_dict[i])
        # print(shapley_dataframe)
        list_each_participant_shapley_in_this_batch = calculate_each_participant_shap_in_one_batch(shapley_dataframe,args)
        for p in range(args.P):
            # print(returned_shapley_outer_dict[i][p])
            # print(list_each_participant_shapley_in_this_batch[p])
            returned_shapley_outer_dict[i][p] = torch.cat(
                (returned_shapley_outer_dict[i][p], list_each_participant_shapley_in_this_batch[p].to(args.device)), dim=0)
            # print(f"回傳後蒐集完的 dict 為:{returned_shapley_outer_dict[i][p]}")
    return returned_shapley_outer_dict

def factorial(n):
    if n == 0:
        return 1
    else:
        return n*factorial(n-1)

# def calculate_each_participant_shap_in_one_batch(shapley_dataframe, args):
#     list_each_participant_shapley_in_this_batch = [0 for p in range(args.P)]
#     for participant in shapley_dataframe.columns:
#         if participant == "Prob":
#             pass
#         else:
#             # print(f"現在正在計算第{participant}的 Shapley Value")
#             participant_in = (shapley_dataframe[participant]==1)
#             participant_out = (shapley_dataframe[participant]==0)
#             participantin = shapley_dataframe[participant_in].reset_index(drop=True).copy()
#             participantout = shapley_dataframe[participant_out].reset_index(drop=True).copy()
            
#             participantin_tocalculate = participantin.drop(participant,axis=1).copy()
#             participantin_tocalculate["WithACC"] = participantin_tocalculate["Prob"]
#             participantin_tocalculate.drop("Prob",axis=1,inplace=True)
#             participantout_tocalculate = participantout.drop(participant,axis=1).copy()
#             participantout_tocalculate["WithoutACC"] = participantout_tocalculate["Prob"]
#             participantout_tocalculate.drop("Prob",axis=1,inplace=True)
            
#             other_participants_list = list(participantin_tocalculate.columns)
#             other_participants_list.remove("WithACC")
#             sheet = pd.merge(participantin_tocalculate,participantout_tocalculate,on=other_participants_list)
            
#             for i in sheet.index:
#                 # print(f"\r現在進度為：{round(i/len(participantin_tocalculate.index),2)*100}%"," "*20,end="")
#                 s=(sheet.loc[i][other_participants_list] == 1).sum()+1 #納入的參與者人數
#                 # print("s為：",s)
#                 n = args.P #特徵總數
#                 # print("n為:",n)
#                 weight = (factorial(s-1)*factorial(n-s)/factorial(n)) #權重公式
#                 contrib = sheet.loc[i,"WithACC"]-sheet.loc[i,"WithoutACC"]
#                 # print(f"第{i}個組合contribution list為:{contrib}")
#                 # print(f"s 為:{s}")
#                 # print(f"weight 為:{weight}")
#                 # print(f"加入 list 的 value值為:{(weight * contrib)}")
#                 list_each_participant_shapley_in_this_batch[int(participant)] += (weight * contrib)
#                 # print(f"目前陣列內容:{list_each_participant_shapley_in_this_batch[participant]}")
#                 # time.sleep(5)
#                 # print("這次的貢獻值為：",weight * contrib)
#             # print(participant,"在此batch中之 Shapley Value 為：",list_each_participant_shapley_in_this_batch[int(participant)])
#             # try:
#             #     print("len")
#             #     print("list長度為:",len(list_each_participant_shapley_in_this_batch[int(participant)]))
#             # except:
#             #     print("shape")
#             #     print("shape長度為:",list_each_participant_shapley_in_this_batch[participant].shape)
#             # print(f"即將進入第{int(participant)+1}個參與者")
#             # time.sleep(10)
    
#     # for k in range(args.P):
#     #     print(f"準備回傳的第{k}個list有{len(list_each_participant_shapley_in_this_batch[k])}筆資料")
#     # time.sleep(1)
            
#     return list_each_participant_shapley_in_this_batch

#原版，沒有套用 P-shapley 

import pandas as pd
import numpy as np
import torch
from math import factorial
from typing import List, Union

TensorLike = Union[torch.Tensor, np.ndarray, float]

def calculate_each_participant_shap_in_one_batch(
        shapley_dataframe: pd.DataFrame,
        args,
        normalize: str | None = None,
        eps: float = 1e-12
    ):
    """
    計算單一 batch 中每位參與者的 Shapley Value。
    
    Parameters
    ----------
    shapley_dataframe : pd.DataFrame
        內容同前；'Prob' 欄可為 tensor、ndarray 或 float。
    args : Namespace
        至少需包含 args.P (參與者總數)。
    normalize : {"proportion", None}
        - None         : 直接回傳 raw Shapley。
        - "proportion" : 先將所有 Shapley 值平移至非負，再以總和做比例化。
    eps : float
        防止除 0 的極小常數。
    """
    # ---------- (0) 推斷向量維度並初始化 ----------
    first_vec = shapley_dataframe["Prob"].iloc[0]
    if isinstance(first_vec, torch.Tensor):
        zero_vec = torch.zeros_like(first_vec, dtype=torch.float64)
    elif isinstance(first_vec, np.ndarray):
        zero_vec = np.zeros_like(first_vec, dtype=np.float64)
    else:                         # scalar
        zero_vec = 0.0

    shap_values: List[TensorLike] = [
        zero_vec.clone() if isinstance(zero_vec, torch.Tensor)
        else (zero_vec.copy() if isinstance(zero_vec, np.ndarray) else 0.0)
        for _ in range(args.P)
    ]

    # ---------- (1) 累積每位參與者的 Shapley ----------
    for participant in shapley_dataframe.columns:
        if participant == "Prob":
            continue

        df_in  = shapley_dataframe[shapley_dataframe[participant] == 1].reset_index(drop=True)
        df_out = shapley_dataframe[shapley_dataframe[participant] == 0].reset_index(drop=True)

        df_in  = df_in .drop(participant, axis=1).copy()
        df_out = df_out.drop(participant, axis=1).copy()
        df_in ["WithACC"]    = df_in .pop("Prob")
        df_out["WithoutACC"] = df_out.pop("Prob")

        other_cols = list(df_in.columns); other_cols.remove("WithACC")
        sheet = pd.merge(df_in, df_out, on=other_cols)

        n = args.P
        for _, row in sheet.iterrows():
            s = (row[other_cols] == 1).sum() + 1
            weight = factorial(s-1) * factorial(n-s) / factorial(n)
            contrib = row["WithACC"] - row["WithoutACC"]
            shap_values[int(participant)] += weight * contrib

    # ---------- (2) proportion 正規化（可選） ----------
    if normalize == "proportion":
        # 堆疊成 (P,d) 或 (P,) 方便向量化
        stacked = torch.stack(shap_values) if isinstance(shap_values[0], torch.Tensor) \
                  else np.stack(shap_values, axis=0)

        # (2‑A) 平移：確保所有值 ≥ 0
        min_val = stacked.min(dim=0).values if isinstance(stacked, torch.Tensor) else stacked.min(axis=0)
        shift   = (-min_val).clamp(min=0) + eps if isinstance(stacked, torch.Tensor) \
                  else np.clip(-min_val, a_min=0, a_max=None) + eps
        shifted = stacked + shift                    # broadcasting

        # (2‑B) 計算佔比
        denom = shifted.sum(dim=0) if isinstance(shifted, torch.Tensor) else shifted.sum(axis=0)
        proportion = shifted / denom                 # broadcasting

        shap_values = [v for v in proportion]        # 轉回 list

    return shap_values

''' 加入 p-shapley
import torch
import numpy as np
import pandas as pd
from math import factorial
from typing import List

# 先定義 Mish 函數
def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

def calculate_each_participant_shap_in_one_batch(
        shapley_dataframe: pd.DataFrame,
        args,
        normalize: str | None = None,
        eps: float = 1e-12
    ):
    """
    計算單一 batch 中每位參與者的 P-Shapley Value（使用 Mish 校正）。

    Parameters
    ----------
    shapley_dataframe : pd.DataFrame
        'Prob' 欄可為 tensor、ndarray 或 float。
    args : Namespace
        至少需包含 args.P (參與者總數)。
    normalize : {"proportion", None}
        - None         : 直接回傳 raw Shapley。
        - "proportion" : 先將所有 Shapley 值平移至非負，再以總和做比例化。
    eps : float
        防止除以 0 的極小常數。
    """
    # ---------- (0) 推斷向量維度並初始化 ----------
    first_vec = shapley_dataframe["Prob"].iloc[0]
    if isinstance(first_vec, torch.Tensor):
        zero_vec = torch.zeros_like(first_vec, dtype=torch.float64)
    elif isinstance(first_vec, np.ndarray):
        zero_vec = np.zeros_like(first_vec, dtype=np.float64)
    else:
        zero_vec = 0.0

    shap_values: List = [
        zero_vec.clone() if isinstance(zero_vec, torch.Tensor)
        else (zero_vec.copy() if isinstance(zero_vec, np.ndarray) else 0.0)
        for _ in range(args.P)
    ]

    # ---------- (1) 累積每位參與者的 Shapley ----------
    for participant in shapley_dataframe.columns:
        if participant == "Prob":
            continue

        df_in  = shapley_dataframe[shapley_dataframe[participant] == 1].reset_index(drop=True)
        df_out = shapley_dataframe[shapley_dataframe[participant] == 0].reset_index(drop=True)

        df_in  = df_in.drop(participant, axis=1).copy()
        df_out = df_out.drop(participant, axis=1).copy()

        # 💡 在這裡做 Mish 校正
        if isinstance(first_vec, torch.Tensor):
            df_in["WithACC"] = df_in.pop("Prob").apply(lambda x: mish(x))
            df_out["WithoutACC"] = df_out.pop("Prob").apply(lambda x: mish(x))
        else:
            df_in["WithACC"] = df_in.pop("Prob")
            df_out["WithoutACC"] = df_out.pop("Prob")


        other_cols = list(df_in.columns); other_cols.remove("WithACC")
        sheet = pd.merge(df_in, df_out, on=other_cols)

        n = args.P
        for _, row in sheet.iterrows():
            s = (row[other_cols] == 1).sum() + 1
            weight = factorial(s-1) * factorial(n-s) / factorial(n)
            contrib = row["WithACC"] - row["WithoutACC"]
            shap_values[int(participant)] += weight * contrib

    # ---------- (2) proportion 正規化（可選） ----------
    if normalize == "proportion":
        stacked = torch.stack(shap_values) if isinstance(shap_values[0], torch.Tensor) \
                  else np.stack(shap_values, axis=0)

        min_val = stacked.min(dim=0).values if isinstance(stacked, torch.Tensor) else stacked.min(axis=0)
        shift   = (-min_val).clamp(min=0) + eps if isinstance(stacked, torch.Tensor) \
                  else np.clip(-min_val, a_min=0, a_max=None) + eps
        shifted = stacked + shift

        denom = shifted.sum(dim=0) if isinstance(shifted, torch.Tensor) else shifted.sum(axis=0)
        proportion = shifted / denom

        shap_values = [v for v in proportion]

    return shap_values
'''


def calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args, tag=0, draw=False, id_in_class=None, steal_id_in_class=None):
    static_class_dict = {outer_key: {} for outer_key in range(args.num_class)}
    static_participant_key = list(range(args.P))
    # 新增 q1 與 q3 四分位數

    for i in range(args.num_class):
        static_class_dict[i] = {key: {} for key in static_participant_key}
        
        if i == args.label and tag == 0:
            steal_index_in_class = [id_ in steal_id_in_class for id_ in id_in_class]
            steal_mask = torch.tensor(steal_index_in_class, dtype=torch.bool)
            non_steal_mask = ~steal_mask
        
        for j in range(args.P):
            values = all_shapley_outer_dict[i][j].cpu().numpy()
            mean_value = np.mean(values)
            median_value = np.median(values)
            std_dev = np.std(values)
            variance = np.var(values)
            mad_value = np.median(np.abs(values - median_value))
            # 計算第一與第三四分位數
            q1_value = np.percentile(values, 25)
            q3_value = np.percentile(values, 75)
            
            up_distance = q3_value-median_value
            down_distance = median_value-q1_value
            
            ratio = math.log(args.P, 2)
            # ratio = 1
            
            # ❶ 假如使用者 CLI 明確給 alpha，就直接採用
            if args.shap_alpha is not None:
                alpha = args.shap_alpha
            # ❷ 否則沿用你原本依資料集決定的預設
            elif args.D == "CIFAR10":
                if args.AM == "TIFS":
                    alpha = 1.7
                elif args.AM == "VILLIAN":
                    alpha = 2.0
            elif args.D == "CINIC10":
                if args.AM == "TIFS":
                    alpha = 1.3
                elif args.AM == "VILLIAN":
                    alpha = 1.5
            else: # BM
                if args.AM == "TIFS":
                    alpha = 1.3
                elif args.AM == "VILLIAN":
                    alpha = 1.5
            
            # # 自創
            up_threshold = median_value + up_distance/(ratio*alpha)
            down_threshold = median_value - down_distance/(ratio*alpha)
            
            #MAD
            # up_threshold = median_value + mad_value
            # down_threshold = median_value
            # down_threshold = median_value - mad_value+6
            
            
            static_class_dict[i][j] = {
                "mean": mean_value,
                "median": median_value,
                "std": std_dev,
                "variance": variance,
                "mad": mad_value,
                "q1": q1_value,
                "q3": q3_value,
                "up_threshold": up_threshold,
                "down_threshold": down_threshold
            }
            
            if draw:
                plt.figure(figsize=(10, 6))
                sns.histplot(values, bins=50, kde=True, color='skyblue')
                
                # 僅標示中位數、median_value ± mad_value 以及第一與第三四分位數
                plt.axvline(median_value, color='blue', linestyle='--', label=f'Median: {median_value:.4f}')
                plt.axvline(down_threshold, color='green', linestyle='-.', label=f'Down Threshold: {down_threshold:.4f}')
                plt.axvline(up_threshold, color='green', linestyle='-.', label=f'Up Threshold: {up_threshold:.4f}')
                # plt.axvline(median_value - mad_value, color='green', linestyle='-.', label=f'Median - MAD: {median_value - mad_value:.4f}')
                # plt.axvline(median_value + mad_value, color='green', linestyle='-.', label=f'Median + MAD: {median_value + mad_value:.4f}')
                # plt.axvline(q1_value, color='purple', linestyle='--', label=f'Q1: {q1_value:.4f}')
                # plt.axvline(q3_value, color='brown', linestyle='--', label=f'Q3: {q3_value:.4f}')
                
                plt.title(f'Shapley Value Distribution (Class {i}, P {j})')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.legend()
                if tag == 0:
                    filename = f'train_class_{i}_P_{j}_distribution.png'
                elif tag == 1:
                    filename = f'test_class_{i}_P_{j}_distribution.png'
                else:
                    filename = f'poisoned_class_{i}_P_{j}_distribution.png'
                plt.savefig(filename)
                plt.close()
                
                if tag == 0 and i == args.label:
                    steal_values = all_shapley_outer_dict[i][j][steal_mask].cpu().numpy()
                    non_steal_values = all_shapley_outer_dict[i][j][non_steal_mask].cpu().numpy()
                    
                    plt.figure(figsize=(10, 6))
                    sns.histplot(steal_values, bins=50, kde=True, color='red', label='Poisoned')
                    sns.histplot(non_steal_values, bins=50, kde=True, color='blue', label='Clean')
                    plt.title(f'Steal vs Non-Steal Samples - Class {i} P {j}')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.savefig(f'class_{i}_P_{j}_steal_vs_non_steal.png')
                    plt.close()
    if tag == 0:
        return static_class_dict
    else:
        return 0

def cal_shapley_distribution(counter_dict, args):
    """
    計算當參與者的 shapley value count 達到標準（例如9）時，
    統計同一筆資料中其他參與者的 count 分佈：
      - 針對所有其他參與者（分佈數量）
      - 分別針對正常參與者與其他攻擊者（除了該攻擊者本身）
    並計算各組別在各個 count 值（例如 0 到標準值）的平均、最大、最小值以及比例（相對於群體總人數）。

    輸入參數：
      counter_dict: dict，鍵為參與者編號，值為長度為 N 的 torch.Tensor（N例如10000），數值代表該筆資料的 count
      args: 具有下列屬性：
            - P: 參與者總數
            - attacker_list: 攻擊者編號列表
            - num_class: 類別數 (例如 10，此時標準值為 num_class - 1，即9)
    """
    attacker_list = args.attacker_list
    total_participants = args.P
    standard = args.num_class - 1  # 例如 9
    counter_np = {p: counter_dict[p].cpu().numpy() for p in range(total_participants)}
    num_samples = len(next(iter(counter_np.values())))
    
    for p in range(total_participants):
        mask = (counter_np[p] == standard)
        indices = np.where(mask)[0]  # 被偵測到的 sample index
        num_detected = len(indices)
        perc = round(num_detected / num_samples * 100, 2)
        # print(f"參與者 {p} 有 {num_detected} 筆資料達到標準 (達標比例: {perc}%)")
        
        if num_detected == 0:
            continue
        
        overall_counts = defaultdict(int)
        normal_counts = defaultdict(int)
        other_attacker_counts = defaultdict(int)
        
        for idx in indices:
            for other_p in range(total_participants):
                if other_p == p:
                    continue
                cnt_val = int(counter_np[other_p][idx])
                cnt_val = max(0, min(cnt_val, standard))
                overall_counts[cnt_val] += 1
                if other_p in attacker_list:
                    other_attacker_counts[cnt_val] += 1
                else:
                    normal_counts[cnt_val] += 1
        
        total_normal = total_participants - len(attacker_list)
        total_other_attacker = max(len(attacker_list) - 1, 0)
        # print(f"\n統計結果(當參與者 {p} 達到標準時)：")
        
        x_vals = list(range(standard + 1))
        overall_arr = np.array([overall_counts[v] for v in x_vals])
        normal_arr = np.array([normal_counts[v] for v in x_vals])
        attacker_arr = np.array([other_attacker_counts[v] for v in x_vals])
        
        plt.figure(figsize=(10, 6))
        plt.bar(x_vals, attacker_arr, color='red', label='Attacker')
        plt.bar(x_vals, normal_arr, bottom=attacker_arr, color='blue', label='Normal participant')
        plt.xlabel('The setting Shapley Value ')
        plt.ylabel('Nmmber of participants')
        plt.title(f'The Distribution when participant {p} over the Shapley Value threshold')
        plt.legend()
        plt.savefig(f'participant_{p}.png')
        plt.close()
        
        # for v in x_vals:
        #     overall_avg = overall_counts[v] / num_detected if num_detected > 0 else 0
        #     normal_avg = normal_counts[v] / num_detected if num_detected > 0 else 0
        #     attacker_avg = other_attacker_counts[v] / num_detected if num_detected > 0 else 0
        #     print(f"--- shapley count = {v} ---")
        #     print(f"全部其他參與者: 總數 = {overall_counts[v]}, 平均數 = {overall_avg:.2f}")
        #     print(f"正常參與者: 總數 = {normal_counts[v]}, 平均數 = {normal_avg:.2f}")
        #     if total_other_attacker > 0:
        #         print(f"其他攻擊者: 總數 = {other_attacker_counts[v]}, 平均數 = {attacker_avg:.2f}")
        #     else:
        #         print("其他攻擊者: 無（僅有一位攻擊者）")
        # print("\n" + "="*50 + "\n")

def do_predicted_class_for_every_sample_shapley(model, vec, args):
    '''
    測試時專用，只計算每筆資料預測類別下各參與者的貢獻度
    input : vec
    output : 這一 batch 中，每個參與者在各筆資料預測類別上的 Shapley value（以 list 表示，每個元素為一個 torch.Tensor）
    '''
    returned_preded_shapley_key = list(range(args.P))
    returned_preded_shapley_dict = {key: torch.empty(0, device=vec.device) for key in returned_preded_shapley_key}
    # 先備份輸入資料
    vec_base = vec.detach().clone()
    
    # 取得 base prediction 與每筆資料的預測類別
    with torch.inference_mode():
        model.eval()
        base_pred = model.top(vec_base)
        softmax = nn.Softmax(dim=1)
        base_softmax = softmax(base_pred)
        # 對每個 sample 取最高機率的類別作為預測類別
        predicted_class = base_softmax.argmax(dim=1)  # shape: (batch_size,)
    
    num_combinations = 2**args.P
    # 建立存放所有組合資訊的字典，鍵為各參與者（0 ~ args.P-1）及 "Prob"
    shapley_dict = {key: [] for key in list(range(args.P)) + ["Prob"]}
    
    # 第一組合：不 mask 任何參與者（皆納入），直接使用 base 的預測機率（注意：取每筆 sample 預測類別的機率）
    base_prob = base_softmax[torch.arange(vec.size(0)), predicted_class].cpu()
    shapley_dict["Prob"].append(base_prob)
    for participant in range(args.P):
        shapley_dict[participant].append(1)  # 1 表示該參與者被納入
    
    # 接下來遍歷所有至少 mask 一位參與者的組合
    for p in range(1, args.P+1):  # p 表示 mask 的參與者數量
        for combination in itertools.combinations(range(args.P), p):
            vec_clone = vec_base.clone()
            # 將每個參與者的輸出當作一個區段，利用 indices 決定哪些區段要 mask
            indices = (torch.arange(args.bottomModel_output_dim * args.P, device=vec.device) //
                       args.bottomModel_output_dim)
            combination_tensor = torch.tensor(combination, dtype=indices.dtype, device=indices.device)
            mask = torch.isin(indices, combination_tensor).view(1, -1)
            mask = mask.expand(vec.size(0), -1)
            vec_clone[mask] = 0
            
            with torch.inference_mode():
                pred = model.top(vec_clone)
                pred_softmax = softmax(pred)
            # 取出每筆 sample 在其預測類別上的機率
            sample_indices = torch.arange(vec.size(0), device=vec.device)
            prob_values = pred_softmax[sample_indices, predicted_class].cpu()
            shapley_dict["Prob"].append(prob_values)
            
            # 記錄各參與者是否有納入：若該參與者在 combination 中，代表被 mask，記 0；否則記 1
            for participant in range(args.P):
                flag = 0 if participant in combination else 1
                shapley_dict[participant].append(flag)
    
    # 將整理好的字典轉為 DataFrame，再利用現有的 Shapley value 計算函式
    shapley_dataframe = pd.DataFrame(shapley_dict)
    list_each_participant_shapley = calculate_each_participant_shap_in_one_batch(shapley_dataframe, args)
    for p in range(args.P):
        # print(returned_shapley_outer_dict[i][p])
        # print(list_each_participant_shapley_in_this_batch[p])
        returned_preded_shapley_dict[p] = torch.cat(
            (returned_preded_shapley_dict[p], list_each_participant_shapley[p].to(args.device)), dim=0)
            # print(f"回傳後蒐集完的 dict 為:{returned_shapley_outer_dict[i][p]}")
    return returned_preded_shapley_dict

def how_many_data_over_threshold(preded_shapley_dict, static_class_dict, pred_c, args):
    """
    作用：對一個 batch 資料計算
    回傳：
      - flagged_passive_dict: list，每筆資料有哪些 passive 被標記為 exceed
      - conflict_sample_count: 有 active below + passive exceed 的樣本數
      - attacker_exceed_counts: 每位 attacker exceed threshold 的次數
      - participant_exceed_in_conflict: 每位參與者在 active below 條件下 exceed 的次數
    """
    active_party = args.active_party
    num_samples = len(pred_c)
    passive_party_list = [p for p in range(args.P) if p != active_party]
    attacker_list = args.attacker_list

    flagged_passive_dict = [set() for _ in range(num_samples)]
    conflict_sample_count = 0
    attacker_exceed_counts = {p: 0 for p in attacker_list}
    participant_exceed_in_conflict = {p: 0 for p in range(args.P)}

    for i in range(num_samples):
        label = pred_c[i].item() if isinstance(pred_c[i], torch.Tensor) else int(pred_c[i])
        sample_status = {}

        for p in range(args.P):
            shap_val = preded_shapley_dict[p][i]
            thresholds = static_class_dict[label][p]
            up_threshold = thresholds['up_threshold']
            down_threshold = thresholds['down_threshold']

            if shap_val > up_threshold:
                sample_status[p] = 'exceed'
                if p in attacker_list:
                    attacker_exceed_counts[p] += 1
            elif shap_val < down_threshold:
                sample_status[p] = 'below'
            else:
                sample_status[p] = 'within'

        # 如果 active 是 below，統計 passive 的 exceed 狀況
        if sample_status[active_party] == 'below':
            for p in passive_party_list:
                if sample_status[p] == 'exceed':
                    flagged_passive_dict[i].add(p)
                    participant_exceed_in_conflict[p] += 1
            if len(flagged_passive_dict[i]) > 0:
                conflict_sample_count += 1

    return (
        flagged_passive_dict,
        conflict_sample_count,
        attacker_exceed_counts,
        participant_exceed_in_conflict
    )
    
def detect_how_much_sample_shapley_s0_in_P(all_shapley_outer_dict, args, tag=0):
    '''
    tag:
        0 : train
        1 : clean testset
        2 : poisoned testset
        3 : take the detected index
    '''
    counter_dict = {key: torch.empty(0, device=args.device) for key in list(range(args.P))}
    for j in range(args.P):
        tensor_list=[]
        for i in range(args.num_class):
            if tag != 0:
                # print(f"現在 tag 是:{tag}，在蒐集第{j}個參與者在第{i}類別的資料")
                tensor_list.append(all_shapley_outer_dict[i][j])
        # print(len(tensor_list))
        # print(tensor_list)
        # time.sleep(3)

        if tag != 0: 
            # 檢查每個 Tensor 小於 0 的位置，轉成布林張量
            masks = [tensor < 0 for tensor in tensor_list]
            '''
            一個參與者在每個 sample 有幾次小於 0，回傳 tensor，index 表示sample編號
            '''
            negative_counts = sum(masks)
            counter_dict[j] = torch.cat((counter_dict[j],negative_counts), dim=0)
            # print(len(negative_counts))
            # if tag != 3:
            #     for k in range(args.num_class+1):
            #         print(f"參與者{j}，有{(negative_counts==k).sum().item()}個 sample 在 {k} 個類別貢獻度小於0")
            #     # print(counter_dict)
    if tag ==3:
        return counter_dict
    else:
        cal_shapley_distribution(counter_dict,args)
        return 0

'''
以下這邊後續都可以刪掉
'''
def plot_embedding_distribution(vec_dict, y_in, num_party, reduction_method='PCA', save_dir='./'):
    """
    vec_dict: dict，每個 key (0 ~ num_party-1) 對應該參與者的 bottom model 輸出, shape: (N, feature_dim)
    y_in: 當前 batch 的 label, shape: (N,)
    reduction_method: 使用降維方法，可選 'PCA' 或 't-SNE'
    save_dir: 圖片儲存的資料夾 (預設為當前資料夾)
    """
    # 將 y_in 轉成 numpy 陣列
    y_np = y_in.detach().cpu().numpy() if hasattr(y_in, 'detach') else np.array(y_in)
    for p in range(num_party):
        # 取得 p 參與者的 embedding，轉成 numpy
        embeddings = vec_dict[p].detach().cpu().numpy() if hasattr(vec_dict[p], 'detach') else np.array(vec_dict[p])
        
        if reduction_method == 'PCA':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings)
            print(reducer.explained_variance_ratio_)
        elif reduction_method == 't-SNE':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError("不支援的降維方法，請選擇 'PCA' 或 't-SNE'")
        
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=y_np, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Label')
        plt.title(f'Participant {p} Embedding Distribution ({reduction_method})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # 定義檔名並儲存圖檔
        filename = f"{save_dir}participant_{p}_{reduction_method}.png"
        plt.savefig(filename)
        print(f"已儲存圖檔：{filename}")
        plt.close()

'''
套用 mean-shift
'''
def filter_representative_vectors_by_corr(vectors, num=100, use_gpu=True):
    """
    根據相關係數篩選代表性向量
    參數:
      - vectors: numpy array, shape (N, dim)
      - num: 目標選取數量 (例如 500)
      - use_gpu: 若為 True，則利用 CuPy 在 GPU 上計算；否則使用 numpy (CPU)
    回傳:
      - 篩選後的向量 (依原始型態返回)
    """
    if use_gpu:
        # 將輸入轉換成 cupy 陣列（若尚未為 cupy.ndarray）
        if not isinstance(vectors, cp.ndarray):
            vectors_gpu = cp.asarray(vectors)
        else:
            vectors_gpu = vectors
        # 利用 cupy 進行相關係數計算
        coef = cp.corrcoef(vectors_gpu)
        rows = cp.sum(coef, axis=1)
        selected = cp.argpartition(rows, -num)[-num:]
        selected = cp.asnumpy(selected)
    else:
        coef = np.corrcoef(vectors)
        rows = np.sum(coef, axis=1)
        selected = np.argpartition(rows, -num)[-num:]
    
    return vectors[selected]


def build_classifier_and_centroids(defense_output, args, classifier_method='knn', k=3, kernel_bandwidth=1000, filter_num=100):
    """
    利用訓練最後一輪資料一次性計算各被動方的群心（採用 mean-shift 算法），
    並建立基於 active party 的分類器。
    
    為降低雜訊，每個類別僅以相關係數篩選後（保留 filter_num 筆）的向量進行群心計算與
    inference phase 的 nearest_centroid 距離計算。
    
    參數:
      - vec_dict_last_epoch: dict，每個 key 為參與者編號，值為訓練最後一輪所有的 embedding tensor (shape: [N, dim])
      - y_in_last_epoch: list 或 1D array，訓練最後一輪所有資料的 label
      - active_party: int，active party 的編號
      - passive_party_list: list，被動方的參與者編號
      - classifier_method: str，分類方法，可選 'knn' 或 'nearest_centroid' (預設 'knn')
      - k: int，當使用 KNN 時的鄰近數 (預設 k=3)
      - kernel_bandwidth : 用於 mean-shift 的 kernel bandwidth (預設 1000)
      - filter_num: int，每個類別選取的代表性向量數量 (例如 100)
      
    回傳:
      - classifier: callable，接受 active party 單筆 embedding (numpy array) 並回傳預測 label
      - passive_centroids_all: dict，key 為被動方編號，value 為該方的群心字典 {label: centroid_vector}
    """
    
    defense_models = defense_output["defense_models"]
    static_class_dict = defense_output["static_class_dict"]
    vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
    y_in_last_epoch = defense_output["y_in_last_epoch"]

    active_party = args.active_party
    passive_party_list = args.passive_party
    labels_train = np.array(y_in_last_epoch)

    # 轉換 labels 成 numpy 陣列
    labels_train = np.array(y_in_last_epoch)
    unique_labels = np.unique(labels_train)
    
    # 取得 active party 的訓練 embedding (轉為 numpy)
    if hasattr(vec_dict_last_epoch[active_party], 'detach'):
        active_train_embeddings = vec_dict_last_epoch[active_party].detach().cpu().numpy()
    else:
        active_train_embeddings = np.array(vec_dict_last_epoch[active_party])
    
    print("正在計算 passive_party 群心")
    # ---------------------------
    # 針對被動方：對每個類別先依相關係數篩選代表性向量，再利用 mean-shift 算法計算群心
    # ---------------------------
    passive_centroids_all = {}
    for p in passive_party_list:
        if hasattr(vec_dict_last_epoch[p], 'detach'):
            passive_train_embeddings = vec_dict_last_epoch[p].detach().cpu().numpy()
        else:
            passive_train_embeddings = np.array(vec_dict_last_epoch[p])
        centroids = {}
        for lbl in unique_labels:
            indices = np.where(labels_train == lbl)[0]
            if len(indices) > 0:
                data = passive_train_embeddings[indices]  # 該類別所有向量，shape: [N_label, dim]
                # 依相關係數篩選代表性向量
                # 若想用 GPU，加速 heavy computation:
                print(f"正在根據相關性從類別{lbl} 蒐集 {filter_num} 筆資料")
                filtered_data = filter_representative_vectors_by_corr(data, num=filter_num, use_gpu=True)
                print(f"正在計算類別 {lbl} 群心")
                # 利用 mean-shift 算群心
                points, shift_points, group = train_mean_shift(filtered_data, kernel_bandwidth)
                centroid = np.array(shift_points[0])
                centroids[lbl] = centroid
            else:
                centroids[lbl] = None
        passive_centroids_all[p] = centroids
    print("=================== passive_party 群心計算完成 =============")

    print("正在計算 active_party 分類器")
    # ---------------------------
    # 建立 active party 的分類器 (以 nearest_centroid 方式)
    # ---------------------------
    if classifier_method == 'nearest_centroid':
        active_centroids = {}
        for lbl in unique_labels:
            indices = np.where(labels_train == lbl)[0]
            if len(indices) > 0:
                data = active_train_embeddings[indices]
                # 依相關係數篩選代表性向量
                # 若想用 GPU，加速 heavy computation:
                print(f"正在根據相關性從類別{lbl} 蒐集 {filter_num} 筆資料")
                filtered_data = filter_representative_vectors_by_corr(data, num=filter_num, use_gpu=True)
                print(f"正在計算類別 {lbl} 群心")
                points, shift_points, group = train_mean_shift(filtered_data, kernel_bandwidth)
                centroid = np.array(shift_points[0])
                active_centroids[lbl] = centroid
            else:
                active_centroids[lbl] = None
        print("=================== active_party 群心計算完成=============")
        
        def classifier(query_vec):
            distances = {}
            for lbl, centroid in active_centroids.items():
                # if centroid is not None:
                #     distances[lbl] = np.linalg.norm(query_vec - centroid)
                if centroid is not None:
                    # 將 centroid 轉成一維向量 (如果它是 2D 的)
                    centroid = np.squeeze(centroid)
                    dot = np.dot(query_vec, centroid)
                    norm_query = np.linalg.norm(query_vec)
                    norm_centroid = np.linalg.norm(centroid)
                    cosine_sim = dot / (norm_query * norm_centroid + 1e-8)
                    cosine_dist = 1 - cosine_sim
                    distances[lbl] = cosine_dist
            return min(distances, key=distances.get)
        
    
    elif classifier_method == 'knn':
        # 對 active party 每個類別篩選代表性向量後合併起來
        filtered_active_list = []
        filtered_active_labels = []
        for lbl in unique_labels:
            indices = np.where(labels_train == lbl)[0]
            if len(indices) > 0:
                data = active_train_embeddings[indices]
                # 依相關係數篩選代表性向量
                # 若想用 GPU，加速 heavy computation:
                print(f"正在根據相關性從類別{lbl} 蒐集 {filter_num} 筆資料")
                filtered_data = filter_representative_vectors_by_corr(data, num=filter_num, use_gpu=True)
                filtered_active_list.append(filtered_data)
                filtered_active_labels.extend([lbl] * filtered_data.shape[0])
        if len(filtered_active_list) > 0:
            filtered_active_train_embeddings = np.concatenate(filtered_active_list, axis=0)
            filtered_active_train_labels = np.array(filtered_active_labels)
        else:
            filtered_active_train_embeddings = active_train_embeddings
            filtered_active_train_labels = labels_train

        def classifier(query_vec):
            distances = np.linalg.norm(filtered_active_train_embeddings - query_vec, axis=1)
            k_indices = np.argsort(distances)[:k]
            k_labels = filtered_active_train_labels[k_indices]
            return np.bincount(k_labels).argmax()
    else:
        raise ValueError("Unsupported classifier_method. Please choose 'knn' or 'nearest_centroid'.")
    
    return classifier, passive_centroids_all

def correct_poisoned_passive_embedding_batch(vec_dict, flagged_passive_dict, classifier, passive_centroids_all, args):
    """
    針對單一 batch 推理，利用預先計算好的分類器與各被動方的群心，
    根據 flagged_passive_dict (字典：key 為資料索引，value 為被標記需修正的 passive party set)
    對於每筆資料，若 active party 被檢測為異常且有某些 passive party 被標記，
    則利用 active party 的向量計算預測 label，並將該筆資料中所有被標記的 passive party 的向量
    替換為對應該 label 的群心。

    參數：
      - vec_dict: dict，每個 key 為參與者編號，值為該 batch 中的 embedding tensor (形狀: [B, dim])
      - flagged_passive_dict: list (或 dict)，索引對應每筆資料，value 為一個 set，包含被標記的 passive party 編號
      - classifier: 預先建立好的分類器，接受 active party 單筆 embedding (numpy array) 並回傳預測 label
      - passive_centroids_all: dict，預先計算好的各被動方群心 {passive_party: {label: centroid_vector}}
      - args: 參數物件，須包含 args.active_party 與 args.passive_party (被動 party 的列表)

    回傳：
      - 更新後的 vec_dict，其中每筆資料中被 flagged 的 passive party 向量均已替換成對應群心。
    """
    active_party = args.active_party

    # 假設 flagged_passive_dict 是一個長度為 batch_size 的列表，
    # 每個元素為一個 set，內含需要替換的 passive party 編號（例如 {1,3}）
    batch_size = vec_dict[active_party].shape[0]

    for i in range(batch_size):
        flagged_set = flagged_passive_dict[i]
        if flagged_set:
            # 以 active party 的第 i 筆 embedding 計算預測 label
            query_vec = vec_dict[active_party][i].detach().cpu().numpy()
            predicted_label = classifier(query_vec)
            # 對於該筆資料中每個被標記的 passive party，進行向量替換
            for p in flagged_set:
                centroid = passive_centroids_all[p][predicted_label]
                centroid_tensor = torch.tensor(centroid, dtype=vec_dict[p].dtype, device=vec_dict[p].device)
                vec_dict[p][i] = centroid_tensor
    return vec_dict