'''
需要接收
1. dataset name
2. batch size

需要回傳
1. data size
2. class number
3. trainloader
4. testloader
''' 

import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from PIL import ImageFile
import os, sys, hashlib, urllib.request, zipfile, subprocess
from pathlib import Path
import scipy.io
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

# data.py  ── 置於頂部或 utils 區域  ─────────────────────────────
import requests, shutil
from tqdm import tqdm
import scipy.io as sio
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 允許載入長度不足的圖片

PARTY_NUM = 1                       # 預設 1；train_new 會改寫它
FEATURE_SLICES = None               # 之後存 [dim_P0, dim_P1, ...]
PARTY_ROOTS    = None               # 之後存 [['age','job',...], ...]
BM_ALLOC  = "active_gets_dur"   # 'equal' | 'dur_to_att' | 'active_gets_dur'

def set_party_num(n: int):
    """由外部 (train_new.py) 指定參與者數。"""
    global PARTY_NUM
    PARTY_NUM = n
    
def set_bm_alloc(mode: str):
    global BM_ALLOC
    BM_ALLOC = mode
    
# ==== NUS-WIDE 專用 ─ helper =================================================
_EXPECTED_DIM = dict(CH=64, CM55=55, CORR=225, EDH=128, WT=162)

def _load_tag(bin_path: Path) -> np.ndarray:
    """讀 1000-D Tag；自動去掉尾端 padding。"""
    raw = np.fromfile(bin_path, dtype=np.uint8)
    usable = (raw.size // 1000) * 1000
    return raw[:usable].reshape(-1, 1000).astype("float32")   # (rows,1000)

def _load_feat(name: str, feat_root: Path) -> np.ndarray:
    """讀單一低階特徵 (CH/CM55/CORR/EDH/WT)。"""
    d   = _EXPECTED_DIM[name]
    raw = np.fromfile(feat_root / f"Normalized_{name}.dat", dtype=np.float32)
    usable = (raw.size // d) * d
    return raw[:usable].reshape(-1, d)                        # (rows,d)

def _load_gt_all(concept: str, root: Path) -> np.ndarray:
    """讀 AllLabels 裡完整 0/1 標籤，長度 = 269 648 列。"""
    f = root / "groundtruth" /"Groundtruth"/ "AllLabels" / f"Labels_{concept}.txt"
    return np.loadtxt(f, dtype=int)                           # (rows,)
# ============================================================================ 


class NonSupportDataset(Exception):
    pass

def get_data(dataname,batch_size):
    global FEATURE_SLICES, PARTY_ROOTS

    if dataname == "CIFAR10":
        # How to calculate mean and std of dataset
        # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
        cifar_mean = [0.49139968, 0.48215827 ,0.44653124]
        cifar_std = [0.24703233,0.24348505,0.26158768]
        
        # transform_for_train = transforms.Compose([
        #     transforms.RandomCrop((32, 32), padding=5),
        #     transforms.RandomRotation(10),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(cifar_mean, cifar_std)
        # ])
        
        transform_for_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std)
        ])
        
        # transform_for_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(cifar_mean, cifar_std)
        # ])
        
        transform_for_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std)
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_for_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_for_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        num_class = 10
        
        trn_x,trn_y = prepared_data(trainset)
        dl_train_set = CIFAR10(trn_x,trn_y)
        val_x,val_y = prepared_data(testset)
        dl_val_set = CIFAR10(val_x,val_y)
        
        poisoned_trainloader = torch.utils.data.DataLoader(dl_train_set, batch_size=batch_size, shuffle=True)
        poisoned_testloader = torch.utils.data.DataLoader(dl_val_set, batch_size=batch_size, shuffle=True)
        
    elif dataname == "CINIC10":
        # https://www.kaggle.com/datasets/mengcius/cinic10
        # https://github.com/AntonFriberg/pytorch-cinic-10/blob/master/main.py
        if os.path.isdir("data/cinic-10"):
            print("CINIC 資料已存在")
        else:
            os.system("mkdir -p data/cinic-10")
            os.system("curl -L \
                        https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz \
                        | tar xz -C data/cinic-10")
        traindir = os.path.join("data/cinic-10", 'train')
        # validatedir = os.path.join("data/cinic-10", 'valid')
        testdir = os.path.join("data/cinic-10", 'test')
        
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        
        train_transform = transforms.Compose([
                        # transforms.RandomCrop(32, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=cinic_mean, std=cinic_std)
                    ])
        
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean, std=cinic_std)
                ])
        
        trainset = torchvision.datasets.ImageFolder(root=traindir, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)

        # validateset = torchvision.datasets.ImageFolder(root=validatedir, transform=transform)
        # validateloader = torch.utils.data.DataLoader(validateset,
        #                                             batch_size=100,
        #                                             shuffle=True,
        #                                             num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=testdir, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)
        
        trn_x,trn_y = prepared_data(trainset)
        dl_train_set = CINIC10(trn_x,trn_y)
        val_x,val_y = prepared_data(testset)
        dl_val_set = CINIC10(val_x,val_y)
        
        poisoned_trainloader = torch.utils.data.DataLoader(dl_train_set, batch_size=batch_size, shuffle=True)
        poisoned_testloader = torch.utils.data.DataLoader(dl_val_set, batch_size=batch_size, shuffle=True)
             
        num_class = 10
        
    elif dataname == "BM":
        # 0. 讀檔＋one-hot（與原來完全相同）
        df = pd.read_csv("./data/bank-market/bank-market.csv")
        for col in df.select_dtypes("object"):
            if df[col].nunique() == 2:
                df[col] = (df[col] == "yes").astype(int)
        df = pd.get_dummies(df).astype(np.float32)

        target = "deposit"                           # ← 目標欄位
        feat_df = df.drop(columns=[target])          # 只留下特徵欄位

        # ---------- ①  建 16 個「原始欄位群組」 -----------------------------
        col_names = feat_df.columns.tolist()
        groups, start, prev = [], 0, col_names[0].split('_')[0]
        for idx, c in enumerate(col_names):
            root = c.split('_')[0]
            if root != prev:
                groups.append((start, idx, prev))    # (開始 idx, 結束 idx, root 名)
                start, prev = idx, root
        groups.append((start, len(col_names), prev)) # 最後一段
        assert len(groups) == 16, f"群組數={len(groups)} ≠ 16，請檢查"
        
        '''
        # ---------- ②  按 PARTY_NUM 均分欄位 --------------------------------
        global FEATURE_SLICES, PARTY_ROOTS          # ← 需宣告 global 才能寫回模組變數
        P   = PARTY_NUM                             # train_new.py 執行前已呼 data.set_party_num()
        per = 16 // P                               # 每人拿幾個原生欄位
        parties_idx, parties_root = [[] for _ in range(P)], [[] for _ in range(P)]
        g_ptr = 0
        for pid in range(P):
            for _ in range(per):
                s, e, root = groups[g_ptr]
                parties_idx[pid].extend(range(s, e))
                parties_root[pid].append(root)
                g_ptr += 1

        # ---------- ③ 重新排欄位，確保同一方的特徵連在一起 ------------------
        reorder_feat = [i for pid in range(P) for i in parties_idx[pid]]
        feat_df = feat_df.iloc[:, reorder_feat].copy()       # (N, F) 只含特徵
        '''
        
        '''
        # duration 給攻擊者
        # ---------- ②-0 先把 duration 指定給攻擊者 ------------------------------
        global FEATURE_SLICES, PARTY_ROOTS
        force_map = {2: 0, 4: 1, 8: 3}            # P → 攻擊者編號
        P         = PARTY_NUM
        force_pid = force_map.get(P, 0)           # 萬一 P 不是 2/4/8，預設 0

        # 找出 duration 這段
        dur_idx   = next(i for i, (_, _, r) in enumerate(groups) if r == "duration")
        dur_seg   = groups.pop(dur_idx)           # (start, end, "duration")

        # 初始化容器
        parties_idx  = [[] for _ in range(P)]
        parties_root = [[] for _ in range(P)]

        # 先把 duration 塞進攻擊者
        s, e, root   = dur_seg
        parties_idx[force_pid].extend(range(s, e))
        parties_root[force_pid].append(root)
        
        # ---------- ②-1 把其餘 15 欄位順序均分 -----------------------------
        per   = 16 // P                       # 每人應拿幾個原始欄位
        quota = [per] * P
        quota[force_pid] -= 1                 # 攻擊者已拿 1 欄

        g_ptr = 0
        for pid in range(P):
            while quota[pid] > 0 and g_ptr < len(groups):
                s, e, root = groups[g_ptr]
                parties_idx[pid].extend(range(s, e))
                parties_root[pid].append(root)
                quota[pid] -= 1
                g_ptr += 1
                
        # ---------- ③ 重新排欄位並記錄切片長度 -----------------------------
        reorder_feat = [i for pid in range(P) for i in parties_idx[pid]]
        feat_df      = feat_df.iloc[:, reorder_feat].copy()
        '''
        
        '''
        #duration 和 poutcome 給攻擊者
        global FEATURE_SLICES, PARTY_ROOTS
        # ---------------- ②-0：鎖定關鍵特徵給攻擊者 -------------------
        FORCE_ROOTS = {"duration", "poutcome"}           # ← 想固定給攻擊者的原生欄位
        force_map   = {2: 0, 4: 1, 8: 3}                 # P → 攻擊者編號
        P   = PARTY_NUM
        force_pid = force_map.get(P, 0)                  # 預設 0

        # 把 list(groups) 轉成待分配池，並把關鍵段移除
        parties_idx  = [[] for _ in range(P)]
        parties_root = [[] for _ in range(P)]

        remain_groups = []
        for seg in groups:
            s, e, root = seg
            if root in FORCE_ROOTS:
                parties_idx[force_pid].extend(range(s, e))
                parties_root[force_pid].append(root)
            else:
                remain_groups.append(seg)

        # ---------------- ②-1：均分剩下 14 個欄位 -----------------------
        per   = 16 // P                       # 每人應拿的原生欄位數
        quota = [per] * P
        quota[force_pid] -= len(FORCE_ROOTS)  # 攻擊者已搶 2 個

        g_ptr = 0
        for pid in range(P):
            while quota[pid] > 0 and g_ptr < len(remain_groups):
                s, e, root = remain_groups[g_ptr]
                parties_idx[pid].extend(range(s, e))
                parties_root[pid].append(root)
                quota[pid] -= 1
                g_ptr += 1

        # ---------------- ③：重新排欄位並寫全域變數 ----------------------
        reorder_feat = [i for pid in range(P) for i in parties_idx[pid]]
        feat_df      = feat_df.iloc[:, reorder_feat].copy()
        '''
        
        '''
        #重要特徵給攻擊者和主動方
        ATTACKER_ROOTS = {"balance"}            # duration → Active
        ACTIVE_ROOTS   = {"duration", "poutcome", "contact"}

        P   = PARTY_NUM
        pid_map = {2:(0,1), 4:(1,2), 8:(3,4)}
        force_att_pid, force_act_pid = pid_map.get(PARTY_NUM, (0,1))

        parties_idx  = [[] for _ in range(PARTY_NUM)]
        parties_root = [[] for _ in range(PARTY_NUM)]

        remain_groups = []

        for seg in groups:                       # groups = [(s,e,root), ...]
            s, e, root = seg
            if root in ATTACKER_ROOTS:
                parties_idx[force_att_pid].extend(range(s, e))
                parties_root[force_att_pid].append(root)
            elif root in ACTIVE_ROOTS:
                parties_idx[force_act_pid].extend(range(s, e))
                parties_root[force_act_pid].append(root)
            else:
                remain_groups.append(seg)        # 先留待平均分配

        # ---------- ②-1：計算每人還剩多少 quota ------------------------------
        per   = 16 // PARTY_NUM                  # 每人目標原生欄位數
        quota = [per] * PARTY_NUM
        quota[force_att_pid] -= len(ATTACKER_ROOTS)
        quota[force_act_pid] -= len(ACTIVE_ROOTS)

        # ---------- ②-2：把剩餘欄位按現順序平均塞滿 --------------------------
        g_ptr = 0
        for pid in range(PARTY_NUM):
            while quota[pid] > 0 and g_ptr < len(remain_groups):
                s, e, root = remain_groups[g_ptr]
                parties_idx[pid].extend(range(s, e))
                parties_root[pid].append(root)
                quota[pid] -= 1
                g_ptr     += 1

        # ---------- ③：重排欄位並寫回全域變數 -------------------------------
        reorder_feat = [i for pid in range(PARTY_NUM) for i in parties_idx[pid]]
        feat_df      = feat_df.iloc[:, reorder_feat].copy()
        '''
        
                # ---------- ②/③：三種分配法分支 → 產生 parties_idx / parties_root ------
        global FEATURE_SLICES, PARTY_ROOTS
        P   = PARTY_NUM
        per = 16 // P

        parties_idx  = [[] for _ in range(P)]
        parties_root = [[] for _ in range(P)]

        def _equal_slice():
            g_ptr = 0
            for pid in range(P):
                for _ in range(per):
                    s,e,r = groups[g_ptr]
                    parties_idx[pid].extend(range(s, e))
                    parties_root[pid].append(r)
                    g_ptr += 1

        def _dur_to_att():
            # 攻擊者鎖定 duration
            force_map = {2:0, 4:1, 8:3}
            att_pid   = force_map.get(P, 0)

            # 找 duration 那段
            dur_i = next(i for i,(_,_,r) in enumerate(groups) if r=="duration")
            s,e,r = groups.pop(dur_i)
            parties_idx[att_pid].extend(range(s, e))
            parties_root[att_pid].append(r)

            # 剩餘均分
            quota = [per]*P
            quota[att_pid] -= 1
            g_ptr = 0
            for pid in range(P):
                while quota[pid] > 0 and g_ptr < len(groups):
                    s,e,r = groups[g_ptr]
                    parties_idx[pid].extend(range(s, e))
                    parties_root[pid].append(r)
                    quota[pid] -= 1
                    g_ptr += 1

        def _active_gets_dur():
            # 訊息雙極配置（論文主設定）：
            # Active 拿 duration、poutcome、contact；Attacker 拿 balance
            pid_map = {2:(0,1), 4:(1,2), 8:(3,4)}   # (att_pid, act_pid)
            att_pid, act_pid = pid_map.get(P, (0,1))
            ATTACKER_ROOTS = {"balance"}
            ACTIVE_ROOTS   = {"duration","poutcome","contact"}

            # 先把指定 root 塞到對應參與者
            remain = []
            for (s,e,r) in groups:
                if r in ATTACKER_ROOTS:
                    parties_idx[att_pid].extend(range(s, e))
                    parties_root[att_pid].append(r)
                elif r in ACTIVE_ROOTS:
                    parties_idx[act_pid].extend(range(s, e))
                    parties_root[act_pid].append(r)
                else:
                    remain.append((s,e,r))

            # 再把剩餘欄位均分補滿 quota
            quota = [per]*P
            quota[att_pid] -= len(ATTACKER_ROOTS)
            quota[act_pid] -= len(ACTIVE_ROOTS)
            g_ptr = 0
            for pid in range(P):
                while quota[pid] > 0 and g_ptr < len(remain):
                    s,e,r = remain[g_ptr]
                    parties_idx[pid].extend(range(s, e))
                    parties_root[pid].append(r)
                    quota[pid] -= 1
                    g_ptr += 1

        # 依 BM_ALLOC 選擇策略
        if BM_ALLOC == "equal":
            _equal_slice()
        elif BM_ALLOC == "dur_to_att":
            _dur_to_att()
        elif BM_ALLOC == "active_gets_dur":
            _active_gets_dur()
        else:
            raise ValueError(f"Unknown BM_ALLOC: {BM_ALLOC}")

        # ---------- ③：重排欄位並寫回全域變數 -------------------------------
        reorder_feat = [i for pid in range(P) for i in parties_idx[pid]]
        feat_df      = feat_df.iloc[:, reorder_feat].copy()
        
        
        # ---------- ④ 把特徵與 target 再 concatenate 回去 -------------------
        df = pd.concat([feat_df, df[target]], axis=1)        # deposit 放在最後一欄

        FEATURE_SLICES = [len(parties_idx[p]) for p in range(P)]
        PARTY_ROOTS    = parties_root                       # [['age','job',...], …]
        

        # ---------- ⑤ train / val split 之後流程與原本相同 ------------------
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[target]),
            df[target],
            test_size=0.2, random_state=42, stratify=df[target])

        # 1. BankMarketing Dataset
        trainset = BankMarketing(X_train, y_train, return_id=True)
        testset  = BankMarketing(X_test,  y_test,  return_id=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False)

        # 2. prepared_data
        trn_x, trn_y = prepared_data(trainset)
        val_x, val_y = prepared_data(testset)

        # 3. poisoned DataLoader
        dl_train_set = BankMarketing(trn_x, trn_y, return_id=True)
        poisoned_trainloader = torch.utils.data.DataLoader(dl_train_set, batch_size=batch_size, shuffle=True)
        dl_val_set  = BankMarketing(val_x, val_y, return_id=True)
        poisoned_testloader = torch.utils.data.DataLoader(dl_val_set, batch_size=batch_size, shuffle=True)

        # 4. 其他欄位
        input_size = torch.Size([X_train.shape[1]])
        num_class  = 2

    elif dataname == "Imagenette":
        Imagenet_mean=[0.485, 0.456, 0.406]
        Imagenet_std=[0.229, 0.224, 0.225]
        
        transform_for_train = transforms.Compose([
            transforms.RandomCrop((224, 224), padding=5),
            transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(Imagenet_mean, Imagenet_std)
        ])
        
        transform_for_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(Imagenet_mean, Imagenet_std)
        ])
        if os.path.isdir("data/imagenette2"):
            trainset = torchvision.datasets.Imagenette(root='./data', split='train',
                                            download=False, transform=transform_for_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)
            testset = torchvision.datasets.Imagenette(root='./data', split='val',
                                                download=False, transform=transform_for_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=False, num_workers=2)
        else:
            trainset = torchvision.datasets.Imagenette(root='./data', split='train',
                                            download=True, transform=transform_for_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)
            testset = torchvision.datasets.Imagenette(root='./data', split='val',
                                                download=False, transform=transform_for_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=False, num_workers=2)
        
        num_class = 10
    
    elif dataname == "NUSWIDE":
        root     = Path("./data/NUS-WIDE")
        low_root = root / "low_level" / "Low_Level_Features"

        # ---------- (a) Tag -----------------------------------------------------
        tag_tr = _load_tag(root /"tags"/"NUS_WID_Tags"/"Train_Tags1k.dat")
        tag_te = _load_tag(root /"tags"/"NUS_WID_Tags"/"Test_Tags1k.dat")
        X_tag  = np.vstack([tag_tr, tag_te])                         # (rows_tag,1000)

        # ---------- (b) 634-D 影像特徵 ------------------------------------------
        feats = {n: _load_feat(n, low_root) for n in ["CH", "CM55", "CORR", "EDH", "WT"]}

        # ---------- (c) 5 類 ground-truth (AllLabels) ---------------------------
        concepts = ["buildings", "grass", "animal", "water", "person"]
        gts = {c: _load_gt_all(c, root) for c in concepts}

        # ---------- (d) 同步列數：取三方最小值 ----------------------------------
        N = min(X_tag.shape[0],
                *(a.shape[0] for a in feats.values()),
                *(a.shape[0] for a in gts.values()))

        X_tag = X_tag[:N]
        X_img = np.hstack([feats[n][:N] for n in ["CH","CM55","CORR","EDH","WT"]]).astype("float32")
        one_hot = np.vstack([gts[c][:N] for c in concepts]).T        # (N,5)

        # ---------- (e) 只保留「單標籤」影像 ------------------------------------
        keep = one_hot.sum(1) == 1
        y    = one_hot[keep].argmax(1).astype("int64")               # (N_keep,)

        scaler  = StandardScaler(with_mean=False)
        X_img   = scaler.fit_transform(X_img[keep])
        X       = np.hstack([X_img, X_tag[keep]]).astype("float32")  # (N_keep,1634)

        # ---------- (f) FEATURE_SLICES / PARTY_ROOTS ----------------------------
        if   PARTY_NUM == 2:
            FEATURE_SLICES = [634, 1000];                       PARTY_ROOTS = [['image'], ['tag']]
        elif PARTY_NUM == 4:
            FEATURE_SLICES = [360, 274, 500, 500];              PARTY_ROOTS = [['image'],['image'],['tag'],['tag']]
        elif PARTY_NUM == 8:
            FEATURE_SLICES = [158,158,158,160, 250,250,250,250]; PARTY_ROOTS = [['image']]*4 + [['tag']]*4
        else:
            raise ValueError("NUSWIDE 只支援 P=2/4/8 (請先 data.set_party_num())")

        # ---------- (g) Train / Test split & DataLoader -------------------------
        idx_tr, idx_te = train_test_split(np.arange(len(y)),
                                        test_size=0.4, stratify=y, random_state=42)

        class NUSD(Dataset):
            def __init__(self, idx, ret_id=True):
                self.X   = torch.from_numpy(X[idx])
                self.y   = torch.from_numpy(y[idx])
                self.ids = torch.from_numpy(idx.astype("int64"))
                self.ret = ret_id
            def __len__(self): return len(self.y)
            def __getitem__(self, k):
                if self.ret:
                    return self.X[k], int(self.y[k]), int(self.ids[k])
                return self.X[k], int(self.y[k])

        trainset   = NUSD(idx_tr, ret_id=False)
        testset    = NUSD(idx_te, ret_id=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2)
        testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2)

        trn_x, trn_y = prepared_data(NUSD(idx_tr, ret_id=True))
        val_x, val_y = prepared_data(NUSD(idx_te, ret_id=True))

        dl_train_set         = NUSD(idx_tr, ret_id=True)
        poisoned_trainloader = DataLoader(dl_train_set, batch_size=batch_size, shuffle=True)
        dl_val_set           = NUSD(idx_te, ret_id=True)
        poisoned_testloader  = DataLoader(dl_val_set, batch_size=batch_size, shuffle=True)

        input_size = torch.Size([1634])
        num_class  = 5
        
    else:
        raise NonSupportDataset("Please use one of the following dataset: CIFAR10, CINIC10, BM, Imagenette")
        # raise NonSupportDataset("Please use one of the following dataset: CIFAR10, CINIC10, BM, Imagenette, NUSWIDE")
    
    if dataname in ["CIFAR10","CINIC10"]:
        egdata, _ = trainset[0]
    else:
        # 取一筆樣本確認形狀
        egdata = trainset[0][0]      # ← 直接取索引 0 的第一個欄位 (x)
        input_size = egdata.shape
    
    input_size = egdata.shape
    # print(input_size)
    
    return trainloader, testloader, input_size, num_class, trn_x, trn_y, dl_train_set, poisoned_trainloader, val_x, val_y,  dl_val_set, poisoned_testloader

def prepared_data(dataset):
    data, label = [], []
    for idx, item in enumerate(dataset):
        if len(item) == 2:
            x, y = item
        else:
            x, y, _ = item     # 忽略 id
        data.append({'id': idx, 'data': x})
        label.append(y)
    return data, label


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self,data,label,transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        x = self.data[item]['data']
        if not(self.transform is None):
            x = self.transform(x)
        y = self.label[item]
        id = self.data[item]['id']

        return x,y,id

    def __len__(self):
        return len(self.label)
    
class CINIC10(torch.utils.data.Dataset):
    def __init__(self,data,label,transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        x = self.data[item]['data']
        if not(self.transform is None):
            x = self.transform(x)
        y = self.label[item]
        id = self.data[item]['id']

        return x,y,id

    def __len__(self):
        return len(self.label)

# --- 專屬 Dataset，回傳 (x,y,id) ---
class BankMarketing(torch.utils.data.Dataset):
    """
    支援兩種初始化方式：
      - 原始 DataFrame  / Series → 用於 trainloader / testloader
      - list[dict]     / list   → 用於 poisoned_trainloader / poisoned_testloader
    """
    def __init__(self, X, y, transform=None, return_id=True):
        self.transform = transform
        self.return_id  = return_id
        # --- 情況 A：原始 DataFrame / Series ---
        if hasattr(X, "values"):                        # DataFrame or ndarray
            self.X = torch.from_numpy(X.values if hasattr(X, "values") else np.asarray(X)).float()
            self.y = torch.from_numpy(y.values if hasattr(y, "values") else np.asarray(y)).long()
            self.ids = torch.arange(len(self.y))
        # --- 情況 B：list 版本 (prepared_data 產生) ---
        else:                                           # list / tuple
            self.X = torch.stack([d["data"] for d in X])
            self.y = torch.tensor(y, dtype=torch.long)
            self.ids = torch.tensor([d["id"] for d in X])
        assert len(self.X) == len(self.y) == len(self.ids)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        if self.return_id:
            return x, self.y[idx], int(self.ids[idx])
        else:                     # 測試/驗證集只回傳 (x, y)
            return x, self.y[idx]

# --------------------------------------------------------------------
#  NUS-WIDE 下載（Hugging Face）+ 解壓 + 前處理 + DataLoader
#  依舊回傳：train_dl, test_dl, poison_tr, poison_te, in_dim, n_class
# --------------------------------------------------------------------
import os, hashlib, shutil, zipfile, subprocess, numpy as np
import torch, scipy.io as sio
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ── 主要函式 ─────────────────────────────────────────────
def load_nuswide(batch: int = 128, root: str = "data/NUS-WIDE", seed: int = 42):
    # ----------- 基本 import -------------------------------------------------
    from pathlib import Path
    import hashlib, zipfile, subprocess, numpy as np, torch, scipy.io as sio
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader
    global PARTY_NUM, FEATURE_SLICES, PARTY_ROOTS
    
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # ----------- 1. 確認三個壓縮檔存在 --------------------------------------
    required = [
        "NUS_WID_Low_Level_Features.rar",
        "NUS_WID_Tags.zip",
        "NUS-WIDE.zip",
    ]
    for fname in required:
        if not (root / fname).exists():
            raise FileNotFoundError(f"{fname} 不在 {root}，請先放入後再執行")

    # ----------- 2. 解壓 -----------------------------------------------------
    if not (root/"low_level").exists():
        _extract_rar(root/"NUS_WID_Low_Level_Features.rar",
                    root/"low_level")

    if not (root/"tags").exists():
        _extract_zip(root/"NUS_WID_Tags.zip",
                    root/"tags")

    if not (root/"groundtruth").exists():
        _extract_zip(root/"NUS-WIDE.zip",
                    root/"groundtruth")

    print("解包完成")
    time.sleep(5)


    # ----------- 3. 讀取特徵 -------------------------------------------------
    mat = sio.loadmat(root / "low_level" / "Low_Level_Features.mat")["Low_Level_Features"][0]
    X_img = np.hstack([mat[i].astype("float32") for i in range(5)])        # (N, 634)

    tags_tr = np.fromfile(root / "tags" / "Train_Tags1k.dat",
                          dtype=np.uint8).reshape(-1, 1000)
    tags_te = np.fromfile(root / "tags" / "Test_Tags1k.dat",
                          dtype=np.uint8).reshape(-1, 1000)
    X_tag   = np.vstack([tags_tr, tags_te]).astype("float32")

    # ----------- 4. 擷取五個單標籤類別 --------------------------------------
    concepts = ["buildings", "grass", "animal", "water", "person"]
    one_hot = []
    for phase in ["Train", "Test"]:
        for c in concepts:
            f = (root / "groundtruth" /
                 "Groundtruth/TrainTestLabels" /
                 f"Labels_{c}_{phase}.txt")
            one_hot.append(np.loadtxt(f, dtype=int))
    one_hot = (np.vstack(one_hot)
                 .reshape(2, len(concepts), -1)
                 .transpose(2, 0, 1)
                 .reshape(-1, 5))
    keep = one_hot.sum(1) == 1
    y    = one_hot[keep].argmax(1).astype("int64")

    scaler = StandardScaler(with_mean=False)
    X_img = scaler.fit_transform(X_img[keep])
    X     = np.hstack([X_img, X_tag[keep]]).astype("float32")              # (N, 1634)

    # ----------- 5. Train / Test DataLoader ---------------------------------
    idx_tr, idx_te = train_test_split(np.arange(len(y)),
                                      test_size=0.2,
                                      stratify=y,
                                      random_state=seed)

    class NUSD(Dataset):
        def __init__(self, idx, with_id=False):
            self.X = torch.from_numpy(X[idx])
            self.y = torch.from_numpy(y[idx])
            self.ids = torch.from_numpy(idx.astype("int64"))
            self.with_id = with_id
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            if self.with_id:
                return {'data': self.X[i],
                        'label': int(self.y[i]),
                        'id':    int(self.ids[i])}
            return self.X[i], int(self.y[i])

    # train_dl  = DataLoader(NUSD(idx_tr),        batch, shuffle=True,  num_workers=4)
    # test_dl   = DataLoader(NUSD(idx_te),        batch, shuffle=False, num_workers=4)
    # poison_tr = DataLoader(NUSD(idx_tr, True),  batch, shuffle=True,  num_workers=4)
    # poison_te = DataLoader(NUSD(idx_te, True),  batch, shuffle=True,  num_workers=4)
    
    # def _nuswide_slices_roots(P):
    #     if P == 2:
    #         return [634, 1000], [['image'], ['tag']]
    #     elif P == 4:
    #         return [360, 274, 500, 500], [['image'], ['image'], ['tag'], ['tag']]
    #     elif P == 8:
    #         img = [158, 158, 158, 160]        # 634 = 158×3 + 160
    #         tag = [250, 250, 250, 250]        # 1000 均分
    #         return img + tag, [['image']]*4 + [['tag']]*4
    #     else:
    #         raise ValueError("NUSWIDE 只支援 P = 2 / 4 / 8")
    # global FEATURE_SLICES, PARTY_ROOTS
    # FEATURE_SLICES, PARTY_ROOTS = _nuswide_slices_roots(PARTY_NUM)

    # return train_dl, test_dl, poison_tr, poison_te, torch.Size([1634]), 5
    
    # ---------- ⑥ 建立「統一格式」的物件 -------------------------------------------
    dl_train_set = NUSD(idx_tr, with_id=True)
    poisoned_trainloader = DataLoader(dl_train_set, batch_size=batch,
                                    shuffle=True, num_workers=4)

    dl_val_set  = NUSD(idx_te, with_id=True)
    poisoned_testloader = DataLoader(dl_val_set, batch_size=batch,
                                    shuffle=True, num_workers=4)

    val_x = torch.from_numpy(X[idx_te]).float()    # 直接給 tensor / 亦可用 ndarray
    val_y = torch.from_numpy(y[idx_te]).long()

    # ---------- ⑦ 寫回 FEATURE_SLICES / PARTY_ROOTS (供 train_new.py 讀) ------------
    def _nuswide_slices_roots(P):
        if P == 2:
            return [634, 1000], [['image'], ['tag']]
        if P == 4:
            return [360, 274, 500, 500], [['image'], ['image'], ['tag'], ['tag']]
        if P == 8:
            img = [158, 158, 158, 160]            # 634 = 158×3 + 160
            tag = [250, 250, 250, 250]
            return img + tag, [['image']]*4 + [['tag']]*4
        raise ValueError("NUS-WIDE 只支援 P = 2 / 4 / 8")
    FEATURE_SLICES, PARTY_ROOTS = _nuswide_slices_roots(PARTY_NUM)

    # ---------- ⑧ 用「統一 return」結束 (與 BM 等資料集相同) -----------------------
    return (dl_train_set,
            poisoned_trainloader,
            val_x, val_y,
            dl_val_set,
            poisoned_testloader)
    
# =============== 通用解包工具 ============================
def _extract_rar(src: Path, dst: Path):
    """穩健解 .rar：unrar → 7z → rarfile 三層退避"""
    dst.mkdir(parents=True, exist_ok=True)

    # ❶ unrar
    try:
        subprocess.run(["unrar", "x", "-idq", str(src), str(dst)],
                    check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # ❷ 7-Zip
    try:
        subprocess.run(["7z", "x", "-y", f"-o{dst}", str(src)],
                    check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # ❸ pure-python - rarfile（需 pip install rarfile & unar / bsdtar）
    try:
        import rarfile
        with rarfile.RarFile(src) as rf:
            rf.extractall(path=dst)
        return
    except Exception as e:
        raise RuntimeError(f"無法解壓 {src} ─ {e}")

def _extract_zip(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)
# =========================================================
