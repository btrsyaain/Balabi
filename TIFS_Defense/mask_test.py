import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime
import json
import argparse
import time
import random
import cal_centers as cc
import generate_vec as gv
import search_vec as sv
import warnings
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import  FuncFormatter

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=int, required=True, help='the target class of your attack')
parser.add_argument('--dup', type=int, required=True, help='the ID for duplicated models of a same setting')
parser.add_argument('--magnification',type=int,required=False,default=10,help='the size of the auxiliary set will be 50*magnification')
parser.add_argument('--multies', type=int, required=False,default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False, default=0.5,help='the feature ratio held by the attacker')
parser.add_argument('--clean-epoch', type=int, required=False,default=80, help='the number of training epochs without poisoning')

args = parser.parse_args()
other_unit = (1-args.unit)/(args.multies-1)

target_num = 50
normal_num = 50
clean_epoch = args.clean_epoch

def prepared_data(set):
    data = []
    label = []
    for idx in range(len(set)):
        x,y = set[idx]
        data.append({'id':idx,'data':x})
        label.append(y)

    return data,label

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

def steal_samples(trn_x,trn_y,t):
    targets = []
    for idx in range(len(trn_y)):
        if trn_y[idx] == t:
            targets.append(trn_x[idx]['id'])
    num = target_num*target_magnification
    print("clean image used for class %d: %d"%(t,num))

    steal_id = random.sample(targets,num)
    data = []
    label = []
    for idx in steal_id:
        data.append(trn_x[idx])
        label.append(trn_y[idx])

    steal_set = CIFAR10(data,label,transform=transform_for_train)
    steal_id = torch.tensor(steal_id)

    return steal_set,steal_id

def design_vec(class_num,model,label,steal_set):
    target_clean_vecs = gv.generate_target_clean_vecs(model.models[0],steal_set,args.unit,bottom_series=0)
    '''
    測試維度影響防禦效果
    '''
    #20dim
    target_clean_vecs = target_clean_vecs.reshape((500,10))
    
    dim = filter_dim(target_clean_vecs)

    center = cc.cal_target_center(target_clean_vecs[dim].copy(),kernel_bandwidth=1000) 
    
    target_vec = sv.search_vec(center,target_clean_vecs,args.unit)

    '''
    測試維度影響防禦效果
    #4096
    target_vec = target_vec.reshape((64,int((int((int(32*args.unit)-2)/2+1)-2)/2+1),8))
    '''
    target_vec = target_vec.reshape(1,10)
    
    return target_vec

def filter_dim(vecs):
    coef = np.corrcoef(vecs)
    rows = np.sum(coef,axis=1)
    selected = np.argpartition(rows,-target_num)[-target_num:]
    # print(np.mean(np.corrcoef(vecs[selected])))
    return selected

def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in,_) in enumerate(dataloader):
        B = x_in.size()[0]
        pred, _, _, _, = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot

def predict_accuracies(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in,_) in enumerate(dataloader):
        B = x_in.size()[0]
        _, _, _, MAE_pred = model(x_in,test_pred_perf=True)
        if is_binary:
            cum_acc += ((MAE_pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = MAE_pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot

def attack_model(model, dataloader, vec_arr,label,multies,unit,other_unit):
    # 嘗試在攻擊中替換，若 embedding vector 與 center 的差距大於平均值，就替換成 prediction model 輸出
    # 查看攻擊成功率以及 main accuracy
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    cum_mask_acc = 0
    tot_mask = 0

    for i, (x_in, y_in,_) in enumerate(dataloader):
        B = x_in.size()[0]

        if args.unit != 1:
            x_list = x_in.split([int(x_in.size()[2]*unit)]+[int(x_in.size()[2]*other_unit) for i in range(multies-2)]+[x_in.size()[2]-int(x_in.size()[2]*unit)-(multies-2)*int(x_in.size()[2]*other_unit)],dim=2)
        
        vec1 = torch.Tensor(np.repeat([vec_arr],B,axis=0)).cuda()

        '''
        測試維度影響防禦效果
        '''
        vec1 = vec1.view(vec1.shape[0],-1)
        

        if args.unit != 1:
            '''
            測試維度影響防禦效果
            #4096
            vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,multies)], dim=2)
            '''
            vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,multies)], dim=1)
        else:
            vec = vec1

        pred = model.top(vec)
        pred_c = pred.max(1)[1].cpu()

        cum_acc += (pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
        tot = tot + B

        # 排除原先 y_in = label 的資料
        mask = (y_in != label)
        cum_mask_acc += (pred_c[mask].eq(label)).sum().item()
        tot_mask += mask.sum().item()
    
    ASR_all = cum_acc / tot
    ASR_mask = cum_mask_acc / tot_mask

    return ASR_all,ASR_mask

def defense_mechanism(model, dataloader, vec_arr,label, multies, unit, other_unit):
    # 嘗試在攻擊中替換，若 embedding vector 與 center 的差距大於平均值，就替換成 prediction model 輸出
    # 查看攻擊成功率以及 main accuracy
    model.eval()
    cum_attack_acc = 0.0
    cum_accurate_acc = 0.0
    tot = 0.0

    cum_nottarget_attack_acc = 0
    cum_nottarget_accurate_acc = 0
    tot_nottarget = 0

    ## min_threshold = min(mutual_distance_means_list)
    # max_threshold = max(Passive_mutual_distance_means_list)
    # print("max threshold:",max_threshold)

    # 測試替換成 center vector
    MAE_cum_accurate_acc = 0
    MAE_cum_nottarget_accurate_acc = 0

    # cos_similarity = torch.nn.CosineSimilarity(dim=1)

    for i, (x_in, y_in,_) in enumerate(dataloader):
        B = x_in.size()[0]

        if args.unit != 1:
            x_list = x_in.split([int(x_in.size()[2]*unit)]+[int(x_in.size()[2]*other_unit) for i in range(multies-2)]+[x_in.size()[2]-int(x_in.size()[2]*unit)-(multies-2)*int(x_in.size()[2]*other_unit)],dim=2)
        
        vec1 = torch.Tensor(np.repeat([vec_arr],B,axis=0)).cuda()

        '''
        測試維度影響防禦效果
        '''
        #20
        vec1 = vec1.view(vec1.shape[0],-1)
            
        if args.unit != 1:
            '''
            測試維度影響防禦效果
            #4096
            vec = torch.cat([vec1]+[model.models[j](x_list[j]) for j in range(1,multies)], dim=2)
            '''
            vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,multies)], dim=1)

        pred = model.top(vec)
        pred_c = pred.max(1)[1].cpu()

        cum_attack_acc += (pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
        cum_accurate_acc += pred_c.eq(y_in).sum().item()
        tot = tot + B

        mask_nottarget = (y_in != label)
        cum_nottarget_attack_acc += (pred_c[mask_nottarget].eq(label)).sum().item()
        cum_nottarget_accurate_acc += (pred_c[mask_nottarget].eq(y_in[mask_nottarget])).sum().item()
        tot_nottarget += mask_nottarget.sum().item()
    
        if args.unit != 1:
            model.predict.set_mode(False)
            MAE_output = model.predict(vec)

        MAE_pred = model.top(MAE_output)
        MAE_pred_c = MAE_pred.max(1)[1].cpu()

        MAE_cum_accurate_acc += MAE_pred_c.eq(y_in).sum().item()
        MAE_cum_nottarget_accurate_acc += (MAE_pred_c[mask_nottarget].eq(y_in[mask_nottarget])).sum().item()
    
    AfterDefense_ASR = cum_attack_acc / tot
    AfterDefense_Acc = cum_accurate_acc / tot
    AfterDefenseNotTarget_ASR = cum_nottarget_attack_acc / tot_nottarget
    AfterDefenseNotTarget_Acc = cum_nottarget_accurate_acc / tot_nottarget
    AfterDefense_Acc_MAE = MAE_cum_accurate_acc / tot
    AfterDefenseNotTarget_Acc_MAE = MAE_cum_nottarget_accurate_acc / tot_nottarget

    return AfterDefense_ASR, AfterDefense_Acc, AfterDefenseNotTarget_ASR, AfterDefenseNotTarget_Acc, AfterDefense_Acc_MAE, AfterDefenseNotTarget_Acc_MAE

if __name__ == '__main__':
    target_magnification = args.magnification

    print("現在執行 dup:",args.dup,",label:",args.label," 防禦實驗")

    GPU = True
    if GPU:
        torch.cuda.manual_seed_all(args.dup)
        random.seed(args.dup)
        torch.manual_seed(args.dup)
        np.random.seed(args.dup)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE = 500
    N_EPOCH = 100
    transform_for_train = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=5),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    transform_for_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True)

    trn_x,trn_y = prepared_data(trainset)
    dl_train_set = CIFAR10(trn_x,trn_y,transform=transform_for_train)
    val_x,val_y = prepared_data(testset)
    dl_val_set = CIFAR10(val_x,val_y,transform=transform_for_test)
    is_binary = False
    need_pad = False

    from cnn_model_multi_relu_0 import Model
    
    input_size = (3, 32, 32)
    class_num = 10

    model = Model(gpu=GPU,multies=args.multies,unit=args.unit)
    trainloader = torch.utils.data.DataLoader(dl_train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(dl_val_set, batch_size=BATCH_SIZE, shuffle=True)

    steal_set,steal_id = steal_samples(trn_x,trn_y,args.label)

    label = args.label
    dup = args.dup

    t1=time.time()

    model.load_state_dict(torch.load('poison_label_%d-%d-%s-%d.model'%(args.dup,args.multies,args.unit,args.label)))

    # train model 有地方需要更改，不應該是在 get_defense_vector 才得到 
    # passive party 的 embedding vector 與 prediction model 的距離平均，要在訓練的最後一輪得到，且要把後門向量包含進來計算
    # 同時也要在最後一輪計算 active party 的 center vector list
    last_vec_arr = torch.from_numpy(np.load('label_%d-%d-%s-%d_vec.npy'%(args.dup,args.multies,args.unit,args.label)))

    cleanacc = eval_model(model, testloader, is_binary=is_binary)
    print('clean acc: %.4f'%cleanacc)

    predacc = predict_accuracies(model, testloader, is_binary=is_binary)
    print('pred acc: %.4f'%predacc)

    ASR_all,ASR_mask= attack_model(model, testloader, last_vec_arr, label, args.multies, args.unit, other_unit)
    print('target label: %d, attack ASR: %.4f, without target class sample attack ASR: %.4f' % (label, ASR_all,ASR_mask))

    print("開始進行防禦測試")
    f = open("voting_record.txt","a")
    f.write("/n開始進行防禦測試/n")
    f.close()

    AfterDefense_ASR, AfterDefense_Acc, AfterDefenseNotTarget_ASR, AfterDefenseNotTarget_Acc, AfterDefense_Acc_C, AfterDefenseNotTarget_Acc_C= defense_mechanism(model, testloader, last_vec_arr, label,args.multies, args.unit, other_unit)
    print('Afer Defense ASR: %.4f, Afer Defense acc: %.4f' % (AfterDefense_ASR,AfterDefense_Acc))
    print('Afer Defense Nontarget ASR: %.4f, Afer Defense Nontarget acc: %.4f' % (AfterDefenseNotTarget_ASR, AfterDefenseNotTarget_Acc))
    print('Afer Defense Plus Replace Active Party vector with center vector, acc: %.4f,  Nontarget acc: %.4f' % (AfterDefense_Acc_C, AfterDefenseNotTarget_Acc_C))
    
    t2 = time.time()
    print("Training a model costs %.4fs."%(t2-t1))

    print("\n","-----NEXT-----"*8+"\n")
