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

def train_model(model, dataloader, class_num,label,steal_set,steal_id,epoch_num,start_epoch=0,is_binary=False, verbose=True):
    model.train()
    #原 optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    # optimizer = torch.optim.Adam([
    #     {'params':model.predict.parameters(),'lr':1e-2,'weight_decay':1e-6},
    #     {'params':model.top.parameters(),'lr':1e-3,'weight_decay':1e-5},
    #     {'params':model.models.parameters(),'lr':1e-3,'weight_decay':1e-5}
    #     ])

    # optimizer_pred = torch.optim.Adam([
    #     {'params':model.predict.parameters(),'lr':1e-2,'weight_decay':1e-6}
    #     ])
    
    optimizer_pred = torch.optim.Adam([
        {'params':model.predict.parameters(),'lr':0.1,'eps':1e-4}
        ])
    
    
    optimizer_clf = torch.optim.Adam([
        {'params':model.top.parameters(),'lr':1e-3,'eps':1e-4},
        {'params':model.models.parameters(),'lr':1e-3,'eps':1e-4}
        ])
    scheduler_pred = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred, mode='min', factor=0.5, patience=5)

    # final_epoch_data = {
    #     'passive_embeddings': {i: [] for i in range(class_num)},  # passive party (model 0) 的 embeddings
    #     'active_embeddings': {i: [] for i in range(class_num)},   # active party (model 1) 的 embeddings
    #     'predicted_embeddings': {i: [] for i in range(class_num)}, # predicted embeddings
    #     'labels': []
    # }

    # class_loss_sum = [0.0 for i in range(class_num)]
    # class_loss_count = [0 for i in range(class_num)]
    # target_class_pornot_average_loss = []


    for epoch in range(start_epoch,epoch_num):
        t1 = time.time()

        cum_loss = 0.0
        cum_acc = 0.0
        cum_MAE_loss = 0.0
        tot = 0.0

        if epoch >= clean_epoch and epoch%10 == 0:
            vec_arr = design_vec(class_num, model, label, steal_set)
        
        for i, (x_in, y_in, id_in) in enumerate(dataloader):
            B = x_in.size()[0]
            # print(B)

            if args.unit != 1:
                x_list = x_in.split([int(x_in.size()[2]*args.unit)]+[int(x_in.size()[2]*other_unit) for i in range(args.multies-2)]+[x_in.size()[2]-int(x_in.size()[2]*args.unit)-(args.multies-2)*int(x_in.size()[2]*other_unit)],dim=2)
            else:
                x_list = [x_in]

            vec1 = model.models[0](x_list[0])
            vec2 = model.models[1](x_list[1])

            if epoch >= clean_epoch:
                condition = []
                for idx in range(B):
                    if id_in[idx] in steal_id:
                        condition.append(idx)
                vec1[condition] = torch.tensor(vec_arr).cuda()

            if args.unit != 1:
                '''
                測試維度影響防禦效果
                #4096
                vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,args.multies)], dim=2)
                '''
                vec = torch.cat([vec1]+[model.models[i](x_list[i]) for i in range(1,args.multies)], dim=1)
                
            else:
                vec = vec1

            pred = model.top(vec)
            # pred, output_1 , _, predicted_output_1 = model.top(vec)
            MAE_output = model.predict(vec)
            
            classification_loss, MAE_loss = model.loss(pred, y_in, vec, MAE_output)
            
            # if epoch == epoch_num-1:
            #     for i in range(class_num):
            #         mask = (y_in == i)
            #         class_loss_sum[i] += classification_loss_each[mask].sum().item()
            #         class_loss_count[i] += mask.sum().item()

            
            optimizer_clf.zero_grad()
            optimizer_pred.zero_grad()
            MAE_loss.backward(retain_graph=True)
            classification_loss.backward(retain_graph=True)
            optimizer_clf.step()
            optimizer_pred.step()
            
            # cum_loss += loss.item() * B
            cum_loss += classification_loss.item() * B
            cum_MAE_loss += MAE_loss.item() * B

            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()

            tot = tot + B

            # # 在最後一個 epoch 收集資料
            # if epoch == epoch_num-1:
            #     with torch.no_grad():
            #         # 收集所有需要的 embeddings
            #         for emb_passive, emb_active, emb_pred, label in zip(vec1, vec, MAE_output, y_in):
            #         # for emb_passive, emb_active, emb_pred, loss, label in zip(vec1, output_1, predicted_output_1, classification_loss , y_in):
            #             final_epoch_data['passive_embeddings'][label.item()].append(emb_passive.cpu())
            #             final_epoch_data['active_embeddings'][label.item()].append(emb_active.cpu())
            #             final_epoch_data['predicted_embeddings'][label.item()].append(emb_pred.cpu())
            #             final_epoch_data['labels'].append(label.item())
            #             # class_loss_sum[label.item()].append(loss)
                    

        avg_loss_MAE = cum_MAE_loss / len(trainloader)
        scheduler_pred.step(avg_loss_MAE)

        if verbose:
            t2 = time.time()
            # print("Epoch %d, class_loss = %.4f, pred_loss=%.4f, acc = %.4f (%.4fs)" % (epoch, cum_loss / tot, cum_pred_loss/tot, cum_acc / tot,t2-t1))
            print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, pred_loss = {cum_MAE_loss/tot:.4f}, acc = {cum_acc/tot:.4f}, pred_learning_rate = {scheduler_pred.get_last_lr()} ({t2-t1:.4f}s)")
    
    # Active_all_vector_center_list, Passive_mutual_distance_means_list = calculate_defense_metrics(final_epoch_data, class_num)
    
    # # 計算 cosine similarity
    # cos_similarity = torch.nn.CosineSimilarity(dim=1)
    # class_mean_similarities = []

    # # 對每個類別計算平均相似度
    # for class_idx in range(class_num):
    #     if len(final_epoch_data['active_embeddings'][class_idx]) > 0:
    #         embeddings = torch.stack(final_epoch_data['active_embeddings'][class_idx]).cuda()
    #         # print("embedding shape:",embeddings.shape)
    #         # print("embedding len:",len(embeddings))
    #         center_vector = torch.tensor(Active_all_vector_center_list[class_idx]).cuda()
    #         # print("center_vector shape:",center_vector.shape)
    #         center_vector = center_vector.unsqueeze(0).repeat(len(embeddings), 1, 1, 1)
    #         # print("center_vector shape:",center_vector.shape)

    #         similarities = cos_similarity(embeddings,center_vector)
            
    #         class_mean_similarities.append(similarities.mean().cpu().numpy())
    #         # print(f"class {class_idx} cos_sim = {class_mean_similarities[-1]}")

    # # 計算閾值
    # cos_threshold = min(class_mean_similarities) * 0.9

    # for i in range(class_num):
    #     class_loss_avg = class_loss_sum[i] / class_loss_count[i] if class_loss_count[i] > 0 else 0
    
    # plt.figure(figsize=(10, 6))
    # x = np.arange(class_num)  # 類別的索引
    # width = 0.15  # 長條的寬度

    # # 在 X 軸上繪製兩組長條圖
    # plt.bar(x, class_loss_avg, width, label='Loss by Class', color='skyblue')

    # # 添加標籤和標題
    # plt.xlabel('Class Number')
    # plt.ylabel('Mean Loss')
    # plt.title('Comparison of Last epoch Mean Loss Values across 10 Classes')
    # plt.xticks(x, [f'Class {i}' for i in x])  # 設置 X 軸標籤
    # plt.legend()

    # picture_name = f'./hope.jpg'

    # plt.savefig(picture_name)
    
    # return vec_arr, Passive_mutual_distance_means_list, Active_all_vector_center_list, cos_threshold
    return vec_arr

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

def get_defense_vector(class_num,model,trainset,unit,bottom_series=0):
    # 計算每個 class bottom model 以及 prediction model 產生 vector 平均值，視覺化呈現差異
    # 由差異計算 threshold，超過這個 threshold 便替換成 prediction model vector（整體是一個值，要看每個類別的效果來決定）
    if (bottom_series==0):
        print("正在計算 passive party center vector")
    elif (bottom_series==1):
        print("正在計算 active party center vector")
    
    all_clean_vector = gv.generate_all_clean_vecs(class_num,model.models[bottom_series],trainset,unit,bottom_series)
    all_vector_center_list = []

    # print(len(all_clean_vector))

    for i in range(len(all_clean_vector)):
        dim = filter_dim(all_clean_vector[i])
        center = cc.cal_target_center(all_clean_vector[i][dim].copy(),kernel_bandwidth=1000) 
        target_vec = sv.search_vec(center,all_clean_vector[i],args.unit)
        all_vector_center_list.append(target_vec.reshape((64,int((int((int(32*args.unit)-2)/2+1)-2)/2+1),8)))
    
    bottom_distance_means_list = []
    predict_distance_means_list = []
    mutual_distance_means_list = []

    for i in range(class_num):
        val_x,val_y = gv.get_data_by_class(testset,i)
        dl_test_set = CIFAR10(val_x,val_y,transform=transform_for_train)
        testloader = torch.utils.data.DataLoader(dl_test_set, batch_size=BATCH_SIZE, shuffle=True)
        bottom_distance_list=[]
        predict_distance_list=[]
        mutual_distance_list=[]

        with torch.no_grad():
            for j, (x_in, y_in,_) in enumerate(testloader):
                _, output_1, _ , predicted_output_1 = model(x_in)
                bottom_distance_list.append((output_1.cpu() - all_vector_center_list[i]) ** 2)
                predict_distance_list.append((predicted_output_1.cpu() - all_vector_center_list[i]) ** 2)
                mutual_distance_list.append((output_1.cpu()-predicted_output_1.cpu())** 2)
    
        bottom_distance_means_list.append(np.mean(bottom_distance_list))
        # print("bottom list 內容:",bottom_distance_means_list)
        predict_distance_means_list.append(np.mean(predict_distance_list))
        # print("predict list 內容:",predict_distance_means_list)
        mutual_distance_means_list.append(np.mean(mutual_distance_list))
        # print("predict list 內容:",mutual_distance_means_list)

    # 繪製雙組長條圖
    plt.figure(figsize=(15, 6))
    x = np.arange(len(bottom_distance_means_list))  # 類別的索引
    width = 0.15  # 長條的寬度

    # 在 X 軸上繪製兩組長條圖
    plt.bar(x - width, bottom_distance_means_list, width, label='BottomCenterDist', color='skyblue')
    plt.bar(x, predict_distance_means_list, width, label='PredictCenterDist', color='orange')
    plt.bar(x + width, mutual_distance_means_list, width, label='MutualDist', color='red')

    # 添加標籤和標題
    plt.xlabel('Class Number')
    plt.ylabel('Mean Distance Value')
    plt.title('Comparison of Mean Distance Values across 10 Classes')
    plt.xticks(x, [f'Class {i}' for i in x])  # 設置 X 軸標籤
    plt.legend()

    picture_name = f'./difference_dup-{args.dup}_class-{args.label}_BM-{bottom_series}.jpg'

    plt.savefig(picture_name)
    # 顯示圖表
    # plt.show()
    # 進行實驗看替換後防禦結果如何
    return all_vector_center_list,mutual_distance_means_list

def calculate_defense_metrics(final_epoch_data, class_num):
    """
    使用收集到的最後一個 epoch 的資料計算防禦指標
    """
    # 計算 center vectors
    Passive_all_vector_center_list = []
    Active_all_vector_center_list = []
    
    for i in range(class_num):
        # Passive party center vectors
        passive_vectors = torch.stack(final_epoch_data['passive_embeddings'][i]).numpy()
        passive_vectors = np.array(passive_vectors)
        shape = passive_vectors.shape
        passive_vectors = passive_vectors.reshape((shape[0], shape[1]*shape[2]*shape[3]))
        # print(passive_vectors.shape)
        dim_passive = filter_dim(passive_vectors)
        center_passive = cc.cal_target_center(passive_vectors[dim_passive].copy(), kernel_bandwidth=1000)
        target_vec_passive = sv.search_vec(center_passive, passive_vectors, args.unit)
        Passive_all_vector_center_list.append(target_vec_passive.reshape((64,int((int((int(32*args.unit)-2)/2+1)-2)/2+1),8)))
        
        # Active party center vectors
        active_vectors = torch.stack(final_epoch_data['active_embeddings'][i]).numpy()
        active_vectors = np.array(active_vectors)
        shape = active_vectors.shape
        active_vectors = active_vectors.reshape((shape[0], shape[1]*shape[2]*shape[3]))
        dim_active = filter_dim(active_vectors)
        center_active = cc.cal_target_center(active_vectors[dim_active].copy(), kernel_bandwidth=1000)
        target_vec_active = sv.search_vec(center_active, active_vectors, args.unit)
        Active_all_vector_center_list.append(target_vec_active.reshape((64,int((int((int(32*args.unit)-2)/2+1)-2)/2+1),8)))
    
    # 計算距離
    Passive_mutual_distance_means_list = []
    Passive_bottom_distance_means_list = []
    Passive_predict_distance_means_list = []
    
    for i in range(class_num):
        if final_epoch_data['passive_embeddings'][i]:
            passive_vectors = torch.stack(final_epoch_data['passive_embeddings'][i])
            active_vectors = torch.stack(final_epoch_data['active_embeddings'][i])
            predicted_vectors = torch.stack(final_epoch_data['predicted_embeddings'][i])
            
            # 計算與 center 的距離
            bottom_dist = ((passive_vectors - Passive_all_vector_center_list[i]) ** 2).mean().item()
            predict_dist = ((predicted_vectors - Passive_all_vector_center_list[i]) ** 2).mean().item()
            mutual_dist = ((passive_vectors - predicted_vectors) ** 2).mean().item()
            
            Passive_bottom_distance_means_list.append(bottom_dist)
            Passive_predict_distance_means_list.append(predict_dist)
            Passive_mutual_distance_means_list.append(mutual_dist)
    
    # 繪製圖表
    plt.figure(figsize=(15, 6))
    x = np.arange(len(Passive_bottom_distance_means_list))
    width = 0.15

    plt.bar(x - width, Passive_bottom_distance_means_list, width, label='BottomCenterDist', color='skyblue')
    plt.bar(x, Passive_predict_distance_means_list, width, label='PredictCenterDist', color='orange')
    plt.bar(x + width, Passive_mutual_distance_means_list, width, label='MutualDist', color='red')

    plt.xlabel('Class Number')
    plt.ylabel('Mean Distance Value')
    plt.title('Comparison of Mean Distance Values across 10 Classes')
    plt.xticks(x, [f'Class {i}' for i in x])
    plt.legend()

    picture_name = f'./difference_dup-{args.dup}_class-{args.label}.jpg'
    plt.savefig(picture_name)
    plt.show()

    return Active_all_vector_center_list, Passive_mutual_distance_means_list

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
        vec1 = vec1.view(vec1.shape[0],-1)
        

        model_outputs = [model.models[j](x_list[j]) for j in range(multies)]
        
        '''用距離計算 ，超出閾值則替換 (old detection)'''
        # # PredictModel 預測的輸出，用於比較距離
        # predicted_output_1 = model.predict(model_outputs[1].detach())
        # vec2 = model.models[1](x_list[1]).cuda()

        # # Calculate L2 distance for each vector pair
        # # Reshape tensors to 2D for easier computation
        # vec1_flat = vec1.view(B, -1)  # [500, 64*4*8]
        # pred_flat = predicted_output_1.view(B, -1)  # [500, 64*4*8]
        
        # # Calculate L2 distance
        # # distances = torch.norm(vec1_flat - pred_flat, dim=1)  # [500]
        # distances = ((vec1_flat - pred_flat) ** 2).sum(dim=1)
        
        # # Create mask based on threshold
        # mask = (distances > max_threshold)  # [500]
        
        # # Expand mask to match original tensor dimensions
        # mask_expanded = mask.view(-1, 1, 1, 1).expand_as(vec1)
        
        # print("Number of vectors exceeding threshold:", mask.sum().item())

        # # Replace vectors where mask is True
        # vec1_detected = torch.where(mask_expanded, predicted_output_1, vec1)
        
        # if args.unit != 1:
        #     vec = torch.cat([vec1_detected]+[model.models[j](x_list[j]) for j in range(1,multies)], dim=2)
            
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
        
        '''根據閾值計算結果同步替換 active party 之 vector (old detection)'''
        # # 測試再加上將 active party embedding vector 替換成 center vector
        # # 成功，但預測時不會先知道 label，因此需要一個方法從 10 個 center vector 裡面挑出一個
        # # 目前想法是選與 active embedding vector 之 Cosine Similarity 最大的那一個

        # # 對於每個被標記的樣本，計算與所有 center vectors 的 cosine 相似度
        # selected_center_vectors = []
        # for j in range(B):
        #     if mask[j].item():
        #         # 將所有 center vectors 堆疊成一個張量
        #         all_centers = torch.stack([torch.from_numpy(center).to(vec2.device) 
        #                                    for center in Active_all_vector_center_list])
        #         # print("all_centers shape:",all_centers.shape)
        #         # print("vec2 shape:",vec2[j].shape)
                
        #         # Reshape for comparison
        #         current_vec = vec2[j].view(1, -1)  # [1, 64*4*8]
        #         # print("current_vec shape:", current_vec.shape)
        #         all_centers_flat = all_centers.view(len(Active_all_vector_center_list), -1)  # [num_centers, 64*4*8]
                
        #         # Cosine
        #         # Calculate Cosine similarities
        #         similarities = F.cosine_similarity(current_vec, all_centers_flat)
                
        #         # 找出最相似的 center vector
        #         max_sim_idx = similarities.argmax()
        #         max_sim_value = similarities[max_sim_idx]

        #         # # 如果最大相似度超過閾值，使用對應的 center vector
        #         # if max_sim_value > min_similarity_threshold:
        #         #     selected_center = all_centers[max_sim_idx]
        #         # else:
        #         #     selected_center = vec2[j]  # 保持原始向量

        #         # 直接替換為最相似的 center vector
        #         selected_center = all_centers[max_sim_idx]
                

        #         '''L2
        #         # Calculate L2 distances between current vector and all centers
        #         # 使用廣播機制計算所有距離
        #         distances = ((current_vec - all_centers_flat) ** 2).sum(dim=1)  # [num_centers]
                
        #         # 找出距離最小的 center vector
        #         min_dist_idx = distances.argmin()
                
        #         # 直接選擇距離最小的 center vector
        #         selected_center = all_centers[min_dist_idx]
        #         '''
        #         '''
        #         # Calculate L1 distances between current vector and all centers
        #         # L1 distance = sum of absolute differences
        #         distances = torch.abs(current_vec - all_centers_flat).sum(dim=1)  # [num_centers]
                
        #         # 找出距離最小的 center vector
        #         min_dist_idx = distances.argmin()
                
        #         # 直接選擇距離最小的 center vector
        #         selected_center = all_centers[min_dist_idx]
        #         '''

        #         '''
        #         訓練階段時額外訓練一個模型，輸入為 active party 的 embedding vector / feature，輸出為類別（或是學習 top model 的輸出向量）
        #         預測時一旦超過閾值就使用此模型進行預測，將 active party 之 embedding vector 替換為對應此模型預測類別之 center vector
        #         '''


        #     else:
        #         selected_center = vec2[j]
                
        #     selected_center_vectors.append(selected_center)

        # # 將選擇的 center vectors 堆疊成批次
        # selected_center_vectors = torch.stack(selected_center_vectors)
        
        # # # 替換 vec2
        # # vec2 = torch.where(mask.unsqueeze(1), selected_center_vectors, vec2)

        # # Replace vectors in vec2
        # mask_expanded_vec2 = mask.view(-1, 1, 1, 1).expand_as(vec2)
        # vec2 = torch.where(mask_expanded_vec2, selected_center_vectors, vec2)

        # # center_vectors = torch.stack([torch.from_numpy(Active_all_vector_center_list[y_in[j]]).to(vec2.device) for j in range(B)])
        # # vec2 = torch.where(mask.unsqueeze(1), center_vectors, vec2)

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

    # model.load_state_dict(torch.load('clean-%d-%d-%s.model'%(args.dup,args.multies,args.unit)))
    model.load_state_dict(torch.load('clean_epoch_%d-%d-%s.model'%(args.dup,args.multies,args.unit)))

    
    # train model 有地方需要更改，不應該是在 get_defense_vector 才得到 
    # passive party 的 embedding vector 與 prediction model 的距離平均，要在訓練的最後一輪得到，且要把後門向量包含進來計算
    # 同時也要在最後一輪計算 active party 的 center vector list
    last_vec_arr = train_model(
        model, trainloader, class_num, label,steal_set,steal_id,epoch_num=N_EPOCH,start_epoch=clean_epoch, is_binary=is_binary,verbose=True
        )
    
    torch.save(model.state_dict(),'poison_label_%d-%d-%s-%d.model'%(args.dup,args.multies,args.unit,args.label))

    cleanacc = eval_model(model, testloader, is_binary=is_binary)
    print('clean acc: %.4f'%cleanacc)

    predacc = predict_accuracies(model, testloader, is_binary=is_binary)
    print('pred acc: %.4f'%predacc)

    # _,Passive_mutual_distance_means_list = get_defense_vector(class_num,model,testset,args.unit)
    
    # Active_all_vector_center_list,_= get_defense_vector(class_num,model,testset,args.unit,bottom_series=1)

    ASR_all,ASR_mask= attack_model(model, testloader, last_vec_arr, label, args.multies, args.unit, other_unit)
    print('target label: %d, attack ASR: %.4f, without target class sample attack ASR: %.4f' % (label, ASR_all,ASR_mask))

    np.save('label_%d-%d-%s-%d_vec'%(args.dup,args.multies,args.unit,args.label),last_vec_arr)

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
