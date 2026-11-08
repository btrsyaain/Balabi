import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from datetime import datetime
import json
import argparse
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--clean-epoch', type=int, required=False,default=80, help='the number of training epochs without poisoning')
parser.add_argument('--dup', type=int, required=True, help='the ID for duplicated models of a same setting')
parser.add_argument('--multies', type=int, required=False,default=2, help='the number of mutiple participants')
parser.add_argument('--unit', type=float, required=False, default=0.5,help='the feature ratio held by the attacker')
parser.add_argument('--eval-interval', type=int, required=False, default=5, help='interval for evaluating prediction model')

# def evaluate_prediction(model, dataloader):
#     model.eval()
#     total_mse = 0.0
#     total_samples = 0
#     with torch.no_grad():
#         for x_in, _ in dataloader:
#             # _, Bottom_output, MAE_output = model(x_in)
#             _, Embedding_Vector, MAE_output, _ = model(x_in)

#             Embedding_Vector = Embedding_Vector.view(Embedding_Vector.size(0), -1)
#             MAE_output = MAE_output['output']

#             mse = torch.mean((Embedding_Vector - MAE_output) ** 2)
#             total_mse += mse.item() * x_in.size(0)
#             total_samples += x_in.size(0)
#     return total_mse / total_samples

def train_model(model, dataloader, testloader, epoch_num, is_binary, eval_interval, verbose=True, done=False, MAE_LR=0.1):
    model.train()

    if done:
        # optimizer_pred = torch.optim.SGD([
        #     {'params':model.predict.parameters(),'lr':MAE_LR}
        #     ]) # 10% 超爛
        optimizer_pred = torch.optim.Adam([
            {'params':model.predict.parameters(),'lr':MAE_LR,'eps':1e-4}
            ])
        # optimizer_pred = torch.optim.Adam([
        #     {'params':model.predict.parameters(),'lr':0.00125,'weight_decay':1e-6}
        #     ])
    else:
        # optimizer_pred = torch.optim.SGD([
        #     {'params':model.predict.parameters(),'lr':MAE_LR}
        #     ])
        optimizer_pred = torch.optim.Adam([
            {'params':model.predict.parameters(),'lr':MAE_LR,'eps':1e-4}
            ])
        # optimizer_pred = torch.optim.Adam([
        #     {'params':model.predict.parameters(),'lr':0.005,'weight_decay':1e-6}
        #     ])
    
    optimizer_clf = torch.optim.Adam([
        {'params':model.top.parameters(),'lr':1e-3,'weight_decay':1e-5,'eps':1e-4},
        {'params':model.models.parameters(),'lr':1e-3,'weight_decay':1e-5,'eps':1e-4}
        ])
    
    # scheduler_clf = torch.optim.lr_scheduler.StepLR(optimizer_clf, step_size=30, gamma=0.9)
    # scheduler_pred = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred, mode='min', factor=0.5, patience=10, cooldown=2)
    # scheduler_pred = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred, mode='min', factor=0.5, patience=5)
    
    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        cum_MAE_loss = 0.0
        tot = 0.0

        if (done) or (epoch >=80):
            F_predict_backward = True
            # print("MAE model 更新")
        else:
            F_predict_backward = False
            # print("MAE model 不更新")

        for i, (x_in, y_in) in enumerate(dataloader):

            B = x_in.size()[0]
            # pred, Bottom_output , MAE_output = model(x_in)
            main_pred, Embedding_Vector, MAE_output, MAE_pred = model(x_in)
            

            if F_predict_backward:
                classification_loss, MAE_loss = model.loss(main_pred, y_in, Embedding_Vector , MAE_output)
            else:
                classification_loss, _ = model.loss(main_pred, y_in, Embedding_Vector , MAE_output)
            
            optimizer_clf.zero_grad()
            
            if F_predict_backward:
                optimizer_pred.zero_grad()
                MAE_loss.backward(retain_graph=True)
                optimizer_pred.step()

            classification_loss.backward(retain_graph=True)
            optimizer_clf.step()

            cum_loss += classification_loss.item() * B

            # scheduler_clf.step()
            if F_predict_backward:
                cum_MAE_loss += MAE_loss.item() * B
                # print("MAE_loss",MAE_loss)
                # avg_loss_pred = cum_pred_loss / len(trainloader)
                # scheduler_pred.step(avg_loss_pred)

            if is_binary:
                cum_acc += ((main_pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = main_pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        
        if F_predict_backward:
            avg_loss_MAE = cum_MAE_loss / len(trainloader)
            # scheduler_pred.step(avg_loss_MAE)
        
        if verbose:
            if F_predict_backward:
                print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, pred_loss = {cum_MAE_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
                # print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, pred_loss = {cum_MAE_loss/tot:.4f}, acc = {cum_acc/tot:.4f}, pred_learning_rate = {scheduler_pred.get_last_lr()}")
            else:
                print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
            
        
        # Evaluate prediction model
        # if F_predict_backward and epoch % eval_interval == 0:
        #     prediction_mse = evaluate_prediction(model, testloader)
        #     print(f"Epoch {epoch+1}, Prediction MSE Loss: {prediction_mse:.6f}")
    
    # return scheduler_pred.get_last_lr()[0]
    return 0.1

def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        # pred, _, _ = model(x_in)
        main_pred, _, _, _ = model(x_in)
        if is_binary:
            cum_acc += ((main_pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = main_pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot

def predict_accuracies(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        # pred, _, _= model(x_in,test_pred_perf=True)
        _, _, _, MAE_pred = model(x_in)
        if is_binary:
            cum_acc += ((MAE_pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = MAE_pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot
   
if __name__ == '__main__':
    args = parser.parse_args()

    GPU = True
    if GPU:
        torch.cuda.manual_seed_all(0)
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
    trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True,
                                            transform=transform_for_train)
    testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True,
                                           transform=transform_for_test)
    is_binary = False
    need_pad = False

    from cnn_model_multi_relu_0 import Model

    input_size = (3, 32, 32)
    class_num = 10

    model = Model(gpu=GPU,multies=args.multies,unit=args.unit)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    if args.clean_epoch:
        t1 = time.time()

        MAE_LR = train_model(model, trainloader, testloader, epoch_num=args.clean_epoch, is_binary=is_binary, eval_interval=args.eval_interval, verbose=True, done=False)
        torch.save(model.state_dict(),'clean_epoch_%d-%d-%s.model'%(args.dup,args.multies,args.unit))
        
        _ = train_model(model, trainloader, testloader, epoch_num=N_EPOCH-args.clean_epoch, is_binary=is_binary, eval_interval=args.eval_interval, verbose=True, done=True, MAE_LR=MAE_LR)
        cleanacc = eval_model(model, testloader, is_binary=is_binary)
        predacc = predict_accuracies(model, testloader, is_binary=is_binary)
        torch.save(model.state_dict(),'clean-%d-%d-%s.model'%(args.dup,args.multies,args.unit))
        print('clean acc: %.4f' % cleanacc)
        print('pred acc: %.4f'%predacc)
        t2 = time.time()
        print("Training a model costs %.4fs." % (t2 - t1))

    else:
        t1 = time.time()

        train_model(model, trainloader, testloader, epoch_num=N_EPOCH, is_binary=is_binary, eval_interval=args.eval_interval, verbose=True)
        cleanacc = eval_model(model, testloader, is_binary=is_binary)
        predacc = predict_accuracies(model, testloader, is_binary=is_binary)
        torch.save(model.state_dict(),'clean%d-%d-%s.model'%(args.dup,args.multies,args.unit))
        print('clean acc: %.4f'%cleanacc)
        print('pred acc: %.4f'%predacc)

        t2 = time.time()
        print("Training a model costs %.4fs." % (t2 - t1))