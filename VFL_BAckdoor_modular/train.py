import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import train_model_structure as tms
import defense_model_structure as dms
import Attack
import Defense
import time
import itertools
import torch.nn.functional as F

def poisoning_train(dl_train_set, poisoned_trainloader, trn_x, trn_y, args):
    # 設定 epoch, 參與者數量
    num_class = args.num_class
    epochs = args.epochs
    start_poison_epoch = args.start_poison_epoch
    start_defense_epoch = args.start_defense_epoch
    attack_method = args.AM 
    num_attacker = args.AN
    defense_method = args.DM
    num_party= args.P
    attacker_target_vec_dict = {}
    vec_dict = {}
    defense_model = None
    contrastive_flag = False
    
    # 指定誰是 attacker
    if args.AL == "F":
        # 前幾位
        attacker_list = np.arange(num_attacker)
    elif args.AL == "L":
        # 後幾位
        attacker_list = np.arange(start=num_party-num_attacker, stop=num_party)
    elif args.AL == "C":
        #中間幾個
        num_redundant = num_party-num_attacker
        first_redundant = int(num_redundant/2)
        last_redundant = num_attacker+first_redundant
        attacker_list = list(range(num_party)[first_redundant:last_redundant])  
    else:
        # 隨機
        attacker_list = np.random.choice(num_party, size=num_attacker, replace=False)
    print(f"在本次訓練中，攻擊者為:{attacker_list}")
    not_attacker_list = list(set(list(range(num_party))) - set(attacker_list))
    #active party 永遠設為攻擊者後一個人
    active_party = attacker_list[num_attacker-1]+1
    print(f"在本次訓練中，active party 為:{active_party}")
    
    other_party = list(range(num_party))
    other_party.remove(active_party)
    passive_party = other_party
    print(f"在本次訓練中 passive party 為：{passive_party}")
    
    if attack_method == "TIFS":
        attack_component = Attack.TIFS_attack()
    elif attack_method == "VILLIAN":
        attack_component = Attack.VILLIAN_attack(args)
    elif attack_method == "BadVFL":
        attack_component = Attack.BadVFL_attack(args)
    steal_set, steal_id = attack_component.steal_samples(trn_x, trn_y, args)
    
    # 設定 tms
    if args.D in ["CIFAR10","CINIC10"]:
        model = tms.Model(args)
    elif args.D == "Imagenette":
        pass
    elif args.D == "BM":
        pass
    elif args.D == "NUSWIDE":
        pass
    
    if args.DM == "VFLIP":
        defense_model = dms.MAE(args,active_party)
        static_class_dict=0
        
    if args.DM == "SHAP":
        if contrastive_flag:
            defense_model = dms.MultiTaskDefenseModelWithoutCls(args, active_party)
            # defense_model = dms.DeepResidualPredictModel(args, active_party)
            print(type(defense_model))
        else:
            # defense_model = dms.PredictModel(args,active_party)
            # 建立 defense_models 與對應的 optimizer 字典
            defense_models = {}
            optimizer_preds = {}
            for p in range(num_party):
                if p == active_party:
                    continue
                defense_models[p] = dms.PredictModel(args, active_party, passive_party=p)
                optimizer_preds[p] = torch.optim.Adam(defense_models[p].parameters(), lr=0.05, eps=1e-4)
            print(type(defense_models))
        
        all_shapley_outer_dict = {outer_key: {} for outer_key in range(num_class)}
        for k in range(num_class):
            all_shapley_inner_key = list(range(args.P))
            all_shapley_outer_dict[k] = {key: torch.empty(0, device=args.device) for key in all_shapley_inner_key}
        into_shapley_count = 0
        id_in_class = []
        steal_id_in_class = []
        
        vec_dict_last_epoch = {}
        for p in range(num_party):
            vec_dict_last_epoch[p] = torch.empty(0, device=args.device)
        y_in_last_epoch = []
        
    
    # 設定 model.train()
    model.train()
    # 設定 lr, optimizer, 
    if args.AM == "VILLIAN":
        if args.D == "Imagenette":
            attacker_LR_mutiply = 1.2
        else:
            attacker_LR_mutiply = 2
            
        optimizer_clf = torch.optim.Adam([
            {'params':model.top.parameters(),'lr':2e-3,'weight_decay':1e-5,'eps':1e-4},
            {'params': [param for idx in not_attacker_list for param in model.bottommodels[idx].parameters()], 'lr': 2e-3, 'weight_decay': 1e-5, 'eps': 1e-4},
            {'params': [param for idx in attacker_list for param in model.bottommodels[idx].parameters()], 'lr': 2e-3*attacker_LR_mutiply, 'weight_decay': 1e-5, 'eps': 1e-4}  # smaller
            ])
    else:
        optimizer_clf = torch.optim.Adam([
            {'params':model.top.parameters(),'lr':2e-3,'weight_decay':1e-5,'eps':1e-4},
            {'params':model.bottommodels.parameters(),'lr':2e-3,'weight_decay':1e-5,'eps':1e-4}
            ])
    if args.DM == "VFLIP":
        optimizer_MAE = torch.optim.Adam([
                {'params':defense_model.parameters(),'lr':0.1,'weight_decay':1e-4,'eps':1e-4}
                ])
        scheduler_MAE = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_MAE, mode='min', factor=0.5, patience=5)
    elif args.DM == "SHAP" and contrastive_flag:
        optimizer_pred = torch.optim.Adam([
            {'params':defense_model.parameters(),'lr':0.05,'eps':1e-4}
            ])
        # scheduler_pred = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred, mode='min', factor=0.5, patience=5)
        
    # 開始 train
    for epoch in range(epochs):
        # 判斷是否開始進行下毒
        if epoch < start_poison_epoch:
            cum_loss = 0.0
            cum_acc = 0.0
            tot = 0.0
            
            cum_defense_loss = 0.0
            avg_loss_defense = 0.0
            
            cum_MAE_loss = 0.0
            avg_loss_MAE = 0.0
                
            for i, (x_in, y_in, _) in enumerate(poisoned_trainloader):
                B = x_in.size()[0]
                x_in = x_in.to(args.device)
                main_pred = model(x_in)
                classification_loss = model.loss(main_pred, y_in)
                
                optimizer_clf.zero_grad()
                classification_loss.backward()
                optimizer_clf.step()
                cum_loss += classification_loss.item() * B

                pred_c = main_pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
                tot = tot + B
                
                # 將圖片 x 平均分成 num_party 份
                x_list = torch.split(x_in, int(x_in.size(2) / num_party), dim=2)
                remainder = x_in.size(2) % num_party
                if remainder > 0:
                    # 將最後一部份補充進 x_list
                    x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                    x_list = x_list[:-2]
                
                for p in range(num_party):
                    # vec_dict[p] = model.bottommodels[p](x_list[p])
                    vec_dict[p] = model.bottommodels[p](x_list[p].clone())
                vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
                
                if epoch >= start_defense_epoch:
                    if args.DM == "VFLIP":
                        defense_model, defense_loss = defense_VFLIP_train(defense_model, vec, optimizer_MAE)
                        cum_defense_loss += defense_loss.item() * B
                    elif args.DM == "SHAP":
                        if contrastive_flag:
                            '''
                            contrastive learning 
                            '''
                            defense_model, defense_loss, passive_party = defense_with_contrast_train(defense_model, vec_dict, optimizer_pred, args, active_party, y_in)
                            cum_defense_loss += defense_loss.item() * B
                        else:
                            '''
                            原先程式碼
                            defense_model, defense_loss, passive_party = defense_SHAP_train(defense_model, vec_dict, optimizer_pred, args, active_party)
                            cum_defense_loss += defense_loss.item() * B
                            '''
                            
                            '''
                            改為多參與者後
                            '''
                            defense_models, defense_loss = defense_SHAP_train_multi(defense_models, optimizer_preds, vec_dict, args, active_party)
                            cum_defense_loss += defense_loss * B  # 累計所有被動方的 loss
                
            if epoch >= start_defense_epoch:
                avg_loss_defense = cum_defense_loss / len(poisoned_trainloader)
                if args.DM == "VFLIP":
                    scheduler_MAE.step(avg_loss_defense)
                    print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                    (defending), MAE_loss = {cum_defense_loss/tot:.4f}")
                elif args.DM == "SHAP":
                    # scheduler_pred.step(avg_loss_defense)
                    print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                    (defending), pred_loss = {cum_defense_loss/tot:.4f}")
                
            else:
                print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
            
        else:
            cum_loss = 0.0
            cum_acc = 0.0
            tot = 0.0
            
            cum_MAE_loss = 0.0
            avg_loss_MAE = 0.0
            
            if attack_method == "TIFS":
                if (epoch-start_poison_epoch)%10 == 0:
                    '''
                    先指定攻擊者是誰
                    後續將攻擊向量製作出來
                    並做成一份 vec
                    '''
                    attacker_target_vec_dict = attack_component.get_design_vec(model, args, steal_set, attacker_list)
                    model.train()
                    
                for i, (x_in, y_in, id_in) in enumerate(poisoned_trainloader):
                    B = x_in.size()[0]
                    x_in = x_in.to(args.device)
                    # 將圖片 x 平均分成 num_party 份
                    x_list = torch.split(x_in, int(x_in.size(2) / num_party), dim=2)
                    remainder = x_in.size(2) % num_party
                    if remainder > 0:
                        # 將最後一部份補充進 x_list
                        x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                        x_list = x_list[:-2]
                    
                    for p in range(num_party):
                        # vec_dict[p] = model.bottommodels[p](x_list[p])
                        vec_dict[p] = model.bottommodels[p](x_list[p].clone())
                    
                    label_condition = []
                    condition = []
                    for idx in range(B):
                        if id_in[idx] in steal_id:
                            condition.append(idx)
                        if y_in[idx] == args.label:
                            label_condition.append(idx)
                    
                    for attacker in attacker_list:
                        vec_dict[attacker][condition] = torch.tensor(attacker_target_vec_dict[attacker]).cuda()
                        
                    vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
                    
                    '''
                    丟入 model 進行訓練
                    '''
                    main_pred = model.top(vec)
                    optimizer_clf.zero_grad()
                    classification_loss = model.loss(main_pred, y_in)
                    
                    # classification_loss.backward(retain_graph=True)
                    classification_loss.backward()
                    optimizer_clf.step()
                    cum_loss += classification_loss.item() * B

                    pred_c = main_pred.max(1)[1].cpu()
                    cum_acc += (pred_c.eq(y_in)).sum().item()
                    
                    if epoch >= start_defense_epoch:
                        if args.DM == "VFLIP":
                            defense_model, defense_loss = defense_VFLIP_train(defense_model, vec, optimizer_MAE)
                            cum_defense_loss += defense_loss.item() * B
                        elif args.DM == "SHAP":
                            if contrastive_flag:
                                '''
                                contrastive learning 
                                '''
                                defense_model, defense_loss, passive_party = defense_with_contrast_train(defense_model, vec_dict, optimizer_pred, args, active_party, y_in)
                                cum_defense_loss += defense_loss.item() * B
                            else:
                                '''
                                原先程式碼
                                defense_model, defense_loss, passive_party = defense_SHAP_train(defense_model, vec_dict, optimizer_pred, args, active_party)
                                cum_defense_loss += defense_loss.item() * B
                                '''
                                
                                '''
                                改為多參與者後
                                '''
                                defense_models, defense_loss = defense_SHAP_train_multi(defense_models, optimizer_preds, vec_dict, args, active_party)
                                cum_defense_loss += defense_loss * B  # 累計所有被動方的 loss
                                
                    tot = tot + B
                    
                    '''
                    shap
                    '''
                    if (args.DM == "SHAP") and (epoch == epochs-1):
                        for index in range(len(condition)):
                            condition[index] = condition[index] + (args.batch_size * into_shapley_count)
                        for index in range(len(label_condition)):
                            label_condition[index] = label_condition[index] + (args.batch_size * into_shapley_count)
                        steal_id_in_class.extend(condition)
                        id_in_class.extend(label_condition)
                        # print(f"len steal_id_in_class: {len(steal_id_in_class)}")
                        # print(f"len id_in_class: {len(id_in_class)}")
                        
                        for p in range(num_party):
                            vec_dict_last_epoch[p] = torch.cat((vec_dict_last_epoch[p], vec_dict[p]), dim=0)
                        y_in_last_epoch.extend(y_in)
                        
                        '''
                        建一個 dict 蒐集每個類別每個參與者的貢獻度分布，do_shapley 計算完之後回傳更新此 dict
                        '''
                        with torch.no_grad():
                            vec_dict_shap = {}
                            for p in range(num_party):
                                vec_dict_shap[p] = model.bottommodels[p](x_list[p].clone().detach())
                            vec_shap = torch.cat([vec_dict_shap[p] for p in range(num_party)], dim=1)
                            returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                        # vec_detached = vec.detach()
                        # returned_shapley_outer_dict = Defense.do_shapley(model, vec_detached, y_in, args)
                        for k in range(num_class):
                            for p in range(num_party): 
                                all_shapley_outer_dict[k][p] = torch.cat(
                                    (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                                # print(f"在主程式這邊，第{p}參與者在第{k}類別目前累積{len(all_shapley_outer_dict[k][p])}筆資料")
                        # print(f"這個 batach 共有 {B} 筆資料")
                        # print(f"dict內容:{all_shapley_outer_dict[0][0]}")
                        print(f"\r現在正在計算訓練時 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(trn_y)/num_class),2)*100}%"," "*20,end="")
                        into_shapley_count+=1
                        # for c in range(num_class):
                        #     print(shapley_outer_dict[c])
                
                '''
                shap
                '''
                if (args.DM == "SHAP") and (epoch == epochs-1):
                    print("shapley 計算完畢，正在繪圖...")
                    static_class_dict = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args, draw=True, id_in_class=id_in_class, steal_id_in_class=steal_id_in_class)
                    # print("使用 PCA 繪製並儲存 embedding 分布圖：")
                    # Defense.plot_embedding_distribution(vec_dict_last_epoch, y_in_last_epoch, num_party, reduction_method='PCA')
                    # print("使用 t-SNE 繪製並儲存 embedding 分布圖：")
                    # Defense.plot_embedding_distribution(vec_dict_last_epoch, y_in_last_epoch, num_party, reduction_method='t-SNE')
                
                if epoch >= start_defense_epoch:
                    avg_loss_defense = cum_defense_loss / len(poisoned_trainloader)
                    if args.DM == "VFLIP":
                        scheduler_MAE.step(avg_loss_defense)
                        print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                        (defending), MAE_loss = {cum_defense_loss/tot:.4f}")
                    elif args.DM == "SHAP":
                        # scheduler_pred.step(avg_loss_defense)
                        print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                        (defending), pred_loss = {cum_defense_loss/tot:.4f}")
                else:
                    print(f"(poisoning) Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
                
            elif attack_method == "VILLIAN":
                attacker_target_vec_dict = attack_component.get_design_vec(model, args, poisoned_trainloader, attacker_list, epoch)
                model.train()
                    
                for i, (x_in, y_in, id_in) in enumerate(poisoned_trainloader):
                    B = x_in.size()[0]
                    x_in = x_in.to(args.device)
                    # 將圖片 x 平均分成 num_party 份
                    x_list = torch.split(x_in, int(x_in.size(2) / num_party), dim=2)
                    remainder = x_in.size(2) % num_party
                    if remainder > 0:
                        # 將最後一部份補充進 x_list
                        x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                        x_list = x_list[:-2]
                    
                    for p in range(num_party):
                        # vec_dict[p] = model.bottommodels[p](x_list[p])
                        vec_dict[p] = model.bottommodels[p](x_list[p].clone())
                    
                    label_condition = []
                    condition = []
                    for idx in range(B):
                        if id_in[idx] in steal_id:
                            condition.append(idx)
                        if y_in[idx] == args.label:
                            label_condition.append(idx)
                    
                    for attacker in attacker_list:
                        
                        vec_dict[attacker][condition] = (
                                    vec_dict[attacker][condition] +
                                    attacker_target_vec_dict[attacker].clone().detach().cuda()
                                    # torch.tensor(attacker_target_vec_dict[attacker], device=args.device)
                                )
                        
                    vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
                    
                    '''
                    丟入 model 進行訓練
                    '''
                    main_pred = model.top(vec)
                    optimizer_clf.zero_grad()
                    classification_loss = model.loss(main_pred, y_in)
                    
                        
                    classification_loss.backward()
                    optimizer_clf.step()
                    cum_loss += classification_loss.item() * B

                    pred_c = main_pred.max(1)[1].cpu()
                    cum_acc += (pred_c.eq(y_in)).sum().item()
                    
                    if epoch >= start_defense_epoch:
                        if args.DM == "VFLIP":
                            defense_model, defense_loss = defense_VFLIP_train(defense_model, vec, optimizer_MAE)
                            cum_defense_loss += defense_loss.item() * B
                        elif args.DM == "SHAP":
                            if contrastive_flag:
                                '''
                                contrastive learning 
                                '''
                                defense_model, defense_loss, passive_party = defense_with_contrast_train(defense_model, vec_dict, optimizer_pred, args, active_party, y_in)
                                cum_defense_loss += defense_loss.item() * B
                            else:
                                '''
                                改為多參與者後
                                '''
                                defense_models, defense_loss = defense_SHAP_train_multi(defense_models, optimizer_preds, vec_dict, args, active_party)
                                cum_defense_loss += defense_loss * B  # 累計所有被動方的 loss
                    
                    tot = tot + B
                    
                    '''
                    shap
                    '''
                    if (args.DM == "SHAP") and (epoch == epochs-1):
                        for index in range(len(condition)):
                            condition[index] = condition[index] + (args.batch_size * into_shapley_count)
                        for index in range(len(label_condition)):
                            label_condition[index] = label_condition[index] + (args.batch_size * into_shapley_count)
                        steal_id_in_class.extend(condition)
                        id_in_class.extend(label_condition)
                        
                        for p in range(num_party):
                            vec_dict_last_epoch[p] = torch.cat((vec_dict_last_epoch[p], vec_dict[p]), dim=0)
                        y_in_last_epoch.extend(y_in)
                        
                        '''
                        建一個 dict 蒐集每個類別每個參與者的貢獻度分布，do_shapley 計算完之後回傳更新此 dict
                        '''
                        with torch.no_grad():
                            vec_dict_shap = {}
                            for p in range(num_party):
                                vec_dict_shap[p] = model.bottommodels[p](x_list[p].clone().detach())
                            vec_shap = torch.cat([vec_dict_shap[p] for p in range(num_party)], dim=1)
                            returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                        for k in range(num_class):
                            for p in range(num_party): 
                                all_shapley_outer_dict[k][p] = torch.cat(
                                    (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                        print(f"\r現在正在計算訓練時 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(trn_y)/num_class),2)*100}%"," "*20,end="")
                        into_shapley_count+=1
                
                '''
                shap
                '''
                if (args.DM == "SHAP") and (epoch == epochs-1):
                    print("shapley 計算完畢，正在繪圖...")
                    static_class_dict = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args, draw=True, id_in_class=id_in_class, steal_id_in_class=steal_id_in_class)
                    # print("使用 PCA 繪製並儲存 embedding 分布圖：")
                    # Defense.plot_embedding_distribution(vec_dict_last_epoch, y_in_last_epoch, num_party, reduction_method='PCA')
                    # print("使用 t-SNE 繪製並儲存 embedding 分布圖：")
                    # Defense.plot_embedding_distribution(vec_dict_last_epoch, y_in_last_epoch, num_party, reduction_method='t-SNE')
                    
                if epoch >= start_defense_epoch:
                    avg_loss_defense = cum_defense_loss / len(poisoned_trainloader)
                    if args.DM == "VFLIP":
                        scheduler_MAE.step(avg_loss_defense)
                        print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                        (defending), MAE_loss = {cum_defense_loss/tot:.4f}")
                    elif args.DM == "SHAP":
                        # scheduler_pred.step(avg_loss_defense)
                        print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                        (defending), pred_loss = {cum_defense_loss/tot:.4f}")
                else:
                    print(f"(poisoning) Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
                    
            elif attack_method == "BadVFL":
                if epoch == start_poison_epoch:
                    new_poisoned_trainloader = attack_component.get_poison_dataloader(args, dl_train_set, attacker_list, steal_id)
                for i, (x_in, y_in, id_in) in enumerate(new_poisoned_trainloader):
                    B = x_in.size()[0]
                    x_in = x_in.to(args.device)
                    # 將圖片 x 平均分成 num_party 份
                    x_list = torch.split(x_in, int(x_in.size(2) / num_party), dim=2)
                    remainder = x_in.size(2) % num_party
                    if remainder > 0:
                        # 將最後一部份補充進 x_list
                        x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                        x_list = x_list[:-2]
                    
                    for p in range(num_party):
                        # vec_dict[p] = model.bottommodels[p](x_list[p])
                        vec_dict[p] = model.bottommodels[p](x_list[p].clone())
                        
                    vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
                    
                    '''
                    丟入 model 進行訓練
                    '''
                    main_pred = model.top(vec)
                    optimizer_clf.zero_grad()
                    classification_loss = model.loss(main_pred, y_in)
                    
                    # with torch.autograd.detect_anomaly():
                    #     classification_loss.backward(retain_graph=True)
                        
                    classification_loss.backward()
                    optimizer_clf.step()
                    cum_loss += classification_loss.item() * B

                    pred_c = main_pred.max(1)[1].cpu()
                    cum_acc += (pred_c.eq(y_in)).sum().item()
                    
                    if epoch >= start_defense_epoch:
                        if args.DM == "VFLIP":
                            defense_model, defense_loss = defense_VFLIP_train(defense_model, vec, optimizer_MAE)
                            cum_defense_loss += defense_loss.item() * B
                        elif args.DM == "SHAP":
                            if contrastive_flag:
                                '''
                                contrastive learning 
                                '''
                                defense_model, defense_loss, passive_party = defense_with_contrast_train(defense_model, vec_dict, optimizer_pred, args, active_party, y_in)
                                cum_defense_loss += defense_loss.item() * B
                            else:
                                '''
                                改為多參與者後
                                '''
                                defense_models, defense_loss = defense_SHAP_train_multi(defense_models, optimizer_preds, vec_dict, args, active_party)
                                cum_defense_loss += defense_loss * B  # 累計所有被動方的 loss
                    
                    tot = tot + B
                
                if epoch >= start_defense_epoch:
                    avg_loss_defense = cum_defense_loss / len(poisoned_trainloader)
                    if args.DM == "VFLIP":
                        scheduler_MAE.step(avg_loss_defense)
                        print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                        (defending), MAE_loss = {cum_defense_loss/tot:.4f}")
                    elif args.DM == "SHAP":
                        # scheduler_pred.step(avg_loss_defense)
                        print(f"Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f};\
                        (defending), pred_loss = {cum_defense_loss/tot:.4f}")
                else:
                    print(f"(poisoning) Epoch {epoch+1}, class_loss = {cum_loss/tot:.4f}, acc = {cum_acc/tot:.4f}")
                
                torch.cuda.empty_cache()
                
    # if args.DM =="SHAP":                
    #     return model, defense_models, attacker_list, attacker_target_vec_dict, active_party, passive_party, static_class_dict, vec_dict_last_epoch, y_in_last_epoch
    # else:
    #     return model, defense_model, attacker_list, attacker_target_vec_dict, active_party, passive_party, static_class_dict, vec_dict_last_epoch, y_in_last_epoch
    
    defense_output = {
        "defense_model": defense_model if args.DM == "VFLIP" else None,
        "defense_models": defense_models if args.DM == "SHAP" else None,
        "static_class_dict": static_class_dict if args.DM == "SHAP" else None,
        "vec_dict_last_epoch": vec_dict_last_epoch if args.DM == "SHAP" else None,
        "y_in_last_epoch": y_in_last_epoch if args.DM == "SHAP" else None
    }

    return model, attacker_list, attacker_target_vec_dict, active_party, passive_party, defense_output, all_shapley_outer_dict

    

def defense_VFLIP_train(defense_model, vec, optimizer_MAE):
    vec = vec.detach()  # 分離計算圖
    MAE_output = defense_model(vec.clone())
    MAE_loss = defense_model.loss(MAE_output)

    # MAE_output = defense_model(vec)
    # MAE_loss = defense_model.loss(MAE_output)
    
    optimizer_MAE.zero_grad()
    
    MAE_loss.backward(retain_graph=False)
    optimizer_MAE.step()
    
    return defense_model, MAE_loss

def defense_SHAP_train(defense_model, vec_dict, optimizer_pred, args, active_party):
    other_party = list(range(args.P))
    other_party.remove(active_party)
    passive_party = other_party.pop()
    
    for key in vec_dict.keys():
        vec_dict[key] = vec_dict[key].detach()
    active_predict = defense_model(vec_dict[active_party])
    mse_loss = defense_model.loss(active_predict, vec_dict[passive_party])
    
    optimizer_pred.zero_grad()
    
    mse_loss.backward(retain_graph=False)
    optimizer_pred.step()
    
    return defense_model, mse_loss, passive_party

def defense_SHAP_train_multi(defense_models, optimizer_preds, vec_dict, args, active_party):
    total_loss = 0.0
    # 對每個被動方進行訓練
    for passive_party, def_model in defense_models.items():
        # 注意：將 active 的向量 detach，防止影響主模型參數
        active_predict = def_model(vec_dict[active_party].detach())
        loss = def_model.loss(active_predict, vec_dict[passive_party].detach())
        optimizer_preds[passive_party].zero_grad()
        loss.backward(retain_graph=False)
        optimizer_preds[passive_party].step()
        total_loss += loss.item()
    return defense_models, total_loss


def defense_with_contrast_train(defense_model, vec_dict, optimizer_pred, args, active_party, batch_labels):
    """
    利用 active party 的 embedding 預測 passive embedding，
    並透過生成負樣本(從同一 batch 中抽取與該樣本不同 label 的 passive embedding)來計算對比損失，
    最後結合 reconstruction loss 與對比損失進行反向傳播。
    
    輸入:
      - vec_dict: dict, 包含各個參與者的 embedding (已 detach)
      - batch_labels: 當前 batch 中的標籤 (用於生成負樣本)
      
    回傳:
      - 更新後的 defense_model
      - 總損失
      - passive_party 的索引
    """
    # 假設 active_party 為防禦模型輸入的索引，其他參與者中取一個作為 passive_party
    other_party = list(range(args.P))
    other_party.remove(active_party)
    passive_party = other_party.pop()

    # detach 防止梯度往下傳
    for key in vec_dict.keys():
        vec_dict[key] = vec_dict[key].detach()
    
    # 取得 active 與 passive 的原始 embedding
    active_embedding = vec_dict[active_party]  # shape: (B, D)
    passive_embedding = vec_dict[passive_party]  # shape: (B, D)
    
    # 使用防禦模型從 active_embedding 預測 passive embedding
    # 此處 defense_model 的 forward 回傳 (pred_embedding, shared_feat)，其中 pred_embedding 用於 reconstruction
    pred_embedding, shared_feat= defense_model(active_embedding)
    
    # 產生對比損失的正樣本對：pred_embedding 與對應的 passive_embedding
    pos_labels = torch.ones(active_embedding.size(0), 1).to(args.device)
    
    # 產生對比損失的負樣本對：
    # 對於 batch 中每一筆資料，選取一筆不同 label的 passive_embedding 作為負樣本
    B = passive_embedding.size(0)
    batch_labels_cpu = batch_labels.detach().cpu().numpy()
    negative_passive_list = []
    for i in range(B):
        # 找出與第 i 筆資料 label 不同的索引
        diff_idx = [j for j in range(B) if batch_labels_cpu[j] != batch_labels_cpu[i]]
        if len(diff_idx) > 0:
            neg_idx = np.random.choice(diff_idx)
            negative_passive_list.append(passive_embedding[neg_idx:neg_idx+1])
        else:
            # 若全 batch 樣本標籤都相同，就用自身 (此時對比損失不會產生額外貢獻)
            negative_passive_list.append(passive_embedding[i:i+1])
    negative_passive = torch.cat(negative_passive_list, dim=0)
    neg_labels = torch.zeros(B, 1).to(args.device)
    
    # 計算正負對比損失：
    # 正樣本對：pred_embedding 與 passive_embedding (label=1)
    pos_contrast_loss = defense_model.contrastive_loss_fn(pred_embedding, passive_embedding, pos_labels)
    # 負樣本對：pred_embedding 與 negative_passive (label=0)
    neg_contrast_loss = defense_model.contrastive_loss_fn(pred_embedding, negative_passive, neg_labels)
    
    # 計算重建損失
    recon_loss = defense_model.reconstruction_loss(pred_embedding, passive_embedding)
    
    # 合併損失，這裡以 0.5 加權重建損失，0.5 加權正負對比損失之和(可依需求調整超參數)
    total_defense_loss = 0.5 * recon_loss + 0.5 * (pos_contrast_loss + neg_contrast_loss)
    
    optimizer_pred.zero_grad()
    total_defense_loss.backward()
    optimizer_pred.step()
    
    return defense_model, total_defense_loss, passive_party