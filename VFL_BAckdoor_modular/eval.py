import torch
import numpy as np
import time
import Defense
import Attack
from NeuralCleanse import NeuralCleanseDefense
from defense_model_structure import CoPurDefense   # NEW
from dualgate_utils import make_dualgate_mask
from vflmonitor import VFLMonitorDefense      # ★ 新增
import json                                   # 若原檔沒載入
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support   # ★ 新增

import os
import matplotlib.pyplot as plt
from pathlib import Path
import time

def _now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def eval_clean_train(model, testloader, args, defense_output):
    print("="*20)
    # with torch.inference_mode(mode=True):
    with torch.no_grad():
        model.eval()
        cum_acc = 0.0
        cum_MAE_acc = 0.0
        cum_SHAP_acc = 0.0
        cum_defense_loss = 0.0
        cum_SHAP_MAE_acc = 0.0
        cum_SHAP_MAE_D_acc = 0.0
        cum_SHAP_MAE_CVPR_acc = 0.0
        cum_NC_acc = 0.0
        cum_CoPur_acc = 0.0
        
        # === 蒐集 label 與預測 (用於 P/R/F1) ===
        all_targets          = []
        all_preds_baseline   = []   # 原始模型
        all_preds_defense    = []   # 目前 args.DM 指定的那個防禦

        
        sum_conflict_sample_count = 0
        attacker_counts_epoch = {}
        participant_conflict_counts_epoch = {}
        
        tot = 0.0
        
        if args.DM == "SHAP":
            defense_models = defense_output["defense_models"]
            static_class_dict = defense_output["static_class_dict"]
            vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
            y_in_last_epoch = defense_output["y_in_last_epoch"]
        elif args.DM == "VFLIP":
            defense_model = defense_output["defense_model"]
        elif args.DM == "SHAP_MAE":
            defense_model = defense_output["defense_model"]
            static_class_dict = defense_output["static_class_dict"]
            vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
            y_in_last_epoch = defense_output["y_in_last_epoch"]
        elif args.DM == "SHAP_MAE_D":
            defense_model = defense_output["defense_model"]
            static_class_dict = defense_output["static_class_dict"]
            vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
            y_in_last_epoch = defense_output["y_in_last_epoch"]
            clean_total_shap_count, clean_total_cos_count = 0, 0
            total_flag_samples, total_flag_pairs = 0, 0
            total_shap_exceed, total_cos_exceed,clean_total_and_count    = 0, 0, 0
            all_cos_list   = []          # 放在迴圈外
            all_cls_idx    = []
            out_dir = Path("cos_plots")
            out_dir.mkdir(exist_ok=True)
        elif args.DM == "SHAP_MAE_CVPR":
            defense_model = defense_output["defense_model"]
            static_class_dict = defense_output["static_class_dict"]
            vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
            y_in_last_epoch = defense_output["y_in_last_epoch"]
        elif args.DM == "NEURAL_CLEANSE":
            defense_model = defense_output["defense_model"]
        elif args.DM == "CoPur":
            defense_model = defense_output["defense_model"]
        elif args.DM == "VFLMonitor":
            out_dir = Path("VFLMonitor")
            out_dir.mkdir(exist_ok=True)
            bank_path = os.path.join(out_dir, 'vflmonitor_bank.pt')
            defense = VFLMonitorDefense(bank_path      = bank_path,
                                proj_num       = args.swd_proj,
                                majority_ratio = 0.5,
                                device         = args.device,
                                tie_rand_seed  = args.tie_rand_seed)
            print(f"[Eval] VFLMonitor loaded bank from {bank_path}")
            
            # ────────── Baseline / Skipped / All 計數器 ──────────
            base_tot = base_hit = 0               # baseline ACC
            base_att_tot = base_att_hit = 0       # baseline ASR

            skip_tot = skip_hit = 0               # skipped ACC
            skip_att_tot = skip_att_hit = 0       # skipped ASR

            all_tot  = all_hit  = 0               # 全部樣本 ACC
            all_att_tot  = all_att_hit  = 0       # 全部樣本 ASR

            target_label = args.label             # ASR 目標 (乾淨測=隨意; 下毒測=攻擊目標)
            # ────────────────────────────────────────────────────

        if args.DM in ["SHAP","SHAP_MAE","SHAP_MAE_CVPR","SHAP_MAE_D"]:
            all_shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}
            for k in range(args.num_class):
                all_shapley_inner_key = list(range(args.P))
                all_shapley_outer_dict[k] = {key: torch.empty(0, device=args.device) for key in all_shapley_inner_key}
            preded_shapley_key = list(range(args.P))
            preded_shapley_dict = {key: torch.empty(0, device=args.device) for key in preded_shapley_key}
            pred_all = torch.empty(0, device=args.device)
        
        #time
        timing_eval = {"model_s": 0.0, "mae_s": 0.0, "shapley_s": 0.0, "n": 0}
        t0 = _now()
        
        for i, (x_in, y_in) in enumerate(testloader):
            B = x_in.size()[0]
            main_pred = model(x_in)
            pred_c = main_pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
            
            #time
            timing_eval["n"] += B
            
            all_targets.extend(y_in.tolist())   # 真實標籤
            all_preds_baseline.extend(pred_c.tolist())

            if args.DM == "VFLIP":
                if args.D == "NUSWIDE":
                    defense_model.set_mode(False)
                    vec_dict = {}
                    device = next(model.parameters()).device   # 取得模型所在裝置
                    x_in = x_in.to(device)                           # 整個 batch 搬到 GPU
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)  # 之後再切片
                else:
                    defense_model.set_mode(False)
                    vec_dict = {}
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)
                
                # x_list = torch.split(x_in, int(x_in.size(2) / args.P), dim=2)
                # remainder = x_in.size(2) % args.P
                    
                # if remainder > 0:
                #     # 將最後一部份補充進 x_list
                #     x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                #     x_list = x_list[:-2]
                
                for p in range(args.P):
                    vec_dict[p] = model.bottommodels[p](x_list[p])
                vec = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)
                MAE_output = defense_model(vec)
                MAE_pred = model.top(MAE_output)
                MAE_pred_c = MAE_pred.max(1)[1].cpu()
                cum_MAE_acc += (MAE_pred_c.eq(y_in)).sum().item()
                all_preds_defense.extend(MAE_pred_c.tolist())
            
            elif args.DM == "SHAP":
                vec_dict = {}
                x_list = torch.split(x_in, int(x_in.size(2) / args.P), dim=2)
                remainder = x_in.size(2) % args.P
                    
                if remainder > 0:
                    # 將最後一部份補充進 x_list
                    x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                    x_list = x_list[:-2]
                
                for p in range(args.P):
                    vec_dict[p] = model.bottommodels[p](x_list[p])
                vec = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)
                
                with torch.no_grad():
                    vec_shap = vec.detach()
                    returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                # returned_shapley_outer_dict = Defense.do_every_class_for_every_sample_shapley(model, vec, args)
                model.eval()

                # '''
                # 1. 取得 counter_dict，裡面是每個參與者在每個sample被偵測到 shapley value <0 的數量
                # 2. 將 counter_dict 引入defense_model，進行 mask 
                # '''
                # counter_dict = Defense.detect_how_much_sample_shapley_s0_in_P(returned_shapley_outer_dict, args, tag=3)
                # # defense_model.set_mode(False)
                # MAE_output = defense_model(vec, counter_dict=counter_dict)
                # MAE_pred = model.top(MAE_output)
                # MAE_pred_c = MAE_pred.max(1)[1].cpu()
                # cum_MAE_acc += (MAE_pred_c.eq(y_in)).sum().item()
                
                for k in range(args.num_class):
                    for p in range(args.P): 
                        all_shapley_outer_dict[k][p] = torch.cat(
                            (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                        # print(f"在主程式這邊，第{p}參與者在第{k}類別目前累積{len(all_shapley_outer_dict[k][p])}筆資料")
                # print(f"這個 batach 共有 {B} 筆資料")
                # print(f"dict內容:{all_shapley_outer_dict[0][0]}")
                print(f"\r現在正在計算測試時的 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(testloader)*args.batch_size),2)*100}%"," "*20,end="")
                # time
                # '''
                # 每筆資料針對每個類別都進行計算
                # 每個類別都會有10000*10筆資料
                # '''
                
                # returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                # for p in range(args.P): 
                #     preded_shapley_dict[p] = torch.cat(
                #         (preded_shapley_dict[p],returned_preded_shapley_dict[p]),dim=0)
                # pred_c = pred_c.to(args.device)
                # pred_all = torch.cat((pred_all, pred_c), dim=0)
                
                returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                '''
                原先雙人環境
                active_below_passive_exceed, conflict_sample_count = Defense.how_many_data_over_threshold(returned_preded_shapley_dict, static_class_dict, pred_c, args, args.active_party)
                '''
                flagged_passive_dict, conflict_sample_count, attacker_exceed_counts_batch, participant_exceed_in_conflict_batch = \
                    Defense.how_many_data_over_threshold(returned_preded_shapley_dict, static_class_dict, pred_c, args)

                # 累加進 epoch 級別
                attacker_counts_epoch, participant_conflict_counts_epoch = accumulate_epoch_stats(
                    attacker_counts_epoch, participant_conflict_counts_epoch,
                    attacker_exceed_counts_batch, participant_exceed_in_conflict_batch
                )
                sum_conflict_sample_count += conflict_sample_count
                
                # 新增功能：依據 active_below_passive_exceed 的標記，利用 defense_model 修正 passive party 的部分
                if defense_models is not None and isinstance(defense_models, dict):
                    with torch.no_grad():
                        active_dim = args.bottomModel_output_dim
                        active_vec = vec_dict[args.active_party]

                        # 先複製一份 vec_dict，準備更新其中異常者
                        corrected_vec_dict = dict(vec_dict)

                        for p in range(args.P):
                            if p == args.active_party:
                                continue

                            mask = torch.tensor([p in flagged for flagged in flagged_passive_dict], dtype=torch.bool, device=args.device)
                            if mask.sum() == 0:
                                continue

                            active_subset = active_vec[mask]
                            predicted_passive = defense_models[p](active_subset)  # 使用第 p 位 passive 對應的防禦模型

                            current_passive = corrected_vec_dict[p].clone()
                            current_passive[mask] = predicted_passive
                            corrected_vec_dict[p] = current_passive

                        # 最後重新拼接成新的 vec
                        vec = torch.cat([corrected_vec_dict[p] for p in range(args.P)], dim=1)
                            
                SHAP_pred = model.top(vec)
                SHAP_pred_c = SHAP_pred.max(1)[1].cpu()
                cum_SHAP_acc += (SHAP_pred_c.eq(y_in)).sum().item()
                
                for p in range(args.P): 
                    preded_shapley_dict[p] = torch.cat(
                        (preded_shapley_dict[p],returned_preded_shapley_dict[p]),dim=0)
                pred_c = pred_c.to(args.device)
                pred_all = torch.cat((pred_all, pred_c), dim=0)

            elif args.DM == "SHAP_MAE":
                defense_model.set_mode(False)
                vec_dict = {}
                if args.D == "NUSWIDE":
                    device = next(model.parameters()).device   # 取得模型所在裝置
                    x_in = x_in.to(device)                           # 整個 batch 搬到 GPU
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)  # 之後再切片
                else:
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)
                
                # x_list = torch.split(x_in, int(x_in.size(2) / args.P), dim=2)
                # remainder = x_in.size(2) % args.P
                    
                # if remainder > 0:
                #     # 將最後一部份補充進 x_list
                #     x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                #     x_list = x_list[:-2]
                
                for p in range(args.P):
                    vec_dict[p] = model.bottommodels[p](x_list[p])
                vec = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)
                
                with torch.no_grad():
                    vec_shap = vec.detach()
                    returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                model.eval()
                
                for k in range(args.num_class):
                    for p in range(args.P): 
                        all_shapley_outer_dict[k][p] = torch.cat(
                            (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                print(f"\r現在正在計算測試時的 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(testloader)*args.batch_size),2)*100}%"," "*20,end="")
                returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                flagged_passive_dict, conflict_sample_count, attacker_exceed_counts_batch, participant_exceed_in_conflict_batch = \
                    Defense.how_many_data_over_threshold(returned_preded_shapley_dict, static_class_dict, pred_c, args)

                # 累加進 epoch 級別
                attacker_counts_epoch, participant_conflict_counts_epoch = accumulate_epoch_stats(
                    attacker_counts_epoch, participant_conflict_counts_epoch,
                    attacker_exceed_counts_batch, participant_exceed_in_conflict_batch
                )
                sum_conflict_sample_count += conflict_sample_count
                
                SHAP_MAE_output = defense_model(vec, flagged_passive_dict)
                SHAP_MAE_pred = model.top(SHAP_MAE_output)
                SHAP_MAE_pred_c = SHAP_MAE_pred.max(1)[1].cpu()
                cum_SHAP_MAE_acc += (SHAP_MAE_pred_c.eq(y_in)).sum().item()
                all_preds_defense.extend(SHAP_MAE_pred_c.tolist())
            
            elif args.DM == "SHAP_MAE_D":
                defense_model.set_mode(False)
                vec_dict = {}
                
                x_list = Attack._split_for_parties(x_in, args.P, args.D)
                
                for p in range(args.P):
                    vec_dict[p] = model.bottommodels[p](x_list[p])
                vec = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)
                
                with torch.no_grad():
                    vec_shap = vec.detach()
                    returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                model.eval()
                
                for k in range(args.num_class):
                    for p in range(args.P): 
                        all_shapley_outer_dict[k][p] = torch.cat(
                            (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                print(f"\r現在正在計算測試時的 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(testloader)*args.batch_size),2)*100}%"," "*20,end="")
                returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                
                shap_mat = torch.stack([returned_preded_shapley_dict[p] for p in range(args.P)], dim=1)  # (B,P)
                
                cos_mat  = defense_model.compute_cos_distance(vec)
                
                # ----- 取出 (B,P) 門檻 -----
                cls_idx   = pred_c.to(torch.long)              # (B,)
                shap_up   = defense_model.shap_up_tbl[cls_idx]   # (B,P)
                shap_down = defense_model.shap_down_tbl[cls_idx]
                
                all_cos_list.append(cos_mat.detach().cpu())
                all_cls_idx .append(cls_idx.detach().cpu())
                
                # ----- 呼叫新版 Dual-Gate -----
                mask_bool, clean_shap_count, clean_cos_count, clean_and_count = make_dualgate_mask(
                    shap_mat, cos_mat,
                    cls_idx,                   # 同樣傳 (B,) 進函式
                    args.active_party,
                    shap_up   = shap_up,
                    shap_down = shap_down,
                    # cos_med   = defense_model.cos_med,
                    # cos_mad   = defense_model.cos_mad,
                    cos_med   = None,
                    cos_mad   = None,
                    return_counts= True)
                
                clean_total_shap_count+=clean_shap_count
                clean_total_cos_count+=clean_cos_count
                clean_total_and_count+=clean_and_count
                
                # ① 計本 batch 有幾筆樣本被 flag
                batch_flag_sample_cnt = (mask_bool.sum(dim=1) > 0).sum().item()
                batch_flag_pair_cnt   = mask_bool.sum().item()       # 已有 true_mask_cnt 亦可
                
                # ② 全局累加（先在函式最外層定義累加器）
                total_flag_samples += batch_flag_sample_cnt
                total_flag_pairs   += batch_flag_pair_cnt
                total_shap_exceed    += clean_shap_count
                total_cos_exceed    += clean_cos_count
                
                # print(f"[DBG] batch {i}, true_mask_cnt =", mask_bool.sum().item())
                
                '''
                檢查
                print(f"[DBG] shap_up[0]   = {shap_up[0]}")
                print(f"[DBG] shap_down[0] = {shap_down[0]}")
                
                # === Step 1 : 立即打印本 batch 的遮蔽統計 ===
                print(f"[DBG] batch {i}  true_mask_cnt =", mask_bool.sum().item())
                # 觀察每個 sample 被遮幾個人（最多 P-1）
                sample_flag_cnt = torch.sum(mask_bool, dim=1)      # shape (B,)
                print(f"[DBG] sample_flag_cnt 直方圖:",
                      torch.bincount(sample_flag_cnt.cpu()).tolist())
                
                # === Step 2 : 只在第一個 batch 印一次 offline 統計量 ===
                if i == 0:
                # 這四組張量理論上在 offline 訓練後就固定
                    print("[DBG] shap_med[:10] =", defense_model.shap_med[:10])
                    print("[DBG] shap_mad[:10] =", defense_model.shap_mad[:10])
                    print("[DBG] cos_med [:10] =", defense_model.cos_med[:10])
                    print("[DBG] cos_mad [:10] =", defense_model.cos_mad[:10])
                '''
                
                # flagged_passive_dict = {p: mask_bool[:, p] for p in range(args.P)}
                # 把 (B,P) Boolean matrix 轉成 per-sample list 形式
                flagged_passive_dict = [
                    torch.where(mask_bool[b])[0].tolist() for b in range(mask_bool.size(0))  # e.g. [1,3]；若無 party 被遮 → []
                ]
                
                y_in = y_in.cpu()
                SHAP_MAE_D_output = defense_model(vec, flagged_passive_dict)
                SHAP_MAE_D_pred = model.top(SHAP_MAE_D_output)
                SHAP_MAE_D_pred_c = SHAP_MAE_D_pred.max(1)[1].cpu()
                cum_SHAP_MAE_D_acc += (SHAP_MAE_D_pred_c.eq(y_in)).sum().item()
            
            elif args.DM == "SHAP_MAE_CVPR":
                defense_model.set_mode(False)
                vec_dict = {}
                x_list = torch.split(x_in, int(x_in.size(2) / args.P), dim=2)
                remainder = x_in.size(2) % args.P
                    
                if remainder > 0:
                    # 將最後一部份補充進 x_list
                    x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                    x_list = x_list[:-2]
                
                for p in range(args.P):
                    vec_dict[p] = model.bottommodels[p](x_list[p])
                vec = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)
                
                with torch.no_grad():
                    vec_shap = vec.detach()
                    returned_shapley_outer_dict = Defense.do_shapley(model, vec_shap, y_in, args)
                model.eval()
                
                for k in range(args.num_class):
                    for p in range(args.P): 
                        all_shapley_outer_dict[k][p] = torch.cat(
                            (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                print(f"\r現在正在計算測試時的 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(testloader)*args.batch_size/args.num_class),2)*100}%"," "*20,end="")
                
                returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                flagged_passive_dict, conflict_sample_count, attacker_exceed_counts_batch, participant_exceed_in_conflict_batch = \
                    Defense.how_many_data_over_threshold(returned_preded_shapley_dict, static_class_dict, pred_c, args)

                # 累加進 epoch 級別
                attacker_counts_epoch, participant_conflict_counts_epoch = accumulate_epoch_stats(
                    attacker_counts_epoch, participant_conflict_counts_epoch,
                    attacker_exceed_counts_batch, participant_exceed_in_conflict_batch
                )
                sum_conflict_sample_count += conflict_sample_count
                
                SHAP_MAE_CVPR_output = defense_model(vec, flagged_passive_dict)
                SHAP_MAE_CVPR_pred = model.top(SHAP_MAE_CVPR_output)
                SHAP_MAE_CVPR_pred_c = SHAP_MAE_CVPR_pred.max(1)[1].cpu()
                cum_SHAP_MAE_CVPR_acc += (SHAP_MAE_CVPR_pred_c.eq(y_in)).sum().item()
            
            elif args.DM == "NEURAL_CLEANSE":
                vec_dict = {}
                if args.D == "NUSWIDE":
                    device = next(model.parameters()).device   # 取得模型所在裝置
                    x_in = x_in.to(device)                           # 整個 batch 搬到 GPU
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)  # 之後再切片
                else:
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)
                
                # x_list = torch.split(x_in, int(x_in.size(2) / args.P), dim=2)
                # remainder = x_in.size(2) % args.P
                # if remainder > 0:                       # 與其他分支做法一致
                #     x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                #     x_list = x_list[:-2]

                for p in range(args.P):                 # ① 先填滿 vec_dict
                    vec_dict[p] = model.bottommodels[p](x_list[p])

                vec    = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)  # ② 再 concat
                logits = model.top(vec)                                          # ③ 原始 logits
                logits = defense_model.filter_and_correct(vec, logits)           # ④ 線上修補
                NC_pred_c = logits.argmax(1).cpu()
                cum_NC_acc += (NC_pred_c.eq(y_in)).sum().item()
                all_preds_defense.extend(NC_pred_c.tolist())
            
            elif args.DM == "CoPur":
                vec_dict = {}
                if args.D == "NUSWIDE":
                    device = next(model.parameters()).device   # 取得模型所在裝置
                    x_in = x_in.to(device)                           # 整個 batch 搬到 GPU
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)  # 之後再切片
                else:
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)
                
                # x_list = torch.split(x_in, int(x_in.size(2) / args.P), dim=2)
                # remainder = x_in.size(2) % args.P
                # if remainder > 0:                       # 與其他分支做法一致
                #     x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
                #     x_list = x_list[:-2]

                for p in range(args.P):                 # ① 先填滿 vec_dict
                    vec_dict[p] = model.bottommodels[p](x_list[p])

                vec = torch.cat([vec_dict[p] for p in range(args.P)], dim=1)  # ② 再 concat
                vec_recovered = defense_model.purify(vec)
                logits = model.top(vec_recovered)                       
                CoPur_pred_c = logits.argmax(1).cpu()
                cum_CoPur_acc += (CoPur_pred_c.eq(y_in)).sum().item()
                all_preds_defense.extend(CoPur_pred_c.tolist())
            
            elif args.DM == "VFLMonitor":
                # # ❶ 取 batch 總嵌入 → 依 party 切回 List[(B,d)]
                # with torch.no_grad():
                #     B, PD = model._cached_embed.shape
                #     d = PD // args.P
                #     party_embeds = [ model._cached_embed[:, p*d:(p+1)*d]  # (B,d)
                #                      for p in range(args.P) ]
                # ---------- 先算出每個 party 嵌入 ----------
                vec_dict = {}
                if args.D == "NUSWIDE":
                    device = next(model.parameters()).device   # 取得模型所在裝置
                    x_in = x_in.to(device)                           # 整個 batch 搬到 GPU
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)  # 之後再切片
                else:
                    x_list = Attack._split_for_parties(x_in, args.P, args.D)   # (either image or tabular)

                for p in range(args.P):
                    vec_dict[p] = model.bottommodels[p](x_list[p])         # (B, d)

                # ---------- 建出 party_embeds list ----------
                with torch.no_grad():
                    d = vec_dict[0].shape[1]               # 向量維度 (假設 P>=1)
                    party_embeds = [vec_dict[p] for p in range(args.P)]     # len = P
                    B = party_embeds[0].size(0)
                    
                # # 直接從 vec_dict 組成 (P,B,d) → 省掉 cached_embed
                # with torch.no_grad():
                #     d = next(iter(vec_dict.values())).shape[1]        # 向量維度
                #     party_embeds = [ vec_dict[p]          # (B,d)  已含攻擊偏移
                #                      for p in range(args.P) ]
                #     B = party_embeds[0].size(0)

                # ❷ 逐樣本執行偵測並更新統計
                for b in range(B):
                    sample_embeds = [ party_embeds[p][b] for p in range(args.P) ]
                    info = defense.detect(sample_embeds)

                    y_true = int(y_in[b])
                    is_target_cls = (y_true == target_label)   # ← 方便後面判斷
                    
                    # ── Baseline ──
                    if info["decided"]:                       
                        base_tot += 1
                        
                        final_pred = info["final_label"]
                        all_preds_defense.append(final_pred)
                        
                        if final_pred == y_true:
                            base_hit += 1
                            
                        # ----- ASR (排除原本就是 target 的樣本) -----
                        if not is_target_cls:
                            base_att_tot += 1
                            if final_pred == target_label:
                                base_att_hit += 1
                        
                        # base_att_tot += 1
                        # if final_pred == target_label:
                        #     base_att_hit += 1

                    # ── Skipped ──
                    else:                                     
                        skip_tot += 1
            
                        # ① 把 4 個 party 的嵌入接回 (1, P*d)
                        concat_emb = torch.cat(sample_embeds, dim=0).unsqueeze(0)  # (1, Pd)
            
                        # ② 交給 top-model 推論
                        with torch.no_grad():
                            logits = model.top(concat_emb.to(args.device))
                        final_pred = logits.argmax(1).item()
                        all_preds_defense.append(final_pred)
                        
                        # ③ 更新 skipped 指標
                        if final_pred == y_true:
                            skip_hit += 1
                            
                        # ----- ASR -----
                        if not is_target_cls:
                            skip_att_tot += 1
                            if final_pred == target_label:
                                skip_att_hit += 1
                                
                        # skip_att_tot += 1
                        # if final_pred == target_label:
                        #     skip_att_hit += 1

                    # -------- all samples (Mix : baseline or top-model) --------
                    all_tot += 1
                    if final_pred == y_true:        # ← 使用剛才得出的 final_pred
                        all_hit += 1
                    
                    if not is_target_cls:                 # ASR
                        all_att_tot += 1
                        if final_pred == target_label:
                            all_att_hit += 1
                            
                    # all_att_tot += 1
                    # if final_pred == target_label:
                    #     all_att_hit += 1
                    
                # 已完成統計，本 batch 不需進其他 DM 分支
                continue
            
        if args.AM == None:
            print(f"乾淨模型 acc on 乾淨測試資料集: {cum_acc/tot:.4f}")
            print("-"*20)
        else:
            print()
            print(f"poisoned 模型 acc on 乾淨測試資料集: {cum_acc/tot:.4f}")
            
        
        if args.DM == "SHAP":
            print("乾淨測試的 shapley 計算完畢，正在繪圖...")
            _ = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args ,tag=1, draw=False)
            print("正在測試乾淨測試集......")
            # Defense.detect_how_much_sample_shapley_s0_in_P(all_shapley_outer_dict, args, tag=1)
            print(f"共有 {sum_conflict_sample_count} 筆資料被偵測到")
            print(f"經過 {args.DM} 淨化後 poisoned 模型 acc on 乾淨測試資料集: {cum_SHAP_acc/tot:.4f}")
            # active_below_passive_exceed, conflict_sample_count = Defense.how_many_data_over_threshold(preded_shapley_dict, static_class_dict, pred_all, args, args.active_party)
        
        elif args.DM == "SHAP_MAE":
            print("乾淨測試的 shapley 計算完畢，正在繪圖...")
            _ = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args ,tag=1, draw=True)
            print("正在測試乾淨測試集......")
            # Defense.detect_how_much_sample_shapley_s0_in_P(all_shapley_outer_dict, args, tag=1)
            print(f"共有 {sum_conflict_sample_count} 筆資料被偵測到")
            print(f"經過 {args.DM} 淨化後 poisoned 模型 acc on 乾淨測試資料集: {cum_SHAP_MAE_acc/tot:.4f}")
        
        elif args.DM == "SHAP_MAE_D":
            cos_full  = torch.cat(all_cos_list , dim=0)   # (N,P)
            cls_full  = torch.cat(all_cls_idx  , dim=0)   # (N,)
            # ───────────── ⑤ 逐 (class c, participant p) 生成直方圖 ─────────────
            C, P = defense_model.args.num_class, defense_model.args.P
            k  = args.k_cos                    # 你在外面傳進 make_dualgate_mask 的 k_cos

            for c in range(C):
                idx = cls_full == c
                if idx.sum() == 0:                # 該類別沒有樣本
                    continue

                v_cos = cos_full[idx]             # (Nc, P)

                for p in range(P):
                    x = v_cos[:, p].numpy()       # 取出該參與者的所有 cos_distance
                    thr  = (defense_model.cos_med[c, p] +
                            k * defense_model.cos_mad[c, p]).item()

                    plt.figure(figsize=(5,3))
                    plt.hist(x, bins=200, density=True, alpha=0.8)
                    plt.axvline(thr, ls='--', lw=1.2, label=f"thr={thr:.4f}")
                    plt.title(f"Class {c}  ·  Participant {p}")
                    plt.xlabel("cos distance"); plt.ylabel("density")
                    plt.legend()
                    plt.tight_layout()

                    # --- 儲存 ---
                    fpath = out_dir / f"cos_clean_class{c}_part{p}.png"
                    plt.savefig(fpath, dpi=150)
                    plt.close()

            print(f"[INFO] 所有圖已存於 {out_dir.resolve()}")
            
            print("乾淨測試的 shapley 計算完畢，正在繪圖...")
            _ = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args ,tag=1, draw=True)
            print("正在測試乾淨測試集......")
            # Defense.detect_how_much_sample_shapley_s0_in_P(all_shapley_outer_dict, args, tag=1)
            # print(f"共有 {sum_conflict_sample_count} 筆資料被偵測到")
            print(f"經過 {args.DM} 淨化後 poisoned 模型 acc on 乾淨測試資料集: {cum_SHAP_MAE_D_acc/tot:.4f}")
            print(f"[STAT] clean_evaluation  S_shap_exceed={clean_total_shap_count}, "
                f"S_cos_exceed={clean_total_cos_count}, S_and_exceed={clean_total_and_count}")
            print(f"[STAT] clean  FlagSample={total_flag_samples}  "
                    f"FlagPair={total_flag_pairs}  "
                    f"S1_exceed={total_shap_exceed}  S2_exceed={total_cos_exceed}")

        elif args.DM == "SHAP_MAE_CVPR":
            print("乾淨測試的 shapley 計算完畢，正在繪圖...")
            _ = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args ,tag=1, draw=True)
            print("正在測試乾淨測試集......")
            # Defense.detect_how_much_sample_shapley_s0_in_P(all_shapley_outer_dict, args, tag=1)
            print(f"共有 {sum_conflict_sample_count} 筆資料被偵測到")
            print(f"經過 {args.DM} 淨化後 poisoned 模型 acc on 乾淨測試資料集: {cum_SHAP_MAE_CVPR_acc/tot:.4f}")
        
        elif args.DM == "VFLIP":
            print(f"經過 {args.DM} 淨化後 poisoned 模型 acc on 乾淨測試資料集: {cum_MAE_acc/tot:.4f}")
            
        elif args.DM == "NEURAL_CLEANSE":
            print(f"經過 NEURAL_CLEANSE 淨化後 acc on 乾淨測試資料集: {cum_NC_acc/tot:.4f}")
        
        elif args.DM == "CoPur":
            print(f"經過 CoPur 淨化後 acc on 乾淨測試資料集: {cum_CoPur_acc/tot:.4f}")
        
        elif args.DM == "VFLMonitor":
            # ────────────── VFLMonitor 指標輸出 ──────────────
            def div(a, b): return 0.0 if b == 0 else a / b

            metrics_baseline = {
                "ACC": div(base_hit, base_tot),
                "ASR": div(base_att_hit, base_att_tot),
                "Skipped": skip_tot
            }
            metrics_skipped  = {
                "ACC": div(skip_hit, skip_tot),
                "ASR": div(skip_att_hit, skip_att_tot)
            }
            metrics_all      = {
                "ACC": div(all_hit, all_tot),
                "ASR": div(all_att_hit, all_att_tot)
            }
            
            print(f"[Baseline]  ACC={metrics_baseline['ACC']:.4f}  "
                f"ASR={metrics_baseline['ASR']:.4f}  "
                f"(samples={base_tot}, skipped={metrics_baseline['Skipped']})")
            print(f"[Skipped ]  ACC={metrics_skipped['ACC']:.4f}  "
                f"ASR={metrics_skipped['ASR']:.4f}  (samples={skip_tot})")
            print(f"[All-Rand] ACC={metrics_all['ACC']:.4f}  "
                f"ASR={metrics_all['ASR']:.4f}  (samples={all_tot})")

            # ---- 儲存 json （可選）----
            save_dir = os.path.join(out_dir, "vflmonitor_metrics")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "baseline.json"), "w") as f:
                json.dump(metrics_baseline, f, indent=2)
            with open(os.path.join(save_dir, "skipped.json"), "w") as f:
                json.dump(metrics_skipped, f, indent=2)
            with open(os.path.join(save_dir, "all_rand.json"), "w") as f:
                json.dump(metrics_all, f, indent=2)
            # -------------------------------------------------
            
    plot_exceed_stats(attacker_counts_epoch, participant_conflict_counts_epoch, save_prefix="clean")
    
    # === Precision / Recall / F1：Baseline ===
    cls_p, cls_r, cls_f1, cls_sup = precision_recall_fscore_support(
        all_targets, all_preds_baseline,
        average=None, labels=list(range(args.num_class)), zero_division=0)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds_baseline, average='macro', zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, all_preds_baseline, average='micro', zero_division=0)

    print("\n[Baseline]  Precision / Recall / F1 (per class)")
    for c in range(args.num_class):
        print(f"  class {c:>2d} | P={cls_p[c]:.4f}  R={cls_r[c]:.4f}  F1={cls_f1[c]:.4f}  (n={cls_sup[c]})")
    print(f"[Baseline]  Macro-avg: P={p_macro:.4f}  R={r_macro:.4f}  F1={f1_macro:.4f}")
    print(f"[Baseline]  Micro-avg: P={p_micro:.4f}  R={r_micro:.4f}  F1={f1_micro:.4f}")

    # === 若有防禦，另外輸出 ===
    if all_preds_defense:        # list 非空才計算
        d_p, d_r, d_f1, d_sup = precision_recall_fscore_support(
            all_targets, all_preds_defense,
            average=None, labels=list(range(args.num_class)), zero_division=0)

        d_p_macro, d_r_macro, d_f1_macro, _ = precision_recall_fscore_support(
            all_targets, all_preds_defense, average='macro', zero_division=0)
        d_p_micro, d_r_micro, d_f1_micro, _ = precision_recall_fscore_support(
            all_targets, all_preds_defense, average='micro', zero_division=0)

        print(f"\n[{args.DM}] Precision / Recall / F1 (per class)")
        for c in range(args.num_class):
            print(f"  class {c:>2d} | P={d_p[c]:.4f}  R={d_r[c]:.4f}  F1={d_f1[c]:.4f}  (n={d_sup[c]})")
        print(f"[{args.DM}] Macro-avg: P={d_p_macro:.4f}  R={d_r_macro:.4f}  F1={d_f1_macro:.4f}")
        print(f"[{args.DM}] Micro-avg: P={d_p_micro:.4f}  R={d_r_micro:.4f}  F1={d_f1_micro:.4f}")
    
    #time
    t1 = _now()
    timing_eval["model_s"] += (t1 - t0)
    n = timing_eval["n"]
    _save_json(f"timings/eval_clean_{args.DM}_P{args.P}.json", {
        "samples": n,
        "total_model_s": timing_eval["model_s"],
        "total_mae_s": timing_eval["mae_s"],
        "total_shapley_s": timing_eval["shapley_s"],
        "avg_model_ms_per_sample": (timing_eval["model_s"] / n) * 1000,
        "avg_mae_ms_per_sample": (timing_eval["mae_s"] / n) * 1000,
        "avg_shapley_ms_per_sample": (timing_eval["shapley_s"] / n) * 1000
    })

    if args.DM in ["SHAP","SHAP_MAE","SHAP_MAE_CVPR","SHAP_MAE_D"]:
        return all_shapley_outer_dict
    else:
        return None
            
def eval_poisoned_train(model, dl_val_set, poisoned_testloader, attacker_target_vec_dict, attacker_list, args, defense_output):
    if args.DM == "SHAP":
        defense_models = defense_output["defense_models"]
        static_class_dict = defense_output["static_class_dict"]
        vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
        y_in_last_epoch = defense_output["y_in_last_epoch"]
    elif args.DM == "VFLIP":
        defense_model = defense_output["defense_model"]
    elif args.DM == "SHAP_MAE":
        defense_model = defense_output["defense_model"]
        static_class_dict = defense_output["static_class_dict"]
        vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
        y_in_last_epoch = defense_output["y_in_last_epoch"]
    elif args.DM == "SHAP_MAE_D":
        defense_model = defense_output["defense_model"]
        static_class_dict = defense_output["static_class_dict"]
        vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
        y_in_last_epoch = defense_output["y_in_last_epoch"]
        poisoned_total_shap_count, poisoned_total_cos_count = 0,0
        total_flag_samples, total_flag_pairs = 0, 0
        total_shap_exceed, total_cos_exceed,poisoned_total_and_count = 0, 0, 0
        all_cos_list   = []          # 放在迴圈外
        all_cls_idx    = []
        out_dir = Path("cos_plots")
        out_dir.mkdir(exist_ok=True)
    elif args.DM == "SHAP_MAE_CVPR":
        defense_model = defense_output["defense_model"]
        static_class_dict = defense_output["static_class_dict"]
        vec_dict_last_epoch = defense_output["vec_dict_last_epoch"]
        y_in_last_epoch = defense_output["y_in_last_epoch"]
    elif args.DM == "NEURAL_CLEANSE":
        defense_model = defense_output["defense_model"]
    elif args.DM == "CoPur":
        defense_model = defense_output["defense_model"]
    elif args.DM == "VFLMonitor":
        out_dir = Path("VFLMonitor")
        out_dir.mkdir(exist_ok=True)
        bank_path = os.path.join(out_dir, 'vflmonitor_bank.pt')
        defense = VFLMonitorDefense(bank_path      = bank_path,
                            proj_num       = args.swd_proj,
                            majority_ratio = 0.5,
                            device         = args.device,
                            tie_rand_seed  = args.tie_rand_seed)
        print(f"[Eval] VFLMonitor loaded bank from {bank_path}")
        
        # ────────── Baseline / Skipped / All 計數器 ──────────
        base_tot = base_hit = 0               # baseline ACC
        base_att_tot = base_att_hit = 0       # baseline ASR

        skip_tot = skip_hit = 0               # skipped ACC
        skip_att_tot = skip_att_hit = 0       # skipped ASR

        all_tot  = all_hit  = 0               # 全部樣本 ACC
        all_att_tot  = all_att_hit  = 0       # 全部樣本 ASR

        target_label = args.label             # ASR 目標 (乾淨測=隨意; 下毒測=攻擊目標)
        # ────────────────────────────────────────────────────
    
    print("="*20)
    label = args.label
    num_party= args.P
    
    cum_acc = 0.0
    cum_ASR = 0.0
    tot = 0.0
    
    cum_mask_ASR = 0
    tot_mask = 0
    
    cum_MAE_acc = 0.0
    cum_MAE_ASR = 0.0
    cum_MAE_mask_ASR = 0.0
    
    cum_SHAP_mask_label_equal_target_count = 0.0
    
    cum_SHAP_MAE_acc = 0.0
    cum_SHAP_MAE_ASR = 0.0
    cum_SHAP_MAE_mask_ASR = 0.0
    cum_SHAP_MAE_mask_label_equal_target_count = 0.0
    
    cum_SHAP_MAE_D_acc = 0.0
    cum_SHAP_MAE_D_ASR = 0.0
    cum_SHAP_MAE_D_mask_ASR = 0.0
    cum_SHAP_MAE_D_mask_label_equal_target_count = 0.0
    
    cum_NC_acc =0.0
    cum_NC_ASR =0.0
    cum_NC_mask_ASR =0.0
    
    cum_CoPur_acc =0.0
    cum_CoPur_ASR =0.0
    cum_CoPur_mask_ASR =0.0
    
    # === P/R/F1 收集器 ===
    all_targets          = []
    all_preds_baseline   = []
    all_preds_defense    = []

    
    sum_conflict_sample_count = 0
    
    attacker_counts_epoch = {}
    participant_conflict_counts_epoch = {}
    vec_dict = {}
    
    if args.DM in ["SHAP","SHAP_MAE","SHAP_MAE_CVPR","SHAP_MAE_D"]:
        all_shapley_outer_dict = {outer_key: {} for outer_key in range(args.num_class)}
        for k in range(args.num_class):
            all_shapley_inner_key = list(range(args.P))
            all_shapley_outer_dict[k] = {key: torch.empty(0, device=args.device) for key in all_shapley_inner_key}
        preded_shapley_key = list(range(args.P))
        preded_shapley_dict = {key: torch.empty(0, device=args.device) for key in preded_shapley_key}
        pred_all = torch.empty(0, device=args.device)
    
    #time
    timing_eval = {"model_s": 0.0, "mae_s": 0.0, "shapley_s": 0.0, "n": 0}
    t0 = _now()
    
    # with torch.inference_mode(mode=True):
    with torch.no_grad():
        model.eval()
        if args.DM in ["VFLIP","SHAP_MAE","SHAP_MAE_CVPR","SHAP_MAE_D"]:
            defense_model.set_mode(False)
            defense_model.eval()
        
        if args.AM == "BadVFL":
            attack_component = Attack.BadVFL_attack(args)
            final_poisoned_testloader = attack_component.get_poison_dataloader(args, dl_val_set, attacker_list)
        else:
            final_poisoned_testloader = poisoned_testloader
        
        for i, (x_in, y_in, id_in) in enumerate(final_poisoned_testloader):
            all_targets.extend(y_in.tolist())
            
            B = x_in.size()[0]
            x_in = x_in.to(args.device)
            x_list = Attack._split_for_parties(x_in, args.P, args.D)
            tot = tot + B
            #time
            timing_eval["n"] += B
            
            # # 將圖片 x 平均分成 self.multies 份
            # x_list = torch.split(x_in, int(x_in.size(2) / num_party), dim=2)
            # remainder = x_in.size(2) % num_party
                
            # if remainder > 0:
            #     # 將最後一部份補充進 x_list
            #     x_list[-2] = torch.cat((x_list[-2], x_list[-1]), dim=2)
            #     x_list = x_list[:-2]
            
            for p in range(num_party):
                vec_dict[p] = model.bottommodels[p](x_list[p])
            
            if args.AM == "TIFS":
                for attacker in attacker_list:
                    vec_dict[attacker][:] = torch.tensor(attacker_target_vec_dict[attacker]).cuda()
                    # vec_dict[attacker][:] = attacker_target_vec_dict[attacker].clone().detach().cuda()
            elif args.AM =="VILLIAN":
                for attacker in attacker_list:
                    vec_dict[attacker][:] = (
                                vec_dict[attacker] +
                                attacker_target_vec_dict[attacker].clone().detach().cuda()
                                # torch.tensor(attacker_target_vec_dict[attacker], device=args.device)
                            )
            elif args.AM == "BadVFL":
                pass
                
            vec = torch.cat([vec_dict[p] for p in range(num_party)], dim=1)
            pred = model.top(vec)
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
            cum_ASR += (pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
            
            all_preds_baseline.extend(pred_c.tolist())
            
            # 排除原先 y_in = label 的資料
            label_mask = (y_in != label)
            cum_mask_ASR += (pred_c[label_mask].eq(label)).sum().item()
            tot_mask += label_mask.sum().item()
            
            if args.DM == "VFLIP":
                MAE_output = defense_model(vec)
                MAE_pred = model.top(MAE_output)
                MAE_pred_c = MAE_pred.max(1)[1].cpu()
                cum_MAE_acc += (MAE_pred_c.eq(y_in)).sum().item()
                cum_MAE_ASR += (MAE_pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
                cum_MAE_mask_ASR += (MAE_pred_c[label_mask].eq(label)).sum().item()
                
                all_preds_defense.extend(MAE_pred_c.tolist())
            
            elif args.DM =="SHAP_MAE":
                returned_shapley_outer_dict = Defense.do_every_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                
                for k in range(args.num_class):
                    for p in range(args.P): 
                        all_shapley_outer_dict[k][p] = torch.cat(
                            (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                print(f"\r現在正在計算測試時的 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(final_poisoned_testloader)*args.batch_size),2)*100}%"," "*20,end="")
                returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                '''
                原先雙人環境
                active_below_passive_exceed, conflict_sample_count = Defense.how_many_data_over_threshold(returned_preded_shapley_dict, static_class_dict, pred_c, args, args.active_party)
                '''
                flagged_passive_dict, conflict_sample_count, attacker_exceed_counts_batch, participant_exceed_in_conflict_batch = \
                    Defense.how_many_data_over_threshold(returned_preded_shapley_dict, static_class_dict, pred_c, args)

                # 累加進 epoch 級別
                attacker_counts_epoch, participant_conflict_counts_epoch = accumulate_epoch_stats(
                    attacker_counts_epoch, participant_conflict_counts_epoch,
                    attacker_exceed_counts_batch, participant_exceed_in_conflict_batch
                )
                
                sum_conflict_sample_count += conflict_sample_count
                
                for p in range(args.P):
                    if p == args.active_party:
                        continue
                    mask = torch.tensor([p in flagged for flagged in flagged_passive_dict], dtype=torch.bool, device=args.device)
                    if mask.sum() == 0:
                        continue
                
                SHAP_MAE_output = defense_model(vec,flagged_passive_dict)
                SHAP_MAE_pred = model.top(SHAP_MAE_output)
                SHAP_MAE_pred_c = SHAP_MAE_pred.max(1)[1].cpu()
                cum_SHAP_MAE_acc += (SHAP_MAE_pred_c.eq(y_in)).sum().item()
                cum_SHAP_MAE_ASR += (SHAP_MAE_pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
                cum_SHAP_MAE_mask_ASR += (SHAP_MAE_pred_c[label_mask].eq(label)).sum().item()
                cum_SHAP_MAE_mask_label_equal_target_count += (y_in[mask.to(y_in.device)].eq(label)).sum().item()
                
                all_preds_defense.extend(SHAP_MAE_pred_c.tolist())
            
            elif args.DM == "SHAP_MAE_D":
                returned_shapley_outer_dict = Defense.do_every_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                
                for k in range(args.num_class):
                    for p in range(args.P): 
                        all_shapley_outer_dict[k][p] = torch.cat(
                            (all_shapley_outer_dict[k][p], returned_shapley_outer_dict[k][p]), dim=0)
                print(f"\r現在正在計算測試時的 shapley value，進度為：{round(len(all_shapley_outer_dict[k][p])/(len(final_poisoned_testloader)*args.batch_size),2)*100}%"," "*20,end="")
                returned_preded_shapley_dict = Defense.do_predicted_class_for_every_sample_shapley(model, vec, args)
                model.eval()
                
                shap_mat = torch.stack([returned_preded_shapley_dict[p] for p in range(args.P)], dim=1)  # (B,P)
                
                cos_mat  = defense_model.compute_cos_distance(vec)
                
                # ----- 取出 (B,P) 門檻 -----
                cls_idx   = pred_c.to(torch.long)              # (B,)
                shap_up   = defense_model.shap_up_tbl[cls_idx]   # (B,P)
                shap_down = defense_model.shap_down_tbl[cls_idx]
                
                # print(f"[DBG] shap_up[0]   = {shap_up[0]}")
                # print(f"[DBG] shap_down[0] = {shap_down[0]}")
                
                all_cos_list.append(cos_mat.detach().cpu())
                all_cls_idx .append(cls_idx.detach().cpu())
                
                # ----- 呼叫新版 Dual-Gate -----
                mask_bool, poisoned_shap_count, poisoned_cos_count, poisoned_and_count = make_dualgate_mask(
                    shap_mat, cos_mat,
                    cls_idx,                   # 同樣傳 (B,) 進函式
                    args.active_party,
                    shap_up   = shap_up,
                    shap_down = shap_down,
                    # cos_med   = defense_model.cos_med,
                    # cos_mad   = defense_model.cos_mad,
                    cos_med   = None,
                    cos_mad   = None,
                    return_counts= True)
                
                poisoned_total_shap_count+=poisoned_shap_count
                poisoned_total_cos_count+=poisoned_cos_count
                poisoned_total_and_count+=poisoned_and_count
                
                # ① 計本 batch 有幾筆樣本被 flag
                batch_flag_sample_cnt = (mask_bool.sum(dim=1) > 0).sum().item()
                batch_flag_pair_cnt   = mask_bool.sum().item()       # 已有 true_mask_cnt 亦可
                
                # ② 全局累加（先在函式最外層定義累加器）
                total_flag_samples += batch_flag_sample_cnt
                total_flag_pairs   += batch_flag_pair_cnt
                total_shap_exceed    += poisoned_shap_count
                total_cos_exceed    += poisoned_cos_count
                
                # print(f"[DBG] batch {i}, true_mask_cnt =", mask_bool.sum().item())
                
                '''
                檢查
                '''
                # # === Step 1 : 立即打印本 batch 的遮蔽統計 ===
                # print(f"[DBG] batch {i}  true_mask_cnt =", mask_bool.sum().item())
                # # 觀察每個 sample 被遮幾個人（最多 P-1）
                # sample_flag_cnt = torch.sum(mask_bool, dim=1)      # shape (B,)
                # print(f"[DBG] sample_flag_cnt 直方圖:",
                #       torch.bincount(sample_flag_cnt.cpu()).tolist())
                
                # # === Step 2 : 只在第一個 batch 印一次 offline 統計量 ===
                # if i == 0:
                # # 這四組張量理論上在 offline 訓練後就固定
                #     print("[DBG] shap_med[:10] =", defense_model.shap_med[:10])
                #     print("[DBG] shap_mad[:10] =", defense_model.shap_mad[:10])
                #     print("[DBG] cos_med [:10] =", defense_model.cos_med[:10])
                #     print("[DBG] cos_mad [:10] =", defense_model.cos_mad[:10])
                
                # flagged_passive_dict = {p: mask_bool[:, p] for p in range(args.P)}
                
                # 把 (B,P) Boolean matrix 轉成 per-sample list 形式
                flagged_passive_dict = [
                    torch.where(mask_bool[b])[0].tolist() for b in range(mask_bool.size(0))  # e.g. [1,3]；若無 party 被遮 → []
                ]

                # 5 送進防禦模型 → 得淨化後向量
                SHAP_MAE_D_output = defense_model(vec, flagged_passive_dict)
                
                for p in range(args.P):
                    if p == args.active_party:
                        continue
                    mask = torch.tensor([p in flagged for flagged in flagged_passive_dict], dtype=torch.bool, device=args.device)
                    if mask.sum() == 0:
                        continue

                # 6 top-model 推論 & 累計指標
                y_in = y_in.cpu()

                SHAP_MAE_D_pred = model.top(SHAP_MAE_D_output)
                SHAP_MAE_D_pred_c  = SHAP_MAE_D_pred.max(1)[1].cpu()
                cum_SHAP_MAE_D_acc += (SHAP_MAE_D_pred_c.eq(y_in)).sum().item()
                cum_SHAP_MAE_D_ASR += (SHAP_MAE_D_pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
                cum_SHAP_MAE_D_mask_ASR += (SHAP_MAE_D_pred_c[label_mask].eq(label)).sum().item()
                cum_SHAP_MAE_D_mask_label_equal_target_count += (y_in[mask.to(y_in.device)].eq(label)).sum().item()

            elif args.DM == "NEURAL_CLEANSE":
                logits = model.top(vec)
                logits = defense_model.filter_and_correct(vec, logits)
                NC_pred_c = logits.max(1)[1].cpu()
                cum_NC_acc += (NC_pred_c.eq(y_in)).sum().item()
                cum_NC_ASR += (NC_pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
                cum_NC_mask_ASR += (NC_pred_c[label_mask].eq(label)).sum().item()
                
                all_preds_defense.extend(NC_pred_c.tolist())
            
            elif args.DM == "CoPur":
                vec_recovered = defense_model.purify(vec)
                logits = model.top(vec_recovered)                       
                CoPur_pred_c = logits.argmax(1).cpu()
                cum_CoPur_acc += (CoPur_pred_c.eq(y_in)).sum().item()
                cum_CoPur_ASR += (CoPur_pred_c.eq(torch.Tensor(np.repeat([label],B,axis=0)))).sum().item()
                cum_CoPur_mask_ASR += (CoPur_pred_c[label_mask].eq(label)).sum().item()
                
                all_preds_defense.extend(CoPur_pred_c.tolist())
            
            elif args.DM == "VFLMonitor":
                # 直接從 vec_dict 組成 (P,B,d) → 省掉 cached_embed
                with torch.no_grad():
                    d = next(iter(vec_dict.values())).shape[1]        # 向量維度
                    party_embeds = [ vec_dict[p]          # (B,d)  已含攻擊偏移
                                     for p in range(args.P) ]
                    B = party_embeds[0].size(0)
                    
                    # if b == 0 and not info["decided"]:
                    #     print("δ-norm =", (party_embeds[attacker][b] -
                    #                     vec_dict_clean[attacker][b]).norm().item())

                # ❷ 逐樣本執行偵測並更新統計
                for b in range(B):
                    sample_embeds = [ party_embeds[p][b] for p in range(args.P) ]
                    info = defense.detect(sample_embeds)

                    y_true = int(y_in[b])
                    is_target_cls = (y_true == target_label)   # ← 方便後面判斷
                    
                    # ── Baseline ──
                    if info["decided"]:                       
                        base_tot += 1
                        
                        final_pred = info["final_label"]
                        all_preds_defense.append(final_pred)
                        
                        if final_pred == y_true:
                            base_hit += 1
                            
                        # ----- ASR (排除原本就是 target 的樣本) -----
                        if not is_target_cls:
                            base_att_tot += 1
                            if final_pred == target_label:
                                base_att_hit += 1
                        
                        # base_att_tot += 1
                        # if final_pred == target_label:
                        #     base_att_hit += 1

                    # ── Skipped ──
                    else:                                     
                        skip_tot += 1
            
                        # ① 把 4 個 party 的嵌入接回 (1, P*d)
                        concat_emb = torch.cat(sample_embeds, dim=0).unsqueeze(0)  # (1, Pd)
            
                        # ② 交給 top-model 推論
                        with torch.no_grad():
                            logits = model.top(concat_emb.to(args.device))
                        final_pred = logits.argmax(1).item()
                        all_preds_defense.append(final_pred)

                        # ③ 更新 skipped 指標
                        if final_pred == y_true:
                            skip_hit += 1
                            
                        # ----- ASR -----
                        if not is_target_cls:
                            skip_att_tot += 1
                            if final_pred == target_label:
                                skip_att_hit += 1
                                
                        # skip_att_tot += 1
                        # if final_pred == target_label:
                        #     skip_att_hit += 1

                    # -------- all samples (Mix : baseline or top-model) --------
                    all_tot += 1
                    if final_pred == y_true:        # ← 使用剛才得出的 final_pred
                        all_hit += 1
                    
                    if not is_target_cls:                 # ASR
                        all_att_tot += 1
                        if final_pred == target_label:
                            all_att_hit += 1
                            
                    # all_att_tot += 1
                    # if final_pred == target_label:
                    #     all_att_hit += 1
                    
                # 已完成統計，本 batch 不需進其他 DM 分支
                continue
                
            
            
        ACC_all = cum_acc / tot
        ASR_all = cum_ASR / tot
        ASR_mask = cum_mask_ASR / tot_mask
        
        if args.DM != "ABL":
            print("未經防禦的情況下:")
            print(f"ACC 為 {ACC_all:.4f}")
            print(f"ASR 為 {ASR_all:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask:.4f}")
            print("-"*20)
        
        if args.DM == "VFLIP":
            defense_acc = cum_MAE_acc/tot
            ASR_defense = cum_MAE_ASR/tot
            ASR_mask_defense = cum_MAE_mask_ASR/tot_mask
            print(f"VFLIP 防禦效果:")
            print(f"acc on 下毒資料集:{defense_acc:.4f}")
            print(f"ASR on 下毒資料集: {ASR_defense:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask_defense:.4f}")
            print("-"*20)
        
        elif args.DM == "SHAP_MAE":
            _ = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args ,tag=2, draw=True)
            print("正在測試下毒測試集......")
            defense_acc = cum_SHAP_MAE_acc/tot
            ASR_defense = cum_SHAP_MAE_ASR/tot
            ASR_mask_defense = cum_SHAP_MAE_mask_ASR/tot_mask
            print(f"SHAP(MAE) 防禦效果:")
            print(f"共有 {sum_conflict_sample_count} 筆資料被偵測到，其中有 {cum_SHAP_mask_label_equal_target_count} 筆資料本來 label 就是{label}")
            print(f"acc on 下毒資料集:{defense_acc:.4f}")
            print(f"ASR on 下毒資料集: {ASR_defense:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask_defense:.4f}")
            print("-"*20)
        
        elif args.DM == "SHAP_MAE_D":
            cos_full  = torch.cat(all_cos_list , dim=0)   # (N,P)
            cls_full  = torch.cat(all_cls_idx  , dim=0)   # (N,)
            
            # ───────────── ⑤ 逐 (class c, participant p) 生成直方圖 ─────────────
            C, P = defense_model.args.num_class, defense_model.args.P
            k  = args.k_cos                    # 你在外面傳進 make_dualgate_mask 的 k_cos

            for c in range(C):
                idx = cls_full == c
                if idx.sum() == 0:                # 該類別沒有樣本
                    continue

                v_cos = cos_full[idx]             # (Nc, P)

                for p in range(P):
                    x = v_cos[:, p].numpy()       # 取出該參與者的所有 cos_distance
                    thr  = (defense_model.cos_med[c, p] +
                            k * defense_model.cos_mad[c, p]).item()

                    plt.figure(figsize=(5,3))
                    plt.hist(x, bins=200, density=True, alpha=0.8)
                    plt.axvline(thr, ls='--', lw=1.2, label=f"thr={thr:.4f}")
                    plt.title(f"Class {c}  ·  Participant {p}")
                    plt.xlabel("cos distance"); plt.ylabel("density")
                    plt.legend()
                    plt.tight_layout()

                    # --- 儲存 ---
                    fpath = out_dir / f"cos_poisoned_class{c}_part{p}.png"
                    plt.savefig(fpath, dpi=150)
                    plt.close()

            print(f"[INFO] 所有圖已存於 {out_dir.resolve()}")
            
            _ = Defense.calculateAndDraw_distribution_of_shapley_value(all_shapley_outer_dict, args ,tag=2, draw=True)
            print("正在測試下毒測試集......")
            defense_acc = cum_SHAP_MAE_D_acc/tot
            ASR_defense = cum_SHAP_MAE_D_ASR/tot
            ASR_mask_defense = cum_SHAP_MAE_D_mask_ASR/tot_mask
            print(f"SHAP(MAE_D) 防禦效果:")
            print(f"acc on 下毒資料集:{defense_acc:.4f}")
            print(f"ASR on 下毒資料集: {ASR_defense:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask_defense:.4f}")
            print(f"[STAT] poisoned_evaluation  S_shap_exceed={poisoned_total_shap_count},"
                f"S_cos_exceed={poisoned_total_cos_count},S_and_exceed={poisoned_total_and_count}")
            print(f"[STAT] poisoned  FlagSample={total_flag_samples}  "
                    f"FlagPair={total_flag_pairs}  "
                    f"S1_exceed={total_shap_exceed}  S2_exceed={total_cos_exceed}")
            print("-"*20)
            
        elif args.DM == "NEURAL_CLEANSE":
            defense_acc = cum_NC_acc/tot
            ASR_defense = cum_NC_ASR/tot
            ASR_mask_defense = cum_NC_mask_ASR/tot_mask
            print(f"NEURAL_CLEANSE 防禦效果:")
            print(f"acc on 下毒資料集:{defense_acc:.4f}")
            print(f"ASR on 下毒資料集: {ASR_defense:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask_defense:.4f}")
            print("-"*20)
            
        elif args.DM == "CoPur":
            defense_acc = cum_CoPur_acc/tot
            ASR_defense = cum_CoPur_ASR/tot
            ASR_mask_defense = cum_CoPur_mask_ASR/tot_mask
            print(f"CoPur 防禦效果:")
            print(f"acc on 下毒資料集:{defense_acc:.4f}")
            print(f"ASR on 下毒資料集: {ASR_defense:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask_defense:.4f}")
            print("-"*20)
            
        elif args.DM == "ABL":
            print("ABL 防禦效果:")
            print(f"acc on 下毒資料集:{ACC_all:.4f}")
            print(f"ASR on 下毒資料集:{ASR_all:.4f}")
            print(f"排除掉原先 label 即為 {label} 之資料後，ASR 為 {ASR_mask:.4f}")
            print("-"*20)
        
        elif args.DM == "VFLMonitor":
            # ────────────── VFLMonitor 指標輸出 ──────────────
            def div(a, b): return 0.0 if b == 0 else a / b

            metrics_baseline = {
                "ACC": div(base_hit, base_tot),
                "ASR": div(base_att_hit, base_att_tot),
                "Skipped": skip_tot
            }
            metrics_skipped  = {
                "ACC": div(skip_hit, skip_tot),
                "ASR": div(skip_att_hit, skip_att_tot)
            }
            metrics_all      = {
                "ACC": div(all_hit, all_tot),
                "ASR": div(all_att_hit, all_att_tot)
            }

            print(f"[Baseline]  ACC={metrics_baseline['ACC']:.4f}  "
                f"ASR={metrics_baseline['ASR']:.4f}  "
                f"(samples={base_tot}, skipped={metrics_baseline['Skipped']})")
            print(f"[Skipped ]  ACC={metrics_skipped['ACC']:.4f}  "
                f"ASR={metrics_skipped['ASR']:.4f}  (samples={skip_tot})")
            print(f"[All-Rand] ACC={metrics_all['ACC']:.4f}  "
                f"ASR={metrics_all['ASR']:.4f}  (samples={all_tot})")

            # ---- 儲存 json （可選）----
            save_dir = os.path.join(out_dir, "vflmonitor_metrics")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "baseline.json"), "w") as f:
                json.dump(metrics_baseline, f, indent=2)
            with open(os.path.join(save_dir, "skipped.json"), "w") as f:
                json.dump(metrics_skipped, f, indent=2)
            with open(os.path.join(save_dir, "all_rand.json"), "w") as f:
                json.dump(metrics_all, f, indent=2)
            # -------------------------------------------------
        
    plot_exceed_stats(attacker_counts_epoch, participant_conflict_counts_epoch, save_prefix="poisoned")
    
    # === Precision / Recall / F1：Baseline ===
    cls_p, cls_r, cls_f1, cls_sup = precision_recall_fscore_support(
        all_targets, all_preds_baseline,
        average=None, labels=list(range(args.num_class)), zero_division=0)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds_baseline, average='macro', zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, all_preds_baseline, average='micro', zero_division=0)

    print("\n[Baseline]  Precision / Recall / F1 (per class)")
    for c in range(args.num_class):
        print(f"  class {c:>2d} | P={cls_p[c]:.4f}  R={cls_r[c]:.4f}  F1={cls_f1[c]:.4f}  (n={cls_sup[c]})")
    print(f"[Baseline]  Macro-avg: P={p_macro:.4f}  R={r_macro:.4f}  F1={f1_macro:.4f}")
    print(f"[Baseline]  Micro-avg: P={p_micro:.4f}  R={r_micro:.4f}  F1={f1_micro:.4f}")

    # === 若有防禦，另外輸出 ===
    if all_preds_defense:        # list 非空才計算
        d_p, d_r, d_f1, d_sup = precision_recall_fscore_support(
            all_targets, all_preds_defense,
            average=None, labels=list(range(args.num_class)), zero_division=0)

        d_p_macro, d_r_macro, d_f1_macro, _ = precision_recall_fscore_support(
            all_targets, all_preds_defense, average='macro', zero_division=0)
        d_p_micro, d_r_micro, d_f1_micro, _ = precision_recall_fscore_support(
            all_targets, all_preds_defense, average='micro', zero_division=0)

        print(f"\n[{args.DM}] Precision / Recall / F1 (per class)")
        for c in range(args.num_class):
            print(f"  class {c:>2d} | P={d_p[c]:.4f}  R={d_r[c]:.4f}  F1={d_f1[c]:.4f}  (n={d_sup[c]})")
        print(f"[{args.DM}] Macro-avg: P={d_p_macro:.4f}  R={d_r_macro:.4f}  F1={d_f1_macro:.4f}")
        print(f"[{args.DM}] Micro-avg: P={d_p_micro:.4f}  R={d_r_micro:.4f}  F1={d_f1_micro:.4f}")

    #time
    t1 = _now()
    timing_eval["model_s"] += (t1 - t0)
    n = timing_eval["n"]
    _save_json(f"timings/eval_poisoned_{args.DM}_P{args.P}.json", {
        "samples": n,
        "total_model_s": timing_eval["model_s"],
        "total_mae_s": timing_eval["mae_s"],
        "total_shapley_s": timing_eval["shapley_s"],
        "avg_model_ms_per_sample": (timing_eval["model_s"] / n) * 1000,
        "avg_mae_ms_per_sample": (timing_eval["mae_s"] / n) * 1000,
        "avg_shapley_ms_per_sample": (timing_eval["shapley_s"] / n) * 1000
    })
    
    if args.DM == "SHAP":
        return all_shapley_outer_dict
    else:
        return None
            
def accumulate_epoch_stats(attacker_counts_epoch, participant_conflict_counts_epoch,
                           attacker_exceed_counts_batch, participant_exceed_in_conflict_batch):
    # 累加 attacker exceed 次數
    for k, v in attacker_exceed_counts_batch.items():
        attacker_counts_epoch[k] = attacker_counts_epoch.get(k, 0) + v

    # 累加所有參與者在 conflict 中 exceed 的次數
    for k, v in participant_exceed_in_conflict_batch.items():
        participant_conflict_counts_epoch[k] = participant_conflict_counts_epoch.get(k, 0) + v

    return attacker_counts_epoch, participant_conflict_counts_epoch

import matplotlib.pyplot as plt

def plot_exceed_stats(attacker_counts_epoch, participant_conflict_counts_epoch, save_prefix=""):
    """
    繪製：
    - 每位攻擊者 exceed 次數（所有樣本）
    - 每位參與者在 conflict 狀況下 exceed 的次數
    """
    # 攻擊者 exceed 次數
    if attacker_counts_epoch:
        plt.figure(figsize=(8, 6))
        plt.bar(attacker_counts_epoch.keys(), attacker_counts_epoch.values(), color='red')
        plt.title('Attacker Exceed Count (All Samples)')
        plt.xlabel('Attacker ID')
        plt.ylabel('Exceed Count')
        plt.savefig(f"{save_prefix}attacker_exceed_count.png")
        plt.close()

    # 所有參與者在 active below 時 exceed 的次數
    if participant_conflict_counts_epoch:
        plt.figure(figsize=(10, 6))
        plt.bar(participant_conflict_counts_epoch.keys(), participant_conflict_counts_epoch.values(), color='blue')
        plt.title('Participant Exceed Count When Active is Below (Conflict Cases)')
        plt.xlabel('Participant ID')
        plt.ylabel('Exceed Count in Conflict Samples')
        plt.savefig(f"{save_prefix}participant_conflict_exceed_count.png")
        plt.close()
