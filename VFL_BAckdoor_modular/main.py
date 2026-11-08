'''
整體流程如下:
1. 程式設定
2. get_data
3. model_train
4. attack
5. eval witt or without defense
'''

import argparse
from ast import arg
import torch
import data
import Attack
import Defense
import train
import train_model_structure as tms
import eval
import Draw

'''
1. 程式設定
    a. dataset
    b. # of participants
    c. # of epoch
    d. attack method
    e. attack label
    f. # of attacker
    g. defense method
'''
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"此程式將在 {device} 上運行")
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--D', type=str, required=True, help='Which dataset to use')
parser.add_argument('--P', type=int, required=False, default=2, help='the number of mutiple participants')
parser.add_argument('--epochs', type=int, required=False,default=100, help='the number of training epoch')
parser.add_argument('--AM', type=str, required=False, default="None", help='Which attack method are used')
parser.add_argument('--label', type=int, required=False, default=0, help='the attack target class')
parser.add_argument('--AN', type=int, required=False, default=1, help='the number of attacker')
parser.add_argument('--AL', type=str, required=False, default="C", help='where the attacker at')  
parser.add_argument('--DM', type=str, required=False, default="None", help='the defense method')
parser.add_argument('--act', type=str, required=False, default=False, help='Whether arrange an active party, and the active party ID is the one after the last attacker')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
if args.AM != "None":
    args.start_poison_epoch = Attack.get_start_poison_epoch(args)
if args.DM != "None":
    args.start_defense_epoch = Defense.get_start_defense_epoch(args)
args.batch_size = 500

print(f"環境中是否有 active party: {args.act}")
print(f"使用 {args.D} 資料集")
trainloader, testloader, input_size, num_class, trn_x, trn_y, dl_train_set, poisoned_trainloader, val_x, val_y,  dl_val_set, poisoned_testloader\
    = data.get_data(args.D, args.batch_size)
args.num_class = num_class
args.bottomModel_output_dim = 60

model, attacker_list, attacker_target_vec_dict, active_party, passive_party, defense_output, train_shap_dict = train.poisoning_train(dl_train_set, poisoned_trainloader, trn_x, trn_y, args)
# 不管是 VFLIP 回傳的 defense_model 還是 SHAP 回傳的 defense_models ，這裡一律命名為 defense_model
# 後續再根據 args.DM 進行動作

args.attacker_list = attacker_list
args.active_party = active_party
args.passive_party = passive_party
# if args.DM == "SHAP":
#     classifier, passive_centroids_all = Defense.build_classifier_and_centroids(defense_output, args, classifier_method='nearest_centroid', kernel_bandwidth=1000, filter_num=100)
# classifier, passive_centroids_all = Defense.build_classifier_and_centroids(vec_dict_last_epoch, y_in_last_epoch, args, classifier_method='knn', kernel_bandwidth=1000, filter_num=100)

clean_test_shap_dict= eval.eval_clean_train(model, testloader, args, defense_output)
if (args.AM != "None"): 
    poisoned_test_shap_dict = eval.eval_poisoned_train(model, dl_val_set, poisoned_testloader, attacker_target_vec_dict, attacker_list, args, defense_output)
    # if args.DM =="SHAP":
    #     eval.eval_centroid_poisoned_train(model, dl_val_set, poisoned_testloader, attacker_target_vec_dict, attacker_list, args, defense_output, classifier, passive_centroids_all) 
        
# 執行：畫圖與統計檢定
# print("🔍 分析攻擊者")
# Draw.plot_bar_comparison(train_shap_dict, clean_test_shap_dict, poisoned_test_shap_dict, args.attacker_list, args, title_suffix='(Attackers)')
# for attacker_id in args.attacker_list:
#     Draw.compute_p_values(train_shap_dict, clean_test_shap_dict, poisoned_test_shap_dict, attacker_id, args)

# print("🧪 分析 Active Party")
# Draw.plot_bar_comparison(train_shap_dict, clean_test_shap_dict, poisoned_test_shap_dict, [args.active_party], args, title_suffix='(Active Party)')
# Draw.compute_p_values(train_shap_dict, clean_test_shap_dict, poisoned_test_shap_dict, args.active_party, args)     

for pid in args.attacker_list + [args.active_party]:
    print(f"\n📌 畫圖中：Participant {pid}")
    Draw.plot_hist_curve_proportion(train_shap_dict, clean_test_shap_dict, poisoned_test_shap_dict, pid, args)
    Draw.plot_shap_boxplot(train_shap_dict, clean_test_shap_dict, poisoned_test_shap_dict, pid, args)
