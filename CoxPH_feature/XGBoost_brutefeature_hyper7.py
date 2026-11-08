import pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import random
from statistics import mean
import CK_function

writefile = './coxph_selectfeature_cindex.txt'

pd.options.mode.chained_assignment = None

used_dataset = "67_33_nandel_240226.csv"

coxph_select_feature_list = ["DM","age1yr","got1yr","platelet1yrR","afp1yrR","albuminR.1"]

cv = 7
cv_criteria = 0.67

f=open(writefile, 'a')
f.write("特徵組合為："+str(coxph_select_feature_list)+"\n"+"==========================================\n")
f.close()

for i in range(5):
    print("cv=",cv,"，第",i+1,"次")
    f=open(writefile, 'a')
    f.write("cv="+str(cv)+"第"+str(i+1)+"次\n")
    f.close()
    CK_function.ini_train_XGBoost(used_dataset,coxph_select_feature_list,i,writefile)
    total_cv,cv,cv_total_trainset_len,cv_part_trainset_len = CK_function.XGBoost_cv(used_dataset,cv,cv_criteria,coxph_select_feature_list,writefile)
    best_ave_model_p,best_ave_model_c = CK_function.XGBoost_hyper_e_m_mi_g_t(total_cv,cv,cv_total_trainset_len,cv_part_trainset_len,writefile)
    best_params = CK_function.XGBoost_hyper_s_c_l_a(total_cv,cv,cv_total_trainset_len,cv_part_trainset_len,best_ave_model_p,best_ave_model_c,writefile)
    CK_function.best_hyper_XGBoost(used_dataset,best_params,coxph_select_feature_list,i,writefile)
    f=open(writefile, 'a')
    f.write("---------------------------------------\n")
    f.close()
print("-"*10)
f=open(writefile, 'a')
f.write("==========================================\n")
f.close()