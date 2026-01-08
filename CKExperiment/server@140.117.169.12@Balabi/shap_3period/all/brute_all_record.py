import numpy as np
import pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import itertools
import time

pd.options.mode.chained_assignment = None

def make_dataset(used_dataset,include_feature_list):
    df = pd.read_csv(used_dataset)
    print("此資料集名稱為：",used_dataset,"，共有",len(df.index),"筆資料")
    df.drop("Unnamed: 0",axis=1,inplace=True)
    # df.drop("ETV1TDFTAF2",axis=1,inplace=True)
    df["HCCdurationwks"] = df["HCCduration (wks)"]
    df.drop("HCCduration (wks)",axis=1,inplace=True)
    exact_feature_list = include_feature_list+['Derivation_or_Validation_group','codenumber','HCC','HCCdurationwks']
    df = df[exact_feature_list]

    trainlist = (df.Derivation_or_Validation_group==1)
    testlist = (df.Derivation_or_Validation_group==2)

    train = df[trainlist].reset_index(drop=True)
    train.drop("Derivation_or_Validation_group",axis=1,inplace=True)
    train.drop("codenumber",axis=1,inplace=True)
    test = df[testlist].reset_index(drop=True)
    test.drop("Derivation_or_Validation_group",axis=1,inplace=True)
    test.drop("codenumber",axis=1,inplace=True)

    X_train = train.drop(["HCC","HCCdurationwks"],axis=1)
    y_train = train[["HCC","HCCdurationwks"]]
    X_test = test.drop(["HCC","HCCdurationwks"],axis=1)
    y_test = test[["HCC","HCCdurationwks"]]
    print("進行訓練特徵為：",X_train.columns)
    return X_train,y_train,X_test,y_test

def calculate_c_index(y_test):
    y_test.loc[:,'HCC'] = y_test['HCC'].map({1:True,0:False})
    cindex = concordance_index_censored(
        y_test['HCC'],
        y_test['HCCdurationwks'],
        y_test["pred"],
    )
    y_test['HCC'] = y_test['HCC'].map({True:1,False:0})
    return round(cindex[0], 3)
    
def do_xgb(X_train,y_train,X_test,y_test):
    history_best=0
    history_best_feature=[]
    variables = X_train.columns
    
    for n in range(len(variables)+1):
        if n==(len(variables)+1):
            pass
        else:
            best_in_iter=0
            best_feature_in_iter = []

            truly_column_number=n+1
            combinations=[]
            combinations.extend(list(itertools.combinations(variables,truly_column_number)))

            file_name = str(truly_column_number)+"_feature_xgboost.txt"
            # platelet_file_name = str(truly_column_number)+"_platelet_xgboost.txt"
            count=0

            for column in combinations:
                iter_number = len(combinations)
                count+=1
                # platelet_warn = False
                # if ("plateletR" in column) or ("platelet1yrR" in column):
                #     platelet_warn = True 
                
                X_train_feature_choose = X_train[list(column)]
                X_test_feature_choose = X_test[list(column)]

                dtrain_reg = xgb.DMatrix(X_train_feature_choose, y_train['HCC'])
                dtest_reg = xgb.DMatrix(X_test_feature_choose, y_test['HCC'])
                best_model_p = {
                            'booster': 'gbtree',
                            'objective': 'binary:logistic',
                            'eta': 0.05,
                            'max_depth': 6,
                            'min_child_weight': 0,
                            'gamma':0,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'lambda': 1,
                            'alpha':0,
                            'tree_method': 'exact',
                            'seed': 1000,
                            'scale_pos_weight' : 1,
                            }
                num_rounds = 30
                plst = list(best_model_p.items())
                model = xgb.train(plst, dtrain_reg, num_rounds)
                y_test.loc[:,'pred'] = model.predict(dtest_reg)
                cindex = calculate_c_index(y_test)

                for feature in variables:
                    if feature in column:
                        shapley_dict[feature].append(1)
                    else:
                        shapley_dict[feature].append(0)
                shapley_dict['Cindex'].append(cindex)
                # print(shapley_dict)
                # time.sleep(20)

                if cindex>best_in_iter:
                    best_in_iter = cindex
                    best_feature_in_iter = list(column)

                f=open(file_store_address+file_name, 'a')
                f.write('XGBOOST 在參數為：'+str(list(column))+'其 c-index 為：'+ str(cindex) +'\n')
                f.close()

                # if platelet_warn==True:
                #     pl=open(file_store_address+platelet_file_name, 'a')
                #     pl.write('XGBOOST 在參數為：'+str(list(column))+'其 c-index 為：'+ str(cindex) +'\n')
                #     pl.close()
                print(f'\r{truly_column_number}個 feature 的目前進度為：{round(count/iter_number,2)*100}%{" "*20}',end="")
                
            print()
            
            if best_in_iter > history_best:
                history_best = best_in_iter
                history_best_feature = best_feature_in_iter
                        
            f=open(file_store_address+file_name, 'a')
            f.write(str("-"*30)+'\nXGBOOST 在'+str(truly_column_number)+'個參數時最佳參數為：'+str(list(best_feature_in_iter))+'其 c-index 為：'+ str(best_in_iter) +'\n')
            f.close()

            # p=open(file_store_address+'permu_result.txt', 'a')
            # p.write('XGBOOST 在'+str(truly_column_number)+'個參數時最佳參數為：'+str(list(best_feature_in_iter))+'其 c-index 為：'+ str(best_in_iter) +'\n')
            # p.close()

        p=open(file_store_address+'permu_result.txt', 'a')
        p.write(str('-'*30)+'\nXGBOOST 在'+str(len(history_best_feature))+'個參數時有最佳參數，為：'+str(list(history_best_feature))+'其 c-index 為：'+ str(history_best) +'\n')
        p.close()

used_dataset = "67_33_nandel_240226.csv"
test_size = 0.33
# includelist_baseline = ['sex1M0F','ETV1TDFTAF2','DM','HTN','NA-naive','HBeAg','logHBVDNA','age1yr','got','gpt','t-bil','INRR','albuminR','plateletR','afpR']
# includelist_1yr = ['sex1M0F','ETV1TDFTAF2','DM','HTN','NA-naive','HBVDNA1yr','age1yr','got1yr','gpt1yr','T-bil1yrR','albuminR.1','platelet1yrR','afp1yrR']
includelist_all = ['sex1M0F','ETV1TDFTAF2','DM','HTN','NA-naive','HBeAg','logHBVDNA','HBVDNA1yr','got','gpt','t-bil','INRR','albuminR','plateletR','afpR','age1yr','got1yr','gpt1yr','T-bil1yrR','albuminR.1','platelet1yrR','afp1yrR']
print("1YR")
print("*"*10+" 測試集大小為：",round(test_size*100),"% "+"*"*10)
# csv_name = str(100-round(test_size*100))+"_"+str(round(test_size*100))+".csv"
# file_store_address = "./"+str(100-round(test_size*100))+"_"+str(round(test_size*100))+"/"
file_store_address = "./"
mid_star = "-"*40
f=open(file_store_address+'permu_result.txt', 'a')
f.write('訓練'+str(100-round(test_size*100))+':測試'+str(round(test_size*100))+'\n'+str(mid_star)+'XGBOOST'+str(mid_star)+'\n')
f.close()
X_train,y_train,X_test,y_test = make_dataset(used_dataset,includelist_all)

shapley_list_key = includelist_all+['Cindex']
feature_number = len(shapley_list_key)
shapley_list_value = [[0] for _ in range(feature_number)]
shapley_dict = {}
for i in range(len(shapley_list_value)):
    shapley_dict[shapley_list_key[i]] = shapley_list_value[i]

do_xgb(X_train,y_train,X_test,y_test)
shapley_dataframe = pd.DataFrame.from_dict(shapley_dict)
shapley_dataframe.to_csv("all_feature_shapley.csv")

