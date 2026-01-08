import pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import random
from statistics import mean

writefile = './3periodcindex.txt'

pd.options.mode.chained_assignment = None

used_dataset = "67_33_nandel_240226.csv"

def make_dataset(used_dataset,include_feature_list):
    df = pd.read_csv(used_dataset)
    print("此資料集名稱為：",used_dataset,"，共有",len(df.index),"筆資料")
    df.drop("Unnamed: 0",axis=1,inplace=True)
    # df.drop("dcompensation",axis=1,inplace=True)
    # df.drop("gpt401yr",axis=1,inplace=True)
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
    return X_train,y_train,X_test,y_test

def ini_train_XGBoost(used_dataset,include_feature_list,i):
    #直接訓練結果
    X_train,y_train,X_test,y_test = make_dataset(used_dataset,include_feature_list)
    print("訓練集內特徵為：",X_train.columns)
    dtrain_reg = xgb.DMatrix(X_train, y_train['HCC'])
    dtest_reg = xgb.DMatrix(X_test, y_test['HCC'])
    params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 0,
            'gamma':0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1,
            'alpha': 0,
            'tree_method': 'exact',
            'scale_pos_weight' : 1,
            'seed': 1000,
            }
    plst = list(params.items())
    model = xgb.train(plst, dtrain_reg,num_boost_round=30)

    y_train.loc[:,'pred'] = model.predict(dtrain_reg)
    y_train.loc[:,'HCC'] = y_train['HCC'].map({1:True,0:False})
    train_cindex = concordance_index_censored(
        y_train['HCC'],
        y_train['HCCdurationwks'],
        y_train["pred"], 
    )
    print("未調整超參數訓練結果：")
    print("trainset C-index",round(train_cindex[0], 3))

    if i==0:
        f=open(writefile, 'a')
        f.write("未調整超參數訓練結果：\n"+"trainset C-index"+str(round(train_cindex[0], 3))+"\n")
        f.close()
    y_train['HCC'] = y_train['HCC'].map({True:1,False:0})

    y_test.loc[:,'pred'] = model.predict(dtest_reg)
    y_test.loc[:,'HCC'] = y_test['HCC'].map({1:True,0:False})
    test_cindex = concordance_index_censored(
        y_test['HCC'],
        y_test['HCCdurationwks'],
        y_test["pred"],
    )
    print("testset C-index",round(test_cindex[0], 3))
    if i==0:
        f=open(writefile, 'a')
        f.write("testset C-index"+str(round(test_cindex[0], 3))+"\n")
        f.close()

    y_test['HCC'] = y_test['HCC'].map({True:1,False:0})
    # print("-"*20)
    y_train.drop("pred",axis=1,inplace=True)
    y_test.drop("pred",axis=1,inplace=True)

def check_train_len(train_0,train_1):
    train_0_len = len(train_0["HCC"])
    train_1_len = len(train_1["HCC"])
    # print("訓練0總長度為：",train_0_len,"訓練1總長度為：",train_1_len)
    return train_0_len,train_1_len

def XGBoost_cv(used_dataset,cv,cv_criteria,include_feature_list):
    df = pd.read_csv(used_dataset)
    df.drop("Unnamed: 0",axis=1,inplace=True)
    # df.drop("dcompensation",axis=1,inplace=True)
    # df.drop("gpt401yr",axis=1,inplace=True)
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

    cv_criteria_adjuster = 0
    while(True):
        #避免達不到 cv 標準，無窮迴圈
        if cv_criteria_adjuster >= 10:
            cv_criteria-=0.01
            f=open(writefile, 'a')
            f.write("無法達到要求，cv 標準降低為："+str(cv_criteria)+"\n")
            f.close()
            cv_criteria_adjuster = 0
            print("無法達到要求，cv 標準降低為：",cv_criteria)
        
        mask0 = (train.HCC==0)
        mask1 = (train.HCC==1)
        train_0 = train[mask0].reset_index(drop=True)
        train_1 = train[mask1].reset_index(drop=True)
        train_0_len,train_1_len = check_train_len(train_0,train_1)
        
        cv_total_trainset_len = train_0_len+train_1_len
        cv_part_trainset_len = cv_total_trainset_len//cv

        for i in range(cv):
            if (i==(cv-1)):
                train_0_cv = train_0
                # print(len(train_0_7["HCC"]))
                train_1_cv = train_1
                # print(len(train_1_7["HCC"]))
                cv_set = pd.concat([train_0_cv,train_1_cv])
                cv_set = cv_set.sample(frac=1).reset_index(drop=True)
            else : 
                # 選擇哪些要在本次被切出來
                train_0_ranlist = random.sample(list(train_0.index),(train_0_len//cv))
                # print(len(train_0_ranlist))
                train_1_ranlist = random.sample(list(train_1.index),(train_1_len//cv))
                # print(len(train_1_ranlist))
                # 被切出來儲存到 train/test_0/1_cv
                train_0_cv = train_0.iloc[train_0_ranlist].reset_index(drop=True)
                train_1_cv = train_1.iloc[train_1_ranlist].reset_index(drop=True)
                # 將取出的從原本0/1資料庫移除
                train_0 = train_0.drop(train_0_ranlist).reset_index(drop=True)
                train_1 = train_1.drop(train_1_ranlist).reset_index(drop=True)
                # 將取出之 0/1 資料合併，並隨機排序
                cv_set = pd.concat([train_0_cv,train_1_cv])
                cv_set = cv_set.sample(frac=1).reset_index(drop=True)
                if (i==0):
                    total_cv = cv_set
            if (i>=1):
                total_cv = pd.concat([total_cv,cv_set]).reset_index(drop=True)
        if test_Is_cv_good(total_cv,cv,cv_part_trainset_len,cv_total_trainset_len,cv_criteria):
            break
        cv_criteria_adjuster+=1
    return total_cv,cv,cv_total_trainset_len,cv_part_trainset_len

def test_Is_cv_good(total_cv,cv,cv_part_trainset_len,cv_total_trainset_len,cv_criteria):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 0,
        'gamma':0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1,
        'alpha': 0,
        'tree_method': 'exact',
        'seed': 1000,
        'scale_pos_weight' : 1,
        }
    plst = list(params.items())
    cv_cindex = []
    for f in range(cv):
        if (f<cv-1):
            #先做 cv 的 test 集
            X_cv_test = total_cv.iloc[f*cv_part_trainset_len:(f+1)*cv_part_trainset_len]
            X_cv_test_train = X_cv_test.drop(["HCC","HCCdurationwks"],axis = 1)
            y_cv_test_train = X_cv_test[["HCC","HCCdurationwks"]]
            #再做 cv 的 train 集
            X_cv_train = total_cv.drop(range(f*cv_part_trainset_len,(f+1)*cv_part_trainset_len))
            X_cv_train_train = X_cv_train.drop(["HCC","HCCdurationwks"],axis = 1)
            y_cv_train_train = X_cv_train[["HCC","HCCdurationwks"]]
        else:
            #先做 cv 的 test 集
            X_cv_test = total_cv.iloc[f*cv_part_trainset_len:(cv_total_trainset_len)]
            X_cv_test_train = X_cv_test.drop(["HCC","HCCdurationwks"],axis = 1)
            y_cv_test_train = X_cv_test[["HCC","HCCdurationwks"]]
            #再做 cv 的 train 集
            X_cv_train = total_cv.drop(range(f*cv_part_trainset_len,(cv_total_trainset_len)))
            X_cv_train_train = X_cv_train.drop(["HCC","HCCdurationwks"],axis = 1)
            y_cv_train_train = X_cv_train[["HCC","HCCdurationwks"]]
        #將其分別轉成XGB的資料結構
        dtrain_cv_reg = xgb.DMatrix(X_cv_train_train, y_cv_train_train['HCC'])
        dtest_cv_reg = xgb.DMatrix(X_cv_test_train, y_cv_test_train['HCC'])
        #開始進行 cv model 訓練，並對 cv testset 進行預測
        model = xgb.train(plst, dtrain_cv_reg,num_boost_round=50)
        y_cv_test_train.loc[:,'pred'] = model.predict(dtest_cv_reg)
        #進行 c-index 檢測
        y_cv_test_train.loc[:,'HCC'] = y_cv_test_train['HCC'].map({1:True,0:False})
        cindex = concordance_index_censored(
            y_cv_test_train['HCC'],
            y_cv_test_train['HCCdurationwks'],
            y_cv_test_train["pred"], 
        )
        y_cv_test_train['HCC'] =y_cv_test_train['HCC'].map({True:1,False:0})
        cv_cindex.append(round(cindex[0], 3))
    if(mean(cv_cindex)>=cv_criteria):
        print(cv_cindex,mean(cv_cindex))
        print("cv完成")

        f=open(writefile, 'a')
        f.write("cv完成，c-index 為："+str(cv_cindex)+"平均為："+str(mean(cv_cindex))+"\n")
        f.close()

        return True
    else:
        print(cv_cindex,mean(cv_cindex))
        return False

def XGBoost_hyper_e_m_mi_g_t(total_cv,cv,cv_total_trainset_len,cv_part_trainset_len):

    etat = [0.001,0.01,0.03,0.05,0.1,0.3]

    max_deptht = [4,5,6,7,8]
    min_child_weightt = [0,1]

    gammat = [0,0.5,1,2,3]
    tree_num = [30,50,100]
    scale_pos_weightt = [1,2,3,4]

    number_of_this_stage = len(etat)*len(max_deptht)*len(min_child_weightt)*len(gammat)*len(tree_num)*len(scale_pos_weightt)

    # subsamplet = [0.8,0.9,1]
    # colsample_bytreet = [0.8,0.9,1]

    # lambdat = [0,0.5,1,3]
    # alphat = [0,0.5,1,2]
    
    count = 0
    best_ave_model_p = {}
    best_ave_model_c = 0

    for e in etat:
        for m in max_deptht:
            for g in gammat:
                for mi in min_child_weightt:
                    for sp in scale_pos_weightt:
                        for t in tree_num:
                            params = {
                            'booster': 'gbtree',
                            'objective': 'binary:logistic',
                            'eta': e,
                            'max_depth': m,
                            'min_child_weight': mi,
                            'gamma':g,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'lambda': 1,
                            'alpha': 0,
                            'tree_method': 'exact',
                            'seed': 1000,
                            'scale_pos_weight' : sp,
                            }
                            plst = list(params.items())
                            cv_cindex = []
                            for f in range(cv):
                                if (f<cv-1):
                                    #先做 cv 的 test 集
                                    X_cv_test = total_cv.iloc[f*cv_part_trainset_len:(f+1)*cv_part_trainset_len]
                                    X_cv_test_train = X_cv_test.drop(["HCC","HCCdurationwks"],axis = 1)
                                    y_cv_test_train = X_cv_test[["HCC","HCCdurationwks"]]
                                    #再做 cv 的 train 集
                                    X_cv_train = total_cv.drop(range(f*cv_part_trainset_len,(f+1)*cv_part_trainset_len))
                                    X_cv_train_train = X_cv_train.drop(["HCC","HCCdurationwks"],axis = 1)
                                    y_cv_train_train = X_cv_train[["HCC","HCCdurationwks"]]
                                else:
                                    #先做 cv 的 test 集
                                    X_cv_test = total_cv.iloc[f*cv_part_trainset_len:(cv_total_trainset_len)]
                                    X_cv_test_train = X_cv_test.drop(["HCC","HCCdurationwks"],axis = 1)
                                    y_cv_test_train = X_cv_test[["HCC","HCCdurationwks"]]
                                    #再做 cv 的 train 集
                                    X_cv_train = total_cv.drop(range(f*cv_part_trainset_len,(cv_total_trainset_len)))
                                    X_cv_train_train = X_cv_train.drop(["HCC","HCCdurationwks"],axis = 1)
                                    y_cv_train_train = X_cv_train[["HCC","HCCdurationwks"]]
                                #將其分別轉成XGB的資料結構
                                dtrain_cv_reg = xgb.DMatrix(X_cv_train_train, y_cv_train_train['HCC'])
                                dtest_cv_reg = xgb.DMatrix(X_cv_test_train, y_cv_test_train['HCC'])
                                #開始進行 cv model 訓練，並對 cv testset 進行預測
                                model = xgb.train(plst, dtrain_cv_reg,num_boost_round=t)
                                y_cv_test_train.loc[:,'pred'] = model.predict(dtest_cv_reg)
                                #進行 c-index 檢測
                                y_cv_test_train.loc[:,'HCC'] = y_cv_test_train['HCC'].map({1:True,0:False})
                                cindex = concordance_index_censored(
                                    y_cv_test_train['HCC'],
                                    y_cv_test_train['HCCdurationwks'],
                                    y_cv_test_train["pred"], 
                                )
                                y_cv_test_train['HCC'] =y_cv_test_train['HCC'].map({True:1,False:0})
                                cv_cindex.append(round(cindex[0], 3))

                            count+=1
                            print(f"\r第一階段超參數搜尋，現在進度:{round(count/number_of_this_stage,2)*100}%"," "*20,end="")

                            if(mean(cv_cindex)>best_ave_model_c):
                                best_ave_model_c = round(mean(cv_cindex), 3)
                                best_ave_model_p["e"]=e
                                best_ave_model_p["m"]=m
                                best_ave_model_p["mi"]=mi
                                best_ave_model_p["g"]=g
                                best_ave_model_p["t"]=t
                                best_ave_model_p["s"]=0.8
                                best_ave_model_p["c"]=0.8
                                best_ave_model_p["l"]=0
                                best_ave_model_p["a"]=0
                                best_ave_model_p["sp"]=sp

    print("\n最佳參數：",best_ave_model_p,"\n最佳平均 c-index 為：",best_ave_model_c)

    f=open(writefile, 'a')
    f.write("第一階段超參數搜尋完成，最佳參數為："+str(best_ave_model_p)+"\n最佳平均 c-index 為："+str(best_ave_model_c)+"\n")
    f.close()

    return best_ave_model_p,best_ave_model_c

def XGBoost_hyper_s_c_l_a(total_cv,cv,cv_total_trainset_len,cv_part_trainset_len,best_ave_model_p,best_ave_model_c):

    # etat = [0.001,0.01,0.03,0.05,0.1,0.3]

    # max_deptht = [4,5,6,7,8]
    # min_child_weightt = [0,1]

    # gammat = [0,0.5,1,2,3]
    # tree_num = [30,50,100]

    subsamplet = [0.8,0.9,1]
    colsample_bytreet = [0.8,0.9,1]

    lambdat = [0,0.5,1,2]
    alphat = [0,0.5,1,2]
    
    number_of_this_stage = len(subsamplet)*len(colsample_bytreet)*len(lambdat)*len(alphat)

    count = 0
    best_ave_model_p_new = best_ave_model_p
    best_ave_model_c_new = best_ave_model_c

    for s in subsamplet:
        for c in colsample_bytreet:
            for l in lambdat:
                for a in alphat:
                        params = {
                        'booster': 'gbtree',
                        'objective': 'binary:logistic',
                        'eta': best_ave_model_p["e"],
                        'max_depth': best_ave_model_p["m"],
                        'min_child_weight': best_ave_model_p["mi"],
                        'gamma':best_ave_model_p["g"],
                        'subsample': s,
                        'colsample_bytree': c,
                        'lambda': l,
                        'alpha': a,
                        'tree_method': 'exact',
                        'seed': 1000,
                        'scale_pos_weight' :best_ave_model_p["sp"] ,
                        }
                        plst = list(params.items())
                        cv_cindex = []
                        for f in range(cv):
                            if (f<cv-1):
                                #先做 cv 的 test 集
                                X_cv_test = total_cv.iloc[f*cv_part_trainset_len:(f+1)*cv_part_trainset_len]
                                X_cv_test_train = X_cv_test.drop(["HCC","HCCdurationwks"],axis = 1)
                                y_cv_test_train = X_cv_test[["HCC","HCCdurationwks"]]
                                #再做 cv 的 train 集
                                X_cv_train = total_cv.drop(range(f*cv_part_trainset_len,(f+1)*cv_part_trainset_len))
                                X_cv_train_train = X_cv_train.drop(["HCC","HCCdurationwks"],axis = 1)
                                y_cv_train_train = X_cv_train[["HCC","HCCdurationwks"]]
                            else:
                                #先做 cv 的 test 集
                                X_cv_test = total_cv.iloc[f*cv_part_trainset_len:(cv_total_trainset_len)]
                                X_cv_test_train = X_cv_test.drop(["HCC","HCCdurationwks"],axis = 1)
                                y_cv_test_train = X_cv_test[["HCC","HCCdurationwks"]]
                                #再做 cv 的 train 集
                                X_cv_train = total_cv.drop(range(f*cv_part_trainset_len,(cv_total_trainset_len)))
                                X_cv_train_train = X_cv_train.drop(["HCC","HCCdurationwks"],axis = 1)
                                y_cv_train_train = X_cv_train[["HCC","HCCdurationwks"]]
                            #將其分別轉成XGB的資料結構
                            dtrain_cv_reg = xgb.DMatrix(X_cv_train_train, y_cv_train_train['HCC'])
                            dtest_cv_reg = xgb.DMatrix(X_cv_test_train, y_cv_test_train['HCC'])
                            #開始進行 cv model 訓練，並對 cv testset 進行預測
                            model = xgb.train(plst, dtrain_cv_reg,num_boost_round=best_ave_model_p["t"])
                            y_cv_test_train.loc[:,'pred'] = model.predict(dtest_cv_reg)
                            #進行 c-index 檢測
                            y_cv_test_train.loc[:,'HCC'] = y_cv_test_train['HCC'].map({1:True,0:False})
                            cindex = concordance_index_censored(
                                y_cv_test_train['HCC'],
                                y_cv_test_train['HCCdurationwks'],
                                y_cv_test_train["pred"], 
                            )
                            y_cv_test_train['HCC'] =y_cv_test_train['HCC'].map({True:1,False:0})
                            cv_cindex.append(round(cindex[0], 3))

                        count+=1
                        print(f"\r第二階段超參數搜尋，現在進度:{round(count/number_of_this_stage,2)*100}%"," "*20,end="")

                        if(mean(cv_cindex)>best_ave_model_c_new):
                            best_ave_model_c_new = round(mean(cv_cindex), 3)
                            best_ave_model_p_new["e"]=best_ave_model_p["e"]
                            best_ave_model_p_new["m"]=best_ave_model_p["m"]
                            best_ave_model_p_new["mi"]=best_ave_model_p["mi"]
                            best_ave_model_p_new["g"]=best_ave_model_p["g"]
                            best_ave_model_p_new["t"]=best_ave_model_p["t"]
                            best_ave_model_p_new["s"]=s
                            best_ave_model_p_new["c"]=c
                            best_ave_model_p_new["l"]=l
                            best_ave_model_p_new["a"]=a
                            best_ave_model_p_new["sp"]=best_ave_model_p["sp"]
    if(best_ave_model_c_new>best_ave_model_c):
        print("\n最佳參數：",best_ave_model_p_new,"\n最佳平均 c-index 為：",best_ave_model_c_new)

        f=open(writefile, 'a')
        f.write("第二階段超參數搜尋完成，最佳參數為："+str(best_ave_model_p_new)+"\n最佳平均 c-index 為："+str(best_ave_model_c_new)+"\n")
        f.close()

        return best_ave_model_p_new
    else:
        print("\n第二階段超參數調整後沒有比較好")

        f=open(writefile, 'a')
        f.write("第二階段超參數調整後沒有比較好\n")
        f.close()

        return best_ave_model_p

def best_hyper_XGBoost(used_dataset,best_params,include_feature_list,mark):
    X_train,y_train,X_test,y_test = make_dataset(used_dataset,include_feature_list)
    dtrain_reg = xgb.DMatrix(X_train, y_train['HCC'])
    dtest_reg = xgb.DMatrix(X_test, y_test['HCC'])
    params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': best_params["e"],
            'max_depth': best_params["m"],
            'min_child_weight': best_params["mi"],
            'gamma':best_params["g"],
            'subsample': best_params["s"],
            'colsample_bytree': best_params["c"],
            'lambda': best_params["l"],
            'alpha': best_params["a"],
            'scale_pos_weight' : best_params["sp"],
            'tree_method': 'exact',
            'seed': 1000,
            }
    
    plst = list(params.items())
    model = xgb.train(plst, dtrain_reg,num_boost_round=best_params["t"])
    y_train.loc[:,'pred'] = model.predict(dtrain_reg)
    y_train.loc[:,'HCC'] = y_train['HCC'].map({1:True,0:False})
    cindex = concordance_index_censored(
        y_train['HCC'],
        y_train['HCCdurationwks'],
        y_train["pred"], 
    )
    print("超參數調整後結果：")
    print("trainset C-index",round(cindex[0], 3))

    f=open(writefile, 'a')
    f.write("超參數調整後結果：\n"+"trainset C-index"+str(round(cindex[0], 3))+"\n")
    f.close()

    y_train['HCC'] = y_train['HCC'].map({True:1,False:0})
    
    y_test.loc[:,'pred'] = model.predict(dtest_reg)
    y_test.loc[:,'HCC'] = y_test['HCC'].map({1:True,0:False})
    cindex = concordance_index_censored(
        y_test['HCC'],
        y_test['HCCdurationwks'],
        y_test["pred"],
    )
    print("testset C-index",round(cindex[0], 3))
    y_test['HCC'] = y_test['HCC'].map({True:1,False:0})
    print("-"*20)

    f=open(writefile, 'a')
    f.write("testset C-index"+str(round(cindex[0], 3))+"\n"+"-------------------------------------------\n")
    f.close()

    y_train.drop("pred",axis=1,inplace=True)
    y_test.drop("pred",axis=1,inplace=True)
    filename = str(mark)+str(round(cindex[0], 3)*1000)+"_xgboost_hyper.json"
    model.save_model(filename)

all_feature_list1 = ['albuminR', 'plateletR', 'age1yr', 'got1yr', 'platelet1yrR', 'afp1yrR']
all_feature_list2 = ['got', 'albuminR', 'plateletR', 'age1yr', 'got1yr', 'albuminR.1', 'platelet1yrR', 'afp1yrR']

yr1_feature_list1 = ['ETV1TDFTAF2', 'DM', 'age1yr', 'got1yr', 'T-bil1yrR', 'albuminR.1', 'platelet1yrR', 'afp1yrR']
yr1_feature_list2 = ['ETV1TDFTAF2', 'DM', 'NA-naive', 'age1yr', 'got1yr', 'T-bil1yrR', 'albuminR.1', 'platelet1yrR', 'afp1yrR']

baseline_feature_list1 = ['ETV1TDFTAF2', 'age', 'got', 'gpt', 't-bil', 'albuminR', 'plateletR', 'afpR']
baseline_feature_list2 = ['ETV1TDFTAF2', 'logHBVDNA', 'age', 'got', 'gpt', 't-bil', 'albuminR', 'plateletR', 'afpR']
baseline_feature_list3 = ['ETV1TDFTAF2', 'NA-naive', 'logHBVDNA', 'age', 'got', 'gpt', 't-bil', 'albuminR', 'plateletR', 'afpR']

# allfeature = [all_feature_list1,all_feature_list2,all_feature_list3,all_feature_list4,all_feature_list5]
allfeature = [all_feature_list1,all_feature_list2]
yr1feature = [yr1_feature_list1,yr1_feature_list2]
baselinefeature = [baseline_feature_list1,baseline_feature_list2,baseline_feature_list3]

feature_set = [allfeature,yr1feature,baselinefeature]
year = {1:"ALL",
        2:"1YR",
        3:"BASELINE"
        }
year_count=0
cv = 7
cv_criteria = 0.67
for year_set in feature_set:
    year_count+=1
    list_count=0
    for feature_list in year_set:
        list_count+=1
        include_feature_list = feature_list
        # for cv in cv_list:
        for i in range(5):
            print("特徵年份：",year[year_count],"第",list_count,"組，cv=",cv,"，第",i+1,"次")
            f=open(writefile, 'a')
            f.write("特徵年份："+str(year[year_count])+"第"+str(list_count)+"組，cv="+str(cv)+"，第"+str(i+1)+"次\n"+"特徵組合為："+str(include_feature_list)+"\n")
            f.close()
            ini_train_XGBoost(used_dataset,include_feature_list,i)
            total_cv,cv,cv_total_trainset_len,cv_part_trainset_len = XGBoost_cv(used_dataset,cv,cv_criteria,include_feature_list)
            best_ave_model_p,best_ave_model_c = XGBoost_hyper_e_m_mi_g_t(total_cv,cv,cv_total_trainset_len,cv_part_trainset_len)
            best_params = XGBoost_hyper_s_c_l_a(total_cv,cv,cv_total_trainset_len,cv_part_trainset_len,best_ave_model_p,best_ave_model_c)
            best_hyper_XGBoost(used_dataset,best_params,include_feature_list,(str(year[year_count])+str(list_count)))
        print("-"*10)
        f=open(writefile, 'a')
        f.write("==========================================\n")
        f.close()