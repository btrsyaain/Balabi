import pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import random
from statistics import mean

pd.options.mode.chained_assignment = None

used_dataset = "67_33_nandel.csv"

def make_dataset(used_dataset,include_feature_list):
    df = pd.read_csv(used_dataset)
    print("此資料集名稱為：",used_dataset,"，共有",len(df.index),"筆資料")
    df.drop("Unnamed: 0",axis=1,inplace=True)
    # df.drop("dcompensation",axis=1,inplace=True)
    # df.drop("gpt401yr",axis=1,inplace=True)
    # df.drop("ETV1TDFTAF2",axis=1,inplace=True)
    df["HCCdurationwks"] = df["HCCduration (wks)"]
    df["age"]=df["age1yr"]-1
    df.drop("age1yr",axis=1,inplace=True)
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

#直接訓練結果
include_feature_list = ['sex1M0F', 'ETV1TDFTAF2', 'HTN', 'logHBVDNA', 'age', 'got', 'gpt', 'albuminR', 'plateletR']
X_train,y_train,X_test,y_test = make_dataset(used_dataset,include_feature_list)
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
model.save_model("0720_xgboosthyper.json")

y_train.loc[:,'pred'] = model.predict(dtrain_reg)
y_train.loc[:,'HCC'] = y_train['HCC'].map({1:True,0:False})
train_cindex = concordance_index_censored(
    y_train['HCC'],
    y_train['HCCdurationwks'],
    y_train["pred"], 
)
print("未調整超參數訓練結果：")
print("trainset C-index",round(train_cindex[0], 3))
y_train['HCC'] = y_train['HCC'].map({True:1,False:0})

y_test.loc[:,'pred'] = model.predict(dtest_reg)
y_test.loc[:,'HCC'] = y_test['HCC'].map({1:True,0:False})
test_cindex = concordance_index_censored(
    y_test['HCC'],
    y_test['HCCdurationwks'],
    y_test["pred"],
)
print("testset C-index",round(test_cindex[0], 3))
y_test['HCC'] = y_test['HCC'].map({True:1,False:0})
# print("-"*20)
y_train.drop("pred",axis=1,inplace=True)
y_test.drop("pred",axis=1,inplace=True)