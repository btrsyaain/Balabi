import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from sksurv.metrics import concordance_index_censored
from xgboost import plot_importance
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

def make_dataset(dataset,include_feature_list):
    df = pd.read_csv(dataset)
    df.drop("Unnamed: 0",axis=1,inplace=True)
    #df.drop("ETV1TDFTAF2",axis=1,inplace=True)
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

def do_xgb(X_train,y_train,X_test,y_test,feature_names):
    dtrain_reg = xgb.DMatrix(X_train, y_train['HCC'])
    dtest_reg = xgb.DMatrix(X_test, y_test['HCC'])

    model = xgb.Booster()
    model.load_model("ALL761_xgboost_hyper.json")
    y_train.loc[:,'pred'] = model.predict(dtrain_reg)
    y_test.loc[:,'pred'] = model.predict(dtest_reg)
    train_cindex = calculate_c_index(y_train)
    test_cindex = calculate_c_index(y_test)

    print("XGBoost 之訓練集 c-index 為：",train_cindex)
    print("XGBoost 之測試集 c-index 為：",test_cindex)

    explainer = shap.TreeExplainer(model,X_train)
    # shap_values = explainer(X_train)
    # shap.initjs()
    # shap.summary_plot(shap_values, X_train,show=False)
    # plt.savefig(file_store_address+file_name+"_train.png")
    # plt.cla()

    shap_values = explainer(X_test)
    shap_values.feature_names = feature_names
    shap.initjs()
    # shap.summary_plot(shap_values, X_test,show=True)
    # plt.savefig("./761shap_feature_test.png")
    shap.summary_plot(shap_values, X_test,show=False)
    plt.savefig("./forpaper_761shap_feature_test.png",dpi=300)
    plt.clf()

dataset="./67_33_nandel_240226.csv"
included_list = ['albuminR', 'plateletR', 'age1yr', 'got1yr', 'platelet1yrR', 'afp1yrR']
feature_names = ['Albumin at baseline', 'Platelet at baseline', 'Age', 'AST at 1 year of NA therapy', 'Platelet at 1 year of NA therapy', 'AFP at 1 year of NA therapy']
X_train,y_train,X_test,y_test = make_dataset(dataset,included_list)
do_xgb(X_train,y_train,X_test,y_test,feature_names)
