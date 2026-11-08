import pandas as pd
import time
start = time.time()
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

df = pd.read_csv("all_feature_shapley.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
feature_dataframe = df.drop("Cindex",axis=1).copy()

shapley_list_key = list(feature_dataframe.columns)
feature_number = len(shapley_list_key)#特徵總數
shapley_list_value = [0 for _ in range(feature_number)]
shapley_dict = {}
for i in range(len(shapley_list_value)):
    shapley_dict[shapley_list_key[i]] = shapley_list_value[i]

for feature in df.columns:
    if feature == "Cindex":
        pass
    else:
        print("現在正在計算",feature,"之 Shapley Value")
        featurein_list = (df[feature]==1)
        featureout_list = (df[feature]==0)
        featurein = df[featurein_list].reset_index(drop=True).copy()
        featureout = df[featureout_list].reset_index(drop=True).copy()
        #把要計算的那一特徵以及 Cindex 排除，以進行剩下的特徵比較
        # featurein_calculate = featurein.drop([feature,"Cindex"],axis=1).copy()
        # featureout_calculate = featureout.drop([feature,"Cindex"],axis=1).copy()
        featurein_calculate = featurein.drop(feature,axis=1).copy()
        featurein_calculate["withCindex"] = featurein_calculate["Cindex"]
        featurein_calculate.drop("Cindex",axis=1,inplace=True)
        # print(featurein_calculate)
        featureout_calculate = featureout.drop(feature,axis=1).copy()
        featureout_calculate["outCindex"] = featureout_calculate["Cindex"]
        featureout_calculate.drop("Cindex",axis=1,inplace=True)
        # print(featureout_calculate)
        other_feature_list = list(featurein_calculate.columns)
        other_feature_list.remove("withCindex")
        sheet = pd.merge(featurein_calculate,featureout_calculate,on=other_feature_list)
        
        for i in sheet.index:
            # print(len(featurein_calculate.index))
            print(f"\r現在進度為：{round(i/len(featurein_calculate.index),2)*100}%"," "*20,end="")
            s=(sheet.loc[i] == 1).sum()+1 #納入的 feature 數
            # print("s為：",s)
            n = feature_number #特徵總數
            # print("n為:",n)
            weight = (factorial(s-1)*factorial(n-s)/factorial(n)) #權重公式
            contrib = sheet.loc[i,"withCindex"]-sheet.loc[i,"outCindex"]
            shapley_dict[feature] += weight * contrib
            # print("這次的貢獻值為：",weight * contrib)
        print(feature,"之 Shapley Value 為：",shapley_dict[feature])
print("-----全部計算完成-----")
end=time.time()
print(start-end)
f=open('all_feature_shap.txt', 'a')
f.write('ALL feature shapley score\n')
f.close()

total = 0
for feature in shapley_dict.keys():
    total+=shapley_dict[feature]
for feature in shapley_dict.keys():
    f=open('all_feature_shap.txt', 'a')
    f.write(str(feature)+" 之 Shapley Value 為："+str(shapley_dict[feature])+'，貢獻度佔總體'+str(round(shapley_dict[feature]/total,2)*100)+'%\n')
    f.close()
    print(feature,"之 Shapley Value 為：",shapley_dict[feature],'，貢獻度佔總體',str(round(shapley_dict[feature]/total,2)*100),'%')

            



