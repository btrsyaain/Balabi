import pandas as pd

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

df = pd.read_csv("1yr_shapley.csv")
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
        featurein_calculate = featurein.drop([feature,"Cindex"],axis=1).copy()
        featureout_calculate = featureout.drop([feature,"Cindex"],axis=1).copy()
        # calculate_featurein_list = featurein_list-[feature,"Cindex"]
        # calculate_featureout_list = featureout_list-[feature,"Cindex"]
        # featurein_calculate = featurein[calculate_featurein_list]
        # featureout_calculate = featureout[calculate_featureout_list]
        for i in featurein_calculate.index:
            # print(len(featurein_calculate.index))
            print(f"\r現在進度為：{round(i/len(featurein_calculate.index),2)*100}%"," "*20,end="")
            for j in featureout_calculate.index:
                if (featurein_calculate.iloc[i]==featureout_calculate.iloc[j]).all():
                    #這邊把加權後的分數加進去總分裡
                    s=(featurein_calculate.loc[i] == 1).sum()+1 #納入的 feature 數
                    # print("s為：",s)
                    n = feature_number #特徵總數
                    # print("n為:",n)
                    weight = (factorial(s-1)*factorial(n-s)/factorial(n)) #權重公式
                    contrib = featurein.loc[i,"Cindex"]-featureout.loc[j,"Cindex"]
                    shapley_dict[feature] += weight * contrib
                    # print("這次的貢獻值為：",weight * contrib)
                else:
                    pass
        print(feature,"之 Shapley Value 為：",shapley_dict[feature])
print("-----全部計算完成-----")
f=open('1yr_feature_shap.txt', 'a')
f.write('1YR feature shapley score\n')
f.close()

total = 0
for feature in shapley_dict.keys():
    total+=shapley_dict[feature]
for feature in shapley_dict.keys():
    f=open('1yr_feature_shap.txt', 'a')
    f.write(str(feature)+" 之 Shapley Value 為："+str(shapley_dict[feature])+'，貢獻度佔總體'+str(round(shapley_dict[feature]/total,2)*100)+'%\n')
    f.close()
    print(feature,"之 Shapley Value 為：",shapley_dict[feature],'，貢獻度佔總體',str(round(shapley_dict[feature]/total,2)*100),'%')

            



