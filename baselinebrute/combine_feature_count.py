import pandas as pd
import itertools
import time
pd.options.mode.chained_assignment = None

df = pd.read_csv("baseline_shapley.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
# df.rename(columns={"got1yr": "Got1yr", "gpt1yr": "Gpt1yr","albuminR.1":"AlbuminR.1","platelet1yrR":"Platelet1yrR","afp1yrR":"Afp1yrR"}, inplace=True)
feature_dataframe = df.drop("Cindex",axis=1).copy()
variables = feature_dataframe.columns

for n in range(1,len(variables)+1):
    combinations=[]
    combinations.extend(list(itertools.combinations(variables,n)))
    print("現在進行到",n,"個特徵")
    for column in combinations:
        columns_to_search = list(column)
        select = df[columns_to_search].eq(1).all(axis=1)
        filter_df = df[select]
        standard_list = [66,67,68,69,70,71,72]
        for standard in standard_list:
            writefilename = "combinefeaturecountover"+str(standard)+".txt"
            fit_number = (filter_df["Cindex"]>(standard/100)).sum()
            with open(writefilename,'a') as w:
                w.write("特徵為："+str(columns_to_search)+"能夠通過"+str(fit_number)+"次\n")