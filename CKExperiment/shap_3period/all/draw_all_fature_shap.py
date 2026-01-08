import matplotlib.pyplot as plt

# 資訊
# features = ['age1yr','sex1M0F', 'ETV1TDFTAF2','DM','HTN','NA-naive','HBeAg','logHBVDNA',
#             'got','gpt','t-bil','INRR','albuminR','plateletR','afpR',
#             'got1yr','gpt1yr','T-bil1yrR','albuminR.1','platelet1yrR','afp1yrR','HBVDNA1yr']
features = ['Age',
            'Sex', 
            'Antiviral treatment type',
            'DM',
            'HTN',
            'NA',
            'HBeAg',
            'HBV DNA titer at baseline',
            'AST at baseline',
            'ALT at baseline',
            'Bilirubin at baseline',
            'INR at baseline',
            'Albumin at baseline',
            'Platelet at baseline',
            'AFP at baseline',
            'AST at 1 year of NA therapy',
            'ALT at 1 year of NA therapy',
            'Bilirubin at 1 year of NA therapy',
            'Albumin at 1 year of NA therapy',
            'Platelet at 1 year of NA therapy',
            'AFP at 1 year of NA therapy',
            'HBV DNA titer at 1 year of NA therapy']
shapley_values = [0.058444540048927886,0.0230838595185367,0.02798939007329306,0.023637580204454802,0.018181005436777992,0.02428365909546479,0.023459782838425335,0.026465511067022905,
                  0.037437737898562314,0.022797292202978627,0.02514381035201584,0.01887483534697506,0.04915469743545606,0.041429403538604276,0.03092755469503501,
                  0.05908614879279521,0.0170001127226885,0.027320656961718898,0.03758562319179292,0.04691826886144095,0.07365158898553134,0.022126940732131024]
# total = 0
# contribution_percentages = []
# for i in range(len(shapley_values)):
#     total+=shapley_values[i]
# for j in range(len(shapley_values)):
#     contribution_percentages.append(round(shapley_values[j]/total,3)*100)

# contribution_percentages = [10.0,8.0,8.0,6.7,6.4,5.6,5.1,5.1,4.2,3.8,3.7,3.6,3.4,3.3,3.2,3.2,3.1,3.1,3.0,2.6,2.5,2.3]

# 將資料依照 Shapley 值大小排序
sorted_features = [feat for _, feat in sorted(zip(shapley_values, features), reverse=False)]
sorted_shapley_values = sorted(shapley_values, reverse=False)
# sorted_contribution_percentages = sorted(contribution_percentages, reverse=False)

# 繪製長條圖
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(sorted_features, sorted_shapley_values, color='navy')

# 添加百分比文本标签
# for bar, percentage in zip(bars, sorted_contribution_percentages):
#     ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{percentage}%',
#             ha='left', va='center', color='black')

ax.set_xlabel('Shapley Value')
ax.set_ylabel('Features')
ax.set_title('Shapley values of features in Combination 3')
plt.savefig('forpaper_allfeature_shap_scoreoutper.png',dpi=300, bbox_inches='tight')