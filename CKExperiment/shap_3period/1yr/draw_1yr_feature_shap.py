import matplotlib.pyplot as plt

# 資訊
features = ['Sex', 
            'Antiviral treatment type', 
            'DM', 
            'HTN', 
            'NA', 
            'HBV DNA titer at 1 year of NA therapy',
            'Age', 
            'AST at 1 year of NA therapy', 
            'ALT at 1 year of NA therapy', 
            'Bilirubin at 1 year of NA therapy', 
            'Albumin at 1 year of NA therapy', 
            'Platelet at 1 year of NA therapy', 
            'AFP at 1 year of NA therapy']
shapley_values = [0.040887076812076914, 0.04502340437340444, 0.04149043179043183, 0.030628635253635322, 0.0412714563214565,0.037612040737040685,
                  0.08613303640803677,0.08074645632145659,0.03236943611943616,0.04528758186258193,0.06226561216561205,0.07222022977022989,0.10706460206460208]
# total = 0
# contribution_percentages = []
# for i in range(len(shapley_values)):
#     total+=shapley_values[i]
# for j in range(len(shapley_values)):
#     contribution_percentages.append(round(shapley_values[j]/total,3)*100)

# contribution_percentages = [14.8, 11.9, 11.2, 10.0, 8.6, 6.3, 6.2, 5.7, 5.7, 5.7, 5.2, 4.5, 4.2]

# 將資料依照 Shapley 值大小排序
sorted_features = [feat for _, feat in sorted(zip(shapley_values, features), reverse=False)]
sorted_shapley_values = sorted(shapley_values, reverse=False)
# sorted_contribution_percentages = sorted(contribution_percentages, reverse=False)

# 繪製長條圖
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(sorted_features, sorted_shapley_values, color='navy')

# 添加百分比文本标签
# for bar, percentage in zip(bars, sorted_contribution_percentages):
#     ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{percentage}%',
#             ha='left', va='center', color='black')

ax.set_xlabel('Shapley Value')
ax.set_ylabel('Features')
ax.set_title('Shapley values of features in Combination 2')
plt.savefig('forpaper_1yrfeature_shap_scoreoutper.png',dpi=300, bbox_inches='tight')