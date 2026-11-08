import matplotlib.pyplot as plt

# 資訊
# features = ['sex1M0F', 'ETV1TDFTAF2', 'DM', 'HTN', 'NA-naive', 'HBeAg','logHBVDNA',
#             'age', 'got', 'gpt', 't-bil', 'INRR', 'albumin', 'platelet', 'afp']
features = ['Sex', 
            'Antiviral treatment type', 
            'DM', 
            'HTN', 
            'NA', 
            'HBeAg',
            'HBV DNA titer at baseline',
            'Age', 
            'AST at baseline', 
            'ALT at baseline', 
            'Bilirubin at baseline', 
            'INR at baseline', 
            'Albumin at baseline', 
            'Platelet at baseline', 
            'AFP at baseline']
shapley_values = [0.030530877455877327, 0.04797314907314953, 0.03088255910755915, 0.031498803973803294, 
                  0.03386084193584186, 0.03244890942390918, 0.03552253024753008, 0.06281319236319216, 0.05869819624819586,
                  0.03881617549117543, 0.03754749694749679, 0.02191138861138864, 0.08145218392718248, 0.06558032245532198, 0.05146337273837282]
# total = 0
# contribution_percentages = []
# for i in range(len(shapley_values)):
#     total+=shapley_values[i]
# for j in range(len(shapley_values)):
#     contribution_percentages.append(round(shapley_values[j]/total,3)*100)

# contribution_percentages = [12.3, 9.9, 9.5, 8.9, 7.8, 7.3, 5.9, 5.7, 5.4, 5.1, 4.9, 4.8, 4.7, 4.6, 3.3]

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
ax.set_title('Shapley values of features in Combination 1')
plt.savefig('forpaper_baselinefeature_shap_score.png',dpi=300, bbox_inches='tight')