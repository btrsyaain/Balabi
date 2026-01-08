import matplotlib.pyplot as plt
import numpy

'''BASELINE'''
# 特徵數量和對應的 c-index
feature_counts = range(1,16)
c_indexes = [0.616,0.618,0.653,0.669,0.669,0.67,0.677,0.686,0.69,0.686,0.673,0.665,0.671,0.674,0.661]
shapley_values = [0.030530877455877327, 0.04797314907314953, 0.03088255910755915, 0.031498803973803294, 
                  0.03386084193584186, 0.03244890942390918, 0.03552253024753008, 0.06281319236319216, 0.05869819624819586,
                  0.03881617549117543, 0.03754749694749679, 0.02191138861138864, 0.08145218392718248, 0.06558032245532198, 0.05146337273837282]
sorted_shapley_values = sorted(shapley_values, reverse=True)
cumulative_shapley_values = []
cumulative_value = 0
for shapley_value in sorted_shapley_values:
    cumulative_value += shapley_value
    cumulative_shapley_values.append(cumulative_value)

# # 繪製折線圖
fig, ax1 = plt.subplots()

ax1.plot(feature_counts, c_indexes, marker='o', linestyle='-')
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('C-Index', color='black')

ax2 = ax1.twinx()
ax2.bar(feature_counts, cumulative_shapley_values, color='skyblue', alpha=0.5)
ax2.set_ylabel('Accumulative Shapley Values', color='black')

# # 加上標籤和標題
# plt.title('Baseline features added in descending way and their C-Index')
plt.title('Features added in descending order and their c-index in Combination 1')
plt.savefig('forpaper_Baselinefeature_cindex.png', dpi=300)
# # 顯示圖表
plt.show()

'''1yr'''
# feature_counts = range(1,14)
# c_indexes = [0.659,0.669,0.721,0.721,0.727,0.728,0.725,0.735,0.733,0.725,0.728,0.722,0.723]
# shapley_values = [0.040887076812076914, 0.04502340437340444, 0.04149043179043183, 0.030628635253635322, 0.0412714563214565,0.037612040737040685,
#                   0.08613303640803677,0.08074645632145659,0.03236943611943616,0.04528758186258193,0.06226561216561205,0.07222022977022989,0.10706460206460208]
# sorted_shapley_values = sorted(shapley_values, reverse=True)
# cumulative_shapley_values = []
# cumulative_value = 0
# for shapley_value in sorted_shapley_values:
#     cumulative_value += shapley_value
#     cumulative_shapley_values.append(cumulative_value)

# fig, ax1 = plt.subplots()

# ax1.plot(feature_counts, c_indexes, marker='o', linestyle='-')
# ax1.set_xlabel('Number of Features')
# ax1.set_ylabel('C-Index', color='black')

# ax2 = ax1.twinx()
# ax2.bar(feature_counts, cumulative_shapley_values, color='skyblue', alpha=0.5)
# ax2.set_ylabel('Accumulative Shapley Values', color='black')

# # # 加上標籤和標題
# # plt.title('1yr features added in descending way and their C-Index')
# plt.title('Features added in descending order and their c-index in Combination 2')
# plt.savefig('forpaper_1yrfeature_cindex.png', dpi=300)
# # # 顯示圖表
# plt.show()

'''ALL'''
# feature_counts = range(1,23)
# c_indexes = [0.659,0.679,0.721,0.721,0.72,0.754,0.758,0.747,0.744,0.739,0.737,0.726,0.731,0.733,0.739,0.749,
# 0.742,0.734,0.742,0.748,0.726,0.735]
# shapley_values = [0.058444540048927886,0.0230838595185367,0.02798939007329306,0.023637580204454802,0.018181005436777992,0.02428365909546479,0.023459782838425335,0.026465511067022905,
#                   0.037437737898562314,0.022797292202978627,0.02514381035201584,0.01887483534697506,0.04915469743545606,0.041429403538604276,0.03092755469503501,
#                   0.05908614879279521,0.0170001127226885,0.027320656961718898,0.03758562319179292,0.04691826886144095,0.07365158898553134,0.022126940732131024]
# sorted_shapley_values = sorted(shapley_values, reverse=True)
# cumulative_shapley_values = []
# cumulative_value = 0
# for shapley_value in sorted_shapley_values:
#     cumulative_value += shapley_value
#     cumulative_shapley_values.append(cumulative_value)

# fig, ax1 = plt.subplots()

# ax1.plot(feature_counts, c_indexes, marker='o', linestyle='-')
# ax1.set_xlabel('Number of Features')
# ax1.set_ylabel('C-Index', color='black')

# ax2 = ax1.twinx()
# ax2.bar(feature_counts, cumulative_shapley_values, color='skyblue', alpha=0.5)
# ax2.set_ylabel('Accumulative Shapley Values', color='black')

# # # 加上標籤和標題
# # plt.title('Baseline + 1yr features added in descending way and their C-Index')
# plt.title('Features added in descending order and their c-index in Combination 3')
# plt.savefig('forpaper_allfeature_cindex.png', dpi=300)
# plt.show()


# ### 3組時間放在一起
# baselinefeature_counts = range(1,16)
# baselinec_indexes = [0.616,0.618,0.653,0.669,0.669,0.67,0.677,0.686,0.69,0.686,0.673,0.665,0.671,0.674,0.661]
# firstyearfeature_counts = range(1,14)
# firstyearc_indexes = [0.659,0.669,0.721,0.721,0.727,0.728,0.725,0.735,0.733,0.725,0.728,0.722,0.723]
# allfeature_counts = range(1,23)
# allc_indexes = [0.659,0.679,0.721,0.721,0.72,0.754,0.758,0.747,0.744,0.739,0.737,0.726,0.731,0.733,0.739,0.749,0.742,0.734,0.742,0.748,0.726,0.735]
# plt.plot(baselinefeature_counts, baselinec_indexes, marker='o', linestyle='-', color='black', label='Baseline')
# plt.plot(firstyearfeature_counts, firstyearc_indexes, marker='o', linestyle='-', color='blue', label='1yr')
# plt.plot(allfeature_counts, allc_indexes, marker='o', linestyle='-', color='red', label='Baseline+1yr')

# plt.xlabel('Number of Features')
# plt.ylabel('C-Index')
# plt.title('Numbers of 3 period features and their C-Index')
# plt.legend()
# plt.savefig('3periodfeature_cindex.png', dpi=300)
# plt.show()
