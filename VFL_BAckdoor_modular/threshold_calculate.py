import numpy as np

def load_sscore_data(filepath):
    """
    讀取每行的 sscore 記錄檔，該行包含 4 個數值：
      [dist_to_participant_0, dist_to_participant_1(=0), dist_to_participant_2, dist_to_participant_3]
    回傳一個 numpy array，元素為該行距離去掉對角線(0)後的平均值。
    """
    results = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        row = line.strip().split()
        # row 例如: ["3.1271945e+01", "0.0000000e+00", "3.5339221e+01", "3.8345466e+01"]
        try:
            row_floats = list(map(float, row))
            # 假設 row_floats[1] 即對角線位置 (participant_1 -> participant_1)
            # 取 [0, 2, 3] 位置的平均
            distances = [row_floats[0], row_floats[2], row_floats[3]]
            mean_val = np.mean(distances)
            results.append(mean_val)
        except ValueError:
            # 若某行轉 float 出錯，可自行 decide 跳過或拋錯
            continue
    
    return np.array(results)

#---- 1) 讀取 clean & poisoned ----
clean_file = "clean_sscore_record.txt"
poison_file = "poisoned_sscore_record.txt"

clean_dist = load_sscore_data(clean_file)      # Participant 1 在 clean 狀況下的距離
poisoned_dist = load_sscore_data(poison_file)  # Participant 1 在 poisoned 狀況下的距離

print("Clean 數量:", len(clean_dist))
print("Poisoned 數量:", len(poisoned_dist))

#---- 2) 建立標籤 (0=clean, 1=poisoned) 以便統計 TPR / FPR 等 ----
X = np.concatenate([clean_dist, poisoned_dist])
y = np.concatenate([np.zeros_like(clean_dist), np.ones_like(poisoned_dist)])

def compute_tpr_fpr(X, y, threshold):
    """
    以 threshold 判斷：若距離 >= threshold → 判為可疑(攻擊 = 1)，否則判為 0
    計算 TPR, FPR
    """
    y_pred = (X >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == 0))
    tn = np.sum((y_pred == 0) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return tpr, fpr

# 先找出 X 的 min/max
min_val, max_val = np.min(X), np.max(X)

# 等間距取 100 個 threshold
thresholds = np.linspace(min_val, max_val, 100)

best_threshold = None
best_youden = -999
results_list = []

for th in thresholds:
    tpr, fpr = compute_tpr_fpr(X, y, th)
    # 以 Youden’s Index (J = TPR - FPR) 作為最佳化依據
    j = tpr - fpr
    if j > best_youden:
        best_youden = j
        best_threshold = th
    results_list.append((th, tpr, fpr, j))

print(f"\n[資訊] 取樣 threshold 範圍：{min_val:.3f} ~ {max_val:.3f}")
print(f"[結果] 最佳門檻 (by Youden’s J) = {best_threshold:.3f}, Youden’s J = {best_youden:.3f}")

# 簡易列印部分資料
print("\n部分 threshold 結果：")
step_show = max(len(results_list)//5, 1)
for i in range(0, len(results_list), step_show):
    th, tpr, fpr, j = results_list[i]
    print(f"threshold={th:.3f}  TPR={tpr:.2%}  FPR={fpr:.2%}  J={j:.3f}")
