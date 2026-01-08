# file: xgb_bankmarket_vfl_shap.py
import pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# 1. 讀檔、前處理 ── 與 VFL pipeline 完全一致
csv_path = pathlib.Path("./data/bank-market/bank-market.csv")
df = pd.read_csv(csv_path)

# (1) 把唯一值=2 的欄位轉成 0/1
for col in df.select_dtypes("object"):
    if df[col].nunique() == 2:
        df[col] = (df[col] == "yes").astype(int)

# (2) 其餘 object → one-hot
df = pd.get_dummies(df).astype(np.float32)

target = "deposit"
X = df.drop(columns=[target])
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. XGBoost 模型
clf = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Test AUC = {clf.score(X_test, y_test):.4f}")

# 3. TreeSHAP
explainer = shap.TreeExplainer(clf)
background = shap.sample(X_train, min(5000, len(X_train)), random_state=42)
shap_values = explainer.shap_values(background)

# 4. (選用) 把同源 dummy 先『加總』，方便 VFL 分配
def group_shap(feature_names, shap_vals):
    groups, scores = {}, {}
    for name, s in zip(feature_names, shap_vals.T):
        # e.g. 'month_may' → group='month', sub='may'
        group = name.split("_")[0]
        groups.setdefault(group, []).append(name)
        scores[group] = scores.get(group, 0) + np.abs(s).mean()
    # 依 |SHAP| 總和排序
    sorted_groups = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_groups, groups

names = X_train.columns.tolist()
group_rank, group_members = group_shap(names, shap_values)

print("\nTop-10 feature groups (mean |SHAP| sum):")
for g, sc in group_rank[:]:
    print(f"{g:<10}  {sc:>.4f}   members={len(group_members[g])}")
    
# === 4b. 把 one-hot 依「原始欄位」合併成群組 SHAP =====================
def build_group_matrix(feature_names, shap_mat):
    """
    feature_names: list[str]，像 'job_admin.'、'month_may'、'age'
    shap_mat    : np.ndarray, 形狀 (n_samples, n_features)
    return:
      S_group : (n_samples, n_groups)  每筆樣本的「群組 SHAP」= 同群 dummy 的 SHAP 相加
      g_names : list[str]              群組名稱（原始欄位名）
      g_cols  : dict[str, list[int]]   每個群組對應到的原本欄位索引
    """
    g_cols = {}
    for j, name in enumerate(feature_names):
        root = name.split('_')[0]       # e.g. 'month_may' → 'month'
        g_cols.setdefault(root, []).append(j)

    g_names = list(g_cols.keys())
    S_group = np.zeros((shap_mat.shape[0], len(g_names)), dtype=np.float32)
    for g, cols in enumerate(g_cols.values()):
        # 關鍵：群組 SHAP = 同一原始欄位的所有 dummy SHAP「逐樣本相加」
        S_group[:, g] = shap_mat[:, cols].sum(axis=1)
    return S_group, g_names, g_cols

S_group, group_names, group_cols = build_group_matrix(names, shap_values)

# === 4c. 依「群組 |SHAP| 的樣本平均」做排序，輸出排名 = 全球重要度 ==========
group_score = np.abs(S_group).mean(axis=0)  # s_g = (1/N) Σ_i |φ_i^(g)|
rank_idx = np.argsort(-group_score)
group_rank_sorted = [(group_names[i], float(group_score[i]), len(group_cols[group_names[i]]))
                     for i in rank_idx]

print("\n[Grouped importance] 依 |SHAP| 樣本平均排序（群組層級）")
for g, sc, m in group_rank_sorted:
    print(f"{g:<10}  {sc:>.4f}   members={m}")

# === 4d. 產出「群組層級」的兩種圖 ================================

# 4d-1. 群組條形圖（全球重要度）──推論用最穩
plt.figure(figsize=(6, 4))
plt.barh([group_names[i] for i in rank_idx[::-1]],
         group_score[rank_idx[::-1]])
plt.xlabel("mean(|SHAP|) per group")
plt.tight_layout()
plt.savefig("shap_group_bar.png", dpi=900)

# 4d-2. 群組 beeswarm（若你也想看分布；顏色對類別意義不大）
# shap.summary_plot 可直接吃群組後矩陣做 bar；beeswarm 也能畫，但顏色不具類別語義
shap.summary_plot(
    S_group[:, rank_idx],                 # 依排名重排
    features=None,                        # 群組層級沒有單一「特徵值」可上色
    feature_names=[group_names[i] for i in rank_idx],
    plot_type=None,                       # 預設 beeswarm
    show=False
)
plt.tight_layout()
plt.savefig("shap_group_beeswarm.png", dpi=900)


# 5. 畫 summary plot（仍用單一 dummy granularity，方便觀察方向）
shap.summary_plot(
    shap_values,
    features=background,
    feature_names=names,
    show=False,
)
plt.tight_layout()
plt.savefig("shap_summary_bankmarket_vfl.png", dpi=900)
print("圖片已輸出 shap_summary_bankmarket_vfl.png")

# 6. 另存 group 排名供 VFL 切分腳本讀取
with open("group_shap_rank.json", "w") as fp:
    json.dump([g for g, _ in group_rank], fp, indent=2)
print("已儲存 feature-group 排序 → group_shap_rank.json")
