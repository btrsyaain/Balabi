import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

def summarize_shapley(shap_values):
    """用中位數來代表 shapley 分布"""
    return shap_values.median().item()

def plot_bar_comparison(train_dict, clean_dict, poisoned_dict, participant_ids, args, title_suffix=''):
    """
    繪製 bar 圖來比較三種資料來源中，各參與者在特定類別下的 Shapley 值中位數。
    """
    sources = ['Train', 'Clean Test', 'Poisoned Test']
    data_dicts = [train_dict, clean_dict, poisoned_dict]

    summary_values = {src: [] for src in sources}
    
    for src_name, shap_dict in zip(sources, data_dicts):
        for pid in participant_ids:
            shap_dist = shap_dict[args.label][pid]
            value = summarize_shapley(shap_dist)
            summary_values[src_name].append(value)

    # 畫圖
    x = np.arange(len(participant_ids))  # bar 的位置
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, src in enumerate(sources):
        ax.bar(x + i * width, summary_values[src], width, label=src)

    ax.set_xlabel('Participant ID')
    ax.set_ylabel('Shapley Median Value')
    ax.set_title(f'Shapley Comparison for Class {args.label} {title_suffix}')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(pid) for pid in participant_ids])
    ax.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f'{title_suffix}_in_three_time_shap_diff.png'
    plt.savefig(filename)

def match_sample_size(a: np.ndarray, b: np.ndarray):
    """將兩組樣本下採樣成相同大小"""
    min_len = min(len(a), len(b))
    a_sampled = np.random.choice(a, min_len, replace=False)
    b_sampled = np.random.choice(b, min_len, replace=False)
    return a_sampled, b_sampled

def compute_p_values(train_dict, clean_dict, poisoned_dict, participant_id, args):
    """
    比較三個資料來源的 Shapley 分布是否有顯著差異（加入下採樣處理）。
    """
    train_vals = train_dict[args.label][participant_id].cpu().numpy()
    clean_vals = clean_dict[args.label][participant_id].cpu().numpy()
    poisoned_vals = poisoned_dict[args.label][participant_id].cpu().numpy()

    print(f"\n📊 [Shapley 數量] Participant {participant_id} in Class {args.label}")
    print(f"Train: {len(train_vals)} samples, Clean: {len(clean_vals)}, Poisoned: {len(poisoned_vals)}")

    # 下採樣處理
    train_vs_clean_a, train_vs_clean_b = match_sample_size(train_vals, clean_vals)
    train_vs_poisoned_a, train_vs_poisoned_b = match_sample_size(train_vals, poisoned_vals)
    clean_vs_poisoned_a, clean_vs_poisoned_b = match_sample_size(clean_vals, poisoned_vals)

    # Mann-Whitney U test
    _, p_train_vs_clean = mannwhitneyu(train_vs_clean_a, train_vs_clean_b, alternative='two-sided')
    _, p_train_vs_poisoned = mannwhitneyu(train_vs_poisoned_a, train_vs_poisoned_b, alternative='two-sided')
    _, p_clean_vs_poisoned = mannwhitneyu(clean_vs_poisoned_a, clean_vs_poisoned_b, alternative='two-sided')

    print(f"🔍 [p 值分析] Participant {participant_id} in Class {args.label}")
    print(f"Train vs Clean Test     : p = {p_train_vs_clean:.4f}")
    print(f"Train vs Poisoned Test  : p = {p_train_vs_poisoned:.4f}")
    print(f"Clean vs Poisoned Test  : p = {p_clean_vs_poisoned:.4f}")
'''=======================================================================================================================''' 
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d  # 平滑用

def get_shap_data_by_participant(shap_dict, label, participant_id):
    return shap_dict[label][participant_id].detach().cpu().numpy()

def plot_overlapping_histogram(train_dict, clean_dict, poisoned_dict, participant_id, args):
    train_vals = get_shap_data_by_participant(train_dict, args.label, participant_id)
    clean_vals = get_shap_data_by_participant(clean_dict, args.label, participant_id)
    poisoned_vals = get_shap_data_by_participant(poisoned_dict, args.label, participant_id)

    plt.figure(figsize=(8, 5))
    bins = np.linspace(
        min(train_vals.min(), clean_vals.min(), poisoned_vals.min()),
        max(train_vals.max(), clean_vals.max(), poisoned_vals.max()),
        50
    )

    plt.hist(clean_vals, bins=bins, alpha=0.7, label="Clean", color='blue', density=True)
    plt.hist(poisoned_vals, bins=bins, alpha=0.7, label="Poisoned", color='red', density=True)
    plt.hist(train_vals, bins=bins, alpha=0.5, label="Train", color='green', density=True)

    plt.xlabel("Shapley Value")
    plt.ylabel("Proportion (Density)")
    plt.title(f"Overlapping Histogram - Participant {participant_id}, Class {args.label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"participant_{participant_id}_overlap.png")

def plot_shap_boxplot(train_dict, clean_dict, poisoned_dict, participant_id, args):
    train_vals = get_shap_data_by_participant(train_dict, args.label, participant_id)
    clean_vals = get_shap_data_by_participant(clean_dict, args.label, participant_id)
    poisoned_vals = get_shap_data_by_participant(poisoned_dict, args.label, participant_id)

    data = [train_vals, clean_vals, poisoned_vals]
    labels = ['Train', 'Clean Test', 'Poisoned Test']

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"Boxplot - Participant {participant_id}, Class {args.label}")
    plt.ylabel("Shapley Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"participant_{participant_id}_box.png")

def plot_kde_density(train_dict, clean_dict, poisoned_dict, participant_id, args):
    train_vals = get_shap_data_by_participant(train_dict, args.label, participant_id)
    clean_vals = get_shap_data_by_participant(clean_dict, args.label, participant_id)
    poisoned_vals = get_shap_data_by_participant(poisoned_dict, args.label, participant_id)

    # 計算 KDE
    kde_train = gaussian_kde(train_vals)
    kde_clean = gaussian_kde(clean_vals)
    kde_poisoned = gaussian_kde(poisoned_vals)

    # 定義 x 軸範圍（所有資料最小最大值範圍內）
    xmin = min(train_vals.min(), clean_vals.min(), poisoned_vals.min())
    xmax = max(train_vals.max(), clean_vals.max(), poisoned_vals.max())
    x_vals = np.linspace(xmin, xmax, 500)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, kde_clean(x_vals), label="Clean", color='blue', linewidth=2)
    plt.plot(x_vals, kde_poisoned(x_vals), label="Poisoned", color='red', linewidth=2)
    plt.plot(x_vals, kde_train(x_vals), label="Train", color='green', linewidth=2)

    plt.xlabel("Shapley Value")
    plt.ylabel("Probability Density")
    plt.title(f"KDE Distribution - Participant {participant_id}, Class {args.label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"participant_{participant_id}_kde.png")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

def plot_hist_curve_proportion(train_dict, clean_dict, poisoned_dict, participant_id, args, bins=50, save_dir="./hist_curves"):

    def get_smoothed_proportion(data, bins):
        hist, bin_edges = np.histogram(data, bins=bins, density=False)
        total = np.sum(hist)
        proportion = hist / total * 100  # 每個 bin 的百分比 (%)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(proportion, sigma=1.2)
        return centers, smoothed

    train_vals = get_shap_data_by_participant(train_dict, args.label, participant_id)
    clean_vals = get_shap_data_by_participant(clean_dict, args.label, participant_id)
    poisoned_vals = get_shap_data_by_participant(poisoned_dict, args.label, participant_id)

    x_clean, y_clean = get_smoothed_proportion(clean_vals, bins)
    x_poisoned, y_poisoned = get_smoothed_proportion(poisoned_vals, bins)
    x_train, y_train = get_smoothed_proportion(train_vals, bins)

    plt.figure(figsize=(8, 5))
    plt.plot(x_clean, y_clean, label="Clean", color='blue', linewidth=2)
    plt.plot(x_poisoned, y_poisoned, label="Poisoned", color='red', linewidth=2)
    plt.plot(x_train, y_train, label="Train", color='green', linewidth=2)

    plt.xlabel("Shapley Value")
    plt.ylabel("Proportion (%)")
    plt.title(f"Smoothed Histogram Curve (Proportion) - Participant {participant_id}, Class {args.label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/participant_{participant_id}_class_{args.label}_histcurve_fixed.png"
    plt.savefig(filename)
    print(f"✅ 已儲存圖：{filename}")
    plt.close()