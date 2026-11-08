#!/usr/bin/env python3
# parse_plot_alpha_multiP_bar.py
#
# 讀取 alpha_test/log_*.txt，解析每次 run 的 ACC / ASR
# 取 (Dataset, AM, P, alpha) 群組中「ACC 中位數 run」作代表
# 依 (Dataset, AM) 畫圖：ACC 折線 + ASR 分組長條（多 P 同圖）
# ---------------------------------------------------------------

import re, pathlib, statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- 可調參數 -----------------
LOG_DIR   = pathlib.Path("alpha_test_BM")            # log 檔所在資料夾
OUT_DIR   = LOG_DIR / "plots_multiP_BM_withlogfun"              # 圖片輸出資料夾
OUT_DIR.mkdir(exist_ok=True)

BAR_RATIO = 0.2      # 長條寬度 = α 最小間距 × BAR_RATIO
MARKERS   = ["o", "s", "^", "D", "v"]             # ACC 折線標記形狀
COLORS    = plt.cm.tab10.colors                  # 預設顏色

# ----------------- 正則樣式 -----------------
file_pat = re.compile(r"log_D(\w+)_P(\d+)_AM(\w+)_alpha([\d.]+)\.txt")
acc_pat  = re.compile(r"acc on 下毒資料集:(\d+\.\d+)")
asr_pat  = re.compile(r"排除掉.*ASR 為 (\d+\.\d+)")

# ----------------- 1. 擷取各 run 資料 -----------------
records = []   # [D, P, AM, alpha, run_idx, ACC, ASR]

for log in LOG_DIR.glob("log_*.txt"):
    m = file_pat.match(log.name)
    if not m:
        continue
    D, P, AM, alpha = m.groups()
    P, alpha = int(P), float(alpha)

    run_idx, acc_tmp, asr_tmp = 0, [], []

    for line in log.read_text().splitlines():
        if line.startswith("▶ Run"):
            # 進入下一 run 之前，先存上一 run（若有）
            if acc_tmp and asr_tmp:
                records.append([D, P, AM, alpha, run_idx,
                                float(acc_tmp[-1]), float(asr_tmp[-1])])
            run_idx += 1
            acc_tmp, asr_tmp = [], []
        else:
            a = acc_pat.search(line)
            if a: acc_tmp.append(a.group(1))
            b = asr_pat.search(line)
            if b: asr_tmp.append(b.group(1))

    # 檔案結束時再存最後一 run
    if acc_tmp and asr_tmp:
        records.append([D, P, AM, alpha, run_idx,
                        float(acc_tmp[-1]), float(asr_tmp[-1])])

df = pd.DataFrame(records,
                  columns=["D", "P", "AM", "alpha", "run", "ACC", "ASR"])

# ----------------- 2. 取 ACC 中位數代表 run -----------------
def choose_median(sub_df: pd.DataFrame) -> pd.Series:
    """給定同一 (D,AM,P,alpha) 的 3 行，選 ACC 最接近中位數者"""
    median_acc = statistics.median(sorted(sub_df["ACC"]))
    idx = (sub_df["ACC"] - median_acc).abs().idxmin()
    return sub_df.loc[idx]

chosen = (
    df.groupby(["D", "AM", "P", "alpha"], group_keys=False)
      .apply(choose_median)
      .reset_index(drop=True)
)

# ----------------- 3. 畫圖 -----------------
for (D, AM), grp in chosen.groupby(["D", "AM"]):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    P_values = sorted(grp["P"].unique())
    nP       = len(P_values)

    for idx, P in enumerate(P_values):
        sub = grp[grp["P"] == P].sort_values("alpha")
        x   = sub["alpha"].values

        # ---------- ACC 折線 ----------
        ax1.plot(x, sub["ACC"],
                 color=COLORS[idx % 10],
                 marker=MARKERS[idx % len(MARKERS)],
                 linestyle="-",
                 label=f"P={P} (ACC)")

        # ---------- ASR 分組長條 ----------
        if len(x) > 1:
            gap = np.diff(np.sort(np.unique(x))).min()    # α 最小間距
            bar_w = BAR_RATIO * gap
        else:
            bar_w = BAR_RATIO                                # 只有一個 α 時

        offset = (idx - (nP - 1) / 2) * bar_w               # 左負右正，置中
        ax2.bar(x + offset, sub["ASR"],
                width=bar_w,
                color=COLORS[idx % 10],
                alpha=0.4,
                label=f"P={P} (ASR)")

    # ---------- 標題與軸 ----------
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("ACC")
    ax2.set_ylabel("ASR (excluded label 0)")
    ax1.set_title(f"{D} | {AM}")

    # ---------- 共同圖例 ----------
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / f"{D}_{AM}_multiP_nolog_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)
