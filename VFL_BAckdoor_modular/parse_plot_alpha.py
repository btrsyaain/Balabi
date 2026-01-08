import re, pathlib, statistics
import pandas as pd
import matplotlib.pyplot as plt

# ========== 參數 ==========
LOG_DIR   = pathlib.Path("alpha_test")        # 你的 log 目錄
OUT_DIR   = pathlib.Path("alpha_test/plots")  # 圖片輸出
OUT_DIR.mkdir(exist_ok=True)

# 正則規則
file_pat  = re.compile(r"log_D(\w+)_P(\d+)_AM(\w+)_alpha([\d.]+)\.txt")
acc_pat   = re.compile(r"acc on 下毒資料集:(\d+\.\d+)")
asr_pat   = re.compile(r"排除掉.*ASR 為 (\d+\.\d+)")

records = []  # 暫存所有 run

# ========== 1. 讀取並擷取 ==========
for log in LOG_DIR.glob("log_*.txt"):
    m = file_pat.match(log.name)
    if not m:
        continue
    D, P, AM, alpha = m.groups()
    alpha = float(alpha)
    run_idx = 0

    acc_tmp, asr_tmp = [], []
    with log.open() as f:
        for line in f:
            if line.startswith("▶ Run"):
                # 進入下一個 run
                if acc_tmp and asr_tmp:  # 存上一輪
                    records.append(dict(D=D, P=int(P), AM=AM,
                                        alpha=alpha, run=run_idx,
                                        ACC=float(acc_tmp[-1]),
                                        ASR=float(asr_tmp[-1])))
                run_idx += 1
                acc_tmp, asr_tmp = [], []
            else:
                a = acc_pat.search(line)
                if a: acc_tmp.append(a.group(1))
                b = asr_pat.search(line)
                if b: asr_tmp.append(b.group(1))

        # 最後一輪
        if acc_tmp and asr_tmp:
            records.append(dict(D=D, P=int(P), AM=AM,
                                alpha=alpha, run=run_idx,
                                ACC=float(acc_tmp[-1]),
                                ASR=float(asr_tmp[-1])))

df = pd.DataFrame(records)

# ========== 2. 取中位數 run ==========
def pick_median(sub):
    # sub: 同一個 (D,P,AM,alpha) 下三筆
    ranks = sorted(sub["ACC"])
    median_acc = statistics.median(ranks)
    # 找到最接近 median 的那一筆 (如兩筆並列，取較小 run id)
    idx = (sub["ACC"] - median_acc).abs().idxmin()
    return sub.loc[idx]

chosen = (
    df.groupby(["D", "P", "AM", "alpha"], group_keys=False)
      .apply(pick_median)
      .reset_index(drop=True)
)

# ========== 3. 畫圖 ==========
for (D, P, AM), group in chosen.groupby(["D", "P", "AM"]):
    g = group.sort_values("alpha")
    x = g["alpha"]

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()

    ax1.plot(x, g["ACC"], marker="o", linestyle="-", label="ACC")
    bar_width = 0.06*(x.max()-x.min()) if len(x) > 1 else 0.1
    ax2.bar(x, g["ASR"], width=bar_width, alpha=0.4, label="ASR")

    ax1.set_xlabel("alpha")
    ax1.set_ylabel("ACC")
    ax2.set_ylabel("ASR (excluded label 0)")
    ax1.set_title(f"{D} | P={P} | {AM}")

    # 合併圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="best")

    plt.tight_layout()
    fname = OUT_DIR / f"{D}_P{P}_{AM}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("saved", fname)
