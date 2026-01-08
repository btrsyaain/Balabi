# import re
# import pandas as pd

# def parse_experiment_results(txt_path):
#     with open(txt_path, 'r', encoding='utf-8') as f:
#         content = f.read()

#     # 分割每一次的實驗紀錄
#     experiments = content.split('🔵 執行第 ')[1:]

#     acc_list = []
#     asr_list = []
#     d_acc_list = []
#     d_asr_list = []

#     for exp in experiments:
#         # ACC: poisoned 模型 acc on 乾淨測試資料集
#         acc_match = re.search(r"poisoned 模型 acc on 乾淨測試資料集:\s*([0-9.]+)", exp)
#         acc = float(acc_match.group(1)) if acc_match else None

#         # ASR: 未經防禦的情況下 → 排除掉原先 label 即為 0 之資料後，ASR 為 ...
#         asr_match = re.search(
#             r"未經防禦的情況下:[\s\S]*?排除掉原先 label 即為 0 之資料後，ASR 為\s*([0-9.]+)", exp)
#         asr = float(asr_match.group(1)) if asr_match else None

#         # D_ACC: acc on 下毒資料集 (防禦區塊中)
#         d_acc_match = re.search(
#             r"(正在測試下毒測試集|SHAP\(MAE\) 防禦效果:|VFLIP 防禦效果:|CoPur 防禦效果:|NEURAL_CLEANSE 防禦效果:|ABL 防禦效果:)[\s\S]*?acc on 下毒資料集:([0-9.]+)", exp)
#         d_acc = float(d_acc_match.group(2)) if d_acc_match else None

#         # D_ASR: 防禦後，排除掉原先 label 為 0 的資料後的 ASR
#         d_asr_match = re.search(
#             r"(正在測試下毒測試集|SHAP\(MAE\) 防禦效果:|VFLIP 防禦效果:|CoPur 防禦效果:|NEURAL_CLEANSE 防禦效果:|ABL 防禦效果:)[\s\S]*?排除掉原先 label 即為 0 之資料後，ASR 為\s*([0-9.]+)", exp)
#         d_asr = float(d_asr_match.group(2)) if d_asr_match else None

#         acc_list.append(acc)
#         asr_list.append(asr)
#         d_acc_list.append(d_acc)
#         d_asr_list.append(d_asr)

#     df = pd.DataFrame({
#         'ACC': acc_list,
#         'ASR': asr_list,
#         'D_ACC': d_acc_list,
#         'D_ASR': d_asr_list
#     })

#     return df


# df = parse_experiment_results("logs/log_DCIFAR10_P4_AN1_AMTIFS_DMSHAP_MAE_D_WoP_7.txt")
# print(df)


# readtxt.py
import re
import pandas as pd

def parse_experiment_results(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    experiments = content.split('🔵 執行第 ')[1:]

    cols = ['ACC', 'ASR',
            'D_ACC', 'D_ASR',
            'D_ACC_Agree', 'A_ASR_Agree',
            'D_ACC_skip',  'A_ASR_skip',
            'D_ACC_total', 'A_ASR_total']
    data = {c: [] for c in cols}

    for exp in experiments:
        # ---------- poisoned ACC on clean ----------
        m = re.search(r"poisoned 模型 acc on 乾淨測試資料集:\s*([0-9.]+)", exp)
        data['ACC'].append(float(m.group(1)) if m else None)

        # ---------- 「未經防禦」區塊 ----------
        # 1) 先切出區塊（從標題到下一條 '----'，或到字串結束）
        nd_match = re.search(r"未經防禦的情況下:[\s\S]*?(?=--------------------------------------|$)", exp)
        nd_block = nd_match.group(0) if nd_match else ""

        # 2) ASR（排除 label=0 後）
        m = re.search(r"排除掉原先 label 即為 0 之資料後，ASR 為\s*([0-9.]+)", nd_block)
        data['ASR'].append(float(m.group(1)) if m else None)

        # 3) Baseline / Skipped / All-Rand 只在 nd_block 裡找
        patterns = {
            'D_ACC_total':  r"\[All-Rand\]\s*ACC\s*=\s*([0-9.]+)",
            'A_ASR_total':  r"\[All-Rand\]\s*ACC\s*=[0-9.]+\s*ASR\s*=\s*([0-9.]+)",
            'D_ACC_Agree':  r"\[Baseline\]\s*ACC\s*=\s*([0-9.]+)",
            'A_ASR_Agree':  r"\[Baseline\]\s*ACC\s*=[0-9.]+\s*ASR\s*=\s*([0-9.]+)",
            'D_ACC_skip':   r"\[Skipped\s*\]\s*ACC\s*=\s*([0-9.]+)",
            'A_ASR_skip':   r"\[Skipped\s*\]\s*ACC\s*=[0-9.]+\s*ASR\s*=\s*([0-9.]+)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, nd_block)
            data[k].append(float(m.group(1)) if m else None)

        # ---------- 舊的防禦區塊 (若有其他防禦) ----------
        d_block_match = re.search(
            r"(正在測試下毒測試集|SHAP\(MAE\) 防禦效果:|VFLIP 防禦效果:|CoPur 防禦效果:|NEURAL_CLEANSE 防禦效果:|ABL 防禦效果:)"
            r"[\s\S]*?(?=未經防禦的情況下:|$)", exp)
        d_block = d_block_match.group(0) if d_block_match else ""

        m = re.search(r"acc on 下毒資料集:([0-9.]+)", d_block)
        data['D_ACC'].append(float(m.group(1)) if m else None)

        m = re.search(r"排除掉原先 label 即為 0 之資料後，ASR 為\s*([0-9.]+)", d_block)
        data['D_ASR'].append(float(m.group(1)) if m else None)

    return pd.DataFrame(data)


# 測試
if __name__ == "__main__":
    df = parse_experiment_results("/home/server/Balabi/VFL_BAckdoor_modular/logs/collected/log_DCIFAR10_P2_AN1_AMTIFS_DMCoPur_WoP_7.txt")
    # 想看的欄位順序
    desired_cols = [
        'D_ACC_total', 'A_ASR_total',
        'D_ACC_Agree', 'A_ASR_Agree',
        'D_ACC_skip',  'A_ASR_skip'
    ]
    
    # step1 ▶ 先抓出「不在 desired_cols 中」的其他欄位，保持原本順序
    head_cols = [c for c in df.columns if c not in desired_cols]

    # step2 ▶ 再把 cols 欄位按你指定的順序接到後面
    final_cols = head_cols + [c for c in desired_cols if c in df.columns]
    print(df[final_cols])
    # print(df)

