#!/usr/bin/env bash
# alpha_test/run_alpha_grid.sh
# 跑 Dataset × P × AM × alpha，每組連續執行 3 次並寫進同一 logfile

set -e

########################
# 固定與掃描之超參數
########################
# D_values=("CIFAR10" "CINIC10" "BM")    # 資料集
# P_values=(2 4 8)                       # 參與方數
# AM_values=("TIFS" "VILLIAN")           # 攻擊方法
# alpha_values=(0.5 1 1.3 1.5 1.7 2 3)   # α 值
# num_runs=3                             # 每組重跑次數

D_values=("BM")    # 資料集
P_values=(2 4 8)                       # 參與方數
AM_values=("TIFS" "VILLIAN")           # 攻擊方法
alpha_values=(0.5 1 1.3 1.5 1.7 2 3)   # α 值
num_runs=1                             # 每組重跑次數
batch_size=128

# 以下還沒測試，圖還沒畫完
# D_values=("CIFAR10")    # 資料集
# P_values=(4)                       # 參與方數
# AM_values=("VILLIAN")           # 攻擊方法
# alpha_values=(0.5)   # α 值

# D_values=("CINIC10")    # 資料集
# P_values=(4)                       # 參與方數
# AM_values=("TIFS")           # 攻擊方法
# alpha_values=(1.3 1.7)   # α 值

# D_values=("CINIC10")    # 資料集
# P_values=(8)                       # 參與方數
# AM_values=("TIFS")           # 攻擊方法
# alpha_values=(1)   # α 值

# num_runs=1                             # 每組重跑次數

DM="SHAP_MAE"   # 防禦方法固定
AN=1
EPOCHS=80

########################
# 輸出資料夾
########################
mkdir -p alpha_test_BM

########################
# 主要迴圈
########################
for D in "${D_values[@]}"; do
  for P in "${P_values[@]}"; do
    for AM in "${AM_values[@]}"; do
      for alpha in "${alpha_values[@]}"; do

        logfile="alpha_test_BM/log_D${D}_P${P}_AM${AM}_alpha${alpha}.txt"

        echo "==================================================" | tee "$logfile"
        echo "Dataset=$D | P=$P | AM=$AM | alpha=$alpha | DM=$DM" | tee -a "$logfile"
        echo "==================================================" | tee -a "$logfile"

        for run in $(seq 1 $num_runs); do
          echo "▶ Run $run / $num_runs  ($(date '+%F %T'))" | tee -a "$logfile"

          python3 train_new.py \
            --D "$D" --P "$P" --epochs "$EPOCHS" \
            --DM "$DM" --AM "$AM" --AN "$AN" \
            --shap_alpha "$alpha" \
            --label 0 --batch_size "$batch_size"\
            2>&1 | tee -a "$logfile"

          echo "----------------------------------------------" | tee -a "$logfile"
        done

        echo "✅ Finished: D=$D P=$P AM=$AM alpha=$alpha" | tee -a "$logfile"
        echo "" | tee -a "$logfile"

      done
    done
  done
done
