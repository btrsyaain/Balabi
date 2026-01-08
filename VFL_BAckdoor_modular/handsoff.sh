#!/usr/bin/env bash
set -e

# 定義參數
P_values=(2 4 8)
AM_values=("TIFS" "VILLIAN")
# AM_values=("VILLIAN")
# DM_values=("VFLIP" "ABL" "NEURAL_CLEANSE" "CoPur" "VFLMonitor")
DM_values=("VFLIP" "SHAP_MAE")
# D_values=("CIFAR10" "CINIC10")
D_values=("CIFAR10" "CINIC10")
num_runs=1  # 每組合執行次數
folder=PRF1

# 建立 logs 資料夾
mkdir -p "${folder}"

for D in "${D_values[@]}"; do
  for P in "${P_values[@]}"; do
    if [[ "$P" -eq 2 || "$P" -eq 4 ]]; then
      AN=1
    else
      AN=1
    fi

    if [ "$D" == "CIFAR10" ]; then
      gamma=0.5
    elif [[ "$D" == "CINIC10" || "$P" -eq 8 ]]; then
      gamma=0.1
    else
      gamma=0.3
    fi

    # if [ "$DM" == "VFLMonitor" ]; then
    #   center_loss_weight=0.003
    # else 
    #   center_loss_weight=0.0
    # fi
    
    for AM in "${AM_values[@]}"; do
      for DM in "${DM_values[@]}"; do
        logfile="${folder}/log_D${D}_P${P}_AN${AN}_AM${AM}_DM${DM}_WoP_7.txt"
        echo "===============================" | tee "$logfile"
        echo "開始組合：D=${D}, P=${P}, AN=${AN}, AM=${AM}, DM=${DM}" | tee -a "$logfile"
        echo "===============================" | tee -a "$logfile"
        
        for ((i=1; i<=num_runs; i++)); do
          echo "🔵 執行第 $i 次：$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logfile"
          python3 train_new.py \
            --D "$D" --P "$P" --epochs 80 --AM "$AM" --AN "$AN" --DM "$DM" --act True \
            --label 0  --abl_gamma "$gamma" --abl_turn_epoch 25 \
            2>&1 | tee -a "$logfile"
          echo "--------------------------------------" | tee -a "$logfile"
        done
        
        echo "✅ 完成組合：D=${D}, P=${P}, AN=${AN}, AM=${AM}, DM=${DM}" | tee -a "$logfile"
        echo "" | tee -a "$logfile"
      done
    done
  done
done

# --label 0  --abl_gamma "$gamma" --abl_turn_epoch 25 --center_loss_weight "$center_loss_weight"\