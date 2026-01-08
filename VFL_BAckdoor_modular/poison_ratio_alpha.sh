#!/usr/bin/env bash
set -euo pipefail

#####################
# 參數區（可自行調整）
#####################

# 掃描的超參數軸
D_values=("CIFAR10" "CINIC10" "BM")          # 資料集
P_values=(8)                              # 參與者數
AM_values=("TIFS" "VILLIAN")                  # 攻擊方法
DM_values=("VFLIP" "SHAP_MAE" "ABL" "NEURAL_CLEANSE" "CoPur" "VFLMonitor")  # 防禦方法
PR_values=(0.01 0.03 0.05 0.07 0.1)           # 下毒比例 poison_ratio

# 其他訓練設定
num_runs=1                                    # 每個參數組合重複次數
epochs=80                                     # 預設總訓練輪數（非 BM 使用）
label=0                                       # 目標類別
AN_default=1                                  # 攻擊者數

# 日誌與彙整輸出位置（固定資料夾，沒有加時間戳）
folder="PR_sweep_logs"
mkdir -p "${folder}"

# 彙整 CSV（若不存在就加表頭）
summary_csv="${folder}/runs.csv"
if [[ ! -f "${summary_csv}" ]]; then
  echo "timestamp,dataset,P,AN,AM,DM,poison_ratio,epochs,label,gamma,logfile,ACC_base,ASR_base,ASRmask_base,ACC_def,ASR_def,ASRmask_def" \
    > "${summary_csv}"
fi

##########################################
# 輔助：依資料集與 P 選 ABL 相關的 gamma 值
##########################################
choose_gamma () {
  local D="$1"
  local P="$2"
  if [[ "$D" == "CIFAR10" ]]; then
    echo "0.5"
  elif [[ "$D" == "CINIC10" || "$P" -eq 8 ]]; then
    echo "0.1"
  else
    echo "0.3"
  fi
}

##########################################
# 主迴圈：枚舉所有組合，跑訓練 + 記錄
##########################################
for D in "${D_values[@]}"; do
  for P in "${P_values[@]}"; do

    # 依你的慣例設定攻擊者數（這裡簡化為固定 1）
    AN="${AN_default}"

    gamma="$(choose_gamma "$D" "$P")"

    # === 這裡是關鍵：針對 BM 覆寫 epochs 為 50，其餘沿用全域 epochs ===
    if [[ "$D" == "BM" ]]; then
      epochs_this=50
    else
      epochs_this="${epochs}"
    fi

    for AM in "${AM_values[@]}"; do
      for DM in "${DM_values[@]}"; do
        for PR in "${PR_values[@]}"; do

          logfile="${folder}/log_D${D}_P${P}_AN${AN}_AM${AM}_DM${DM}_PR${PR}.txt"

          {
            echo "==============================="
            echo "開始組合：D=${D}, P=${P}, AN=${AN}, AM=${AM}, DM=${DM}, PR=${PR}"
            echo "epochs=${epochs_this}, label=${label}, gamma=${gamma}"
            echo "時間：$(date '+%Y-%m-%d %H:%M:%S')"
            echo "==============================="
          } | tee "${logfile}"

          for ((run=1; run<=num_runs; run++)); do
            echo "🔵 Run $run：$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${logfile}"

            # 核心訓練指令：這裡改用 epochs_this
            python3 train_new.py \
              --D "${D}" --P "${P}" --epochs "${epochs_this}" \
              --AM "${AM}" --AN "${AN}" --DM "${DM}" --act True \
              --label "${label}" \
              --abl_gamma "${gamma}" --abl_turn_epoch 25 \
              --poison_ratio "${PR}" \
              2>&1 | tee -a "${logfile}"

            echo "--------------------------------------" | tee -a "${logfile}"
          done

          #############################################
          # 從 log 抓關鍵指標（抓不到就留空，不會中斷）
          #############################################
          # baseline（未啟用防禦）
          ACC_base="$(grep -F "未經防禦的情況下:" -A 3 "${logfile}" | grep -E -m1 "ACC 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASR_base="$(grep -F "未經防禦的情況下:" -A 3 "${logfile}" | grep -E -m1 "ASR 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASRmask_base="$(grep -F "未經防禦的情況下:" -A 5 "${logfile}" | grep -E -m1 "排除掉原先|ASR_mask" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"

          # defense（依 DM 擷取）
          ACC_def="$(grep -F "${DM} 防禦效果:" -A 3 "${logfile}" | grep -E -m1 "acc on 下毒資料集" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASR_def="$(grep -F "${DM} 防禦效果:" -A 5 "${logfile}" | grep -E -m1 "ASR on 下毒資料集" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASRmask_def="$(grep -F "${DM} 防禦效果:" -A 6 "${logfile}" | grep -E -m1 "排除掉原先" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"

          # 追加一筆到彙整 CSV（也改為寫入 epochs_this）
          printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$(date '+%Y-%m-%d %H:%M:%S')" \
            "${D}" "${P}" "${AN}" "${AM}" "${DM}" "${PR}" \
            "${epochs_this}" "${label}" "${gamma}" "${logfile}" \
            "${ACC_base}" "${ASR_base}" "${ASRmask_base}" \
            "${ACC_def}" "${ASR_def}" "${ASRmask_def}" \
            >> "${summary_csv}"

          echo "✅ 完成：D=${D}, P=${P}, AN=${AN}, AM=${AM}, DM=${DM}, PR=${PR}" | tee -a "${logfile}"
          echo "" | tee -a "${logfile}"

        done
      done
    done
  done
done