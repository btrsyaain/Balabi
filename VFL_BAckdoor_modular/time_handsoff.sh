#!/usr/bin/env bash
set -e

P_values=(2 4 8)
# DM_values=("VFLIP" "SHAP_MAE")
DM_values=("ABL" "NEURAL_CLEANSE" "CoPur" "VFLMonitor")
D_values=("CIFAR10" "CINIC10" "BM")
AM_value="VILLIAN"
num_runs=1
folder="timing_logs"

mkdir -p "${folder}"
summary_csv="timing_summary.csv"
echo "phase,method,P,dataset,total_model_s,total_mae_s,total_shapley_s,avg_model_ms,avg_mae_ms,avg_shapley_ms" > "$summary_csv"

extract_timing() {
    local json_file="$1"
    local phase="$2"
    local method="$3"
    local p="$4"
    local dataset="$5"
    if [ -f "$json_file" ]; then
        model=$(jq '.total_model_s' "$json_file")
        mae=$(jq '.total_mae_s' "$json_file")
        shapley=$(jq '.total_shapley_s' "$json_file")
        avg_model=$(jq '.avg_model_ms_per_sample' "$json_file")
        avg_mae=$(jq '.avg_mae_ms_per_sample' "$json_file")
        avg_shapley=$(jq '.avg_shapley_ms_per_sample' "$json_file")
        echo "$phase,$method,$p,$dataset,$model,$mae,$shapley,$avg_model,$avg_mae,$avg_shapley" >> "$summary_csv"
    else
        echo "⚠️ 找不到 $json_file"
    fi
}

for D in "${D_values[@]}"; do
  for P in "${P_values[@]}"; do
    AN=1
    if [ "$D" == "CIFAR10" ]; then
      gamma=0.5
    elif [[ "$D" == "CINIC10" || "$P" -eq 8 ]]; then
      gamma=0.1
    else
      gamma=0.3
    fi

    for DM in "${DM_values[@]}"; do
      logfile="${folder}/log_D${D}_P${P}_AM${AM_value}_DM${DM}.txt"
      echo "===============================" | tee "$logfile"
      echo "開始組合：D=${D}, P=${P}, AM=${AM_value}, DM=${DM}" | tee -a "$logfile"
      echo "===============================" | tee -a "$logfile"

      for ((i=1; i<=num_runs; i++)); do
        echo "🔵 執行第 $i 次：$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logfile"

        python3 train_new.py \
          --D "$D" --P "$P" --epochs 80 --AM "$AM_value" --AN "$AN" --DM "$DM" --act True \
          --label 0 --abl_gamma "$gamma" --abl_turn_epoch 25 \
          2>&1 | tee -a "$logfile"

        extract_timing "timings/train_overall_${DM}_P${P}.json" "train" "$DM" "$P" "$D"
        extract_timing "timings/eval_clean_${DM}_P${P}.json" "eval_clean" "$DM" "$P" "$D"
        extract_timing "timings/eval_poisoned_${DM}_P${P}.json" "eval_poisoned" "$DM" "$P" "$D"

        echo "--------------------------------------" | tee -a "$logfile"
      done
    done
  done
done
