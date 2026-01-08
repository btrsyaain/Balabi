#!/usr/bin/env bash
set -euo pipefail

#####################
# 參數區（可自行調整）
#####################

D_values=("CIFAR10" "CINIC10" "BM")          # 資料集
P_values=(4)                                  # 參與者數
AM_values=("TIFS" "VILLIAN")                  # 攻擊方法
DM_values=("VFLIP" "SHAP_MAE" "ABL" "NEURAL_CLEANSE" "CoPur" "VFLMonitor")  # 防禦方法
PR_values=(0.01 0.03 0.05 0.07 0.1)           # 下毒比例 poison_ratio

num_runs=1
epochs=50                                     # 非 BM 使用
label=0
AN_default=1

# 斷點（含本組合一起跑；若你要換，就改這五個）
RESUME_FROM_D="BM"
RESUME_FROM_P=4
RESUME_FROM_AM="TIFS"
RESUME_FROM_DM="SHAP_MAE"
RESUME_FROM_PR="0.07"

# 輸出
folder="PR_sweep_logs"
mkdir -p "${folder}"

summary_csv="${folder}/runs.csv"
if [[ ! -f "${summary_csv}" ]]; then
  echo "timestamp,dataset,P,AN,AM,DM,poison_ratio,epochs,label,gamma,logfile,ACC_base,ASR_base,ASRmask_base,ACC_def,ASR_def,ASRmask_def" \
    > "${summary_csv}"
fi

##########################################
# ABL gamma 規則
##########################################
choose_gamma () {
  local D="$1"; local P="$2"
  if [[ "$D" == "CIFAR10" ]]; then
    echo "0.5"
  elif [[ "$D" == "CINIC10" || "$P" -eq 8 ]]; then
    echo "0.1"
  else
    echo "0.3"
  fi
}

##########################################
# 建立值→索引對照，供排序判斷
##########################################
declare -A IDX_D IDX_AM IDX_DM IDX_PR
for i in "${!D_values[@]}";   do IDX_D["${D_values[$i]}"]=$i; done
for i in "${!AM_values[@]}";  do IDX_AM["${AM_values[$i]}"]=$i; done
for i in "${!DM_values[@]}";  do IDX_DM["${DM_values[$i]}"]=$i; done
for i in "${!PR_values[@]}";  do IDX_PR["${PR_values[$i]}"]=$i; done

bp_d_idx=${IDX_D[$RESUME_FROM_D]:-}
bp_p=${RESUME_FROM_P}
bp_am_idx=${IDX_AM[$RESUME_FROM_AM]:-}
bp_dm_idx=${IDX_DM[$RESUME_FROM_DM]:-}
bp_pr_idx=${IDX_PR[$RESUME_FROM_PR]:-}

# 斷點防呆
if [[ -z "${bp_d_idx}" || -z "${bp_am_idx}" || -z "${bp_dm_idx}" || -z "${bp_pr_idx}" ]]; then
  echo "❌ 斷點設定不在你的列表中，請檢查 RESUME_FROM_* 與 D/AM/DM/PR 的陣列。" >&2
  exit 1
fi

##########################################
# 判斷 (D,P,AM,DM,PR) 是否「≥ 斷點」
# 依序比較：D → P → AM → DM → PR（字典序）
##########################################
tuple_ge_breakpoint () {
  local d_idx="$1" p="$2" am_idx="$3" dm_idx="$4" pr_idx="$5"
  if   (( d_idx > bp_d_idx )); then return 0
  elif (( d_idx < bp_d_idx )); then return 1
  fi
  if   (( p > bp_p )); then return 0
  elif (( p < bp_p )); then return 1
  fi
  if   (( am_idx > bp_am_idx )); then return 0
  elif (( am_idx < bp_am_idx )); then return 1
  fi
  if   (( dm_idx > bp_dm_idx )); then return 0
  elif (( dm_idx < bp_dm_idx )); then return 1
  fi
  if (( pr_idx >= bp_pr_idx )); then return 0; else return 1; fi
}

##########################################
# 主迴圈
##########################################
for D in "${D_values[@]}"; do
  for P in "${P_values[@]}"; do

    AN="${AN_default}"
    gamma="$(choose_gamma "$D" "$P")"
    if [[ "$D" == "BM" ]]; then epochs_this=50; else epochs_this="${epochs}"; fi

    for AM in "${AM_values[@]}"; do
      for DM in "${DM_values[@]}"; do
        for PR in "${PR_values[@]}"; do

          d_idx=${IDX_D[$D]}
          am_idx=${IDX_AM[$AM]}
          dm_idx=${IDX_DM[$DM]}
          pr_idx=${IDX_PR[$PR]}

          # 只跑「≥ 斷點」的組合（含斷點本身）
          if ! tuple_ge_breakpoint "$d_idx" "$P" "$am_idx" "$dm_idx" "$pr_idx"; then
            echo "⏭️  Skip (< breakpoint): D=${D}, P=${P}, AM=${AM}, DM=${DM}, PR=${PR}"
            continue
          fi

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

            # 關鍵：直接執行；因為 set -e -o pipefail，若 python 非 0 就會整個腳本停止
            python3 train_new.py \
              --D "${D}" --P "${P}" --epochs "${epochs_this}" \
              --AM "${AM}" --AN "${AN}" --DM "${DM}" --act True \
              --label "${label}" \
              --abl_gamma "${gamma}" --abl_turn_epoch 25 \
              --poison_ratio "${PR}" \
              2>&1 | tee -a "${logfile}"

            echo "--------------------------------------" | tee -a "${logfile}"
          done

          # 指標擷取（抓不到就留空；只有訓練成功才會到這裡）
          ACC_base="$(grep -F "未經防禦的情況下:" -A 3 "${logfile}" | grep -E -m1 "ACC 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASR_base="$(grep -F "未經防禦的情況下:" -A 3 "${logfile}" | grep -E -m1 "ASR 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASRmask_base="$(grep -F "未經防禦的情況下:" -A 5 "${logfile}" | grep -E -m1 "排除掉原先|ASR_mask" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"

          ACC_def="$(grep -F "防禦效果:" -A 3 "${logfile}" | grep -E -m1 "acc on 下毒資料集" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASR_def="$(grep -F "防禦效果:" -A 5 "${logfile}" | grep -E -m1 "ASR on 下毒資料集" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
          ASRmask_def="$(grep -F "防禦效果:" -A 6 "${logfile}" | grep -E -m1 "排除掉原先" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"

          printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$(date '+%Y-%m-%d %H:%M:%S')" \
            "${D}" "${P}" "${AN}" "${AM}" "${DM}" "${PR}" \
            "${epochs_this}" "${label}" "${gamma}" "${logfile}" \
            "${ACC_base}" "${ASR_base}" "${ASRmask_base}" \
            "${ACC_def}" "${ASR_def}" "${ASRmask_def}" \
            >> "${summary_csv}"

          echo "📝 已記錄：D=${D}, P=${P}, AN=${AN}, AM=${AM}, DM=${DM}, PR=${PR}" | tee -a "${logfile}"
          echo "" | tee -a "${logfile}"

        done
      done
    done
  done
done
