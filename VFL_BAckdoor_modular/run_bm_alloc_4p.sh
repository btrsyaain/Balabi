#!/usr/bin/env bash
set -euo pipefail

# 固定參數
D="BM"
P=4
AM="VILLIAN"        # 或 TIFS
DM="VFLIP"          # 或 SHAP_MAE
EPOCHS=80
AN=1
LABEL=0

ALLOC_MODES=("equal" "dur_to_att" "active_gets_dur")
NUM_RUNS=3

outdir="bm_alloc_runs_p4"
mkdir -p "$outdir"
summary="${outdir}/summary.csv"
if [[ ! -f "$summary" ]]; then
  echo "timestamp,alloc,run,log,ACC_base,ASR_base,ACC_def,ASR_def" > "$summary"
fi

for alloc in "${ALLOC_MODES[@]}"; do
  for run in $(seq 1 $NUM_RUNS); do
    log="${outdir}/log_alloc-${alloc}_run-${run}.txt"
    echo "===== alloc=${alloc} run=${run} ===== $(date '+%F %T')" | tee "$log"

    python3 train_new.py \
      --D "$D" --P "$P" --epochs "$EPOCHS" \
      --AM "$AM" --AN "$AN" --DM "$DM" --act True \
      --label "$LABEL" --bm_alloc "$alloc" \
      2>&1 | tee -a "$log"

    # 指標擷取（依你現有日誌關鍵字，可視需要調整）
    ACC_base="$(grep -F "未經防禦的情況下:" -A 3 "$log" | grep -E -m1 "ACC 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
    ASR_base="$(grep -F "未經防禦的情況下:" -A 3 "$log" | grep -E -m1 "ASR 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
    ACC_def="$(grep -F "防禦效果:" -A 3 "$log" | grep -E -m1 "acc on 下毒資料集" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"
    ASR_def="$(grep -F "防禦效果:" -A 5 "$log" | grep -E -m1 "排除掉原先 label 即為 0 之資料後，ASR 為" | sed -E 's/.*[:： ]([0-9.]+).*/\1/' || true)"

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$(date '+%F %T')" "$alloc" "$run" "$log" \
      "${ACC_base}" "${ASR_base}" "${ACC_def}" "${ASR_def}" \
      >> "$summary"

    echo "📝 記錄完成 → $summary"
    echo
  done
done
