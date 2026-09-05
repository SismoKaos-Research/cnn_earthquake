#!/usr/bin/env bash
# Finish the normalised-dataset grid: the two arms killed on 2026-08-20.
#
#   1d_lstm       -- the reproduction control. Seed 42 landed at 0.9139
#                    (report: 0,9144) before the kill; seeds 43-44 never ran,
#                    and seed 43's checkpoint is a partial, so the arm is rerun
#                    clean rather than resumed. The old checkpoints are in
#                    trained_model_norm_stale_partial/.
#   all_cnn-lstm  -- fusion on amplitude-deleted data. Never started.
#
# Sequential: three concurrent --channels all runs OOMed this 3.68 GiB card.
# cnn and cnn-lstm are already complete (0.9146 and 0.9309) and are NOT rerun.
set -u
cd /home/hogib/Projects/model_cnn_lstm

DATA=/home/hogib/Projects/Sismokaos/data_downloader/raw/data/dataset_specdual_6s
SAVE=trained_model_norm_branch1d
mkdir -p "$SAVE" logs

run () {
  local name=$1; shift
  local LOG=logs/norm_${name}.log
  local CMD=(python3 src/detection/cnn_lstm_classify.py
             --dataset-dir "$DATA" --save-dir "$SAVE"
             --seq-transform asinh --ensemble-seeds 42,43,44 --num-workers 2 "$@")
  { echo "# ${CMD[*]}"; echo "# started $(date -Is)"; } > "$LOG"
  echo "running $name -> $LOG"
  "${CMD[@]}" >> "$LOG" 2>&1
  echo "   $name done $(date -Is)"
}

run 1d_lstm      --channels 1d  --branch-1d lstm
run all_cnn-lstm --channels all --fusion linear --branch-1d cnn-lstm
echo "NORMALISED GRID COMPLETE $(date -Is)"
