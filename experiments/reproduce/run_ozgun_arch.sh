#!/usr/bin/env bash
# The new 1D architectures on the AMPLITUDE-PRESERVED Ozgun dataset.
#
# The Ozgun re-check only ran --branch-1d lstm, so the report's 1B row there
# (0,9443, i.e. -0,0018 BELOW the 0,9461 seq abs-max floor) still reflects the
# evrisimsiz branch alone. That row is what carries finding (i).
#
# On every other dataset measured today cnn-lstm beat plain lstm:
#   hard negatives      0,9896 vs 0,9883   (+0,0013)
#   per-window norm.    0,9309 vs 0,9165   (+0,0144)
# If it also clears 0,9461 here, finding (i) fails on the amplitude-PRESERVED
# set too, not just the amplitude-deleted one -- a strictly stronger result.
#
# Sequential: 1d arms are small but the GPU is 3.68 GiB and norm_2d may still
# be finishing when this starts.
set -u
# Repo root from this script's own location, not an absolute path: the
# checkout moves and a hardcoded `cd` then lands nowhere.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WAIT_PID=${1:-}
if [[ -n "$WAIT_PID" ]]; then
  echo "waiting for pid $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

DATA=/home/hogib/Projects/Sismokaos/seismic_cli/raw/data/dataset_specdual_6s_baseline
SAVE=trained_model_ozgun_arch
mkdir -p "$SAVE" logs

run () {
  local name=$1; shift
  local LOG=logs/ozgun_arch_${name}.log
  local CMD=(python3 src/detection/cnn_lstm_classify.py
             --dataset-dir "$DATA" --save-dir "$SAVE"
             --seq-transform asinh --ensemble-seeds 42,43,44 --num-workers 2 "$@")
  { echo "# ${CMD[*]}"; echo "# started $(date -Is)"; } > "$LOG"
  echo "running $name -> $LOG"
  "${CMD[@]}" >> "$LOG" 2>&1
  echo "   $name done $(date -Is)"
}

run 1d_cnn-lstm --channels 1d --branch-1d cnn-lstm
run 1d_cnn      --channels 1d --branch-1d cnn
echo "OZGUN ARCH COMPLETE $(date -Is)"
