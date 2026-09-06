#!/usr/bin/env bash
# The new 1D architectures on the PER-WINDOW NORMALISED dataset.
#
# Why this dataset is the interesting one: per-window standardisation deletes
# absolute amplitude, so the `seq` abs-max floor collapses and whatever the 1D
# branch scores is **waveform shape**, not loudness. The report's row here is
# 0,9144 against a 0,9205 floor -- i.e. BELOW floor, the basis for the claim
# that the branch only re-learns the scalar it was denied.
#
# That row was measured with the evrisimsiz (no-conv) branch. If cnn-lstm
# clears the floor here, the claim does not survive: it would mean local
# waveform structure IS extractable once the architecture can reach it, and
# that the old result measured the branch's reach rather than the data's
# information content.
#
# This dataset scanned CLEAN for fp16 (max 21.6), so asinh is a no-op here --
# passed only to keep the pipeline identical to every other arm.
set -u
# Repo root from this script's own location, not an absolute path: the
# checkout moves and a hardcoded `cd` then lands nowhere.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WAIT_PID=${1:-}
if [[ -n "$WAIT_PID" ]]; then
  echo "waiting for ozgun driver pid $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "ozgun re-check finished at $(date -Is); starting normalised grid"
fi

DATA=/home/hogib/Projects/Sismokaos/seismic_cli/raw/data/dataset_specdual_6s
SAVE=trained_model_norm_branch1d
mkdir -p "$SAVE" logs

launch () {
  local name=$1; shift
  local LOG=logs/norm_${name}.log
  local CMD=(python3 src/sismokaos/detection/cnn_lstm_classify.py
             --dataset-dir "$DATA" --save-dir "$SAVE"
             --seq-transform asinh --ensemble-seeds 42,43,44 --num-workers 2 "$@")
  { echo "# ${CMD[*]}"; echo "# started $(date -Is)"; } > "$LOG"
  "${CMD[@]}" >> "$LOG" 2>&1 &
  echo "launched $name -> pid $! -> $LOG"
}

# The three 1D arms are small enough to share this 3.68 GiB GPU (measured
# ~2.1 GiB for three concurrent 1d runs). The fusion arm is NOT -- three
# concurrent --channels all runs OOMed earlier tonight -- so it goes after.
launch 1d_lstm     --channels 1d --branch-1d lstm
launch 1d_cnn-lstm --channels 1d --branch-1d cnn-lstm
launch 1d_cnn      --channels 1d --branch-1d cnn
wait
echo "normalised 1D arms done $(date -Is)"

launch all_cnn-lstm --channels all --fusion linear --branch-1d cnn-lstm
wait
echo "NORMALISED GRID COMPLETE $(date -Is)"
