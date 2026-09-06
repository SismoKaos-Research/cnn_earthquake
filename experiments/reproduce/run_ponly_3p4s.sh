#!/usr/bin/env bash
# P-only (3.4 s = 2.0 s pre-P + 1.4 s post-P) detector grid.
# Sequential, not parallel: three concurrent runs have OOM'd this 3.68 GiB card.
set -u
D=/home/hogib/Projects/Sismokaos/seismic_cli/dataset_specdual_ponly_3p4s_hard
S=trained_model_ponly_3p4s
mkdir -p "$S" logs
for arm in "1d cnn-lstm" "2d cnn-lstm"; do
    set -- $arm
    ch=$1; br=$2
    echo "=== channels=$ch branch-1d=$br ==="
    .venv/bin/python src/sismokaos/detection/cnn_lstm_classify.py \
        --dataset-dir "$D" --save-dir "$S" \
        --channels "$ch" --branch-1d "$br" --fusion linear \
        --seq-transform asinh --batch-size 32 \
        --ensemble-seeds 42,43,44 \
        2>&1 | tee "logs/ponly_3p4s_${ch}_${br}.log"
done
echo "ALL ARMS DONE"
