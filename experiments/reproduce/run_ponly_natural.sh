#!/usr/bin/env bash
# P-only 3.4 s windows with NATURAL negatives -- no hard-negative mining, so
# noise follows the pool's own amplitude density. This is the regime a deployed
# station actually sees.
#
# The complement to run_ponly_matched.sh. Models trained on amplitude-matched
# negatives capture 61-65% of headroom on matched and band but only 12-16% on
# natural, adding ~+0.03 AUC over a plain loudness scalar there. Two things
# could produce that and they need separating: those models never saw the quiet
# noise that dominates natural (a train/test mismatch), and natural leaves less
# headroom to begin with (0.212 vs 0.332, so part of the drop is arithmetic).
# Training on natural and scoring all four regimes separates them.
#
# Sequential, not parallel: three concurrent runs have OOM'd this 3.68 GiB card.
set -u
D=/home/hogib/Projects/Sismokaos/data_downloader/dataset_specdual_ponly_3p4s_natural
S=trained_model_ponly_natural
mkdir -p "$S" logs
for ch in 1d 2d all; do
    echo "=== channels=$ch branch-1d=cnn-lstm ==="
    .venv/bin/python src/detection/cnn_lstm_classify.py \
        --dataset-dir "$D" --save-dir "$S" \
        --channels "$ch" --branch-1d cnn-lstm --fusion linear \
        --seq-transform asinh --batch-size 32 \
        --ensemble-seeds 42,43,44 \
        2>&1 | tee "logs/ponly_natural_${ch}_cnn-lstm.log"
done
echo "ALL NATURAL ARMS DONE"
