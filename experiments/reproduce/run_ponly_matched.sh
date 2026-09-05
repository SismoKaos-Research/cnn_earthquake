#!/usr/bin/env bash
# P-only 3.4 s windows (2.0 s pre-P + 1.4 s post-P) with AMPLITUDE-MATCHED
# negatives.
#
# This is the build whose floor is honest. Under the default 75-99 percentile
# band the negatives carry a hard amplitude floor that the positives lack, so
# P(event|amplitude) came out U-shaped -- 0.67 in the quietest decile -- and a
# monotone ROC-AUC floor read 0.6447 when a single-feature tree on the same
# scalar reached 0.7461. Matching the noise amplitude DISTRIBUTION to the
# events' flattens that: quietest decile 0.399, and the monotone/non-monotone
# gap closes to -0.0021.
#
# Sequential, not parallel: three concurrent runs have OOM'd this 3.68 GiB card.
set -u
D=/home/hogib/Projects/Sismokaos/seismic_cli/dataset_specdual_ponly_3p4s_matched
S=trained_model_ponly_matched
mkdir -p "$S" logs
for ch in 1d 2d; do
    echo "=== channels=$ch branch-1d=cnn-lstm ==="
    .venv/bin/python src/detection/cnn_lstm_classify.py \
        --dataset-dir "$D" --save-dir "$S" \
        --channels "$ch" --branch-1d cnn-lstm --fusion linear \
        --seq-transform asinh --batch-size 32 \
        --ensemble-seeds 42,43,44 \
        2>&1 | tee "logs/ponly_matched_${ch}_cnn-lstm.log"
done
echo "ALL ARMS DONE"
