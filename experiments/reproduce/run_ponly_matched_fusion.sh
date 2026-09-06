#!/usr/bin/env bash
# Fusion arm on the amplitude-matched P-only set, completing the pair with the
# 1d/2d arms already run. Linear fusion, matching the 6 s hard-negative config
# where linear beat gated (0.9908 vs the gated arm's collapse on ozgun).
set -u
D=/home/hogib/Projects/Sismokaos/seismic_cli/dataset_specdual_ponly_3p4s_matched
S=trained_model_ponly_matched
.venv/bin/python src/sismokaos/detection/cnn_lstm_classify.py \
    --dataset-dir "$D" --save-dir "$S" \
    --channels all --branch-1d cnn-lstm --fusion linear \
    --seq-transform asinh --batch-size 32 \
    --ensemble-seeds 42,43,44 2>&1 | tee logs/ponly_matched_all_cnn-lstm.log
echo "FUSION DONE"
