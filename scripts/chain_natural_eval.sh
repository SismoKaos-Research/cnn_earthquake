#!/usr/bin/env bash
# Waits for the natural grid, then scores it against all four negative regimes.
# Watches the runner's sentinel line rather than pgrep: matching a process by
# command-line substring also matches the watcher itself, which has stalled
# three chains today.
set -u
cd /home/hogib/Projects/model_cnn_lstm
until grep -q "ALL NATURAL ARMS DONE" logs/ponly_natural_all.log 2>/dev/null; do sleep 60; done
echo "=== grid finished, scoring four regimes ==="
B=/home/hogib/Projects/Sismokaos/data_downloader
.venv/bin/python src/detection/negative_regime_transfer.py \
    --ckpt-dir trained_model_ponly_natural \
    --datasets matched=$B/dataset_specdual_ponly_3p4s_matched \
               band=$B/dataset_specdual_ponly_3p4s_hard \
               wideband=$B/dataset_specdual_ponly_3p4s_wideband \
               natural=$B/dataset_specdual_ponly_3p4s_natural
echo "=== NATURAL TRANSFER MATRIX DONE ==="
