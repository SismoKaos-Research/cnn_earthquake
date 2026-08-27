#!/usr/bin/env bash
# GPD (Ross et al. 2018) on this project's P-only test windows, scored against
# the same conditional amplitude floor the local models are measured against.
#
# Four configurations, because a single number here would hide the two choices
# the comparison actually rests on:
#   1  headline      -- GPD's own preprocessing, P on its prediction sample,
#                       with this repo's detector re-scored on the SAME rows
#   2  alignment     -- P shifted 60 samples off the prediction sample
#   3  preprocessing -- the corpus's 1-45 Hz bandpass instead of GPD's 2 Hz HP
#   4  natural       -- the deployment-realistic negative regime
#
# Sequential: the mseed rebuild is I/O bound and wants the page cache to itself.
set -u
B=/home/hogib/Projects/Sismokaos/data_downloader
S=src/detection/pretrained_picker_baseline.py
mkdir -p logs

echo "=== [1/4] headline: matched, GPD preprocessing, head-to-head ==="
.venv/bin/python $S --dataset-dir $B/dataset_specdual_ponly_3p4s_matched \
  --data-root $B --weights original \
  --local-ckpt-dir trained_model_ponly_matched 2>&1 | tee logs/picker_matched_headline.log

echo "=== [2/4] alignment sensitivity: P off the prediction sample ==="
.venv/bin/python $S --dataset-dir $B/dataset_specdual_ponly_3p4s_matched \
  --data-root $B --weights original --front-pad 60 2>&1 | tee logs/picker_matched_frontpad.log

echo "=== [3/4] preprocessing sensitivity: corpus band, not GPD's ==="
.venv/bin/python $S --dataset-dir $B/dataset_specdual_ponly_3p4s_matched \
  --data-root $B --weights original --preprocess pipeline 2>&1 | tee logs/picker_matched_pipeline.log

echo "=== [4/4] natural negatives ==="
.venv/bin/python $S --dataset-dir $B/dataset_specdual_ponly_3p4s_natural \
  --data-root $B --weights original \
  --local-ckpt-dir trained_model_ponly_natural 2>&1 | tee logs/picker_natural_headline.log

echo "PICKER BASELINE DONE"
