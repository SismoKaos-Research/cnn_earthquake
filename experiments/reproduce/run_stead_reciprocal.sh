#!/usr/bin/env bash
# The other half of the GPD comparison: our detector on STEAD, against a floor
# computed on STEAD. A locally-trained model winning on local data is the
# expected result; this is the run that can complicate it.
#
# Both Aegean-trained arms are scored against all four corpora/regimes, so the
# cross-corpus drop is read next to the in-domain number rather than alone.
set -u
B=/home/hogib/Projects/Sismokaos/seismic_cli
SETS=(aegean_matched=$B/dataset_specdual_ponly_3p4s_matched
      aegean_natural=$B/dataset_specdual_ponly_3p4s_natural
      stead_matched=$B/dataset_specdual_stead_ponly_matched
      stead_natural=$B/dataset_specdual_stead_ponly_natural)
mkdir -p logs

for ckpt in trained_model_ponly_natural trained_model_ponly_matched; do
    echo "=== trained on: $ckpt ==="
    .venv/bin/python src/detection/negative_regime_transfer.py \
        --ckpt-dir $ckpt --datasets "${SETS[@]}" 2>&1 | tee "logs/stead_transfer_${ckpt}.log"
done

echo "=== GPD on the STEAD windows, same protocol as the Aegean run ==="
.venv/bin/python src/detection/pretrained_picker_baseline.py \
    --dataset-dir $B/dataset_specdual_stead_ponly_natural --data-root $B \
    --weights original,geofon \
    --local-ckpt-dir trained_model_ponly_natural 2>&1 | tee logs/picker_stead_natural.log

echo "STEAD RECIPROCAL DONE"
