#!/usr/bin/env bash
# STEAD recast into this project's P-only geometry, built by the SAME generator
# command as the Aegean corpus so nothing about the preprocessing can drift.
# Only --eq-dir/--noise-dir/--output-dir differ from build_ponly_variants.sh
# and build_ponly_natural.sh.
set -u
cd /home/hogib/Projects/Sismokaos/data_downloader
common=(--eq-dir raw/data/batched_waveforms/stead_ponly_3p4s
        --noise-dir raw/data/batched_noise_waveforms/stead_noise
        --window-seconds 3.4 --fs 100 --max --baseline
        --n-fft 64 --hop-length 16)

echo "=== [1/2] STEAD amplitude-matched ==="
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset "${common[@]}" \
    --hard-negatives --match-negative-amplitude \
    --output-dir dataset_specdual_stead_ponly_matched

echo "=== [2/2] STEAD natural ==="
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset "${common[@]}" \
    --no-hard-negatives \
    --output-dir dataset_specdual_stead_ponly_natural
echo "STEAD BUILDS DONE"
