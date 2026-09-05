#!/usr/bin/env bash
# Two negative-selection variants over the SAME P-only event windows, so the
# only thing that differs is which noise is kept.
#   matched  : noise amplitude distribution mirrors the events' (the fix)
#   wideband : band's lower edge removed only (the diagnostic)
# Sequential -- each build wants all cores.
set -u
cd /home/hogib/Projects/Sismokaos/data_downloader
common=(--eq-dir raw/data/batched_waveforms/window_post_3.4s_ponly
        --noise-dir raw/data/batched_noise_waveforms/noise_pre_3h
        --window-seconds 3.4 --fs 100 --max --baseline
        --hard-negatives --n-fft 64 --hop-length 16)

echo "=== [1/2] amplitude-matched ==="
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset "${common[@]}" \
    --match-negative-amplitude \
    --output-dir dataset_specdual_ponly_3p4s_matched
echo "=== [2/2] wideband diagnostic ==="
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset "${common[@]}" \
    --hard-negative-band 0.0 0.99 \
    --output-dir dataset_specdual_ponly_3p4s_wideband
echo "=== BOTH BUILDS DONE ==="
