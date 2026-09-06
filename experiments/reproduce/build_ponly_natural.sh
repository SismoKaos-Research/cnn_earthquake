#!/usr/bin/env bash
# Natural-noise negatives over the SAME P-only event windows: no hard-negative
# mining at all, so noise is drawn evenly across each file and follows the
# pool's own amplitude density. This is the deployment-realistic negative
# regime -- the one a station actually sees -- and the third column of the
# transfer matrix alongside matched (amplitude-neutralised) and band (loud).
set -u
cd /home/hogib/Projects/Sismokaos/seismic_cli
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset \
  --eq-dir raw/data/batched_waveforms/window_post_3.4s_ponly \
  --noise-dir raw/data/batched_noise_waveforms/noise_pre_3h \
  --output-dir dataset_specdual_ponly_3p4s_natural \
  --window-seconds 3.4 --fs 100 --max --baseline \
  --no-hard-negatives \
  --n-fft 64 --hop-length 16
echo "NATURAL BUILD DONE"
