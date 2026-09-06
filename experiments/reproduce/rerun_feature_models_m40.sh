#!/usr/bin/env bash
# feature_lstm / feature_gru_tcn at an evaluable operating point.
#
# At the original M>=4.5 / 30 d these models cannot be scored: fold 1's
# validation split and fold 5's test split hold zero positives, so early
# stopping has no signal and two AUCs are undefined. See
# docs/experiment_neural_forecasters_2026-08-30.md.
#
# scripts/probe_forecast_horizons.py swept magnitude x horizon on label
# composition alone. M>=4.0 gives 5/5 evaluable folds at 3, 7 and 14 days and
# lifts independent episodes from 31 to 70. Lower thresholds overshoot: M>=3.0
# at 7 d has a base rate of 0.991, degenerate in the other direction.
#
# 14 d matches catalog_mlp's horizon so the two families are comparable.
set -uo pipefail
cd "$(dirname "$0")/.."
CAT=~/Projects/Sismokaos/seismic_cli/catalogs/catalog_current.csv
FEAT=~/Projects/Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy
mkdir -p logs

for s in feature_lstm_forecast feature_gru_tcn; do
    log="logs/${s}_m40_h14.log"
    echo "=== $s  M>=4.0 14d  $(date +%H:%M:%S) ==="
    uv run python "src/forecasting/$s.py" --features-csv "$FEAT" --catalog-path "$CAT" \
        --threshold 4.0 --horizon-days 14 --cv-folds 5 > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || grep -q "\[ERROR\]" "$log"; then
        echo "    FAILED (exit $rc)"; grep -m2 "\[ERROR\]" "$log" | sed 's/^/    /'
    else
        echo "    ok -> $log"
    fi
done
echo "=== done $(date +%H:%M:%S) ==="
