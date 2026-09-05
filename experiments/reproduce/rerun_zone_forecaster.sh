#!/usr/bin/env bash
# Re-derives report.md §11's per-zone forecaster numbers on the corrected catalogue.
#
# Not new forecasting research -- these numbers are already published in the
# report (pooled AUC 0.733 vs the retired scalar's 0.723, block-level 0.62 EAFZ
# / 0.60 AEGEAN) and they predate the catalogue rebuild.
#
# The dataset bakes catalogue-derived features into its manifest (log_rate,
# b_value, mean_mag, days_since_prev_major), so retraining alone would not fix
# them -- the dataset has to be rebuilt per arm.
#
# --data-downloader-root is REQUIRED, not optional: without it block-level
# evaluation is silently skipped and only pooled window-level AUC is printed,
# which is inflated by the 8:1 overlap between windows (64 events, stride 8).
# The published 0.62/0.60 figures are block-level.
#
# Both arms use catalogues spanning 2000-2026, so only completeness differs.
# NOTE the published 0.733 came from deprem_katalog_utc.csv, which starts in
# 2010; it is therefore not span-matched to either arm here.
set -uo pipefail
cd "$(dirname "$0")/.."
DD=~/Projects/Sismokaos/data_downloader
CATS=$DD/catalogs
OLD=$CATS/archive_superseded_2026-08-30/data_large.csv
NEW=$CATS/catalog_current.csv
mkdir -p logs

for which in old new; do
    [ "$which" = old ] && cat=$OLD || cat=$NEW
    out=$DD/raw/data/dataset_catalog_forecast_${which}
    echo "=== build $which  $(date +%H:%M:%S) ==="
    if [ -f "$out/manifest.csv" ]; then
        echo "    exists, skipping build"
    else
        (cd "$DD" && uv run seismic-cli generate-catalog-forecast-dataset \
            --catalog-path "$cat" --output-dir "$out") \
            > "logs/zone_build_${which}.log" 2>&1 \
          && echo "    built" || { echo "    BUILD FAILED"; tail -3 "logs/zone_build_${which}.log"; continue; }
    fi
    for seed in 42 43 44; do
        log="logs/zone_forecast_${which}_seed${seed}.log"
        echo "=== train $which seed $seed  $(date +%H:%M:%S) ==="
        uv run python src/forecasting/cnn_lstm_forecast.py \
            --dataset-dir "$out" --catalog-path "$cat" --seed $seed \
            --data-downloader-root "$DD" > "$log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || grep -q "\[ERROR\]" "$log"; then
            echo "    FAILED (exit $rc)"; grep -m2 "\[ERROR\]" "$log" | sed 's/^/    /'
        else
            echo "    ok -> $log"
        fi
    done
done
echo "=== done $(date +%H:%M:%S) ==="
