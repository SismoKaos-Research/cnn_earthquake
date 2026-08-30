#!/usr/bin/env bash
# Re-runs the neural forecasters against the corrected catalogue, paired.
#
# Each model is run twice with IDENTICAL settings and identical fixed seeds --
# once against the archived catalogue that produced the original numbers, once
# against the rebuilt one.
#
# The baseline is data_large.csv, NOT deprem_katalog_utc.csv. Every neural
# forecaster's documented usage pointed at data_large; only the chaos scripts
# used deprem_katalog_utc. This matters: deprem_katalog_utc starts 2010-01-01
# with zero rows before it, so running it under --catalog-span 2000-01-01 hands
# the model a decade-long hole at the front of the span and fold 1 collapses to
# AUC 0.31. Only the labels differ, so the difference is
# attributable. Fixed seeds matter here: the original headline used 5 *random*
# seeds, and per-seed AUC spread on this data is ~0.17, so an unpaired
# comparison would be dominated by seed noise rather than the label change.
set -uo pipefail
cd "$(dirname "$0")/.."

CATS=~/Projects/Sismokaos/data_downloader/catalogs
OLD=$CATS/archive_superseded_2026-08-30/data_large.csv
NEW=$CATS/catalog_current.csv
FE=~/Projects/Sismokaos/feature-extract/results
SEEDS=42,43,44,45,46
mkdir -p logs

# Optional stage filter: `rerun_neural_forecasters.sh feature_lstm feature_gru_tcn`
WANT="$*"

pair () {   # pair <name> <script> <args...>
    local name=$1 script=$2; shift 2
    if [ -n "$WANT" ] && ! printf '%s\n' $WANT | grep -qx "$name"; then
        echo "--- skipping $name ---"; return 0
    fi
    for which in old new; do
        local cat; [ "$which" = old ] && cat=$OLD || cat=$NEW
        local log="logs/neural_${name}_${which}.log"
        echo "=== $name / $which  $(date +%H:%M:%S) ==="
        uv run python "$script" --catalog-path "$cat" "$@" > "$log" 2>&1
        local rc=$?
        # Exit status alone is not enough: these scripts print [ERROR] and still
        # exit 0, which once let an empty-split run be recorded as a success.
        if [ $rc -ne 0 ] || grep -q "\[ERROR\]" "$log"; then
            echo "    FAILED (exit $rc)"; grep -m2 "\[ERROR\]" "$log" | sed 's/^/    /'
            tail -3 "$log" | sed 's/^/    /'
        else
            echo "    ok -> $log"
        fi
    done
}

# 1. The headline: catalog-only MLP, the 0.5886 configuration
pair catalog_mlp src/forecasting/cnn_lstm_catalog_waveform_fusion.py \
    --channels catalog --catalog-span 2000-01-01 2026-08-12 \
    --horizon-days 14 --cv-folds 2 --bg-min-mag 3.0 --batch-size 128 \
    --keep-features log1p_dsp mean_mag_30d cv_interevent_90d mag_deficit_90d \
    --ensemble-seeds $SEEDS

# 2/3. Hand-feature sequence models on the BODT continuous archive
pair feature_lstm src/forecasting/feature_lstm_forecast.py \
    --features-csv $FE/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy --cv-folds 5
pair feature_gru_tcn src/forecasting/feature_gru_tcn.py \
    --features-csv $FE/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy --cv-folds 5

echo "=== all done $(date +%H:%M:%S) ==="
