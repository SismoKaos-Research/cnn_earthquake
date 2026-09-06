#!/usr/bin/env bash
# Re-runs the chaos forecasting suite against the corrected catalogue.
#
# The original logs (logs/chaos_*.log, no suffix) were produced against
# deprem_katalog_utc.csv, which held 51 of 1,256 February 2025 events in the
# region. They are the comparison baseline, so this writes *_corrected.log
# beside them rather than over them.
#
# Sequential by design: 528-feature frames plus LightGBM on 12 cores and 11 GB
# free contend badly if these overlap.
set -uo pipefail
cd "$(dirname "$0")/.."

CAT=~/Projects/Sismokaos/seismic_cli/catalogs/catalog_current.csv
CLI=~/Projects/Sismokaos/sismokaos-cli/dataset_features_chaos_q1_5hz
BODT=$CLI/bodt_q1_chaos_5hz_features.parquet
DAT=$CLI/dat_q1_chaos_5hz_features.parquet
mkdir -p logs

run () {   # run <logname> <args...>
    local name=$1; shift
    echo "=== $name  $(date +%H:%M:%S) ==="
    if uv run python "$@" > "logs/${name}_corrected.log" 2>&1; then
        echo "    ok -> logs/${name}_corrected.log"
    else
        echo "    FAILED (exit $?) -- see logs/${name}_corrected.log"
        tail -5 "logs/${name}_corrected.log" | sed 's/^/    /'
    fi
}

run chaos_screen      src/forecasting/chaos_univariate_screen.py \
                          --parquet "$BODT" --catalog "$CAT"
run chaos_shape       src/forecasting/chaos_univariate_screen.py \
                          --parquet "$BODT" --catalog "$CAT" --shape
run chaos_lags        src/forecasting/chaos_univariate_screen.py \
                          --parquet "$BODT" --catalog "$CAT" --lags
run chaos_replication src/forecasting/chaos_station_replication.py \
                          --catalog "$CAT" \
                          --station "BODT=$BODT" --station "DAT=$DAT"
run chaos_forecast    src/forecasting/chaos_forecast.py \
                          --parquet "$BODT" --catalog "$CAT"
run chaos_config_sweep src/forecasting/chaos_config_sweep.py \
                          --parquet "$BODT" --catalog "$CAT" \
                          --out-csv logs/chaos_config_sweep_corrected.csv

echo "=== all done $(date +%H:%M:%S) ==="
