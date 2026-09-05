#!/usr/bin/env bash
# Quarter batch (first 181 of 723 days) of BODT chaotic features at 5 Hz.
# Step 0 of docs/HANDOFF_chaos_forecasting.md -- the run that was killed at 38%
# for a machine move that never happened, leaving nothing behind.
#
# The run is ALL-OR-NOTHING: FeatureWriter accumulates in memory and writes once
# at the end, so an interrupt produces zero output, not a partial parquet.
#
# The output filename is derived from the DATA-DIR basename, not the out-dir, so
# two runs over same-named input directories silently overwrite each other.
# Here: bodt_q1_chaos_5hz -> dataset_features_chaos_q1_5hz/bodt_q1_chaos_5hz_features.parquet
set -u
cd /home/hogib/Projects/Sismokaos/sismokaos-cli
./target/release/sismokaos-cli --config config_chaos_5hz.json \
    run --data-dir ./bodt_q1_chaos_5hz --out-dir ./dataset_features_chaos_q1_5hz
echo "CHAOS Q1 DONE"
