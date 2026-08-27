#!/usr/bin/env bash
# DAT quarter batch, date-matched to BODT's 2024-05-01..2024-10-28 window.
#
# NOT a sample-size increase. DAT sits 43.8 km from BODT, so at the 400 km label
# radius the two see 95.3% of the same events -- DAT adds 3 to BODT's 232. This
# is a REPLICATION CHECK and a source of cross-station features, and it should
# not be described as more data.
#
# Config is identical to the BODT run (5 Hz, chaos_tau 5, step 50 s) so the two
# feature sets are directly comparable. Only --station differs.
#
# All-or-nothing: FeatureWriter accumulates in memory and writes once at the end.
# The output filename comes from the DATA-DIR basename, hence dat_q1_chaos_5hz.
set -u
cd /home/hogib/Projects/sismokaos-cli
./target/release/sismokaos-cli --config config_chaos_5hz.json \
    run --data-dir ./dat_q1_chaos_5hz --out-dir ./dataset_features_chaos_q1_5hz \
    --station DAT
echo "CHAOS DAT Q1 DONE"
