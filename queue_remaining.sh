#!/usr/bin/env bash
# Baseline + 6 s detector scan for every station not yet done.
# Order unlocks the best pairs earliest: SEMS first because ELBA is already
# scored, so ELBA-SEMS (114 km, 357 joint days) is testable the moment it ends.
set -u

ARCHIVE=../tdvms/afad_raw
OUT=../archive_pipeline/out/stations
ARM='6s:6.0:trained_model_branch1d_asinh:cnn-lstm'

for STN in SEMS KAND BAND CMH KURT KIRK VIZE; do
    # Matching the date explicitly, not ${STN}_*.zip: SEMS carries nine
    # SEMS_<date>.dup<id>.zip redeliveries, byte-identical to chunks already
    # held. Scoring those would count the same nine spans twice and inflate
    # every per-day denominator downstream.
    zips="$ARCHIVE/$STN/${STN}_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].zip"
    n=$(ls $zips 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && { echo "== $STN: no chunks, skipping"; continue; }
    echo "== $STN: $n chunks  $(date '+%F %T')"

    mkdir -p "$OUT/$STN"
    if [ ! -f "$OUT/$STN/baseline.json" ]; then
        uv run sk falsealarm baseline --zips "$zips" \
            --out "$OUT/$STN/baseline.json" || { echo "!! $STN baseline failed"; continue; }
    fi

    # scan skips chunks it has already written, so re-running resumes.
    uv run sk falsealarm scan --zips "$zips" \
        --baseline-json "$OUT/$STN/baseline.json" \
        --arm "$ARM" --out-dir "$OUT/$STN/scores" \
        || echo "!! $STN scan failed"
    echo "== $STN done  $(date '+%F %T')"
done
