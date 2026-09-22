#!/usr/bin/env bash
# Coincidence tables for the pairs that span the separation range.
# CPU/IO only -- reads scored .npz, decodes nothing -- so it can run alongside
# a scan.
set -u
OUT=../archive_pipeline/out
S=$OUT/stations

pair () {   # $1=A $2=B
    d="$OUT/pairs/$1-$2"
    mkdir -p "$d"
    [ -f "$d/6s_coincidence.csv" ] && { echo "== $1-$2 already done"; return; }
    echo "== $1-$2  $(date '+%T')"
    uv run sk falsealarm coincidence \
        --scores-a "$S/$1/scores/6s/*.npz" --station-a "$1" \
        --scores-b "$S/$2/scores/6s/*.npz" --station-b "$2" \
        --stations-csv catalogs/istasyon_katalog.csv \
        --catalog catalogs/catalog_current.csv \
        --window-seconds 6.0 --out-prefix "$d/6s" \
        || echo "!! $1-$2 failed"
}

# Ordered by separation, to trace excess coincidence against distance.
pair BAND CMH      #  39.8 km
pair SEMS KAND     #  45.4 km
pair SEMS KURT     #  54.5 km
pair KIRK VIZE     #  65.7 km
pair ELBA VIZE     #  68.8 km
pair ELBA KURT     #  90.1 km
pair ELBA BAND     #  92.3 km
pair ELBA SEMS     # 114.0 km
pair ELBA CMH      # 132.1 km
pair CMH  MANT     # 176.5 km
