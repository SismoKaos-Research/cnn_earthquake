"""Builds an hourly catalog-feature table over an arbitrary span, no waveform.

`build_offline_features.py` derives its hourly index from a preprocessed
waveform Parquet, so every table it writes is confined to the seismometer
archive's own window -- 2024-05 .. 2026-03 for the Aegean BODT stream, which
holds 34 M>=4.5 events out of the catalog's 261. The catalog branch never reads
the waveform, so that confinement buys nothing and costs an order of magnitude
in events. This script cuts the tie: the hourly index comes from `--start` and
`--end`, and the catalog alone fills it.

The output is a Parquet with a `DatetimeIndex`, which is what
`sismokaos.rust_io.RustData.open(features=...)` accepts, so it drops straight
into `forecasting/gru_cnn_train.py` in place of the 114-column table.

Columns are `catalog_feature_processor.build_catalog_features`' rolling
7/30/90-day block, plus `log1p_dsp` -- log1p of the days elapsed since the
previous M>=`--threshold` event. That last one is the quantity the persistence
floor ranks by, and no table built by this repo carried it before: the model
was being asked to beat persistence while blind to it. Hours before the
catalog's first qualifying event have no previous event; they take
`dsp = 3650` days, matching the convention in
`forecasting/cnn_lstm_catalog_waveform_fusion.py`.

    python3 src/sismokaos/features/build_catalog_span_features.py \\
        --catalog-path ../seismic_cli/catalogs/catalog_current.csv \\
        --start 2000-01-01 --end 2026-08-12 \\
        --out-path catalog_features_2000_2026.parquet

`--mc` sets the completeness threshold for the rolling statistics (2.5 by
default, as in the 114d table) and is deliberately separate from
`--threshold`, which defines the rare events that drive labels, persistence and
`log1p_dsp`. b-value estimation needs far more events than M>=4.5 alone
supplies.
"""

import argparse

import numpy as np
import pandas as pd

from sismokaos.features.catalog_feature_processor import build_catalog_features
from sismokaos.catalog import AEGEAN_BBOX, days_since_prev_major, load_aegean_events


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-path", required=True,
                   help="Event catalog CSV (data_large.csv format).")
    p.add_argument("--out-path", required=True, help="Parquet to write.")
    p.add_argument("--start", required=True, help="First hour, e.g. 2000-01-01.")
    p.add_argument("--end", required=True, help="Last hour, e.g. 2026-08-12.")
    p.add_argument("--threshold", type=float, default=4.5,
                   help="Magnitude defining a qualifying event, for log1p_dsp. "
                        "Match whatever gru_cnn_train.py will label with.")
    p.add_argument("--mc", type=float, default=2.5,
                   help="Completeness threshold for the rolling statistics.")
    p.add_argument("--no-dsp", action="store_true",
                   help="Omit log1p_dsp, for an ablation against the table that "
                        "lacks it.")
    return p.parse_args()


def load_catalog_frame(catalog_path, mc):
    """Aegean events at or above `mc`, as the frame `build_catalog_features` wants.

    `sismokaos.catalog`'s loaders return arrays rather than a DataFrame with a
    `dt` column, so the bounding box and date parsing are repeated here against
    the same constant rather than reshaping their output.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    keep = (cat.Latitude.between(lat0, lat1) & cat.Longitude.between(lon0, lon1)
            & (cat.Magnitude >= mc) & cat.dt.notna())
    return cat[keep].sort_values("dt").reset_index(drop=True)


def main():
    args = parse_args()

    hours = pd.date_range(args.start, args.end, freq="h")
    cat = load_catalog_frame(args.catalog_path, args.mc)
    major = load_aegean_events(args.catalog_path, args.threshold)

    print("=" * 66)
    print(f"catalog span features | {hours[0]} .. {hours[-1]}  ({len(hours)} hours)")
    print(f"  background M>={args.mc}: {len(cat)} events "
          f"({cat.dt.iloc[0]} .. {cat.dt.iloc[-1]})")
    print(f"  qualifying M>={args.threshold}: {len(major)} events")
    in_span = ((major >= np.datetime64(hours[0])) & (major <= np.datetime64(hours[-1]))).sum()
    print(f"    of which inside the span: {in_span}")
    print("=" * 66)

    if in_span == 0:
        raise SystemExit(
            f"No M>={args.threshold} events fall inside {args.start}..{args.end}; "
            f"there is nothing for labels or log1p_dsp to key off.")

    print("building rolling 7/30/90d features ...")
    feats = build_catalog_features(cat, hours, Mc=args.mc)

    if not args.no_dsp:
        print("building log1p_dsp ...")
        dsp = days_since_prev_major(hours, major)
        n_before = int(np.isnan(dsp).sum())
        # 3650 days stands in for "no previous event on record", the same
        # convention the fusion script uses, so the two are comparable.
        feats["log1p_dsp"] = np.log1p(np.nan_to_num(dsp, nan=3650.0)).astype(np.float32)
        if n_before:
            print(f"  {n_before} hour(s) precede the first M>={args.threshold} event "
                  f"and take dsp=3650 d")

    feats.to_parquet(args.out_path)
    print(f"\nwrote {args.out_path}  {feats.shape[0]} rows x {feats.shape[1]} cols")
    print(f"  columns: {', '.join(feats.columns)}")


if __name__ == "__main__":
    main()
