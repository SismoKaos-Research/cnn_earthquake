"""Does anything found at one station reproduce at another?

The BODT screen turned up a best feature at 0.5726 against a 0.5503 floor --
marginally past a permutation null, and the model then failed to reproduce it
out of sample. A second station settles which of those two readings is right,
and it settles it in a way no amount of re-analysis at BODT can.

**A failed replication is the stronger result here.** "You only looked at one
station" is the first objection anyone raises against a single-station null, and
answering it in advance is worth more than another architecture.

**Each station gets its OWN label**, computed from its own coordinates, so this
compares two independent measurements rather than two views of one arbitrary
point. At 400 km the two event sets overlap 95%, so the label is nearly shared
by construction -- what genuinely differs is the waveform, which is the thing
under test.

**Rank correlation is the statistic, not the top-1 feature.** With ~500 columns,
the single best feature at either station is a draw from a wide distribution;
whether the whole ranking agrees is the question that has an answer.

Usage:
    python3 src/sismokaos/forecasting/chaos_station_replication.py \\
        --catalog ~/Projects/Sismokaos/seismic_cli/catalogs/catalog_current.csv \\
        --station BODT=~/Projects/Sismokaos/sismokaos-cli/dataset_features_chaos_q1_5hz/bodt_q1_chaos_5hz_features.parquet \\
        --station DAT=~/Projects/Sismokaos/sismokaos-cli/dataset_features_chaos_q1_5hz/dat_q1_chaos_5hz_features.parquet
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sismokaos.forecasting.chaos_dataset import (HORIZON_HOURS, MIN_MAGNITUDE, RADIUS_KM,
                                       build, persistence_scores)
from sismokaos.metrics import safe_auc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--station", action="append", required=True, metavar="NAME=PARQUET",
                   help="Repeatable. NAME must be a key in STATION_COORDS.")
    p.add_argument("--horizon-hours", type=float, default=HORIZON_HOURS)
    p.add_argument("--top", type=int, default=15)
    return p.parse_args()


def screen_one(name, parquet, catalog, horizon_hours):
    """Per-feature oriented AUC at one station, against its own local label.

    Returns:
        Tuple of (Series of AUC indexed by feature, floor, n_hours, positive rate).
    """
    feats, y, dsp, idx = build(os.path.expanduser(parquet), os.path.expanduser(catalog),
                               horizon_hours=horizon_hours, station=name)
    floor = safe_auc(y, persistence_scores(dsp), oriented=True)
    out = {}
    for c in feats.columns:
        v = feats[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 100 or len(np.unique(y[ok])) < 2 or np.nanstd(v[ok]) == 0:
            continue
        out[c] = safe_auc(y[ok], v[ok], oriented=True)
    return pd.Series(out).sort_values(ascending=False), floor, len(y), float(y.mean())


def main():
    args = parse_args()
    res = {}
    for spec in args.station:
        name, parquet = spec.split("=", 1)
        aucs, floor, n, pos = screen_one(name, parquet, args.catalog, args.horizon_hours)
        res[name] = (aucs, floor)
        print(f"{name:6} {n:,} hours  {pos:.1%} positive  floor {floor:.4f}  "
              f"best {aucs.iloc[0]:.4f} ({aucs.index[0]})")

    names = list(res)
    if len(names) < 2:
        raise SystemExit("need at least two stations to compare")

    a, b = names[0], names[1]
    (aa, fa), (bb, fb) = res[a], res[b]
    common = aa.index.intersection(bb.index)
    print(f"\n{len(common):,} features scored at both stations")

    rho = spearmanr(aa[common], bb[common]).statistic
    print(f"\nSpearman rank correlation of the two AUC vectors: {rho:+.4f}")
    print("  near 0  -> the rankings are unrelated; each station's leaders are noise")
    print("  high    -> the same features lead at both; something is being measured")

    # Does the first station's shortlist survive at the second?
    top = aa.head(args.top).index
    print(f"\n{a}'s top {args.top}, and where they land at {b}:")
    print(f"{'feature':36}{a + ' AUC':>12}{b + ' AUC':>12}{'  ' + b + ' rank':>14}")
    order = {c: i + 1 for i, c in enumerate(bb.index)}
    for c in top:
        if c not in bb.index:
            continue
        print(f"{c:36}{aa[c]:>12.4f}{bb[c]:>12.4f}{order[c]:>12,} /{len(bb):,}")

    held = sum(1 for c in top if c in bb.index and bb[c] > fb)
    print(f"\n{held} of {len(top)} of {a}'s leaders also beat {b}'s own floor ({fb:.4f})")
    # Under a null, a feature drawn at random clears the floor at the base rate.
    base = float((bb > fb).mean())
    print(f"base rate at {b}: {base:.1%} of all features clear it, so ~{base * len(top):.1f} "
          f"of {len(top)} would be expected by chance")


if __name__ == "__main__":
    main()
