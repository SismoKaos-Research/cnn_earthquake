"""Does any single chaos feature carry marginal association with the label?

Run before writing training code, not after a model disappoints. If no
individual feature ranks the label better than the persistence floor, a model
combining them is unlikely to, and the screen costs minutes against a day.

**Every feature is scored with `oriented=True`.** These are candidate baselines,
not trained models: a feature anti-correlated with the label is exactly as
exploitable as a correlated one, so 0.43 is a score of 0.57. Reporting the raw
value would understate half the features and, worse, understate them
asymmetrically.

**The bar is the persistence floor, not 0.5.** On the chosen cell (M>=2.5,
400 km, 6 h) the trivial "an event happened recently" rule scores ~0.54, and
0.5 is not a bar any forecaster has to clear.

**Multiple comparisons are the obvious objection and are handled by reporting,
not by a correction.** Screening ~536 aggregated columns against one label will
produce apparent winners from noise alone; the expected best-of-N under the null
is printed beside the observed best so the two can be compared directly.

Usage:
    python3 src/sismokaos/forecasting/chaos_univariate_screen.py \\
        --parquet ~/Projects/Sismokaos/sismokaos-cli/.../bodt_q1_chaos_5hz_features.parquet \\
        --catalog ~/Projects/Sismokaos/seismic_cli/catalogs/catalog_current.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from sismokaos.forecasting.chaos_dataset import (HORIZON_HOURS, MAGNITUDE_BANDS,
                                       MIN_MAGNITUDE, RADIUS_KM,
                                       build, persistence_scores)
from sismokaos.metrics import safe_auc


def resolve_bands(args):
    """The band tuple this run should use, or None for the flat label.

    Kept out of the default so every published number stays reproducible: the
    graded scheme changes which events are positive AND what the persistence
    floor is, so a silent switch would move both sides of every comparison.
    """
    if getattr(args, "band_spec", None):
        return tuple(tuple(float(x) for x in part.split(":"))
                     for part in args.band_spec.split(","))
    return MAGNITUDE_BANDS if getattr(args, "bands", False) else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--horizon-hours", type=float, default=HORIZON_HOURS)
    p.add_argument("--shape", action="store_true",
                   help="Add within-hour slope/half-difference/argmax/lag-1 autocorrelation. Tests whether the trajectory inside an hour carries anything the four summary statistics discard.")
    p.add_argument("--lags", action="store_true",
                   help="Add lagged levels and deltas (1/3/6/12/24 h) for the top-scoring columns. Stands in for the recurrent half of a CNN+LSTM: whether the trajectory ACROSS hours carries anything the current hour does not.")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--permutations", type=int, default=200,
                   help="Null draws for the best-of-N reference. 0 skips it.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bands", action="store_true",
                   help="distance-graded magnitude label (MAGNITUDE_BANDS) instead "
                        "of the flat M>=%.1f / %.0f km one. Moves the positive class "
                        "and the persistence floor together." % (MIN_MAGNITUDE, RADIUS_KM))
    p.add_argument("--band-spec", default=None,
                   help="custom bands, 'radius:mag,...' e.g. "
                        "'100:2.0,300:3.0,500:5.0,1000:6.0'. Implies --bands.")
    return p.parse_args()


def best_under_null(y, ranks, n_draws, rng):
    """Expected best oriented AUC when the label is scrambled.

    Two choices here decide whether the bar means anything.

    **The REAL features are scored, not random ones.** Chaos features are
    heavily autocorrelated; white noise is not, and two autocorrelated series
    agree spuriously far more often than one does with noise. Permuting the
    label and re-scoring the actual columns keeps that structure on both sides.

    **Labels are shuffled in contiguous DAY-LONG blocks.** One event marks six
    consecutive positive hours, so an element-wise shuffle destroys the label's
    own autocorrelation and produces a null far tighter than reality.

    Ranks are computed once outside this function: permuting the label does not
    change a feature's ranking, so each draw is a sum over the positive rows
    rather than a fresh AUC, which is what makes a real permutation test
    affordable over ~500 columns.
    """
    n = len(y)
    block = 24
    n1 = int(y.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return np.asarray([])
    const = n1 * (n1 + 1) / 2.0
    best = []
    for _ in range(n_draws):
        blocks = [y[i:i + block] for i in range(0, n, block)]
        rng.shuffle(blocks)
        ys = np.concatenate(blocks)[:n]
        if ys.sum() == 0 or ys.sum() == n:
            continue
        k1 = int(ys.sum())
        auc = (ranks[:, ys == 1].sum(axis=1) - k1 * (k1 + 1) / 2.0) / (k1 * (n - k1))
        best.append(float(np.nanmax(np.maximum(auc, 1.0 - auc))))
    return np.asarray(best)


def main():
    args = parse_args()
    bands = resolve_bands(args)
    feats, y, dsp, idx = build(os.path.expanduser(args.parquet),
                               os.path.expanduser(args.catalog),
                               horizon_hours=args.horizon_hours, shape=args.shape,
                               lags=args.lags, bands=bands)
    floor = safe_auc(y, persistence_scores(dsp), oriented=True)

    cell = (f"M>={MIN_MAGNITUDE}, {RADIUS_KM:g} km" if bands is None
            else "graded " + " ".join(f"<{r:g}km:M>={m:g}" for r, m in bands))
    print(f"cell        {cell}, {args.horizon_hours:g} h")
    print(f"hours       {len(y):,}  ({idx[0]} .. {idx[-1]})")
    print(f"positives   {int(y.sum()):,} ({y.mean():.1%})")
    print(f"features    {feats.shape[1]:,} aggregated columns")
    print(f"floor       {floor:.4f}  (persistence: days since previous qualifying event)\n")

    rows = []
    for c in feats.columns:
        v = feats[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 100 or len(np.unique(y[ok])) < 2 or np.nanstd(v[ok]) == 0:
            continue
        rows.append((c, safe_auc(y[ok], v[ok], oriented=True), int(ok.sum())))
    res = pd.DataFrame(rows, columns=["feature", "auc", "n"]).sort_values(
        "auc", ascending=False).reset_index(drop=True)

    print(f"{'feature':38}{'oriented AUC':>14}{'vs floor':>10}")
    for _, r in res.head(args.top).iterrows():
        print(f"{r.feature:38}{r.auc:>14.4f}{r.auc - floor:>+10.4f}")

    over = int((res.auc > floor).sum())
    print(f"\n{over} of {len(res)} features beat the floor "
          f"({over / max(1, len(res)):.1%}); best {res.auc.iloc[0]:.4f} "
          f"({res.auc.iloc[0] - floor:+.4f})")

    if args.permutations:
        rng = np.random.default_rng(args.seed)
        # Rank the real columns once; NaNs are ranked last, consistently.
        mat = feats[res.feature.tolist()].to_numpy(dtype=float).T
        med = np.nanmedian(mat, axis=1, keepdims=True)
        mat = np.where(np.isfinite(mat), mat, med)
        ranks = np.apply_along_axis(
            lambda v: pd.Series(v).rank().to_numpy(), 1, mat)
        null = best_under_null(y, ranks, args.permutations, rng)
        if len(null):
            print(f"\nbest-of-{len(res)} under a block-shuffled null: "
                  f"median {np.median(null):.4f}, 95th pct {np.quantile(null, 0.95):.4f} "
                  f"({len(null)} draws)")
            verdict = ("ABOVE the null's 95th percentile -- worth modelling"
                       if res.auc.iloc[0] > np.quantile(null, 0.95)
                       else "WITHIN what screening this many columns produces from noise")
            print(f"observed best {res.auc.iloc[0]:.4f} is {verdict}")


if __name__ == "__main__":
    main()
