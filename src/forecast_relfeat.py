"""
Phase B: zone-relative (anomaly) features.

**The problem being fixed.** `catalog_report.md` 4.3 diagnosed it and never
acted on it: a single pooled model "must represent two opposite temporal
dependences at once." The features are all ABSOLUTE -- `log_rate`,
`log_total_energy`, `max_mag` -- but baseline seismicity differs several-fold
between zones (AEGEAN carries 73,703 events, CENTRAL 16,841). So a pooled model
partly learns WHICH ZONE THIS IS rather than whether this zone is currently
unusual FOR ITSELF, which is the thing a precursor signal would actually be.

The fix is a trailing z-score: express every feature relative to that zone's own
history up to (and excluding) the current window.

    rel_f = (f - trailing_mean_f) / trailing_std_f

**Causality is the whole risk here.** `.expanding().shift(1)` is deliberate --
expanding uses only past rows, and the shift excludes the current window from
its own normalisation. Using the full-series mean would leak future statistics
backwards into every training row and would inflate the result silently, which
is this project's characteristic failure mode.

**How this is judged.** Not against Phase A's point estimate, and not against
Phase A's confidence interval either -- both are underpowered. The correct test
is a PAIRED bootstrap over the same blocks: resample block indices once, score
both models on that resample, and take the distribution of the DIFFERENCE. The
models share nearly all their variance (same blocks, same catalog, same target),
so the difference is far better determined than either AUC alone. If that
interval excludes zero, the improvement is real.

Pre-registered decision rule: **adopt only if the paired difference CI excludes
zero in a zone that is itself established** (EAFZ, AEGEAN per 4.7). An
improvement in NAFZ or CENTRAL, whose AUCs are indistinguishable from chance,
is not evidence of anything.

Usage:
    python forecast_relfeat.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_downloader"))
from seismic_cli.catalog import load_catalog  # noqa: E402
from seismic_cli.forecast import FAULT_ZONES, FEATURES, build_dataset  # noqa: E402

from forecast_backtest import CHANCE, fit_logistic  # noqa: E402
from forecast_blocks import (bootstrap_auc, brier_skill, build_blocks,  # noqa: E402
                             calibrate_prospective, information_gain)

REL_FEATURES = [f"rel_{f}" for f in FEATURES]


def add_relative_features(d, min_std=1e-9):
    """
    Trailing z-score of every feature within its own zone.

    `.expanding()` sees only rows at or before the current one; `.shift(1)` then
    drops the current row, so a window is never part of the statistics used to
    normalise it. Early windows have too little history and come out NaN --
    left as NaN rather than filled, so the count of usable rows is visible.
    """
    parts = []
    for z, g in d.groupby("region", sort=False):
        g = g.sort_values("end_time").copy()
        for f in FEATURES:
            s = g[f].astype(float)
            mu = s.expanding().mean().shift(1)
            sd = s.expanding().std().shift(1)
            g[f"rel_{f}"] = (s - mu) / sd.where(sd > min_std)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values("end_time").reset_index(drop=True)


def walk_forward(d, blocks, zone, feats, horizon_days, min_train=400, step=4):
    """As forecast_blocks.walk_forward_blocks, but over an arbitrary feature list."""
    b = blocks[blocks.region == zone].sort_values("block_start").reset_index(drop=True)
    g = d[d.region == zone].sort_values("end_time").reset_index(drop=True)
    if b.empty or g.empty:
        return pd.DataFrame()

    emb = pd.Timedelta(days=horizon_days)
    scores, model, fitted_at = np.full(len(b), np.nan), None, -10**9
    for i, row in b.iterrows():
        if model is None or i - fitted_at >= step:
            tr = g[g.end_time <= row.block_start - emb]
            if len(tr) >= min_train and tr.label.nunique() > 1:
                model = (tr[feats].to_numpy(float), tr.label.to_numpy())
                fitted_at = i
        if model is None:
            continue
        scores[i] = fit_logistic(model[0], model[1],
                                 g.loc[[row.fc_index], feats].to_numpy(float))[0]
    b = b.copy()
    b["score"] = scores
    return b[np.isfinite(b.score)]


def paired_bootstrap_delta(y, s_a, s_b, n_boot=4000, seed=0):
    """
    CI on AUC(b) - AUC(a) over the SAME resampled blocks.

    Pairing is what makes this test usable at n~190: both models see identical
    resamples, so the shared variance cancels and only the difference is left.
    """
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], s_b[idx]) - roc_auc_score(y[idx], s_a[idx]))
    if not d:
        return float("nan"), float("nan"), float("nan")
    return (roc_auc_score(y, s_b) - roc_auc_score(y, s_a),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def cal_scores(bz):
    bc = calibrate_prospective(bz)
    bc = bc[np.isfinite(bc.p_cal)]
    if len(bc) < 20 or bc.label.nunique() < 2:
        return float("nan"), float("nan")
    y, p = bc.label.to_numpy(), bc.p_cal.to_numpy()
    base = float(y.mean())
    return brier_skill(y, p, base), information_gain(y, p, base)


def main():
    p = argparse.ArgumentParser(description="Zone-relative features vs absolute features.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--window-events", type=int, default=64)
    p.add_argument("--stride-events", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=4000)
    args = p.parse_args()

    d = build_dataset(args.catalog, FAULT_ZONES, window_events=args.window_events,
                      stride_events=args.stride_events, threshold=args.threshold,
                      horizon_days=args.horizon_days)
    d = add_relative_features(d)
    zones = [z for z in FAULT_ZONES if (d.region == z).any()]

    ok = d[REL_FEATURES].notna().all(axis=1).sum()
    print(f"\n[relfeat] {len(REL_FEATURES)} trailing z-scored features; "
          f"{ok}/{len(d)} rows fully defined (early windows lack history)")

    cat = load_catalog(args.catalog, min_magnitude=args.threshold)
    major = {}
    for z in zones:
        la0, la1, lo0, lo1 = FAULT_ZONES[z]
        major[z] = cat[cat.lat.between(la0, la1) & cat.lon.between(lo0, lo1)].time.to_numpy()
    blocks = pd.concat([build_blocks(d, z, args.horizon_days, d.end_time.max(), major[z])
                        for z in zones], ignore_index=True)

    sets = {"absolute (Phase A)": FEATURES,
            "relative only": REL_FEATURES,
            "absolute + relative": FEATURES + REL_FEATURES}

    print(f"\n{'='*98}")
    print("BLOCK-LEVEL AUC BY FEATURE SET  (paired bootstrap vs the Phase A absolute model)")
    print(f"{'='*98}")
    print(f"{'zone':9s} {'feature set':22s} {'blocks':>6s} {'AUC':>8s} {'95% CI':>16s} "
          f"| {'delta':>7s} {'delta 95% CI':>17s} | {'BSScal':>7s}")
    print("-" * 98)

    verdicts = {}
    for z in zones:
        base_b = walk_forward(d, blocks, z, FEATURES, args.horizon_days)
        if base_b.empty or base_b.label.nunique() < 2:
            print(f"{z:9s} (insufficient blocks)")
            continue
        y = base_b.label.to_numpy()
        s_base = base_b.score.to_numpy()

        for name, feats in sets.items():
            bz = walk_forward(d, blocks, z, feats, args.horizon_days)
            # Compare only on blocks both models scored.
            m = base_b.block_start.isin(bz.block_start)
            bz = bz[bz.block_start.isin(base_b.block_start)]
            yy, sa, sb = y[m.to_numpy()], s_base[m.to_numpy()], bz.score.to_numpy()
            auc, lo, hi = bootstrap_auc(yy, sb, args.n_boot // 2)
            bss_c, _ = cal_scores(bz)
            if name == "absolute (Phase A)":
                print(f"{z:9s} {name:22s} {len(bz):6d} {auc:8.4f} [{lo:.3f}, {hi:.3f}] "
                      f"| {'--':>7s} {'--':>17s} | {bss_c:+7.3f}")
                continue
            dl, dlo, dhi = paired_bootstrap_delta(yy, sa, sb, args.n_boot)
            star = "  *" if (dlo > 0 or dhi < 0) else ""
            print(f"{z:9s} {name:22s} {len(bz):6d} {auc:8.4f} [{lo:.3f}, {hi:.3f}] "
                  f"| {dl:+7.4f} [{dlo:+.3f}, {dhi:+.3f}]{star:3s} | {bss_c:+7.3f}")
            verdicts[(z, name)] = (dl, dlo, dhi)
        print("-" * 98)

    print("  * = paired 95 % CI on the difference excludes zero.\n")

    print(f"{'='*98}\nDECISION (rule set before running: adopt only if the paired CI")
    print("excludes zero in a zone that is itself established -- EAFZ or AEGEAN)")
    print(f"{'='*98}")
    established = {"EAFZ", "AEGEAN"}
    adopt = False
    for (z, name), (dl, lo, hi) in verdicts.items():
        if z not in established:
            continue
        sig = lo > 0
        if sig:
            adopt = True
        print(f"  {z:8s} {name:22s} delta {dl:+.4f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{'IMPROVES' if sig else 'no significant change' if hi > 0 else 'HURTS'}")
    print(f"\n  => {'ADOPT zone-relative features' if adopt else 'DO NOT ADOPT -- no significant gain in an established zone'}")


if __name__ == "__main__":
    main()
