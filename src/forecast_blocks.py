"""
Block-level evaluation: what the forecaster's numbers look like at the honest
sample size.

**The defect this exists to fix.** `forecast.py` slides a 64-event window with
`stride_events=8`, so consecutive windows share 56 of their 64 events. In AEGEAN
consecutive windows end 0.35 days apart while the label looks forward 30 DAYS --
so consecutive labels share ~99 % of their horizon. Measured redundancy:

    zone      windows   independent 30-day blocks   inflation
    AEGEAN      9193              ~200                 46x
    EAFZ        6224              ~200                 31x
    NAFZ        3828              ~199                 19x
    CENTRAL     2094              ~199                 11x

Every AUC in `catalog_report.md` is therefore computed over samples 11-46x
redundant. The point estimates are not necessarily biased, but any confidence
read off `n` is wrong by ~sqrt(inflation) -- for AEGEAN about 7x. The reported
"n_test = 862, AUC 0.8035" rests on roughly TEN independent forecast
opportunities, not 862.

This is the same silent-wrong-number class as the rest of the project's defect
list: nothing crashes, the number simply is not what it appears to be.

**What this does instead.** Partition each zone's timeline into consecutive
DISJOINT `horizon_days` blocks. Each block gets exactly one forecast -- made
from the last window ending strictly BEFORE the block opens, so no window can
see any part of the interval it is predicting -- and one binary outcome. Walking
forward over blocks, every block is predicted once and out-of-sample, giving
~200 genuinely independent evaluations per zone.

Reported with bootstrap confidence intervals, and with the probabilistic scores
that AUC cannot supply: Brier skill score, information gain per block against a
Poisson benchmark, and a reliability curve. None of the seven papers in
`literature_review.md` reports any of these; they are standard in the actual
earthquake-forecasting literature (CSEP). AUC is rank-only -- it cannot tell you
whether a probability means anything, and a forecast that gets acted on needs
calibrated probabilities.

The aggregation idea is Basar & Celik (Sensors 2026), the one paper in that
batch with a protocol worth borrowing. Applied here it found a defect rather
than an improvement.

Usage:
    python forecast_blocks.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
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

from forecast_backtest import CHANCE, auc_or_nan, fit_logistic  # noqa: E402

EPS = 1e-6   # keeps log-loss finite when a model is confidently wrong


def build_blocks(d, zone, horizon_days, catalog_end, major_times):
    """
    Disjoint consecutive blocks for one zone, each with one forecast time and
    one outcome.

    A block [t, t+H) is positive iff a qualifying event's ORIGIN TIME falls
    inside it. The forecast for it is the score of the last window ending
    STRICTLY BEFORE t -- the information a forecaster would actually hold at the
    moment the block opens.

    `major_times` comes straight from the catalog rather than from window
    labels. Inheriting the labels would be wrong: a window ending at
    `block_start + 25d` carries a horizon reaching `block_start + 55d`, so
    aggregating window labels marks a block positive for events occurring up to
    a full horizon AFTER it closes. That inflates the base rate, and the base
    rate is the reference for both Brier skill and information gain -- so the
    "usable / not usable" verdicts would be computed against a moved goalpost.
    """
    g = d[d.region == zone].sort_values("end_time").reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()

    t0 = g.end_time.min()
    H = pd.Timedelta(days=horizon_days)
    edges = []
    t = t0
    while t + H <= min(g.end_time.max(), catalog_end):
        edges.append(t)
        t = t + H

    ends = g.end_time.to_numpy()
    mt = np.sort(np.asarray(major_times, dtype="datetime64[ns]"))
    rows = []
    for lo in edges:
        hi = lo + H
        # Forecast made from the last window ending strictly before the block.
        prior = np.searchsorted(ends, np.datetime64(lo), side="left") - 1
        if prior < 0:
            continue
        i0 = np.searchsorted(mt, np.datetime64(lo), side="left")
        i1 = np.searchsorted(mt, np.datetime64(hi), side="left")
        rows.append(dict(
            region=zone, block_start=lo, block_end=hi,
            fc_index=int(prior), fc_time=g.end_time.iloc[prior],
            label=int(i1 > i0),
            n_major=int(i1 - i0),
        ))
    return pd.DataFrame(rows)


def bootstrap_auc(y, s, n_boot=2000, seed=0):
    """Percentile CI. Resamples blocks, which are the independent unit."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], s[idx]))
    if not stats:
        return float("nan"), float("nan"), float("nan")
    return (roc_auc_score(y, s), float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def brier_skill(y, p, base):
    """BSS vs always predicting the zone's own base rate. >0 means useful."""
    bs = np.mean((p - y) ** 2)
    bs_ref = np.mean((base - y) ** 2)
    return 1.0 - bs / bs_ref if bs_ref > 0 else float("nan")


def information_gain(y, p, base):
    """
    Mean information gain per block in BITS over a Poisson (base-rate)
    benchmark. This is the CSEP-standard forecast score and the one number that
    says whether the probabilities themselves are worth anything.
    """
    p = np.clip(p, EPS, 1 - EPS)
    b = np.clip(base, EPS, 1 - EPS)
    ll = y * np.log2(p) + (1 - y) * np.log2(1 - p)
    ll_ref = y * np.log2(b) + (1 - y) * np.log2(1 - b)
    return float(np.mean(ll - ll_ref))


def reliability(y, p, n_bins=5):
    """Predicted vs observed frequency. Reported even when it is bad."""
    out = []
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() < 3:
            continue
        out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def walk_forward_blocks(d, blocks, zone, horizon_days, min_train=400, step=4):
    """
    Predict every block out-of-sample.

    Refits every `step` blocks rather than every block -- the model is logistic
    over 11 features, so refitting is cheap, but at ~200 blocks per zone every
    block would be 200 fits per zone for no measurable difference.
    """
    b = blocks[blocks.region == zone].sort_values("block_start").reset_index(drop=True)
    g = d[d.region == zone].sort_values("end_time").reset_index(drop=True)
    if b.empty or g.empty:
        return pd.DataFrame()

    emb = pd.Timedelta(days=horizon_days)
    scores, model, fitted_at = np.full(len(b), np.nan), None, -10**9

    for i, row in b.iterrows():
        # Training windows must be fully resolved before this block opens:
        # end_time + horizon <= block_start, i.e. end_time <= block_start - emb.
        if model is None or i - fitted_at >= step:
            tr = g[g.end_time <= row.block_start - emb]
            if len(tr) >= min_train and tr.label.nunique() > 1:
                model = (tr[FEATURES].to_numpy(float), tr.label.to_numpy())
                fitted_at = i
        if model is None:
            continue
        x = g.loc[[row.fc_index], FEATURES].to_numpy(float)
        scores[i] = fit_logistic(model[0], model[1], x)[0]

    b = b.copy()
    b["score"] = scores
    return b[np.isfinite(b.score)]


def calibrate_prospective(b, min_hist=40):
    """
    Map raw scores to usable probabilities using only PAST blocks.

    The raw model is trained on WINDOW labels but scored against BLOCK outcomes,
    which aggregate by max and therefore carry a different base rate (AEGEAN
    0.563 vs 0.620). That mismatch alone guarantees miscalibration, and it is
    invisible to AUC because AUC is rank-only -- ranking survives any monotone
    transform, calibration does not.

    Platt scaling (a 1-D logistic on the raw score) rather than isotonic: with
    ~190 blocks and only ~40 of history before the first calibrated prediction,
    isotonic would overfit its own step function. Refit at every block on all
    prior blocks, so the calibrator never sees the block it is calibrating.

    Blocks before `min_hist` get no calibrated probability rather than a
    fallback -- a silently-defaulted probability is exactly the kind of number
    this project keeps finding.
    """
    from sklearn.linear_model import LogisticRegression

    b = b.sort_values("block_start").reset_index(drop=True)
    out = np.full(len(b), np.nan)
    s, y = b.score.to_numpy(), b.label.to_numpy()
    for i in range(min_hist, len(b)):
        ys, ss = y[:i], s[:i]
        if len(np.unique(ys)) < 2:
            continue
        m = LogisticRegression(max_iter=1000).fit(ss.reshape(-1, 1), ys)
        out[i] = m.predict_proba(s[i].reshape(1, 1))[0, 1]
    b = b.copy()
    b["p_cal"] = out
    return b


def main():
    p = argparse.ArgumentParser(description="Block-level (non-overlapping) forecaster evaluation.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--window-events", type=int, default=64)
    p.add_argument("--stride-events", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    d = build_dataset(args.catalog, FAULT_ZONES, window_events=args.window_events,
                      stride_events=args.stride_events, threshold=args.threshold,
                      horizon_days=args.horizon_days)
    zones = [z for z in FAULT_ZONES if (d.region == z).any()]
    catalog_end = d.end_time.max()

    # Block outcomes come from the catalog directly, never from window labels.
    cat = load_catalog(args.catalog, min_magnitude=args.threshold)
    major = {}
    for z in zones:
        la0, la1, lo0, lo1 = FAULT_ZONES[z]
        major[z] = cat[cat.lat.between(la0, la1) & cat.lon.between(lo0, lo1)] \
            .time.to_numpy()

    all_blocks = pd.concat([build_blocks(d, z, args.horizon_days, catalog_end, major[z])
                            for z in zones], ignore_index=True)

    # --- correctness assertions: this is the leak that would fabricate a result
    for z in zones:
        b = all_blocks[all_blocks.region == z].sort_values("block_start")
        if b.empty:
            continue
        assert (b.block_start.values[1:] >= b.block_end.values[:-1]).all(), \
            f"{z}: blocks overlap"
        assert (b.fc_time < b.block_start).all(), \
            f"{z}: a forecast used a window ending inside the block it predicts"
    print(f"\n[blocks] disjoint, forecast strictly precedes block: OK")
    print(f"[blocks] " + "  ".join(
        f"{z}={int((all_blocks.region == z).sum())}" for z in zones))

    print(f"\n{'='*94}")
    print("BLOCK-LEVEL RESULTS  (one forecast per disjoint 30-day block)")
    print(f"{'='*94}")
    print(f"{'zone':9s} {'blocks':>6s} {'pos':>6s} | "
          f"{'BLOCK AUC':>9s} {'95% CI':>16s} | {'BSS':>7s} {'IG bits':>8s}")
    print("-" * 94)

    summary = []
    for z in zones:
        bz = walk_forward_blocks(d, all_blocks, z, args.horizon_days)
        if bz.empty or bz.label.nunique() < 2:
            print(f"{z:9s} {len(bz):6d}   (insufficient blocks or single class)")
            continue
        y, s = bz.label.to_numpy(), bz.score.to_numpy()
        auc, lo, hi = bootstrap_auc(y, s, args.n_boot)
        base = float(y.mean())
        bss = brier_skill(y, s, base)
        ig = information_gain(y, s, base)

        sig = "" if (lo <= CHANCE <= hi) else "  *"
        print(f"{z:9s} {len(bz):6d} {base:6.3f} | "
              f"{auc:9.4f} [{lo:.3f}, {hi:.3f}]{sig:3s} | {bss:+7.3f} {ig:+8.3f}")

        # Prospective recalibration -- fixes the probabilities without touching
        # the ranking, so AUC is unchanged by construction and only BSS/IG move.
        bc = calibrate_prospective(bz)
        bc = bc[np.isfinite(bc.p_cal)]
        if len(bc) > 20 and bc.label.nunique() > 1:
            yc, pc = bc.label.to_numpy(), bc.p_cal.to_numpy()
            base_c = float(yc.mean())
            bss_c, ig_c = brier_skill(yc, pc, base_c), information_gain(yc, pc, base_c)
        else:
            yc = pc = None
            bss_c = ig_c = float("nan")

        summary.append(dict(zone=z, n=len(bz), base=base, auc=auc, lo=lo, hi=hi,
                            bss=bss, ig=ig, bss_cal=bss_c, ig_cal=ig_c,
                            n_cal=len(bc),
                            rel=reliability(y, s),
                            rel_cal=(reliability(yc, pc) if yc is not None else [])))

    print("-" * 94)
    print("  * = 95 % CI excludes chance. No star means the effect is NOT")
    print("      statistically established at the honest sample size.")
    print("  BSS > 0 and IG > 0 mean the probabilities beat the base rate.")

    print(f"\n{'='*94}")
    print("PROSPECTIVE RECALIBRATION (Platt on prior blocks only; ranking untouched)")
    print(f"{'='*94}")
    print(f"{'zone':9s} {'n_cal':>6s} | {'BSS raw':>8s} {'BSS cal':>8s} | "
          f"{'IG raw':>7s} {'IG cal':>7s} | usable?")
    print("-" * 94)
    for s in summary:
        usable = "YES" if (s["bss_cal"] > 0 and s["ig_cal"] > 0) else "no"
        print(f"{s['zone']:9s} {s['n_cal']:6d} | {s['bss']:+8.3f} {s['bss_cal']:+8.3f} | "
              f"{s['ig']:+7.3f} {s['ig_cal']:+7.3f} | {usable}")
    print("-" * 94)
    print("  'usable' = calibrated probabilities beat the base rate on BOTH scores.")
    print("  AUC is identical before and after: Platt is monotone, so this changes")
    print("  only whether the numbers mean anything, not how well they rank.")

    print(f"\n{'='*94}\nRELIABILITY (are the probabilities calibrated?)\n{'='*94}")
    for s in summary:
        print(f"\n  {s['zone']}  (base rate {s['base']:.3f})")
        print(f"    {'predicted':>10s} {'observed':>9s} {'n':>5s}   |  after calibration")
        rc = {i: r for i, r in enumerate(s["rel_cal"])}
        for i, (pred, obs, n) in enumerate(s["rel"]):
            flag = "" if abs(pred - obs) < 0.15 else " <- off"
            c = (f"   {rc[i][0]:.3f} -> {rc[i][1]:.3f}" if i in rc else "")
            print(f"    {pred:10.3f} {obs:9.3f} {n:5d}{flag:8s}{c}")

    print(f"\n{'='*94}\nVERDICT\n{'='*94}")
    for s in summary:
        est = ("ESTABLISHED" if s["lo"] > CHANCE else
               "NOT ESTABLISHED -- CI includes chance" if s["hi"] > CHANCE else
               "BELOW CHANCE")
        print(f"  {s['zone']:9s} block AUC {s['auc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] "
              f"on {s['n']:3d} independent blocks  ->  {est}")

    out = Path(__file__).resolve().parent / "forecast_blocks_results.csv"
    pd.DataFrame([{k: v for k, v in s.items() if k != "rel"} for s in summary]).to_csv(
        out, index=False)
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
