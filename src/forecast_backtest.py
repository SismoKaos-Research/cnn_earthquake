"""
Rolling-origin backtest for the regional forecaster.

**Why this exists.** `forecast_eval.py` and `forecast_perzone.py` both rest on a
SINGLE chronological cut, and `catalog_report.md` builds its headline on it:
"M >= 4.5 within 30 days is forecastable in AEGEAN (test AUC 0.798 against a
0.640 persistence floor)". With one test era that claim cannot be distinguished
from "the 2023-2026 window happened to favour the Aegean" -- especially since
that window contains the 2023 Kahramanmaras sequence, which is why its positive
rate is 0.782 against training's 0.414, and since CENTRAL and NAFZ are judged
on 142 and 320 test windows respectively.

This walks the origin forward and reports each zone's AUC as a DISTRIBUTION
over origins rather than a point. The discriminating question it answers:

    at how many origins does a zone beat its persistence floor by > 0.02?

If AEGEAN is 1/N, the headline is an artifact of one era and has to be
weakened. If it is most of N, the result is real. That question gates whether
any feature work is worth doing: adding a feature now would mean comparing a
new point estimate against an old point estimate whose noise has never been
measured, which is exactly the failure `report.md` 6.6 documented when a
single-seed margin reversed sign on re-running.

Rolling origins also handle Kahramanmaras structurally instead of by assertion:
the sequence falls in test for some origins and in train for others, so its
influence becomes visible rather than argued about.

Splitting at each origin O (embargo `emb` = one horizon, the rule already used
by `forecast.chronological_split`):

    train   end_time <= O - val_len - 2*emb      (or a sliding window of it)
    val     O - val_len - emb  <  end_time <= O - emb
    test    O                  <  end_time <= O + test_len

The val slice exists to pick the decision threshold, which is the one idea
worth taking from Basar & Celik (Sensors 2026) -- the only paper in the
reviewed batch that calibrates its operating point on validation instead of
thresholding at 0.5. AUC is threshold-free; a forecast someone acts on is not.

Usage:
    python forecast_backtest.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
    python forecast_backtest.py --catalog ... --train-window-years 8    # sliding
    python forecast_backtest.py --catalog ... --sensitivity             # threshold x horizon grid
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_downloader"))
from seismic_cli.forecast import FAULT_ZONES, FEATURES, build_dataset  # noqa: E402

BEAT_MARGIN = 0.02   # same margin forecast_perzone.py uses to call a win

# A model must clear BOTH floors, and the bar is max(chance, persistence).
#
# Persistence alone is not sufficient: in NAFZ and CENTRAL it scores 0.18-0.34,
# i.e. far BELOW chance, because a recent qualifying event there predicts FEWER
# events, not more. Scoring "better than persistence" against a floor that is
# itself anti-predictive is the identical trap to the LOEO majority baseline in
# catalog_report.md 2(b) -- it turns a sub-chance model into an apparent win.
# A forecast that loses to a coin flip is not a forecast.
CHANCE = 0.5


def auc_or_nan(y, s):
    """AUC, or NaN when the slice has a single class -- never a fake 0.5."""
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def persistence_score(part):
    """
    Domain floor: recency of the last qualifying event.

    Identical to `forecast_eval.persistence_scores`, restated here so the
    backtest has no import-order dependency on that script. Higher score =
    more recent = predicted more likely. NaN (no prior major on record) sorts
    to the bottom rather than being dropped.
    """
    ds = part.days_since_prev_major.to_numpy(dtype=float)
    return np.where(np.isfinite(ds), -ds, -1e9)


def fit_logistic(Xtr, ytr, Xte):
    if len(np.unique(ytr)) < 2 or len(Xte) == 0:
        return np.full(len(Xte), float(ytr.mean()) if len(ytr) else 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(max_iter=2000).fit(np.nan_to_num(Xtr, nan=0.0), ytr)
    return m.predict_proba(np.nan_to_num(Xte, nan=0.0))[:, 1]


def best_f1_threshold(y, s):
    """
    Pick the operating point on VALIDATION, not 0.5.

    Returns 0.5 when validation is unusable (empty or single-class), so the
    caller degrades to the old behaviour rather than silently reporting a
    threshold fitted on nothing.
    """
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.quantile(s, np.linspace(0.02, 0.98, 49)):
        pred = (s > t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        if tp == 0:
            continue
        prec = tp / max(int((pred == 1).sum()), 1)
        rec = tp / max(int((y == 1).sum()), 1)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def make_origins(d, n_origins, test_len, val_len, emb, train_window_years=None,
                 min_train=400):
    """
    Evenly spaced origins across the part of the span that can support a split.

    An origin needs enough history behind it for train+val+embargoes and a full
    test slice ahead of it, so the usable range is strictly interior.
    """
    t0, t1 = d.end_time.min(), d.end_time.max()
    need_back = pd.Timedelta(days=val_len + 2 * emb + 365 * 2)   # + 2y minimum history
    first = t0 + need_back
    last = t1 - pd.Timedelta(days=test_len)
    if last <= first:
        raise SystemExit("catalog span too short for a rolling backtest")
    return list(pd.date_range(first, last, periods=n_origins))


def run_backtest(d, origins, test_len, val_len, emb, train_window_years, zones,
                 min_train=400, verbose=True):
    """
    One row per (origin, zone). Origins that cannot be split are skipped and
    counted -- never silently dropped.
    """
    rows, skipped = [], []
    for oi, O in enumerate(origins):
        te_lo, te_hi = O, O + pd.Timedelta(days=test_len)
        va_hi = O - pd.Timedelta(days=emb)
        va_lo = va_hi - pd.Timedelta(days=val_len)
        tr_hi = va_lo - pd.Timedelta(days=emb)
        tr_lo = (tr_hi - pd.Timedelta(days=365.25 * train_window_years)
                 if train_window_years else d.end_time.min() - pd.Timedelta(days=1))

        tr = d[(d.end_time > tr_lo) & (d.end_time <= tr_hi)]
        va = d[(d.end_time > va_lo) & (d.end_time <= va_hi)]
        te = d[(d.end_time > te_lo) & (d.end_time <= te_hi)]

        # Correctness assertions -- the embargo is the whole point of the split.
        assert tr.empty or te.empty or tr.end_time.max() <= te.end_time.min(), "train/test overlap"
        assert tr.empty or va.empty or (va.end_time.min() - tr.end_time.max()).days >= emb - 1, \
            "train/val embargo violated"
        assert va.empty or te.empty or (te.end_time.min() - va.end_time.max()).days >= emb - 1, \
            "val/test embargo violated"

        if len(tr) < min_train or te.empty or tr.label.nunique() < 2:
            skipped.append((O.date(), len(tr), len(te)))
            continue

        Xtr, ytr = tr[FEATURES].to_numpy(float), tr.label.to_numpy()
        s_te_pooled = fit_logistic(Xtr, ytr, te[FEATURES].to_numpy(float))
        s_va_pooled = fit_logistic(Xtr, ytr, va[FEATURES].to_numpy(float))
        thr = best_f1_threshold(va.label.to_numpy(), s_va_pooled) if len(va) else 0.5

        te = te.copy()
        te["s_pooled"] = s_te_pooled

        for z in zones:
            tez, trz = te[te.region == z], tr[tr.region == z]
            if tez.empty:
                continue
            y = tez.label.to_numpy()
            s_pz = fit_logistic(trz[FEATURES].to_numpy(float), trz.label.to_numpy(),
                                tez[FEATURES].to_numpy(float)) if len(trz) >= 100 else \
                np.full(len(tez), np.nan)

            sp = tez.s_pooled.to_numpy()
            pred = (sp > thr).astype(int)
            rows.append(dict(
                origin=O, oi=oi, zone=z, n_test=len(tez), n_train=len(trz),
                pos_rate=float(y.mean()) if len(y) else np.nan,
                auc_persist=auc_or_nan(y, persistence_score(tez)),
                auc_pooled=auc_or_nan(y, sp),
                auc_perzone=(auc_or_nan(y, s_pz) if np.isfinite(s_pz).all() else np.nan),
                thr=thr,
                precision=(precision_score(y, pred, zero_division=0) if len(np.unique(y)) > 1 else np.nan),
                recall=(recall_score(y, pred, zero_division=0) if len(np.unique(y)) > 1 else np.nan),
            ))

    if verbose and skipped:
        print(f"\n[skip] {len(skipped)} origin(s) unusable (train<{min_train} or empty test):")
        for s in skipped:
            print(f"       {s[0]}  n_train={s[1]}  n_test={s[2]}")
    return pd.DataFrame(rows)


def summarise(bt, label):
    print(f"\n{'='*84}")
    print(f"PER-ZONE AUC ACROSS ORIGINS  --  {label}")
    print(f"{'='*84}")
    print(f"{'zone':9s} {'N':>3s} {'pos':>6s} | {'persist med':>11s} "
          f"{'pooled med':>10s} {'IQR':>13s} {'per-zone med':>12s} | "
          f"{'>persist':>9s} {'>BOTH floors':>13s}")
    print("-" * 96)

    out = []
    for z, g in bt.groupby("zone"):
        g = g[np.isfinite(g.auc_pooled) & np.isfinite(g.auc_persist)]
        if g.empty:
            print(f"{z:9s}   0  (no origin with both classes in test)")
            continue
        q1, q3 = g.auc_pooled.quantile(0.25), g.auc_pooled.quantile(0.75)
        wins_p = int((g.auc_pooled > g.auc_persist + BEAT_MARGIN).sum())
        bar = np.maximum(g.auc_persist.to_numpy(), CHANCE)
        wins = int((g.auc_pooled.to_numpy() > bar + BEAT_MARGIN).sum())
        print(f"{z:9s} {len(g):3d} {g.pos_rate.mean():6.3f} | "
              f"{g.auc_persist.median():11.4f} {g.auc_pooled.median():10.4f} "
              f"[{q1:5.3f},{q3:5.3f}] {g.auc_perzone.median():12.4f} | "
              f"{wins_p:4d}/{len(g):<4d} {wins:6d}/{len(g):<3d} {wins/len(g):5.0%}")
        out.append(dict(zone=z, n=len(g), persist=g.auc_persist.median(),
                        pooled=g.auc_pooled.median(), wins_persist=wins_p,
                        wins=wins, total=len(g)))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description="Rolling-origin backtest of the regional forecaster.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--window-events", type=int, default=64)
    p.add_argument("--stride-events", type=int, default=8)
    p.add_argument("--n-origins", type=int, default=12)
    p.add_argument("--test-len-days", type=float, default=365.0)
    p.add_argument("--val-len-days", type=float, default=365.0)
    p.add_argument("--train-window-years", type=float, default=None,
                   help="sliding-window training span; omit for expanding (all history)")
    p.add_argument("--sensitivity", action="store_true",
                   help="sweep magnitude threshold x horizon instead of the main run")
    args = p.parse_args()

    if args.sensitivity:
        run_sensitivity(args)
        return

    d = build_dataset(args.catalog, FAULT_ZONES, window_events=args.window_events,
                      stride_events=args.stride_events, threshold=args.threshold,
                      horizon_days=args.horizon_days)
    zones = [z for z in FAULT_ZONES if (d.region == z).any()]
    origins = make_origins(d, args.n_origins, args.test_len_days,
                           args.val_len_days, args.horizon_days)
    print(f"\n[origins] {len(origins)} from {origins[0].date()} to {origins[-1].date()}, "
          f"{args.test_len_days:.0f}-day test slices, {args.horizon_days:.0f}-day embargo")

    # Expanding window: all history up to the origin.
    bt_exp = run_backtest(d, origins, args.test_len_days, args.val_len_days,
                          args.horizon_days, None, zones)
    s_exp = summarise(bt_exp, "EXPANDING training window (all history)")

    # Sliding window: recent history only. If this wins, an ever-growing
    # training set is actively hurting, which IS the non-stationarity claim.
    slide_years = args.train_window_years or 8.0
    bt_sld = run_backtest(d, origins, args.test_len_days, args.val_len_days,
                          args.horizon_days, slide_years, zones, verbose=False)
    s_sld = summarise(bt_sld, f"SLIDING training window ({slide_years:.0f} years)")

    print(f"\n{'='*84}\nTHE QUESTION THIS RUN EXISTS TO ANSWER\n{'='*84}")
    print("  Bar = max(chance 0.5, persistence) + 0.02. Beating an anti-predictive")
    print("  persistence floor while sitting below chance is not a forecast.\n")
    for _, r in s_exp.iterrows():
        verdict = ("FORECASTABLE -- clears both floors at most origins"
                   if r.wins / r.total >= 0.5 else
                   "NOT FORECASTABLE -- below chance despite 'beating' persistence"
                   if r.pooled < CHANCE else "MARGINAL")
        print(f"  {r.zone:9s} median AUC {r.pooled:.3f} | clears both floors at "
              f"{r.wins}/{r.total} origins (persistence-only: {r.wins_persist}/{r.total})"
              f"\n            -> {verdict}")

    if not s_exp.empty and not s_sld.empty:
        m = s_exp.merge(s_sld, on="zone", suffixes=("_exp", "_sld"))
        better = int((m.pooled_sld > m.pooled_exp).sum())
        print(f"\n  Sliding beats expanding in {better}/{len(m)} zones "
              f"({'non-stationary: older data hurts' if better > len(m)/2 else 'more history helps'})")

    # Operating point actually chosen, and what it buys.
    ok = bt_exp[np.isfinite(bt_exp.precision)]
    if not ok.empty:
        print(f"\n  Val-calibrated threshold: median {ok.thr.median():.3f} "
              f"(vs the 0.5 used previously) -> precision {ok.precision.median():.3f}, "
              f"recall {ok.recall.median():.3f}")

    out = Path(__file__).resolve().parent / "forecast_backtest_results.csv"
    bt_exp.assign(mode="expanding").to_csv(out, index=False)
    print(f"\n[write] per-origin rows -> {out}")


def run_sensitivity(args):
    """
    Threshold x horizon grid. Answers catalog_report.md 5's "single threshold,
    single horizon" limitation directly. Only the per-zone MEDIAN AUC and its
    persistence floor are reported -- the grid is for spotting whether the
    result is specific to M>=4.5/30d, not for picking a winner post hoc.
    """
    print(f"\n{'='*84}\nSENSITIVITY: magnitude threshold x horizon\n{'='*84}")
    print(f"{'M>=':>4s} {'horiz':>6s} {'zone':9s} {'orig':>5s} {'pos':>6s} "
          f"{'persist':>8s} {'pooled':>8s} {'delta':>8s}")
    print("-" * 84)
    for thr in (4.0, 4.5, 5.0):
        for hz in (15.0, 30.0, 60.0):
            d = build_dataset(args.catalog, FAULT_ZONES, window_events=args.window_events,
                              stride_events=args.stride_events, threshold=thr,
                              horizon_days=hz)
            zones = [z for z in FAULT_ZONES if (d.region == z).any()]
            origins = make_origins(d, args.n_origins, args.test_len_days,
                                   args.val_len_days, hz)
            bt = run_backtest(d, origins, args.test_len_days, args.val_len_days,
                              hz, None, zones, verbose=False)
            for z, g in bt.groupby("zone"):
                g = g[np.isfinite(g.auc_pooled) & np.isfinite(g.auc_persist)]
                if g.empty:
                    print(f"{thr:4.1f} {hz:6.0f} {z:9s} {0:5d}    -- single class in test --")
                    continue
                pe, po = g.auc_persist.median(), g.auc_pooled.median()
                print(f"{thr:4.1f} {hz:6.0f} {z:9s} {len(g):5d} {g.pos_rate.mean():6.3f} "
                      f"{pe:8.4f} {po:8.4f} {po - pe:+8.4f}")


if __name__ == "__main__":
    main()
