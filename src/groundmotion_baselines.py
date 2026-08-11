"""
Non-neural floors for the peak-ground-motion task, run BEFORE any network.

Nurtas et al. (ACDSA 2025) report validation MAE 2.61 gal / R2 0.714 for a
CNN-BiLSTM+attention model predicting PGA from the first 3 s of waveform, and
compare it against... an ANN, an LSTM, and nothing else. All three are neural.
The paper never asks the obvious question: PGA is an amplitude, and the input
window contains amplitude, so how much of that 0.714 does the peak amplitude of
the input window deliver on its own?

That question is this project's recurring one and the answer has three times out
of four been "most of it" (`report.md` 8.5 has a two-scalar model beating a CNN
by nine points on a related task). `report.md` 12 defect 16 records what happens
when the floor is computed after the headline instead of before it, so this runs
first and its numbers are written down before `cnn_groundmotion.py` exists.

**The predictors are restricted to what is knowable at inference.** Magnitude is
NOT a legitimate predictor here -- in an early-warning setting the whole point is
that you are trying to characterise the shaking before the source is
characterised -- so it appears only as a clearly-marked ORACLE row that bounds
what any model could achieve. Distance is included, matching the plan and the
local-magnitude relation `cnn_regression.py` already uses.

**Metrics are reported in log AND linear space, deliberately.** The paper trains
on MSE in log space and reports R2 in linear space on a heavy-tailed target,
which is almost certainly what produced its ANN R2 of -10.08: one badly missed
large value dominates a linear-space sum of squares. Reporting both makes the
discrepancy visible instead of mysterious. Note the linear-space predictions are
back-transformed as 10**pred, which is the median rather than the mean of the
implied lognormal -- an understatement by construction, and one that is NOT
corrected here because correcting it would silently change what the log-space
model claims.

Usage:
    python groundmotion_baselines.py \
        --manifest ../../data_downloader/data/dataset_groundmotion_3s/manifest.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from metrics import regression_report

# target key -> (linear column, log column, unit, matched-unit input peak, degenerate?)
#
# The amplitude predictor is matched to the target's units: velocity input for a
# PGV target, acceleration input for a PGA target. Predicting gal from cm/s
# works, but it understates what amplitude alone can do.
#
# `degenerate` marks the targets whose window CONTAINS the input window, so the
# input peak is a mathematical lower bound on the target rather than a predictor
# of it. Verified on the corpus: log_peak_input_vel <= log_pgv_full in 100.00 %
# of rows and is EXACTLY EQUAL in 32.05 %, which is exactly the `peak_in_input`
# fraction. For a third of the corpus the "baseline" is the answer, bit for bit.
# These are reported separately and must never be quoted as a like-for-like
# result against the `_fwd` numbers.
TARGETS = {
    "pgv_fwd":  ("pgv_cms_fwd",  "log_pgv_fwd",  "cm/s", "log_peak_input_vel", False),
    "pga_fwd":  ("pga_gal_fwd",  "log_pga_fwd",  "gal",  "log_peak_input_acc", False),
    "pgv_full": ("pgv_cms_full", "log_pgv_full", "cm/s", "log_peak_input_vel", True),
    "pga_full": ("pga_gal_full", "log_pga_full", "gal",  "log_peak_input_acc", True),
}

# Predictor sets, cheapest first. The second is THE floor the paper omits.
# "AMP" is substituted with the target's matched-unit input peak column.
PREDICTORS = {
    "predict-the-median":        [],
    "log peak amplitude":        ["AMP"],
    "amplitude + log distance":  ["AMP", "log_dist"],
    "gbm (same 2 features)":     ["AMP", "log_dist"],
    "ORACLE +magnitude":         ["AMP", "log_dist", "magnitude"],
}
# Given the SAME two features as the linear model, so this isolates functional
# form -- it is not a stronger feature set, and should not be read as one.
GBM_KEYS = {"gbm (same 2 features)"}


def load(manifest, drop_truncated=True, drop_clipped=True):
    """
    Read the manifest and apply the label-independent quality rules.

    Every exclusion here is decided by a flag that was computed without looking
    at the target, so this cannot select the rows that happen to fit.
    """
    d = pd.read_csv(manifest)
    n0 = len(d)
    keep = d.response_ok.astype(bool) & (d.sens_mismatch <= 0.05)
    reasons = [("response failed / untrustworthy sensitivity", int((~keep).sum()))]
    if drop_truncated:
        m = ~d.label_truncated.astype(bool)
        reasons.append(("forward label truncated", int((keep & ~m).sum())))
        keep &= m
    if drop_clipped:
        m = ~d.clipped.astype(bool)
        reasons.append(("clipped at the digitizer rail", int((keep & ~m).sum())))
        keep &= m
    d = d[keep].copy()
    d["log_dist"] = np.log10(d.distance_km.clip(lower=1.0))
    print(f"[data] {n0} rows -> {len(d)} after label-independent quality rules")
    for why, n in reasons:
        print(f"         -{n:6d}  {why}")

    # Splits are event-disjoint, but stations are not, and site response is a
    # per-station multiplicative term on amplitude. A linear amplitude->amplitude
    # fit is fairly robust to that; a GBM can memorise per-station offsets. Same
    # residual-overlap caveat `regression.py` reports for the magnitude dataset.
    shared_ev = sum(g.split.nunique() > 1 for _, g in d.groupby("event_id"))
    shared_st = sum(g.split.nunique() > 1 for _, g in d.groupby("station_key"))
    print(f"[split] events in >1 split : {shared_ev}/{d.event_id.nunique()}  (must be 0)")
    print(f"[split] stations in >1 split: {shared_st}/{d.station_key.nunique()}  "
          f"(expected; site response is shared across splits)")
    return d


def metrics(y_log_true, y_log_pred, unit):
    """MAE and R2 in log space and, after back-transforming, in linear space."""
    lin_true, lin_pred = 10.0 ** y_log_true, 10.0 ** y_log_pred
    return {
        "MAE_log": mean_absolute_error(y_log_true, y_log_pred),
        "R2_log": r2_score(y_log_true, y_log_pred),
        f"MAE_{unit}": mean_absolute_error(lin_true, lin_pred),
        "R2_lin": r2_score(lin_true, lin_pred),
    }


def fit_predict(name, feats, tr, te, tcol):
    """Train on `tr`, predict `te`. Everything is fitted in log space."""
    if not feats:                       # predict-the-median
        return np.full(len(te), float(np.median(tr[tcol])))
    Xtr, Xte = tr[feats].to_numpy(float), te[feats].to_numpy(float)
    if name in GBM_KEYS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          random_state=0)
    else:
        m = LinearRegression()
    return m.fit(Xtr, tr[tcol].to_numpy(float)).predict(Xte)


def run_target(d, tkey, stratify=True):
    lin_col, log_col, unit, amp_col, degenerate = TARGETS[tkey]
    sub = d.dropna(subset=[log_col, amp_col, "log_dist", "magnitude"])
    tr = sub[sub.split == "train"]
    te = sub[sub.split == "test"]
    if len(tr) < 50 or len(te) < 50:
        print(f"\n[{tkey}] insufficient data (train {len(tr)}, test {len(te)})")
        return None

    print(f"\n{'='*94}")
    print(f"TARGET {tkey}  ({lin_col}, {unit})   train {len(tr)}  test {len(te)}  "
          f"test median {10**np.median(te[log_col]):.4g} {unit}")
    if degenerate:
        bound = float((sub[amp_col] <= sub[log_col] + 1e-12).mean())
        exact = float((np.abs(sub[amp_col] - sub[log_col]) < 1e-9).mean())
        print(f"  !! DEGENERATE: the target window CONTAINS the input window.")
        print(f"     {amp_col} <= {log_col} in {bound:.2%} of rows and is EXACTLY")
        print(f"     EQUAL in {exact:.2%}. The amplitude 'baseline' is a lower bound on the")
        print(f"     target, not a predictor of it. These numbers are NOT comparable to the")
        print(f"     _fwd table and must not be quoted as a like-for-like result.")
    print(f"{'='*94}")
    print(f"{'baseline':28s} {'MAE_log':>9s} {'R2_log':>8s} {'MAE_'+unit:>12s} {'R2_lin':>9s}")
    print("-" * 94)

    out, preds = [], {}
    for name, spec in PREDICTORS.items():
        feats = [amp_col if f == "AMP" else f for f in spec]
        p = fit_predict(name, feats, tr, te, log_col)
        preds[name] = p
        m = metrics(te[log_col].to_numpy(float), p, unit)
        out.append({"target": tkey, "baseline": name, "degenerate": degenerate, **m})
        print(f"{name:28s} {m['MAE_log']:9.4f} {m['R2_log']:8.4f} "
              f"{m['MAE_'+unit]:12.5g} {m['R2_lin']:9.4f}")
        rr = regression_report(te[log_col].to_numpy(float), p)
        print(f"{'':28s}  (log-space RMSE {rr['RMSE']:.4f}  median-AE {rr['median_AE']:.4f}"
              f"  max-error {rr['max_error']:.4f})")

    if stratify and not degenerate:
        _stratify(te, log_col, preds, unit)
    return out


def _stratify(te, log_col, preds, unit):
    """
    Error by target decile and magnitude band.

    A single aggregate MAE on a distribution this skewed is exactly the number
    the paper reports, and it is the one that hides the failure mode: a model
    that nails the common small values and misses every large one still looks
    good on the mean.
    """
    y = te[log_col].to_numpy(float)
    key = "log peak amplitude"
    if key not in preds:
        return
    err = np.abs(preds[key] - y)

    print(f"\n  MAE_log of '{key}' by target decile (does it hold at the top?):")
    print("     n_events matters more than n_rows in the top deciles: one large")
    print("     earthquake supplies many correlated rows, so a decile spanning a")
    print("     handful of events is that many events' worth of noise, not a trend.")
    dec = pd.qcut(y, 10, labels=False, duplicates="drop")
    ev = te.event_id.to_numpy()
    for i in sorted(set(dec)):
        m = dec == i
        print(f"     decile {i+1:2d}  n={int(m.sum()):5d}  events={len(set(ev[m])):4d}  "
              f"true median {10**np.median(y[m]):10.4g} {unit:5s}  MAE_log {err[m].mean():.4f}")

    if "magnitude" in te:
        print("\n  MAE_log by magnitude band:")
        band = pd.cut(te.magnitude, [0, 2.5, 3.0, 3.5, 4.0, 10.0])
        for iv, idx in te.groupby(band, observed=True).groups.items():
            m = te.index.isin(idx)
            if m.sum() > 20:
                print(f"     M {str(iv):12s} n={int(m.sum()):5d}  MAE_log {err[m].mean():.4f}")


def main():
    p = argparse.ArgumentParser(description="Non-neural floors for peak ground motion.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--keep-truncated", action="store_true",
                   help="Keep rows whose forward label ran past the record end.")
    p.add_argument("--keep-clipped", action="store_true",
                   help="Keep rows with a raw sample at the digitizer rail.")
    p.add_argument("--out-csv", default="groundmotion_baseline_results.csv")
    args = p.parse_args()

    d = load(args.manifest, drop_truncated=not args.keep_truncated,
             drop_clipped=not args.keep_clipped)

    rows = []
    for tkey in TARGETS:
        r = run_target(d, tkey)
        if r:
            rows.extend(r)

    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}")

    print(f"\n{'='*94}")
    print("THE NUMBER THE NETWORK HAS TO BEAT (R2_log of 'log peak amplitude', test split)")
    print(f"{'='*94}")
    floor = res[res.baseline == "log peak amplitude"].set_index("target")
    best = res[res.baseline != "ORACLE +magnitude"].groupby("target").R2_log.max()
    for tkey, (_, _, _, _, degen) in TARGETS.items():
        if tkey not in floor.index:
            continue
        tag = "  [DEGENERATE -- not a real task]" if degen else ""
        print(f"  {tkey:10s} amplitude-only R2_log {floor.loc[tkey,'R2_log']:+.4f}   "
              f"best non-oracle {best[tkey]:+.4f}{tag}")
    print("\n  Any CNN result reported without these next to it is unreported (report.md 8).")
    print("\n  Neither target is a clean 'forecast peak ground motion from 3 s' task on this")
    print("  corpus, and both flaws are measured rather than suspected:")
    print("    _full  the paper's quantity -- its window CONTAINS the input, so amplitude")
    print("           bounds it below and equals it outright for ~32 % of rows.")
    print("    _fwd   no overlap with the input, but confounded by S-P moveout: the input")
    print("           closes at a fixed +2.4 s while the S wave moves out with distance,")
    print("           so the label is partly a measure of whether S landed in the window.")
    print("  Report both flaws. Do not present either number as the clean version.")


if __name__ == "__main__":
    main()
