"""
Per-zone vs. pooled forecasting.

The pooled model reaches test AUC 0.72 but that number is carried by a single
fault zone: AEGEAN scores 0.751 while EAFZ (0.484) and NAFZ (0.470) sit at or
below chance. A single model over four seismotectonic provinces assumes one
mapping from seismicity statistics to hazard applies to all of them, which is
physically doubtful -- the Aegean is an extensional province, the NAFZ and
EAFZ are strike-slip, and their b-values, rates and recurrence differ.

This compares four strategies, evaluated PER ZONE so the pooled number can
never hide a zone that does not work:

  persistence   a qualifying event in the previous horizon -> predict another
  pooled        one model over all zones (the current forecaster)
  pooled+zone   pooled, with the zone as an explicit categorical feature
  per-zone      a separate model fit on that zone alone

Split dates are computed ONCE on the pooled data and applied to every zone, so
all zones are tested on the same calendar period and the comparison is not
confounded by one zone being evaluated on a quieter era than another.

Usage:
    python forecast_perzone.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_downloader"))
from seismic_cli.forecast import FAULT_ZONES, FEATURES, build_dataset  # noqa: E402


def fit_predict(kind, Xtr, ytr, Xte, seed):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean()))
    if kind == "logistic":
        m = LogisticRegression(max_iter=2000).fit(np.nan_to_num(Xtr, nan=0.0), ytr)
        return m.predict_proba(np.nan_to_num(Xte, nan=0.0))[:, 1]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=15, l2_regularization=1.0,
                                       early_stopping=True, random_state=seed).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def auc_or_nan(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def main():
    p = argparse.ArgumentParser(description="Per-zone vs pooled forecasting.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--window-events", type=int, default=64)
    p.add_argument("--stride-events", type=int, default=8)
    p.add_argument("--model", default="logistic", choices=["logistic", "gradboost"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = build_dataset(args.catalog, FAULT_ZONES, window_events=args.window_events,
                      stride_events=args.stride_events, threshold=args.threshold,
                      horizon_days=args.horizon_days)

    # One set of split dates for everybody.
    t0, t1 = d.end_time.min(), d.end_time.max()
    span = (t1 - t0).total_seconds() / 86400.0
    cut_tr = t0 + pd.Timedelta(days=span * 0.70)
    cut_va = t0 + pd.Timedelta(days=span * 0.85)
    emb = pd.Timedelta(days=args.horizon_days)
    d = d.copy()
    d["split"] = np.where(d.end_time <= cut_tr - emb, "train",
                  np.where((d.end_time > cut_tr) & (d.end_time <= cut_va - emb), "val",
                  np.where(d.end_time > cut_va, "test", "drop")))
    print(f"\n[split] shared cuts: train <= {cut_tr.date()} | test > {cut_va.date()} "
          f"| {args.horizon_days:.0f}-day embargo | dropped {(d.split=='drop').sum()}")

    tr_all, te_all = d[d.split == "train"], d[d.split == "test"]
    zones = [z for z in FAULT_ZONES if (te_all.region == z).any()]

    # pooled, and pooled + zone one-hot
    Xtr = tr_all[FEATURES].to_numpy(float); ytr = tr_all.label.to_numpy()
    Xte = te_all[FEATURES].to_numpy(float)
    s_pooled = fit_predict(args.model, Xtr, ytr, Xte, args.seed)

    oh_tr = pd.get_dummies(tr_all.region).reindex(columns=zones, fill_value=0).to_numpy(float)
    oh_te = pd.get_dummies(te_all.region).reindex(columns=zones, fill_value=0).to_numpy(float)
    s_zonefeat = fit_predict(args.model, np.hstack([Xtr, oh_tr]), ytr,
                             np.hstack([Xte, oh_te]), args.seed)

    te_all = te_all.copy()
    te_all["s_pooled"] = s_pooled
    te_all["s_zonefeat"] = s_zonefeat

    print(f"\n{'='*78}")
    print(f"PER-ZONE TEST AUC   (model = {args.model})")
    print(f"{'='*78}")
    print(f"{'zone':9s} {'n_test':>7s} {'pos':>6s} {'n_train':>8s} | "
          f"{'persist':>8s} {'pooled':>8s} {'pool+zone':>9s} {'per-zone':>9s}")
    print("-" * 78)

    summary = []
    for z in zones:
        te = te_all[te_all.region == z]
        tr = tr_all[tr_all.region == z]
        y = te.label.to_numpy()

        ds = te.days_since_prev_major.to_numpy(float)
        a_persist = auc_or_nan(y, np.where(np.isfinite(ds), -ds, -1e9))
        a_pooled = auc_or_nan(y, te.s_pooled.to_numpy())
        a_zf = auc_or_nan(y, te.s_zonefeat.to_numpy())

        s_pz = fit_predict(args.model, tr[FEATURES].to_numpy(float), tr.label.to_numpy(),
                           te[FEATURES].to_numpy(float), args.seed)
        a_pz = auc_or_nan(y, s_pz)

        print(f"{z:9s} {len(te):7d} {y.mean():6.3f} {len(tr):8d} | "
              f"{a_persist:8.4f} {a_pooled:8.4f} {a_zf:9.4f} {a_pz:9.4f}")
        summary.append(dict(zone=z, n=len(te), persist=a_persist, pooled=a_pooled,
                            zonefeat=a_zf, perzone=a_pz))

    sm = pd.DataFrame(summary)
    print("-" * 78)
    # Macro = each zone counts once. Micro = window-weighted, which is what the
    # pooled headline reports and is why one large zone can carry it.
    print(f"{'MACRO avg':9s} {'':7s} {'':6s} {'':8s} | "
          f"{sm.persist.mean():8.4f} {sm.pooled.mean():8.4f} "
          f"{sm.zonefeat.mean():9.4f} {sm.perzone.mean():9.4f}")
    w = sm.n / sm.n.sum()
    print(f"{'MICRO avg':9s} {'':7s} {'':6s} {'':8s} | "
          f"{(sm.persist*w).sum():8.4f} {(sm.pooled*w).sum():8.4f} "
          f"{(sm.zonefeat*w).sum():9.4f} {(sm.perzone*w).sum():9.4f}")

    print("\nZones where each strategy beats persistence by >0.02 AUC:")
    for col in ("pooled", "zonefeat", "perzone"):
        wins = sm[sm[col] > sm.persist + 0.02].zone.tolist()
        print(f"  {col:9s} {len(wins)}/{len(sm)}  {wins}")

    best = max(("pooled", "zonefeat", "perzone"), key=lambda c: sm[c].mean())
    print(f"\nBest by MACRO average (every zone counts once): {best} "
          f"({sm[best].mean():.4f})")


if __name__ == "__main__":
    main()
