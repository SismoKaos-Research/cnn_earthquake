"""
Regional earthquake forecasting: evaluation with the floors attached.

Target: will a M >= threshold event occur in this fault zone within the next
N days? See `data_downloader/seismic_cli/forecast.py` for why this replaces
the time-to-next-mainshock formulation, which measured at chance.

**Every model is reported next to two floors**, because this project has twice
been burned by a headline with nothing beneath it (report.md 12, defects 16
and the anti-predictive LOEO majority baseline):

  * base rate -- always predict the majority class of the training period
  * persistence -- predict "yes" iff a qualifying event occurred in the
    PREVIOUS `horizon` days. This is the honest domain floor: earthquakes
    cluster, so "it happened recently, so it will happen again" is free and
    surprisingly strong. A model that cannot beat persistence has not learned
    seismology, it has learned that seismicity is bursty.

AUC is the headline rather than accuracy: the positive rate varies from 0.24
to 0.61 across zones, so accuracy is dominated by the base rate and a model
predicting one class can look competent.

Usage:
    python forecast_eval.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_downloader"))
from seismic_cli.forecast import (FAULT_ZONES, FEATURES, build_dataset,  # noqa: E402
                                  chronological_split)


def persistence_scores(part: pd.DataFrame, full: pd.DataFrame, horizon_days: float):
    """
    Domain floor: did a qualifying event occur in the previous `horizon` days?

    Derived from `days_since_prev_major`, which is measured at the window's own
    end time and therefore uses no future information.
    """
    ds = part.days_since_prev_major.to_numpy(dtype=float)
    # score = recency (closer -> higher risk); NaN (no prior major) -> lowest
    score = np.where(np.isfinite(ds), -ds, -1e9)
    pred = (np.isfinite(ds) & (ds <= horizon_days)).astype(int)
    return score, pred


def report(name, y, score, pred, out):
    try:
        auc = roc_auc_score(y, score)
    except ValueError:
        auc = float("nan")
    row = dict(model=name, auc=auc, acc=accuracy_score(y, pred),
               mcc=matthews_corrcoef(y, pred))
    out.append(row)
    print(f"  {name:28s} AUC {row['auc']:.4f}   acc {row['acc']:.4f}   MCC {row['mcc']:+.4f}")
    return row


def main():
    p = argparse.ArgumentParser(description="Regional earthquake forecasting evaluation.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--window-events", type=int, default=64)
    p.add_argument("--stride-events", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = build_dataset(args.catalog, FAULT_ZONES, window_events=args.window_events,
                      stride_events=args.stride_events, threshold=args.threshold,
                      horizon_days=args.horizon_days)
    train, val, test = chronological_split(d, args.horizon_days)
    if test.empty or train.empty:
        raise SystemExit("empty split")

    Xtr = train[FEATURES].to_numpy(float); ytr = train.label.to_numpy()
    Xva = val[FEATURES].to_numpy(float);   yva = val.label.to_numpy()
    Xte = test[FEATURES].to_numpy(float);  yte = test.label.to_numpy()

    print(f"\n{'='*66}\nTEST SET ({len(test)} windows, positive rate {yte.mean():.3f})\n{'='*66}")
    rows = []

    # --- floor 1: base rate ------------------------------------------------
    maj = int(round(ytr.mean()))
    report("base rate (majority)", yte, np.full(len(yte), ytr.mean()),
           np.full(len(yte), maj), rows)

    # --- floor 2: persistence ---------------------------------------------
    sc, pr = persistence_scores(test, d, args.horizon_days)
    report("persistence (recency)", yte, sc, pr, rows)

    # --- models ------------------------------------------------------------
    lg = LogisticRegression(max_iter=2000).fit(np.nan_to_num(Xtr, nan=0.0), ytr)
    sc = lg.predict_proba(np.nan_to_num(Xte, nan=0.0))[:, 1]
    report("logistic (all features)", yte, sc, (sc > 0.5).astype(int), rows)

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        max_leaf_nodes=15, l2_regularization=1.0,
                                        early_stopping=True, random_state=args.seed)
    gb.fit(Xtr, ytr)   # handles NaN natively
    sc_gb = gb.predict_proba(Xte)[:, 1]
    report("gradient boosting", yte, sc_gb, (sc_gb > 0.5).astype(int), rows)

    # --- ablation: does days_since_prev_major carry it? --------------------
    keep = [i for i, f in enumerate(FEATURES) if f != "days_since_prev_major"]
    gb2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=15, l2_regularization=1.0,
                                         early_stopping=True, random_state=args.seed)
    gb2.fit(Xtr[:, keep], ytr)
    sc2 = gb2.predict_proba(Xte[:, keep])[:, 1]
    report("gradboost, no recency feat", yte, sc2, (sc2 > 0.5).astype(int), rows)

    best = max(rows, key=lambda r: (r["auc"] if np.isfinite(r["auc"]) else -1))
    print(f"\nBest by test AUC: {best['model']} ({best['auc']:.4f})")
    gain = best["auc"] - next(r["auc"] for r in rows if r["model"].startswith("persistence"))
    print(f"  vs persistence floor: {gain:+.4f} AUC")
    if gain <= 0.01:
        print("  [!] Does not meaningfully beat persistence. The model has learned that")
        print("      seismicity is bursty, not how to forecast it.")

    print(f"\nConfusion (gradient boosting), rows=true:\n"
          f"{confusion_matrix(yte, (sc_gb > 0.5).astype(int))}")
    print(f"Brier score (gradboost): {brier_score_loss(yte, sc_gb):.4f}  "
          f"(base rate {brier_score_loss(yte, np.full(len(yte), ytr.mean())):.4f})")

    print("\nPer-zone test AUC (gradient boosting):")
    test = test.copy(); test["score"] = sc_gb
    for z, g in test.groupby("region"):
        if g.label.nunique() < 2:
            print(f"  {z:9s} n={len(g):5d}  (single class in test)")
            continue
        print(f"  {z:9s} n={len(g):5d}  positive {g.label.mean():.3f}  "
              f"AUC {roc_auc_score(g.label, g.score):.4f}")


if __name__ == "__main__":
    main()
