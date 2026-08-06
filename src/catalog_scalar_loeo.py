"""
Scalar baseline for the catalog risk task, under identical leave-one-event-out
folds to `cnn_lstm_loeo.py`.

**Why this exists.** `cnn_lstm_loeo.py` compares the dual-channel model only
against per-fold majority-class prediction. On the three-class waveform risk
task that turned out to be nowhere near a sufficient floor: a gradient-boosted
model over two physical scalars beat the CNN by nine accuracy points, and the
gap went unnoticed at first because the scalar baseline crashed silently
(report.md 8.4, defect 16). The catalog task is more exposed to the same
failure, not less: its nine `aux` features (`b_value`, `lyapunov`, `log_rate`,
`log_total_energy`, ...) are exactly the "seismicity indicators" that
comparable published CNN-LSTM forecasting work builds its entire model from.
If those nine numbers already carry the signal, the sequence and image channels
are decoration, and the only way to know is to run them under the same folds.

Every window's aux features are already columns in `manifest.csv`, so this
reads no tensors and runs in seconds.

Folds are grouped by `(region, target_time)` exactly as `cnn_lstm_loeo.py`
groups them, so the two scripts' pooled numbers are directly comparable.

**Two pooled numbers are reported, and they answer different questions.**
Fold sizes are extremely uneven on this dataset (1 to 398 windows, median 14),
so pooling every held-out window weights a single large aftershock-rich
episode ~400x a small one. Window-weighted pooling answers "of all windows,
how many were right"; the per-fold mean answers "of all target events, how
well did we do on a typical one". The second is the one that matches the LOEO
design, and both are printed.

Usage:
    python catalog_scalar_loeo.py --dataset-dir ../../data_downloader/data/catalog_dataset_pooled
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)

from cnn_lstm import risk_classes_from_manifest

AUX_FEATURES = ["n_events", "log_duration_days", "log_rate", "mean_mag", "max_mag",
                "log_total_energy", "b_value", "lyapunov", "mag_std"]


def make_model(name: str, seed: int):
    if name == "gradboost":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=seed)
    if name == "logistic":
        return LogisticRegression(max_iter=2000)
    raise ValueError(f"unknown model {name!r}")


def parse_args():
    p = argparse.ArgumentParser(description="Scalar LOEO baseline for the catalog risk task.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory containing manifest.csv from `generate-catalog-dataset`.")
    p.add_argument("--model", default="gradboost", choices=["gradboost", "logistic"])
    p.add_argument("--min-fold-test", type=int, default=1,
                   help="Skip target events with fewer than this many windows "
                        "(match cnn_lstm_loeo.py's value for a like-for-like comparison).")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    for col in ("region", "target_time", "risk_class"):
        if col not in manifest.columns:
            raise SystemExit(f"manifest.csv is missing '{col}'.")
    manifest["target_time"] = pd.to_datetime(manifest.target_time)

    classes, class_to_idx = risk_classes_from_manifest(manifest)
    y_all = manifest.risk_class.map(class_to_idx).to_numpy()
    X_all = manifest[AUX_FEATURES].to_numpy(dtype=np.float64)

    print("[classes] ordered by actual time-to-next-mainshock:")
    spans = manifest.groupby("risk_class").days_to_major.agg(["min", "max"])
    for c in classes:
        print(f"            {c:12s} {spans.loc[c, 'min']:8.1f} - {spans.loc[c, 'max']:8.1f} days")

    groups = [(k, g) for k, g in manifest.groupby(["region", "target_time"])
              if len(g) >= args.min_fold_test]
    if args.max_folds:
        groups = groups[:args.max_folds]
    if len(groups) < 3:
        raise SystemExit(f"Only {len(groups)} usable fold(s).")

    print("=" * 64)
    print(f"Scalar LOEO baseline | model='{args.model}' | {len(groups)} folds | "
          f"{len(AUX_FEATURES)} features")
    print("=" * 64)

    all_pos = np.arange(len(manifest))
    yts, pts, majs, fold_rows = [], [], [], []
    for i, ((region, target_time), test_rows) in enumerate(groups):
        test_idx = test_rows.index.to_numpy()
        train_idx = np.setdiff1d(all_pos, test_idx, assume_unique=True)

        clf = make_model(args.model, args.seed + i).fit(X_all[train_idx], y_all[train_idx])
        pt = clf.predict(X_all[test_idx])
        yt = y_all[test_idx]
        # Per-fold majority from THIS fold's training pool -- identical
        # definition to cnn_lstm_loeo.py's, so the floors line up too.
        maj = int(np.bincount(y_all[train_idx], minlength=len(classes)).argmax())
        maj_pred = np.full_like(yt, maj)

        yts.append(yt); pts.append(pt); majs.append(maj_pred)
        fold_rows.append((region, target_time, len(test_rows),
                          float((yt == pt).mean()), float((yt == maj_pred).mean())))

    yt = np.concatenate(yts); pt = np.concatenate(pts); maj = np.concatenate(majs)
    acc, bal = float((yt == pt).mean()), balanced_accuracy_score(yt, pt)
    kappa = cohen_kappa_score(yt, pt)
    maj_acc, maj_bal = float((yt == maj).mean()), balanced_accuracy_score(yt, maj)
    chance = 1.0 / len(classes)

    print("\n" + "=" * 64)
    print(f"POOLED (window-weighted) across {len(groups)} folds / {len(yt)} held-out windows")
    print("=" * 64)
    print(f"  chance ({len(classes)} classes, globally balanced)   acc {chance:.4f}"
          f"   <- THE floor that matters")
    print(f"  majority-class (per-fold train mode)  acc {maj_acc:.4f}   balanced {maj_bal:.4f}")
    print(f"  scalar {args.model:<22s}         acc {acc:.4f}   balanced {bal:.4f}   "
          f"kappa {kappa:+.4f}")
    print(f"  vs chance:         {acc - chance:+.4f} accuracy")
    print(f"  vs majority-class: {acc - maj_acc:+.4f} accuracy, {bal - maj_bal:+.4f} balanced")

    # The per-fold majority baseline is ANTI-predictive on this dataset and
    # must not be quoted as the floor. Classes are globally balanced, so
    # removing one fold tips the remaining counts AWAY from that fold's own
    # dominant class; the resulting "majority" is the class the fold has
    # LEAST of. Measured on the pooled 4-region set: it matches the fold's
    # true mode in 2/264 folds and the fold's rarest class in 175/264.
    # Beating it is therefore evidence of nothing.
    if maj_acc < chance - 0.05:
        print(f"\n  [!] The majority-class figure ({maj_acc:.4f}) is far BELOW chance "
              f"({chance:.4f}).")
        print("      That is an artifact of balanced classes under leave-one-event-out:")
        print("      removing a fold tips the training pool away from that fold's own")
        print("      dominant class, so the 'majority' is the class the fold has least of.")
        print("      Compare against chance, not against this number.")
    if kappa <= 0.0:
        print(f"\n  [!] kappa {kappa:+.4f} <= 0: agreement is no better than chance.")
    elif acc < chance:
        print(f"\n  [!] accuracy is below the {chance:.4f} chance rate.")

    print("\nConfusion matrix (rows = true, cols = predicted), pooled:")
    print(pd.DataFrame(confusion_matrix(yt, pt, labels=range(len(classes))),
                       index=classes, columns=classes))
    print("\n" + classification_report(yt, pt, labels=range(len(classes)),
                                       target_names=classes, digits=4, zero_division=0))

    fdf = pd.DataFrame(fold_rows, columns=["region", "target_time", "n_test", "acc", "majority_acc"])
    print(f"PER-FOLD (event-weighted) accuracy: mean {fdf.acc.mean():.4f}  "
          f"std {fdf.acc.std():.4f}   (majority-baseline mean {fdf.majority_acc.mean():.4f})")
    print(f"  fold sizes: min {fdf.n_test.min()}  median {fdf.n_test.median():.0f}  "
          f"max {fdf.n_test.max()} -- window-weighted pooling favours the large folds")
    print("\nPer-region breakdown:")
    for region, g in fdf.groupby("region"):
        print(f"  {region:10s} folds={len(g):3d}  mean acc {g.acc.mean():.4f}  "
              f"(majority-baseline {g.majority_acc.mean():.4f})")


if __name__ == "__main__":
    main()
