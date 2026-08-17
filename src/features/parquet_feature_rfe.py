"""
Recursive Feature Elimination operating on pre-computed Parquet features.
Bypasses all feature-extraction overhead to run rapid LightGBM subsets.

Usage:
    python parquet_feature_rfe.py \
        --parquet-path combined_features_114d.parquet \
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \
        --horizon-days 14 --cv-folds 3
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from seismolib.catalog import (days_since_prev_major, label_hours,
                               load_aegean_events)
from seismolib.metrics import safe_auc
from seismolib.splits import walk_forward_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-path", required=True, help="Pre-computed features from build_offline_features.py")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--embargo-hours", type=int, default=24)
    p.add_argument("--num-boost-round", type=int, default=200)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def fit_and_score(cat_features, labels, dsp, folds, feature_idx, args):
    """Fits LightGBM on the current subset, returning test AUC and feature gains."""
    x = cat_features[:, feature_idx]
    test_aucs, floor_aucs = [], []
    total_gain = np.zeros(len(feature_idx), dtype=np.float64)
    n_used = 0

    for train_idx, val_idx, test_idx in folds:
        if len(train_idx) < 10 or len(test_idx) < 5 or len(val_idx) < 5:
            continue
            
        train_set = lgb.Dataset(x[train_idx], label=labels[train_idx])
        val_set = lgb.Dataset(x[val_idx], label=labels[val_idx], reference=train_set)
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": args.seed,
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "learning_rate": 0.05,
            "is_unbalance": True,
        }
        
        booster = lgb.train(params, train_set, num_boost_round=args.num_boost_round,
                            valid_sets=[val_set],
                            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                                       lgb.log_evaluation(0)])

        test_score = booster.predict(x[test_idx], num_iteration=booster.best_iteration)
        yt = labels[test_idx]
        test_auc = safe_auc(yt, test_score)

        pos_tr = labels[train_idx].mean()
        base_pred = np.full_like(yt, int(round(pos_tr)), dtype=np.float64)
        base_auc = safe_auc(yt, base_pred)
        pers_dsp = dsp[test_idx]
        pers_pred = np.where(np.isnan(pers_dsp), 0,
                             (pers_dsp <= args.horizon_days).astype(int)).astype(np.float64)
        pers_auc = safe_auc(yt, pers_pred)
        floor = max(0.5, base_auc, pers_auc)

        if np.isfinite(test_auc):
            test_aucs.append(test_auc)
            floor_aucs.append(floor)
            n_used += 1
        total_gain += booster.feature_importance(importance_type="gain")

    if n_used == 0:
        return float("nan"), float("nan"), 0, 0, total_gain

    test_aucs = np.array(test_aucs)
    floor_aucs = np.array(floor_aucs)
    beats_floor = int((test_aucs > floor_aucs).sum())
    return float(test_aucs.mean()), float(floor_aucs.mean()), beats_floor, n_used, total_gain

def main():
    args = parse_args()

    print(f"Loading pre-computed Parquet features from {args.parquet_path}...")
    df_features = pd.read_parquet(args.parquet_path)
    hour_index = df_features.index
    feature_names = df_features.columns.tolist()
    cat_features = df_features.to_numpy(dtype=np.float32)

    print("Loading catalog and building labels...")
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    dsp = days_since_prev_major(hour_index, major_times)
    labels = label_hours(hour_index, major_times, args.horizon_days)
    
    print(f"  {len(hour_index)} hourly records, {len(feature_names)} starting features")
    print(f"  hourly positive rate: {labels.mean():.3f}\n")

    embargo = args.embargo_hours + int(round(args.horizon_days * 24))
    folds = walk_forward_splits(np.arange(len(hour_index)), args.cv_folds, embargo=embargo)

    remaining = list(range(len(feature_names)))
    history = []

    print(f"{'=' * 78}\nRecursive feature elimination ({args.cv_folds}-fold walk-forward CV)\n{'=' * 78}")
    header = f"{'n_feat':>6}  {'mean test AUC':>14}  {'mean floor':>11}  {'beats floor':>12}  dropped"
    print(header)

    while remaining:
        mean_auc, mean_floor, beats, n_used, gains = fit_and_score(
            cat_features, labels, dsp, folds, remaining, args)
        
        history.append((len(remaining), [feature_names[i] for i in remaining], mean_auc,
                        mean_floor, beats, n_used))

        if len(remaining) == 1:
            print(f"{len(remaining):>6}  {mean_auc:>14.4f}  {mean_floor:>11.4f}  "
                 f"{beats:>9}/{n_used}  -- (last feature, stopping)")
            break

        worst_pos = int(np.argmin(gains))
        dropped_feature = feature_names[remaining[worst_pos]]
        print(f"{len(remaining):>6}  {mean_auc:>14.4f}  {mean_floor:>11.4f}  "
             f"{beats:>9}/{n_used}  {dropped_feature}")
        remaining.pop(worst_pos)

    print(f"\n{'=' * 78}\nSummary: mean test AUC by subset size\n{'=' * 78}")
    best = max(history, key=lambda h: (h[2] if np.isfinite(h[2]) else -1))
    for n_feat, feats, mean_auc, mean_floor, beats, n_used in history:
        marker = "  <-- best" if (n_feat, feats) == (best[0], best[1]) else ""
        print(f"  n={n_feat:>3}  AUC {mean_auc:.4f}  floor {mean_floor:.4f}  "
             f"beats {beats}/{n_used} folds{marker}")

    print(f"\nBest subset (n={best[0]}, mean test AUC {best[2]:.4f}, "
         f"beats floor in {best[4]}/{best[5]} folds):")
    for f in best[1]:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
