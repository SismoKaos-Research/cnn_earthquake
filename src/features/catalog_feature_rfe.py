"""
Recursive feature elimination over the 11 catalog features in
`cnn_lstm_catalog_waveform_fusion.build_catalog_features`, using LightGBM
gain-importance as the ranking signal.

Motivated by the rich-feature catalog-only run underperforming the old
4-feature baseline (ensemble test AUC 0.4916 vs 0.5756) despite one seed
individually beating it -- a sign the extra features may be adding noise
rather than signal for a training set this size (~3.5k rows/fold). Two of
the 11 (`count_7d`, `count_30d`) already showed ~0 LightGBM gain in every
fold tried so far.

At each step: fit LightGBM on the current feature subset across the
walk-forward CV folds, record mean test AUC, then drop whichever remaining
feature has the lowest total gain-importance summed across folds. Repeats
until 1 feature is left, so the full AUC-vs-subset-size trajectory can be
read off and the peak picked by hand.

Usage:
    python catalog_feature_rfe.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --horizon-days 14 --cv-folds 3

Not imported by anything else -- standalone script.
"""

import argparse

import lightgbm as lgb
import numpy as np

from forecasting.cnn_lstm_catalog_waveform_fusion import CATALOG_DIM, FEATURE_NAMES, build_catalog_features
from seismolib.catalog import days_since_prev_major, label_hours, load_aegean_events, load_aegean_events_with_location, truncate_to_reliable_catalog_end
from seismolib.metrics import safe_auc
from seismolib.splits import walk_forward_splits
from seismolib.waveform import load_hourly_raw, load_hourly_raw_consolidated

assert len(FEATURE_NAMES) == CATALOG_DIM


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Only used to get hour_index/n_hours -- "
                  "raw waveform itself is never touched.")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--bg-min-mag", type=float, default=3.0)
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--cv-folds", type=int, default=3,
                  help="More folds than the usual 2 for a steadier importance/AUC signal -- "
                       "RFE is only as good as the ranking it eliminates on.")
    p.add_argument("--embargo-hours", type=int, default=24)
    p.add_argument("--num-boost-round", type=int, default=200)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_and_score(cat_features, labels, dsp, folds, feature_idx, args):
    """Fits LightGBM on one feature subset across all CV folds.

    Args:
        cat_features: Full (n_hours, CATALOG_DIM) feature array.
        labels: Per-hour binary labels.
        dsp: Days-since-previous-major-event array, for the persistence floor.
        folds: List of (train_idx, val_idx, test_idx) from walk_forward_splits.
        feature_idx: Column indices of the features to keep for this subset.
        args: Parsed CLI args.

    Returns:
        Tuple of (mean_test_auc, mean_floor_auc, beats_floor_count, n_folds_used,
        total_gain_per_feature) where total_gain_per_feature is an array aligned
        with feature_idx, summed across folds.
    """
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
    """Loads catalog features once, then runs the elimination loop."""
    args = parse_args()

    print("Loading catalog and building hourly catalog features...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    bg_times, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag)
    print(f"  {len(hour_index)} hourly timestamps, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events, {len(bg_times)} M>={args.bg_min_mag} background events")

    hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                        buffer_days=args.horizon_days)
    del raw

    dsp = days_since_prev_major(hour_index, major_times)
    cat_features = build_catalog_features(hour_index, major_times, dsp, bg_times, bg_mags,
                                          args.bg_min_mag, bg_lats, bg_lons)
    labels = label_hours(hour_index, major_times, args.horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}\n")

    n = len(hour_index)
    # See catalog_lgbm_forecast.py: --embargo-hours alone leaves the label's
    # horizon_days forward window straddling block boundaries, so train labels encode
    # what happens in val and val labels encode what happens in test. RFE is the most
    # leak-sensitive tool here -- it picks which features survive, so ranking under a
    # leak selects for the wrong objective.
    embargo = args.embargo_hours + int(round(args.horizon_days * 24))
    folds = walk_forward_splits(np.arange(n), args.cv_folds, embargo=embargo)

    remaining = list(range(CATALOG_DIM))
    history = []

    print(f"{'=' * 78}\nRecursive feature elimination ({args.cv_folds}-fold walk-forward CV)\n{'=' * 78}")
    header = f"{'n_feat':>6}  {'mean test AUC':>14}  {'mean floor':>11}  {'beats floor':>12}  dropped"
    print(header)

    while remaining:
        mean_auc, mean_floor, beats, n_used, gains = fit_and_score(
            cat_features, labels, dsp, folds, remaining, args)
        history.append((len(remaining), [FEATURE_NAMES[i] for i in remaining], mean_auc,
                        mean_floor, beats, n_used))

        if len(remaining) == 1:
            print(f"{len(remaining):>6}  {mean_auc:>14.4f}  {mean_floor:>11.4f}  "
                 f"{beats:>9}/{n_used}  -- (last feature, stopping)")
            break

        worst_pos = int(np.argmin(gains))
        dropped_feature = FEATURE_NAMES[remaining[worst_pos]]
        print(f"{len(remaining):>6}  {mean_auc:>14.4f}  {mean_floor:>11.4f}  "
             f"{beats:>9}/{n_used}  {dropped_feature}")
        remaining.pop(worst_pos)

    print(f"\n{'=' * 78}\nSummary: mean test AUC by subset size\n{'=' * 78}")
    best = max(history, key=lambda h: (h[2] if np.isfinite(h[2]) else -1))
    for n_feat, feats, mean_auc, mean_floor, beats, n_used in history:
        marker = "  <-- best" if (n_feat, feats) == (best[0], best[1]) else ""
        print(f"  n={n_feat:>2}  AUC {mean_auc:.4f}  floor {mean_floor:.4f}  "
             f"beats {beats}/{n_used} folds{marker}")

    print(f"\nBest subset (n={best[0]}, mean test AUC {best[2]:.4f}, "
         f"beats floor in {best[4]}/{best[5]} folds):")
    for f in best[1]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
