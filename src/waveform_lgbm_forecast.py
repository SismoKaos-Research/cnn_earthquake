"""
LightGBM on the DWT/spectral waveform features from `waveform_dwt_features.py`
(Bhatia et al. 2023's time/frequency/time-frequency domain feature formulas) --
a quick, non-neural sanity check of whether hand-engineered waveform features
carry any forecasting signal at all, before investing in wiring them into the
fusion model's wave_branch as an alternative to the CNN's learned embedding.

Same role as `catalog_lgbm_forecast.py` played for the catalog branch: cheap
(seconds to minutes, no GPU/epochs) and a useful floor-check before trusting
a much slower neural training loop's result.

102 raw features (34/channel x 3 channels) is almost certainly too many for
this dataset's size -- same lesson as the 11-feature catalog overfit. Run
`--rfe` to prune via the same LightGBM-gain elimination idea as
`catalog_feature_rfe.py` instead of trusting the all-102 number.

Usage:
    python waveform_lgbm_forecast.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --horizon-days 14 --cv-folds 2 --n-jobs 6

Not imported by anything else -- standalone script.
"""

import argparse

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss

from feature_lstm_forecast import (days_since_prev_major, label_hours,
                                   load_aegean_events, print_split_diagnostics,
                                   safe_auc, truncate_to_reliable_catalog_end,
                                   walk_forward_splits)
from metrics import binary_report, print_report
from raw_cnn_lstm_forecast import load_hourly_raw, load_hourly_raw_consolidated
from waveform_dwt_features import build_hourly_waveform_features, feature_names


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--max-days", type=int, default=None,
                  help="Cap on days of data loaded -- useful for a fast smoke test before "
                       "committing to a full-archive feature-extraction pass.")
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1,
                  help="Worker processes for feature extraction (ApEn is the bottleneck, "
                       "~50ms/channel-hour serial). Leave headroom if other GPU jobs are "
                       "running and using dataloader worker threads.")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--embargo-hours", type=int, default=24)
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    p.add_argument("--num-boost-round", type=int, default=200)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rfe", action="store_true",
                  help="Instead of one full-feature run, recursively eliminate the "
                       "lowest-gain feature and re-score, like catalog_feature_rfe.py.")
    return p.parse_args()


def run_fold(fold_label, wave_features, labels, dsp, hour_index, train_idx, val_idx, test_idx,
            args, names):
    """Fits one LightGBM model on one split and reports it. Mirrors
    catalog_lgbm_forecast.py's run_fold, on waveform features instead.
    """
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    train_set = lgb.Dataset(wave_features[train_idx], label=labels[train_idx])
    val_set = lgb.Dataset(wave_features[val_idx], label=labels[val_idx], reference=train_set)

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

    test_score = booster.predict(wave_features[test_idx], num_iteration=booster.best_iteration)
    yt = labels[test_idx]

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt, base_pred)
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(yt)}")
    pers_dsp = dsp[test_idx]
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= args.horizon_days).astype(int)).astype(np.float64)
    pers_auc = safe_auc(yt, pers_pred)
    single_class = len(np.unique(yt)) < 2
    pers_brier = float("nan") if single_class else float(brier_score_loss(yt, pers_pred))
    print(f"  persistence             AUC {pers_auc:.4f}   Brier {pers_brier:.4f}   n={len(yt)}")

    test_auc = safe_auc(yt, test_score)
    print(f"\n--- Waveform DWT LightGBM ---")
    print(f"  best_iteration={booster.best_iteration}   test AUC {test_auc:.4f}   n={len(yt)}")

    floor = max(0.5, base_auc, pers_auc)
    report = binary_report(yt, test_score)
    bss = (float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
          else 1.0 - report["brier"] / pers_brier)
    report["brier_skill_score_vs_persistence"] = bss
    print_report(f"Waveform DWT LightGBM ({fold_label}, test set)", report)

    gains = booster.feature_importance(importance_type="gain")
    top = sorted(zip(names, gains), key=lambda kv: -kv[1])[:10]
    print(f"  top-10 feature importance (gain): {top}")
    return test_auc, floor, report, gains


def fit_and_score_subset(wave_features, labels, dsp, folds, feature_idx, args):
    """Fits/scores one feature subset across all folds -- used by --rfe."""
    x = wave_features[:, feature_idx]
    test_aucs, floor_aucs = [], []
    total_gain = np.zeros(len(feature_idx), dtype=np.float64)
    n_used = 0
    for train_idx, val_idx, test_idx in folds:
        if len(train_idx) < 10 or len(test_idx) < 5 or len(val_idx) < 5:
            continue
        train_set = lgb.Dataset(x[train_idx], label=labels[train_idx])
        val_set = lgb.Dataset(x[val_idx], label=labels[val_idx], reference=train_set)
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1, "seed": args.seed,
            "num_leaves": 15, "min_data_in_leaf": 20, "learning_rate": 0.05, "is_unbalance": True,
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
    test_aucs, floor_aucs = np.array(test_aucs), np.array(floor_aucs)
    beats_floor = int((test_aucs > floor_aucs).sum())
    return float(test_aucs.mean()), float(floor_aucs.mean()), beats_floor, n_used, total_gain


def run_rfe(wave_features, labels, dsp, folds, names, args):
    """Recursive feature elimination over the 102 waveform features, LightGBM-gain ranked."""
    remaining = list(range(wave_features.shape[1]))
    history = []
    print(f"{'=' * 78}\nWaveform-feature RFE ({len(folds)}-fold walk-forward CV, "
         f"{len(remaining)} features)\n{'=' * 78}")
    header = f"{'n_feat':>6}  {'mean test AUC':>14}  {'mean floor':>11}  {'beats floor':>12}  dropped"
    print(header)
    while remaining:
        mean_auc, mean_floor, beats, n_used, gains = fit_and_score_subset(
            wave_features, labels, dsp, folds, remaining, args)
        history.append((len(remaining), [names[i] for i in remaining], mean_auc, mean_floor,
                        beats, n_used))
        if len(remaining) == 1 or len(remaining) <= 2:
            print(f"{len(remaining):>6}  {mean_auc:>14.4f}  {mean_floor:>11.4f}  "
                 f"{beats:>9}/{n_used}  -- (stopping)")
            break
        worst_pos = int(np.argmin(gains))
        dropped = names[remaining[worst_pos]]
        print(f"{len(remaining):>6}  {mean_auc:>14.4f}  {mean_floor:>11.4f}  "
             f"{beats:>9}/{n_used}  {dropped}")
        remaining.pop(worst_pos)

    print(f"\n{'=' * 78}\nSummary: mean test AUC by subset size\n{'=' * 78}")
    best = max(history, key=lambda h: (h[2] if np.isfinite(h[2]) else -1))
    for n_feat, feats, mean_auc, mean_floor, beats, n_used in history:
        marker = "  <-- best" if (n_feat, feats) == (best[0], best[1]) else ""
        print(f"  n={n_feat:>3}  AUC {mean_auc:.4f}  floor {mean_floor:.4f}  "
             f"beats {beats}/{n_used} folds{marker}")
    print(f"\nBest subset (n={best[0]}, mean test AUC {best[2]:.4f}):")
    for f in best[1][:20]:
        print(f"  - {f}")
    if best[0] > 20:
        print(f"  ... ({best[0] - 20} more)")


def main():
    """Loads the archive/catalog, builds waveform DWT features, and runs the fold sweep."""
    args = parse_args()

    print("Loading raw waveform archive...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly timestamps, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events")

    hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                        buffer_days=args.horizon_days)

    dsp = days_since_prev_major(hour_index, major_times)
    labels = label_hours(hour_index, major_times, args.horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    print(f"Building DWT/spectral waveform features ({len(hour_index)} hours, "
         f"n_jobs={args.n_jobs})...")
    wave_features = build_hourly_waveform_features(raw, n_jobs=args.n_jobs)
    del raw
    names = feature_names()
    print(f"  {wave_features.shape[1]} features/hour; any non-finite: "
         f"{bool(np.any(~np.isfinite(wave_features)))}")

    n = len(hour_index)
    valid_indices = np.arange(n)
    if args.cv_folds <= 1:
        i_train = int(n * args.train_frac)
        i_val = int(n * (args.train_frac + args.val_frac))
        folds = [(valid_indices[:i_train], valid_indices[i_train + args.embargo_hours:i_val],
                 valid_indices[i_val + args.embargo_hours:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_indices, args.cv_folds, embargo=args.embargo_hours)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    if args.rfe:
        run_rfe(wave_features, labels, dsp, folds, names, args)
        return

    skip = set(args.skip)
    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            print(f"\n[skip] {fold_label} (--skip)")
            continue
        result = run_fold(fold_label, wave_features, labels, dsp, hour_index,
                          train_idx, val_idx, test_idx, args, names)
        if result is not None:
            results.append(result)

    if args.cv_folds > 1 and results:
        aucs = np.array([r[0] for r in results])
        floors = np.array([r[1] for r in results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(results)}/{args.cv_folds} folds)\n{'=' * 64}")
        print(f"  test AUC per fold: {[f'{a:.4f}' for a in aucs]}")
        print(f"  test AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}")
        print(f"  floor AUC: mean {floors.mean():.4f}  std {floors.std():.4f}")
        print(f"  beats its own fold's floor in {int((aucs > floors).sum())}/{len(results)} folds")


if __name__ == "__main__":
    main()
