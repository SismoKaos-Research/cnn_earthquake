"""
Regression to the next major event: predict days until the next M>=threshold
AEGEAN event for each hour, using catalog features and optionally DWT waveform
features.

Target:
    days_until_next_major = min(
        (next_major_event_time - current_hour).total_hours() / 24.0,
        --max-horizon-days
    )

Hours whose next major event lies beyond --max-horizon-days are assigned
--max-horizon-days (right-censored but kept in the regression).

Usage:
    python next_event_regression.py \
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \
        --consolidated --max-horizon-days 30 --cv-folds 3 --n-jobs 4
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cnn_lstm_catalog_waveform_fusion import (FEATURE_NAMES,
                                              build_catalog_features)
from seismolib.catalog import days_since_prev_major, load_aegean_events, load_aegean_events_with_location, truncate_to_reliable_catalog_end
from seismolib.splits import walk_forward_splits
from seismolib.waveform import load_hourly_raw, load_hourly_raw_consolidated
from waveform_dwt_features import build_hourly_waveform_features
from waveform_dwt_features import feature_names as dwt_feature_names


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--bg-min-mag", type=float, default=3.0)
    p.add_argument("--max-horizon-days", type=float, default=30.0)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--max-days", type=int, default=None,
                   help="Only used if --consolidated is not given.")
    p.add_argument("--waveform", action="store_true",
                   help="Also append DWT/spectral waveform features (102 per hour).")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Workers for DWT feature extraction (only used with --waveform).")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--embargo-hours", type=int, default=24)
    p.add_argument("--log-target", action="store_true",
                   help="Train on log1p(days_until_next) instead of raw days.")
    p.add_argument("--num-boost-round", type=int, default=200)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def days_until_next_major(hour_index, major_times, max_horizon_days):
    """Computes regression target: days until next major event, capped."""
    major_arr = np.asarray(pd.to_datetime(major_times).to_numpy(dtype="datetime64[ns]"))
    hour_arr = np.asarray(hour_index.to_numpy(dtype="datetime64[ns]"))
    idx = np.searchsorted(major_arr, hour_arr, side="right")
    target = np.full(len(hour_index), max_horizon_days, dtype=np.float64)
    valid = idx < len(major_arr)
    if np.any(valid):
        delta_ns = major_arr[idx[valid]] - hour_arr[valid]
        delta_days = delta_ns.astype(np.float64) / (1e9 * 3600 * 24)
        target[valid] = np.minimum(delta_days, max_horizon_days)
    return target


def count_teeth(hour_index, idx, major_times):
    """Distinct major events inside a block -- the block's *effective* sample size.

    The regression target is a sawtooth that resets at every major event, so
    3000 hourly rows spanning 5 events carry roughly 5 independent
    observations, not 3000. Reporting MAE without this number makes a fold
    look far better powered than it is.
    """
    if not len(idx):
        return 0
    block = hour_index[idx]
    major_arr = pd.to_datetime(major_times).to_numpy(dtype="datetime64[ns]")
    lo = block[0].to_datetime64()
    hi = block[-1].to_datetime64()
    return int(np.sum((major_arr >= lo) & (major_arr <= hi)))


def run_fold(fold_label, X, y, hour_index, train_idx, val_idx, test_idx, args,
             major_times, dsp_col):
    """Trains one LightGBM regressor on one split and reports metrics."""
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: target mean {y[idx].mean():.2f} days, "
                  f"median {np.median(y[idx]):.2f} days, "
                  f"censored {100.0 * np.mean(y[idx] >= args.max_horizon_days):.1f}%, "
                  f"distinct events (teeth) {count_teeth(hour_index, idx, major_times)}")

    if len(train_idx) < 20 or len(test_idx) < 10 or len(val_idx) < 10:
        print("[ERROR] Not enough data for a meaningful split.")
        return None

    y_train_raw = y[train_idx]
    y_val_raw = y[val_idx]
    y_test_raw = y[test_idx]

    if args.log_target:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw

    train_set = lgb.Dataset(X[train_idx], label=y_train)
    val_set = lgb.Dataset(X[val_idx], label=y_val, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "seed": args.seed,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "learning_rate": 0.05,
    }
    booster = lgb.train(params, train_set, num_boost_round=args.num_boost_round,
                        valid_sets=[val_set],
                        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                                   lgb.log_evaluation(0)])

    pred = booster.predict(X[test_idx], num_iteration=booster.best_iteration)
    if args.log_target:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, args.max_horizon_days)

    yt = y_test_raw

    model_mae = mean_absolute_error(yt, pred)
    model_rmse = np.sqrt(mean_squared_error(yt, pred))
    model_r2 = r2_score(yt, pred)

    # Constant baseline. MAE is minimised by the median and RMSE by the mean,
    # so scoring a mean-predictor on MAE understates the baseline and inflates
    # the model's apparent edge. Each metric gets its own optimal constant.
    const_mae = mean_absolute_error(yt, np.full_like(yt, np.median(y_train_raw)))
    const_rmse = np.sqrt(mean_squared_error(yt, np.full_like(yt, y_train_raw.mean())))

    # Conditional baseline -- the real floor. `days_until_next` is the mirror of
    # `days_since_prev`, which is already a model input, so a model can look
    # skilful purely by relearning the marginal inter-event distribution. This
    # fits E[target | days_since_prev] alone; the full feature set has to beat
    # it to have contributed anything. Same role the Omori persistence floor
    # plays in the classification branch.
    floor_train = lgb.Dataset(X[train_idx, dsp_col].reshape(-1, 1), label=y_train)
    floor_val = lgb.Dataset(X[val_idx, dsp_col].reshape(-1, 1), label=y_val,
                            reference=floor_train)
    floor_booster = lgb.train(params, floor_train, num_boost_round=args.num_boost_round,
                              valid_sets=[floor_val],
                              callbacks=[lgb.early_stopping(args.early_stopping_rounds,
                                                            verbose=False),
                                         lgb.log_evaluation(0)])
    floor_pred = floor_booster.predict(X[test_idx, dsp_col].reshape(-1, 1),
                                       num_iteration=floor_booster.best_iteration)
    if args.log_target:
        floor_pred = np.expm1(floor_pred)
    floor_pred = np.clip(floor_pred, 0.0, args.max_horizon_days)
    floor_mae = mean_absolute_error(yt, floor_pred)
    floor_rmse = np.sqrt(mean_squared_error(yt, floor_pred))
    floor_r2 = r2_score(yt, floor_pred)

    beats_floor = model_mae < floor_mae

    print(f"\n--- Regression results ({fold_label}) ---")
    print(f"  LightGBM (all feats)  MAE {model_mae:.3f} days, RMSE {model_rmse:.3f}, R² {model_r2:.3f}")
    print(f"  FLOOR (dsp only)      MAE {floor_mae:.3f} days, RMSE {floor_rmse:.3f}, R² {floor_r2:.3f}")
    print(f"  Constant (median/mean) MAE {const_mae:.3f} days, RMSE {const_rmse:.3f}")
    print(f"  --> beats conditional floor: {'YES' if beats_floor else 'NO'} "
          f"({model_mae - floor_mae:+.3f} days MAE)")

    return {
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "model_r2": model_r2,
        "floor_mae": floor_mae,
        "floor_r2": floor_r2,
        "const_mae": const_mae,
        "const_rmse": const_rmse,
        "beats_floor": beats_floor,
        "test_teeth": count_teeth(hour_index, test_idx, major_times),
        "best_iter": booster.best_iteration,
    }


def main():
    args = parse_args()

    print("Loading catalog and hourly index...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)

    major_times = load_aegean_events(args.catalog_path, args.threshold)
    bg_times, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag)

    print(f"  {len(hour_index)} hourly timestamps, {len(major_times)} M>={args.threshold} "
          f"events, {len(bg_times)} M>={args.bg_min_mag} background events")

    hour_index, raw = truncate_to_reliable_catalog_end(
        hour_index, raw, major_times, buffer_days=args.max_horizon_days)

    y = days_until_next_major(hour_index, major_times, args.max_horizon_days)
    print(f"  target stats: mean {y.mean():.3f} days, median {np.median(y):.3f}, "
          f"min {y.min():.3f}, max {y.max():.3f}")

    dsp = days_since_prev_major(hour_index, major_times)
    X_cat = build_catalog_features(hour_index, major_times, dsp, bg_times, bg_mags,
                                   args.bg_min_mag, bg_lats, bg_lons)
    feature_names = list(FEATURE_NAMES)

    if args.waveform:
        print(f"Building DWT waveform features (n_jobs={args.n_jobs})...")
        X_wav = build_hourly_waveform_features(raw, n_jobs=args.n_jobs)
        X = np.hstack([X_cat, X_wav])
        feature_names += dwt_feature_names()
    else:
        X = X_cat
        del raw

    dsp_col = feature_names.index("log1p_dsp")

    n = len(hour_index)
    valid_indices = np.arange(n)

    # Embargo must cover both input window overlap (seq_hours - 1) and the
    # forward-looking target (max_horizon_days).
    embargo = args.embargo_hours + int(round(args.max_horizon_days * 24))

    if args.cv_folds <= 1:
        i_train = int(n * args.train_frac)
        i_val = int(n * (args.train_frac + args.val_frac))
        folds = [(valid_indices[:i_train],
                  valid_indices[i_train + embargo:i_val],
                  valid_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_indices, args.cv_folds, embargo=embargo)
        fold_labels = [f"fold {k+1}/{args.cv_folds}" for k in range(args.cv_folds)]

    results = []
    for fold_label, (train_idx, val_idx, test_idx) in zip(fold_labels, folds):
        if len(train_idx) < 20 or len(val_idx) < 10 or len(test_idx) < 10:
            print(f"\n[skip] {fold_label}: split too small after embargo.")
            continue
        res = run_fold(fold_label, X, y, hour_index, train_idx, val_idx, test_idx, args,
                       major_times, dsp_col)
        if res is not None:
            results.append(res)

    if results:
        print(f"\n{'=' * 64}\nSummary across {len(results)} folds\n{'=' * 64}")
        maes = np.array([r["model_mae"] for r in results])
        rmses = np.array([r["model_rmse"] for r in results])
        floor_maes = np.array([r["floor_mae"] for r in results])
        const_maes = np.array([r["const_mae"] for r in results])
        n_beat = sum(r["beats_floor"] for r in results)
        print(f"  LightGBM MAE : mean {maes.mean():.3f} ± {maes.std():.3f} days")
        print(f"  LightGBM RMSE: mean {rmses.mean():.3f} ± {rmses.std():.3f} days")
        print(f"  FLOOR MAE    : mean {floor_maes.mean():.3f} ± {floor_maes.std():.3f} days (dsp only)")
        print(f"  Constant MAE : mean {const_maes.mean():.3f} ± {const_maes.std():.3f} days (train median)")
        print(f"  MAE gain over floor:    {((floor_maes - maes) / floor_maes * 100).mean():+.1f}%")
        print(f"  MAE gain over constant: {((const_maes - maes) / const_maes * 100).mean():+.1f}%")
        print(f"  beats conditional floor in {n_beat}/{len(results)} folds")
        print(f"  test-set distinct events (teeth) per fold: "
              f"{[r['test_teeth'] for r in results]}")


if __name__ == "__main__":
    main()
