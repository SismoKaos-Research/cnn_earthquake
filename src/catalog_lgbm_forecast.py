"""
LightGBM directly on the same catalog features `cnn_lstm_catalog_waveform_fusion.py`'s
MLP catalog branch uses -- a non-neural comparison, matching this project's
existing ~0.73 pooled-AUC catalog-based baseline (report.md) more closely
than a small hand-built MLP does. Cheap: seconds, not epochs, no GPU, no
windowing (each row is one hour's catalog-feature snapshot, not a multi-hour
sequence -- GBMs don't need the sequence framing neural branches do).

Motivated by tonight's finding that the catalog features are genuinely
informative once given the right tool (MLP beat an LSTM badly-suited to
near-constant-within-window inputs; AUC 0.5756 on fold 1, first clean
floor-beat of the day). A GBM is the more natural fit for small tabular
feature sets like this one and is worth checking before assuming the MLP
result is close to the ceiling.

Usage:
    python catalog_lgbm_forecast.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --horizon-days 14 --cv-folds 2

Not imported by anything else -- standalone script.
"""

import argparse

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss

from cnn_lstm_catalog_waveform_fusion import FEATURE_NAMES, build_catalog_features
from seismolib.catalog import compute_beta_statistic, days_since_prev_major, label_hours, label_hours_beta_precursor, load_aegean_events, load_aegean_events_with_location, truncate_to_reliable_catalog_end
from seismolib.metrics import safe_auc
from seismolib.splits import print_split_diagnostics, walk_forward_splits
from seismolib.metrics import binary_report, print_report
from seismolib.waveform import load_hourly_raw, load_hourly_raw_consolidated


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Only used to get hour_index/n_hours -- "
                  "raw waveform itself is never touched.")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--bg-min-mag", type=float, default=3.0,
                  help="Minimum magnitude for the background catalog used in the richer "
                       "b-value/energy/deficit features (separate from --threshold, which "
                       "defines the 'major event' being forecast).")
    p.add_argument("--keep-features", nargs="+", default=None, choices=FEATURE_NAMES,
                  metavar="FEATURE", help="Restrict to this subset of features (by name, from "
                       "FEATURE_NAMES) instead of all 11 -- e.g. an RFE-picked subset. "
                       "Default: keep all.")
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--beta-label", action="store_true",
                  help="Convertito et al. 2024-style precursor labeling: narrow the standard "
                       "'M>=threshold within horizon_days' positive class to hours that also "
                       "show a statistically significant seismicity-rate acceleration "
                       "(beta-statistic), instead of every hour within the horizon.")
    p.add_argument("--beta-recent-days", type=float, default=7.0)
    p.add_argument("--beta-baseline-days", type=float, default=30.0)
    p.add_argument("--beta-threshold", type=float, default=1.645)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--embargo-hours", type=int, default=24,
                  help="Small embargo between chronological blocks -- softer justification "
                       "than the raw-waveform scripts' window-overlap leakage (catalog "
                       "features are single-hour snapshots, not windows), but adjacent hours "
                       "are still highly autocorrelated so a modest gap is cheap insurance.")
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    p.add_argument("--num-boost-round", type=int, default=200)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_fold(fold_label, cat_features, labels, dsp, hour_index, train_idx, val_idx, test_idx,
            args, feature_names):
    """Fits one LightGBM model on one split and reports it.

    Args:
        fold_label: Header string printed above this fold's report.
        cat_features: Per-hour catalog feature array.
        labels: Per-hour binary labels.
        dsp: Days-since-previous-major-event array, for the persistence floor.
        hour_index: DatetimeIndex of hour starts, for split diagnostics.
        train_idx: Hour indices for the training split.
        val_idx: Hour indices for the validation split.
        test_idx: Hour indices for the test split.
        args: Parsed CLI args.

    Returns:
        Tuple of (test_auc, floor_auc, report_dict), or None if the split
        is too thin (fewer than 10 train or 5 test rows).
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

    train_set = lgb.Dataset(cat_features[train_idx], label=labels[train_idx])
    val_set = lgb.Dataset(cat_features[val_idx], label=labels[val_idx], reference=train_set)

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

    test_score = booster.predict(cat_features[test_idx], num_iteration=booster.best_iteration)
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
    print(f"\n--- Catalog LightGBM ---")
    print(f"  best_iteration={booster.best_iteration}   test AUC {test_auc:.4f}   n={len(yt)}")

    floor = max(0.5, base_auc, pers_auc)
    report = binary_report(yt, test_score)
    bss = (float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
          else 1.0 - report["brier"] / pers_brier)
    report["brier_skill_score_vs_persistence"] = bss
    print_report(f"Catalog LightGBM ({fold_label}, test set)", report)

    importances = dict(zip(feature_names, booster.feature_importance(importance_type="gain")))
    print(f"  feature importance (gain): {importances}")
    return test_auc, floor, report


def main():
    """Loads the archive/catalog, builds catalog features, and runs the fold sweep."""
    args = parse_args()

    print("Loading catalog and building hourly catalog features...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    bg_times, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag)
    # See cnn_lstm_catalog_waveform_fusion.py: skip the O(n_bg * NND_LOOKBACK) NND
    # precompute when no location-derived feature is being kept.
    if args.keep_features is not None and not any(
            f in args.keep_features for f in ("nnd_log_eta_90d", "shannon_entropy_90d")):
        bg_lats = bg_lons = None
    print(f"  {len(hour_index)} hourly timestamps, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog, {len(bg_times)} M>={args.bg_min_mag} background events")

    hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                        buffer_days=args.horizon_days)
    del raw  # only needed hour_index; drop the (possibly large) waveform array immediately

    dsp = days_since_prev_major(hour_index, major_times)
    cat_features = build_catalog_features(hour_index, major_times, dsp, bg_times, bg_mags,
                                          args.bg_min_mag, bg_lats, bg_lons)
    if args.beta_label:
        beta = compute_beta_statistic(hour_index, bg_times, args.beta_recent_days,
                                      args.beta_baseline_days)
        labels = label_hours_beta_precursor(hour_index, major_times, args.horizon_days, beta,
                                            args.beta_threshold)
        print(f"  beta-precursor labeling: recent={args.beta_recent_days}d "
             f"baseline={args.beta_baseline_days}d threshold={args.beta_threshold} "
             f"-> {(beta > args.beta_threshold).mean():.3f} of hours pass the beta test")
    else:
        labels = label_hours(hour_index, major_times, args.horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    if args.keep_features is not None:
        keep_idx = [FEATURE_NAMES.index(f) for f in args.keep_features]
        cat_features = cat_features[:, keep_idx]
        feature_names = args.keep_features
        print(f"  restricting to {len(keep_idx)} feature(s): {feature_names}")
    else:
        feature_names = FEATURE_NAMES

    n = len(hour_index)
    valid_indices = np.arange(n)

    # --embargo-hours covers autocorrelation between adjacent hourly snapshots; the
    # label additionally looks horizon_days FORWARD, so without the horizon term the
    # last ~horizon_days of each block carry labels determined by events in the NEXT
    # block (train labels encoding val, val labels encoding test). Closing that
    # overlapping-label leak matters here specifically because this script and
    # catalog_feature_rfe.py are what rank features -- ranking under the leak would
    # select features for the wrong objective (Lopez de Prado, AFML Ch. 7).
    embargo = args.embargo_hours + int(round(args.horizon_days * 24))
    if args.cv_folds <= 1:
        i_train = int(n * args.train_frac)
        i_val = int(n * (args.train_frac + args.val_frac))
        folds = [(valid_indices[:i_train], valid_indices[i_train + embargo:i_val],
                 valid_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_indices, args.cv_folds, embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, cat_features, labels, dsp, hour_index,
                          train_idx, val_idx, test_idx, args, feature_names)
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
