"""
LSTM forecaster trained directly on Sismokaos-featureExtract's hand-crafted
continuous features (STA/LTA, Hjorth, permutation entropy, spectral
centroid, cross-axis correlation -- computed on real KO.GEDZ continuous
data, Aegean zone).

Features:
- Robust absolute-time parsing via Unix Epoch (Zaman_Dk) for gap preservation.
- Walk-forward CV for non-stationary timeline evaluation.
- Dual-logging (Terminal + File).
- Real-time epoch tracking.

Usage:
    python feature_lstm_forecast.py \
        --features-csv ../../Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy \
        --catalog-path ../../Sismokaos/data_downloader/catalogs/catalog_current.csv \
        --cv-folds 5
"""

import argparse
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from seismolib.catalog import (days_since_prev_major, label_hours,
                               load_aegean_events, load_hourly_features)
from seismolib.logging import DualLogger
from seismolib.metrics import (binary_report, print_report,  # noqa: F401
                               safe_auc)
from seismolib.model.registry import add_model_args, spec_from_args
from seismolib.model.sequence import SequenceHeadNet
from seismolib.splits import print_split_diagnostics, walk_forward_splits
from seismolib.training import seed_everything


class HourlySeqDataset(Dataset):
    """Windows of `seq_hours` consecutive hourly feature vectors, z-normalized."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_hours: int,
                 indices: np.ndarray, stats=None):
        self.features = features
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
        if stats is None:
            train_feats = np.concatenate([features[max(0, i - seq_hours + 1):i + 1] for i in indices], axis=0)
            with np.errstate(invalid="ignore"):
                mu, sd = np.nanmean(train_feats, axis=0), np.nanstd(train_feats, axis=0) + 1e-6
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            stats = (mu, sd)
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        seq = self.features[start:end + 1].copy()
        mu, sd = self.stats
        seq = (seq - mu) / sd
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return (torch.from_numpy(seq).float(),
                torch.tensor(self.labels[end], dtype=torch.float32))


def parse_args():
    p = argparse.ArgumentParser(description="LSTM forecaster on hand-crafted continuous features.")
    p.add_argument("--features-csv", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=None)
    p.add_argument("--seq-hours", type=int, default=168)
    # The whole sequence family, not just the LSTM: all three take (B, T, F)
    # per-step feature vectors and return one logit, so the training loop below
    # is already generic over them. Until now this script could only build the
    # LSTM, and comparing it against a GRU or a TCN meant a separate script
    # with its own split logic.
    add_model_args(p, family="sequence")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    return p.parse_args()


class ForecastLSTM(SequenceHeadNet):
    """`SequenceHeadNet` with no per-step encoder."""
    def __init__(self, feat_dim, hidden=64, dropout=0.3):
        super().__init__(feat_dim, hidden=hidden, dropout=dropout)


def train_one_seed(args, seed, feature_cols, features, labels,
                   train_idx, val_idx, test_idx, device):
    seed_everything(seed)
    train_ds = HourlySeqDataset(features, labels, args.seq_hours, train_idx)
    val_ds = HourlySeqDataset(features, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = HourlySeqDataset(features, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model = spec_from_args(args).build(feat_dim=len(feature_cols)).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = labels[train_idx].mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for seq, y in loader:
                seq, y = seq.to(device), y.to(device)
                logit = model(seq)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    yv0, _, _ = evaluate(val_loader)
    use_loss_fallback = len(np.unique(yv0)) < 2
    if use_loss_fallback:
        print(f"  [seed {seed}] val split is single-class (positive rate {yv0.mean():.3f}) -- "
               "checkpointing on train AUC instead of val loss.")

    best = -1.0
    no_improve, best_state = 0, None
    for epoch in range(args.epochs):
        model.train()
        for seq, y in train_loader:
            seq, y = seq.to(device), y.to(device)
            loss = criterion(model(seq), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()

        yv, sv, val_loss = evaluate(val_loader)
        val_auc = safe_auc(yv, sv)
        if use_loss_fallback:
            ytr, tr_scores, _ = evaluate(train_loader)
            metric = safe_auc(ytr, tr_scores)
        else:
            metric = val_auc
            
        print(f"  [seed {seed}] ForecastLSTM epoch {epoch+1:02d}/{args.epochs} "
               f"val AUC {val_auc:.4f} val loss {val_loss:.4f}"
               + (f" train AUC {metric:.4f}" if use_loss_fallback else ""))
               
        improved = metric > best
        if improved:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [seed {seed}] Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = evaluate(test_loader)
    print(f"  [seed {seed}] test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def run_fold(fold_label, args, feature_cols, features, labels, dsp, hourly_index,
             train_idx, val_idx, test_idx, seeds, device, horizon_days=None):
    horizon_days = args.horizon_days if horizon_days is None else horizon_days
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hourly_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split -- need more days of features.")
        return None

    print(f"\nTraining ForecastLSTM ({len(seeds)} seed(s)): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, feature_cols, features, labels,
                                train_idx, val_idx, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    ensemble_score = np.mean(per_seed_scores, axis=0)

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(yt_ref)}")
    pers_dsp = dsp[test_idx]
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= horizon_days).astype(int)).astype(np.float64)
    pers_raw = safe_auc(yt_ref, pers_pred)
    # Orientation-corrected, i.e. max(a, 1-a). A persistence rule that scores
    # below 0.5 ranks *inversely*, and an inverted rule is exactly as
    # exploitable as a correct one -- so the bar it sets is 1-a, not a.
    # Without this, any persistence AUC under 0.5 collapsed the floor to a
    # vacuous 0.5: an earlier run of this script scored persistence 0.343,
    # so the floor should have been 0.657 and the ensemble's 0.558 was 0.099
    # *below* it, not 0.058 above. That result was written up in a cheat
    # sheet which has since been deleted rather than corrected.
    pers_auc = max(pers_raw, 1.0 - pers_raw) if pers_raw == pers_raw else pers_raw
    print(f"  persistence            AUC {pers_auc:.4f}   n={len(yt_ref)}"
          f"   (raw {pers_raw:.4f}, oriented)")

    print(f"\n--- Hand-feature LSTM ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
           f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, pers_auc)
    if ensemble_auc <= floor + 1e-9:
        print("\n  [!] Ensemble does NOT clear max(chance, persistence) -- not evidence of forecasting skill.")
    else:
        print(f"\n  Ensemble beats max(chance, persistence) by {ensemble_auc - floor:+.4f} AUC.")

    report = binary_report(yt_ref, ensemble_score)
    print_report(f"Hand-feature LSTM ensemble ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def run_horizon(horizon_days, args, hourly_index, feature_cols, features, major_times, dsp,
                folds, fold_labels, skip, seeds, device):
    print(f"\n{'#' * 64}\n# horizon = {horizon_days:.0f} days\n{'#' * 64}")
    labels = label_hours(hourly_index, major_times, horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, feature_cols, features, labels, dsp, hourly_index,
                          train_idx, val_idx, test_idx, seeds, device, horizon_days=horizon_days)
        if result is not None:
            results.append(result)

    if args.cv_folds <= 1:
        if results:
            print(f"\n  [!] Single station, ~10 months, {len(seeds)}-seed ensemble -- treat as a first look.")
    elif results:
        aucs = np.array([r[0] for r in results])
        floors = np.array([r[1] for r in results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(results)}/{args.cv_folds} folds, "
               f"horizon={horizon_days:.0f}d)\n{'=' * 64}")
        print(f"  ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
        print(f"  ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}"
               f"  ({np.isnan(aucs).sum()} fold(s) undefined -- single-class test set)"
               if np.isnan(aucs).any() else
               f"  ensemble AUC:  mean {aucs.mean():.4f}  std {aucs.std():.4f}")
        print(f"  floor AUC:     mean {floors.mean():.4f}  std {floors.std():.4f}")
        beats = (aucs > floors + 1e-9).sum()
        print(f"  beats its own fold's floor in {beats}/{len(results)} folds")

    return results


def main():
    args = parse_args()
    
    # Setup Dual Logging
    log_filename = f"lstm_forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("=" * 64)
    print(f"Logging initialized. Saving all terminal output to: {log_filename}")
    print("=" * 64)
    
    print("Loading hand-crafted continuous features and building hourly labels...")
    hourly = load_hourly_features(args.features_csv)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hourly)} hourly feature vectors, {len(major_times)} M>={args.threshold} "
           f"AEGEAN events in the full catalog")

    dsp = days_since_prev_major(hourly.index, major_times)
    feature_cols = list(hourly.columns)
    features = hourly[feature_cols].to_numpy(dtype=np.float64)

    n = len(hourly)
    valid_end_indices = np.arange(args.seq_hours - 1, n)

    embargo = args.seq_hours - 1

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train], valid_end_indices[i_train + embargo:i_val],
                  valid_end_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_end_indices, args.cv_folds, embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    horizons = ([float(h) for h in args.horizons.split(",")] if args.horizons
                else [args.horizon_days])

    per_horizon = {}
    for horizon_days in horizons:
        per_horizon[horizon_days] = run_horizon(horizon_days, args, hourly.index, feature_cols,
                                                features, major_times, dsp, folds, fold_labels,
                                                skip, seeds, device)

    if len(horizons) > 1:
        print(f"\n{'#' * 64}\n# Cross-horizon summary\n{'#' * 64}")
        print(f"  {'horizon (d)':>12s}  {'ensemble AUC (mean)':>20s}  {'floor AUC (mean)':>16s}  folds")
        for horizon_days in horizons:
            results = per_horizon[horizon_days]
            if not results:
                print(f"  {horizon_days:12.0f}  {'(no folds ran)':>20s}")
                continue
            aucs = np.array([r[0] for r in results])
            floors = np.array([r[1] for r in results])
            print(f"  {horizon_days:12.0f}  {np.nanmean(aucs):20.4f}  {floors.mean():16.4f}  {len(results)}")

    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
