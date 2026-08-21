"""
Raw-waveform CNN-LSTM forecaster (or feature CSV loader for massive files).

Features:
- Dual-logging (Terminal + File).
- Chunked streaming reader for really long CSV files (prevents OOM on huge datasets).
- Robust absolute-time parsing via Unix Epoch (Zaman_Dk).
- Walk-forward CV for non-stationary timeline evaluation.
- Real-time epoch tracking.

Usage:
    python raw_cnn_lstm_forecast.py \
        --data-root path/to/massive_features.csv \
        --catalog-path ../../data_downloader/catalogs/data_large.csv \
        --cv-folds 5
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader, Dataset

from seismolib.catalog import (days_since_prev_major, label_hours,
                               load_aegean_events,
                               truncate_to_reliable_catalog_end)
from seismolib.logging import DualLogger
from seismolib.metrics import binary_report, print_report, safe_auc
from seismolib.model.sequence import SequenceHeadNet
from seismolib.splits import print_split_diagnostics, walk_forward_splits
from seismolib.training import seed_everything
from seismolib.waveform import (RawSeqDataset, RawWaveformEncoder,
                                load_hourly_raw, load_hourly_raw_consolidated)


class RawCNNLSTM(SequenceHeadNet):
    """`SequenceHeadNet` with a `RawWaveformEncoder` embedding each hour's raw waveform."""

    def __init__(self, cnn_out=32, hidden=16, dropout=0.5):
        super().__init__(cnn_out, hidden=hidden, dropout=dropout,
                         encoder=RawWaveformEncoder(out_dim=cnn_out, dropout=dropout))


def parse_args():
    p = argparse.ArgumentParser(description="Raw-waveform CNN-LSTM forecaster (with massive CSV support).")
    p.add_argument("--data-root", required=True,
                   help="Path to preprocessed dir (data/<EARTHQUAKE_NAME>) OR a really long CSV file.")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--stations", nargs="+", default=["BODT", "DAT"], metavar="NAME")
    p.add_argument("--max-station-dist-km", type=float, default=None)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=None)
    p.add_argument("--seq-hours", type=int, default=24)
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    p.add_argument("--shift-diagnostic", action="store_true")
    p.add_argument("--balanced-folds", action="store_true")
    return p.parse_args()


def train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device, model_cls=None):
    seed_everything(seed)
    train_ds = RawSeqDataset(raw, labels, args.seq_hours, train_idx)
    val_ds = RawSeqDataset(raw, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = RawSeqDataset(raw, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model_cls = model_cls or RawCNNLSTM
    
    # If input is tabular (CSV features), adjust input dimension for the model
    if train_ds.is_tabular:
        feat_dim = raw.shape[1] if isinstance(raw, pd.DataFrame) or isinstance(raw, np.ndarray) else len(raw.columns)
        # If tabular, we bypass the 1D CNN encoder and feed features directly to SequenceHeadNet
        model = SequenceHeadNet(feat_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    else:
        model = model_cls(cnn_out=args.cnn_out, hidden=args.hidden, dropout=args.dropout).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=2)
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
        print(f"  [seed {seed}] val split is single-class -- checkpointing on train AUC instead.")

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
        improved = metric > best
        print(f"  [seed {seed}] epoch {epoch+1:02d}/{args.epochs} val AUC {val_auc:.4f} val loss {val_loss:.4f}"
             + (f" train AUC {metric:.4f}" if use_loss_fallback else ""))
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


def run_fold(fold_label, args, raw, labels, dsp, hour_index, train_idx, val_idx, test_idx,
             seeds, device, model_cls=None, horizon_days=None):
    horizon_days = args.horizon_days if horizon_days is None else horizon_days
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device,
                                model_cls=model_cls)
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
    # Orientation-corrected: a persistence rule scoring below 0.5 ranks
    # inversely, and an inverted rule is exactly as exploitable as a correct
    # one, so the bar it sets is max(a, 1-a). Without this the floor silently
    # collapsed to a vacuous 0.5 whenever persistence fell under chance.
    pers_auc = max(pers_raw, 1.0 - pers_raw) if pers_raw == pers_raw else pers_raw
    single_class = len(np.unique(yt_ref)) < 2
    pers_brier = float("nan") if single_class else float(brier_score_loss(yt_ref, pers_pred))
    print(f"  persistence            AUC {pers_auc:.4f}   Brier {pers_brier:.4f}   n={len(yt_ref)}")

    print(f"\n--- Forecaster Ensemble ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
           f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, pers_auc)
    report = binary_report(yt_ref, ensemble_score)
    bss = (float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
           else 1.0 - report["brier"] / pers_brier)
    report["brier_skill_score_vs_persistence"] = bss
    print_report(f"Ensemble ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def run_horizon(horizon_days, args, hour_index, raw, major_times, dsp, folds, fold_labels,
                skip, seeds, device, model_cls=None):
    print(f"\n{'#' * 64}\n# horizon = {horizon_days:.0f} days\n{'#' * 64}")
    labels = label_hours(hour_index, major_times, horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, raw, labels, dsp, hour_index,
                          train_idx, val_idx, test_idx, seeds, device,
                          model_cls=model_cls, horizon_days=horizon_days)
        if result is not None:
            results.append(result)

    if args.cv_folds > 1 and results:
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
    log_filename = f"raw_forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("=" * 64)
    print(f"Logging initialized. Saving all terminal output to: {log_filename}")
    print("=" * 64)

    print("Loading input data (Directory or Massive CSV)...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
        
    major_times = load_aegean_events(args.catalog_path, args.threshold,
                                     stations=args.stations,
                                     max_dist_km=args.max_station_dist_km)
    if args.max_station_dist_km:
        print(f"  [distance cap] events restricted to <= {args.max_station_dist_km:.0f} km "
               f"from nearest of {args.stations}")
    print(f"  {len(hour_index)} hourly records loaded, {len(major_times)} M>={args.threshold} "
           f"AEGEAN events in the full catalog")

    max_horizon = max([float(h) for h in args.horizons.split(",")] if args.horizons
                      else [args.horizon_days])
    hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                       buffer_days=max_horizon)

    dsp = days_since_prev_major(hour_index, major_times)

    n = len(hour_index)
    valid_end_indices = np.arange(args.seq_hours - 1, n)

    horizons = ([float(h) for h in args.horizons.split(",")] if args.horizons
                else [args.horizon_days])

    embargo = args.seq_hours - 1 + int(round(max(horizons) * 24))

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train], valid_end_indices[i_train + embargo:i_val],
                  valid_end_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    elif args.balanced_folds:
        ref_labels = label_hours(hour_index, major_times, horizons[0])
        print(f"  [balanced-folds] placing block boundaries by positive-label mass at "
               f"horizon={horizons[0]:.0f}d")
        folds = walk_forward_splits(valid_end_indices, args.cv_folds,
                                    labels=ref_labels[valid_end_indices], embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]
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

    per_horizon = {}
    for horizon_days in horizons:
        per_horizon[horizon_days] = run_horizon(horizon_days, args, hour_index, raw, major_times,
                                                dsp, folds, fold_labels, skip, seeds, device)

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
