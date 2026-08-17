"""
GRU and TCN forecasters trained directly on Sismokaos-featureExtract's hand-crafted
continuous features. 

Features:
- Robust absolute-time parsing (Unix Epoch) for gap preservation.
- Walk-forward CV for non-stationary timeline evaluation.
- Dual-logging (Terminal + File).
- Real-time epoch tracking.

Usage:
    python feature_gru_tcn_forecast.py \
        --features-csv ../../Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy \
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \
        --cv-folds 5
"""

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from forecasting.feature_lstm_forecast import HourlySeqDataset
from seismolib.catalog import days_since_prev_major, label_hours, load_aegean_events
from seismolib.splits import print_split_diagnostics, walk_forward_splits
from seismolib.metrics import binary_report, print_report, safe_auc
from seismolib.training import seed_everything
from seismolib.logging import DualLogger




def load_hourly_features(features_csv: str) -> pd.DataFrame:
    """Loads the combined features file and aggregates it to hourly means."""
    if str(features_csv).endswith(".npy"):
        df = pd.DataFrame.from_records(np.load(features_csv, allow_pickle=False))
    else:
        df = pd.read_csv(features_csv)

    # Vectorized absolute time assignment using Zaman_Dk minutes
    exact_times = pd.to_datetime(df["Zaman_Dk"], unit="m")
    
    # .copy() prevents Pandas fragmentation warnings before assigning the new column
    # "h" is lowercase to comply with Pandas 2.2+ frequency alias deprecations
    df = df.copy().assign(hour_start=exact_times.dt.floor("h"))
    
    feature_cols = [c for c in df.columns if c not in ("Pencere_ID", "Zaman_Dk", "hour_start", "index")]
    hourly = df.groupby("hour_start")[feature_cols].mean().sort_index()
    return hourly


class ForecastGRU(nn.Module):
    """GRU branch for sequence forecasting."""
    def __init__(self, feat_dim, hidden=64, dropout=0.3):
        super().__init__()
        # Dropout set to 0 here because PyTorch GRU dropout only applies *between* 
        # layers in a multi-layer stack. We use the head's dropout instead.
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True, dropout=0)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.head(context).squeeze(-1)


class Chomp1d(nn.Module):
    """Removes padding from the end of a sequence for causal 1D convolutions."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single residual block for the TCN."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ForecastTCN(nn.Module):
    """Temporal Convolutional Network for sequence forecasting."""
    def __init__(self, feat_dim, num_channels=[64, 64, 64], kernel_size=3, dropout=0.3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feat_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser(description="GRU and TCN forecasters on hand-crafted continuous features.")
    p.add_argument("--features-csv", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=None)
    p.add_argument("--seq-hours", type=int, default=168)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
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


def train_one_seed(args, seed, feature_cols, features, labels,
                   train_idx, val_idx, test_idx, device, model_class):
    seed_everything(seed)
    train_ds = HourlySeqDataset(features, labels, args.seq_hours, train_idx)
    val_ds = HourlySeqDataset(features, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = HourlySeqDataset(features, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    if model_class == ForecastGRU:
        model = ForecastGRU(len(feature_cols), hidden=args.hidden, dropout=args.dropout).to(device)
    elif model_class == ForecastTCN:
        model = ForecastTCN(len(feature_cols), num_channels=[args.hidden, args.hidden, args.hidden], dropout=args.dropout).to(device)

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
            
        # REAL-TIME EPOCH PRINTING
        print(f"  [seed {seed}] {model_class.__name__} epoch {epoch+1:02d}/{args.epochs} "
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
    print(f"  [seed {seed}] {model_class.__name__} test AUC {safe_auc(yt, st):.4f}")
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

    results = {}
    for model_class in [ForecastGRU, ForecastTCN]:
        print(f"\nTraining {model_class.__name__} ({len(seeds)} seed(s)): {seeds}")
        per_seed_scores = []
        yt_ref = None
        for seed in seeds:
            yt, st = train_one_seed(args, seed, feature_cols, features, labels,
                                    train_idx, val_idx, test_idx, device, model_class)
            if yt_ref is None:
                yt_ref = yt
            per_seed_scores.append(st)

        ensemble_score = np.mean(per_seed_scores, axis=0)
        
        per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
        print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
              f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
        ensemble_auc = safe_auc(yt_ref, ensemble_score)
        print(f"  ENSEMBLE AUC {ensemble_auc:.4f}   n={len(yt_ref)}")
        results[model_class.__name__] = (ensemble_auc, ensemble_score, yt_ref)

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(yt_ref)}")
    pers_dsp = dsp[test_idx]
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= horizon_days).astype(int)).astype(np.float64)
    pers_auc = safe_auc(yt_ref, pers_pred)
    print(f"  persistence            AUC {pers_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, pers_auc)
    
    for model_name, (ensemble_auc, ensemble_score, yt_ref) in results.items():
        if ensemble_auc <= floor + 1e-9:
            print(f"\n  [!] {model_name} Ensemble does NOT clear max(chance, persistence).")
        else:
            print(f"\n  {model_name} Ensemble beats max(chance, persistence) by {ensemble_auc - floor:+.4f} AUC.")
        
        report = binary_report(yt_ref, ensemble_score)
        print_report(f"{model_name} ({fold_label}, test set)", report)
    
    return results, floor


def run_horizon(horizon_days, args, hourly_index, feature_cols, features, major_times, dsp,
                folds, fold_labels, skip, seeds, device):
    print(f"\n{'#' * 64}\n# horizon = {horizon_days:.0f} days\n{'#' * 64}")
    labels = label_hours(hourly_index, major_times, horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    all_results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, feature_cols, features, labels, dsp, hourly_index,
                          train_idx, val_idx, test_idx, seeds, device, horizon_days=horizon_days)
        if result is not None:
            all_results.append(result)

    if args.cv_folds > 1 and all_results:
        floors = np.array([r[1] for r in all_results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(all_results)}/{args.cv_folds} folds, "
              f"horizon={horizon_days:.0f}d)\n{'=' * 64}")
        
        for model_name in ["ForecastGRU", "ForecastTCN"]:
            aucs = np.array([r[0][model_name][0] for r in all_results])
            print(f"  {model_name} ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
            if np.isnan(aucs).any():
                print(f"  {model_name} ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}"
                      f"  ({np.isnan(aucs).sum()} fold(s) undefined)")
            else:
                print(f"  {model_name} ensemble AUC:  mean {aucs.mean():.4f}  std {aucs.std():.4f}")
            beats = (aucs > floors + 1e-9).sum()
            print(f"  {model_name} beats floor in {beats}/{len(all_results)} folds\n")
            
        print(f"  floor AUC:     mean {floors.mean():.4f}  std {floors.std():.4f}")

    return all_results


def main():
    args = parse_args()
    
    # Setup Dual Logging
    log_filename = f"forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        print(f"  {'horizon (d)':>12s}  {'GRU AUC':>12s}  {'TCN AUC':>12s}  {'floor AUC':>12s}  folds")
        for horizon_days in horizons:
            results = per_horizon[horizon_days]
            if not results:
                print(f"  {horizon_days:12.0f}  {'(no folds ran)':>26s}")
                continue
            gru_aucs = np.array([r[0]["ForecastGRU"][0] for r in results])
            tcn_aucs = np.array([r[0]["ForecastTCN"][0] for r in results])
            floors = np.array([r[1] for r in results])
            print(f"  {horizon_days:12.0f}  {np.nanmean(gru_aucs):12.4f}  {np.nanmean(tcn_aucs):12.4f}  {floors.mean():12.4f}  {len(results)}")

    # Restore standard output just to be clean
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
