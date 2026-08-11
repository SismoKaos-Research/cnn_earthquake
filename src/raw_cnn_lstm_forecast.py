"""
Same setup as feature_lstm_forecast.py (KO.GEDZ, May 2024-Feb 2025, M>=4.5
dense forecast target, 24h sequences, same split, same 3-seed ensemble) but
the per-hour input is a small 1D CNN over the raw waveform (3, 18000)
instead of the hand-crafted feature vector. Reads directly from
Sismokaos-featureExtract's preprocessor.preprocess output
(data/aegean_2024_2025/YYYY_MM_DD/*.npy), not the feature CSV.

Usage:
    python raw_cnn_lstm_forecast.py \\
        --data-root ../../Sismokaos-featureExtract/data/aegean_2024_2025 \\
        --catalog-path ../../data_downloader/catalogs/data_large.csv
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from feature_lstm_forecast import (days_since_prev_major, label_hours,
                                   load_aegean_events, safe_auc)
from metrics import binary_report, print_report
from model.sequence import SequenceHeadNet
from training import seed_everything

_DATE_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
HOUR_SAMPLES = 18000  # 3600s * 5Hz


def load_hourly_raw(data_root: str):
    """Returns (DatetimeIndex, raw) where raw is (n_hours, 3, HOUR_SAMPLES)
    float32. Gaps (usually a couple percent per hour) are linearly
    interpolated, then anything left over is zeroed."""
    root = Path(data_root)
    date_dirs = sorted(d for d in root.iterdir() if d.is_dir() and _DATE_DIR_RE.match(d.name))

    hours, arrs = [], []
    for date_dir in date_dirs:
        for npy_path in sorted(date_dir.glob("*.npy")):
            parts = npy_path.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                hour_dt = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
            except ValueError:
                continue
            struct = np.load(npy_path)
            comp_arr = np.stack(
                [struct[c] if c in struct.dtype.names else np.full(len(struct), np.nan)
                 for c in ("E", "N", "Z")], axis=0)
            if comp_arr.shape[1] != HOUR_SAMPLES:
                fixed = np.full((3, HOUR_SAMPLES), np.nan)
                n = min(HOUR_SAMPLES, comp_arr.shape[1])
                fixed[:, :n] = comp_arr[:, :n]
                comp_arr = fixed
            hours.append(hour_dt)
            arrs.append(comp_arr)

    order = np.argsort(hours)
    hour_index = pd.DatetimeIndex([hours[i] for i in order])
    raw = np.stack([arrs[i] for i in order], axis=0).astype(np.float32)

    for h in range(raw.shape[0]):
        for c in range(3):
            x = raw[h, c]
            nan = np.isnan(x)
            if nan.any() and (~nan).sum() > 3:
                x[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), x[~nan])
            raw[h, c] = np.nan_to_num(x, nan=0.0)
    return hour_index, raw


class RawSeqDataset(Dataset):
    def __init__(self, raw: np.ndarray, labels: np.ndarray, seq_hours: int, indices: np.ndarray,
                stats=None):
        self.raw = raw
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
        if stats is None:
            sub = np.concatenate([raw[max(0, i - seq_hours + 1):i + 1] for i in indices[:50]], axis=0)
            mu = sub.mean(axis=(0, 2), keepdims=True)
            sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
            stats = (mu[0], sd[0])  # (3,1) each
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        seq = self.raw[start:end + 1]  # (T, 3, HOUR_SAMPLES)
        mu, sd = self.stats
        seq = (seq - mu[None]) / sd[None]
        return (torch.from_numpy(seq).float(),
               torch.tensor(self.labels[end], dtype=torch.float32))


class RawWaveformEncoder(nn.Module):
    """1D CNN that embeds one hour's raw 3-component waveform."""

    def __init__(self, out_dim=32, dropout=0.3):
        super().__init__()

        def block(cin, cout, k, s):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                                 nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(dropout))

        self.net = nn.Sequential(
            block(3, 16, 7, 4),
            block(16, 32, 5, 4),
            block(32, 32, 5, 4),
            block(32, out_dim, 3, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RawCNNLSTM(SequenceHeadNet):
    def __init__(self, cnn_out=32, hidden=16, dropout=0.5):
        super().__init__(cnn_out, hidden=hidden, dropout=dropout,
                         encoder=RawWaveformEncoder(out_dim=cnn_out, dropout=dropout))


def parse_args():
    p = argparse.ArgumentParser(description="Raw-waveform CNN-LSTM forecaster (vs. hand features).")
    p.add_argument("--data-root", required=True,
                  help="Sismokaos-featureExtract preprocessed dir (data/<EARTHQUAKE_NAME>).")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--seq-hours", type=int, default=24)
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
    return p.parse_args()


def train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device):
    seed_everything(seed)
    train_ds = RawSeqDataset(raw, labels, args.seq_hours, train_idx)
    val_ds = RawSeqDataset(raw, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = RawSeqDataset(raw, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model = RawCNNLSTM(cnn_out=args.cnn_out, hidden=args.hidden, dropout=args.dropout).to(device)

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
        print(f"  [seed {seed}] val split is single-class, checkpointing on val loss instead of AUC")

    best = float("inf") if use_loss_fallback else -1.0
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
        metric = val_loss if use_loss_fallback else val_auc
        improved = metric < best if use_loss_fallback else metric > best
        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val AUC {val_auc:.4f} val loss {val_loss:.4f}")
        if improved:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = evaluate(test_loader)
    print(f"  [seed {seed}] test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def main():
    args = parse_args()

    print("Loading raw preprocessed waveform and building hourly labels...")
    hour_index, raw = load_hourly_raw(args.data_root)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    labels = label_hours(hour_index, major_times, args.horizon_days)
    dsp = days_since_prev_major(hour_index, major_times)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    n = len(hour_index)
    valid_end_indices = np.arange(args.seq_hours - 1, n)
    n_valid = len(valid_end_indices)
    i_train = int(n_valid * args.train_frac)
    i_val = int(n_valid * (args.train_frac + args.val_frac))
    train_idx = valid_end_indices[:i_train]
    val_idx = valid_end_indices[i_train:i_val]
    test_idx = valid_end_indices[i_val:]
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds = [int(s) for s in args.ensemble_seeds.split(",")]
    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device)
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
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= args.horizon_days).astype(int)).astype(np.float64)
    pers_auc = safe_auc(yt_ref, pers_pred)
    print(f"  persistence             AUC {pers_auc:.4f}   n={len(yt_ref)}")

    print(f"\n--- Raw-waveform CNN-LSTM ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    print_report("Raw-waveform CNN-LSTM ensemble (test set)", binary_report(yt_ref, ensemble_score))


if __name__ == "__main__":
    main()
