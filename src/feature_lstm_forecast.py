"""
LSTM forecaster trained directly on Sismokaos-featureExtract's hand-crafted
continuous features (STA/LTA, Hjorth, permutation entropy, spectral
centroid, cross-axis correlation -- computed on real KO.GEDZ continuous
data, Aegean zone), instead of the catalog-derived sequences
cnn_lstm_forecast.py uses. Same validated target as that script: will a
M >= threshold event occur in this zone within horizon_days? -- reusing the
target definition, not the catalog features, so this is a fair test of
whether the hand-crafted continuous features carry the same (or different)
forecasting signal.

Pipeline:
  1. Load the combined features CSV (Sismokaos-featureExtract/results/GEDZ/*.csv).
  2. Reconstruct each row's approximate absolute UTC time from Pencere_ID
     (date + hour + within-file window index; ~1 window / 50s resolution).
  3. Aggregate to HOURLY mean feature vectors -- the label doesn't move
     faster than daily, and raw 50s-resolution would make the LSTM's
     sequence length intractable for a 100-day span (~172k raw steps vs.
     ~2.4k hourly steps).
  4. Label each hour: will a M>=threshold AEGEAN event occur within the next
     horizon_days (dense target, no declustering -- same choice
     catalog.py's build_dense_windows makes and the same reason: a
     clustered sequence is signal, not noise to define away).
  5. Sliding windows of `seq_hours` consecutive hourly vectors, chronological
     train/val/test split (no shuffling across time).
  6. Train an LSTM + attention (reuses cnn_lstm.py's LSTMAttentionBranch)
     binary classifier; report against base-rate and persistence floors,
     matching cnn_lstm_forecast.py's convention exactly so the two are
     directly comparable.

Usage:
    python feature_lstm_forecast.py \\
        --features-csv ../../Sismokaos-featureExtract/results/GEDZ/GEDZ_2024_11_15-2025_02_23_ENZ_features.csv \\
        --catalog-path ../../data_downloader/catalogs/data_large.csv
"""

import argparse
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from cnn_lstm import LSTMAttentionBranch
from training import seed_everything

AEGEAN_BBOX = (36.0, 40.0, 25.0, 30.0)  # lat0, lat1, lon0, lon1 -- matches forecast.py's FAULT_ZONES


def parse_hour_start(pencere_id: str):
    """'2024_11_15_00_w01' -> datetime(2024,11,15,0,0,0). Ignores the small
    (<=150s) stitching offset from PREV_LEN carry-over -- negligible against
    a 30-day forecast horizon."""
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_w(\d+)", pencere_id)
    if not m:
        return None, None
    y, mo, d, h, w = m.groups()
    return datetime(int(y), int(mo), int(d), int(h)), int(w)


def load_hourly_features(features_csv: str) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    parsed = df["Pencere_ID"].apply(parse_hour_start)
    df["hour_start"] = [p[0] for p in parsed]
    df = df.dropna(subset=["hour_start"])
    feature_cols = [c for c in df.columns if c not in ("Pencere_ID", "Zaman_Dk", "hour_start")]
    hourly = df.groupby("hour_start")[feature_cols].mean().sort_index()
    return hourly


def load_aegean_events(catalog_path: str, min_magnitude: float = 4.5) -> np.ndarray:
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    return np.sort(aegean.dt.to_numpy())


def label_hours(hourly_index: pd.DatetimeIndex, major_times: np.ndarray, horizon_days: float) -> np.ndarray:
    horizon = np.timedelta64(int(horizon_days), "D")
    t = hourly_index.to_numpy()
    labels = np.zeros(len(t), dtype=np.int64)
    for i, ti in enumerate(t):
        fut = major_times[(major_times > ti) & (major_times <= ti + horizon)]
        labels[i] = int(len(fut) > 0)
    return labels


def days_since_prev_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        prev = major_times[major_times < ti]
        if len(prev):
            out[i] = (ti - prev[-1]) / np.timedelta64(1, "D")
    return out


class HourlySeqDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_hours: int,
                indices: np.ndarray, stats=None):
        self.features = features
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices  # end-index (inclusive) of each window, into `features`
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


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def parse_args():
    p = argparse.ArgumentParser(description="LSTM forecaster on hand-crafted continuous features.")
    p.add_argument("--features-csv", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--seq-hours", type=int, default=168, help="7 days of hourly context per sample.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44",
                  help="Comma-separated seeds; trains one model per seed and reports both "
                       "per-seed AUC and the ensemble (mean-probability) AUC.")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


class ForecastLSTM(nn.Module):
    def __init__(self, feat_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.branch = LSTMAttentionBranch(feat_dim, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.branch.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.branch.out_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq):
        return self.head(self.branch(seq)).squeeze(-1)


def train_one_seed(args, seed, feature_cols, features, labels,
                   train_idx, val_idx, test_idx, device):
    seed_everything(seed)
    train_ds = HourlySeqDataset(features, labels, args.seq_hours, train_idx)
    val_ds = HourlySeqDataset(features, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = HourlySeqDataset(features, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model = ForecastLSTM(len(feature_cols), hidden=args.hidden, dropout=args.dropout).to(device)

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

    # Checkpoint selection prefers val AUC, but this task's val split can be a
    # single-class stretch (e.g. entirely inside a sustained swarm's 30-day
    # lookahead) where AUC is undefined for every epoch, not just some --
    # detected once up front rather than falling back per-epoch, since a
    # date-range split's class composition doesn't change epoch to epoch.
    yv0, _, _ = evaluate(val_loader)
    use_loss_fallback = len(np.unique(yv0)) < 2
    if use_loss_fallback:
        print(f"  [seed {seed}] val split is single-class (positive rate {yv0.mean():.3f}) -- "
             "falling back to val loss for checkpoint selection.")

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

    print("=" * 64)
    print("Loading hand-crafted continuous features and building hourly labels...")
    hourly = load_hourly_features(args.features_csv)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hourly)} hourly feature vectors, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    labels = label_hours(hourly.index, major_times, args.horizon_days)
    dsp = days_since_prev_major(hourly.index, major_times)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    feature_cols = list(hourly.columns)
    features = hourly[feature_cols].to_numpy(dtype=np.float64)

    n = len(hourly)
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

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split -- need more days of features.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds = [int(s) for s in args.ensemble_seeds.split(",")]
    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
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
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= args.horizon_days).astype(int)).astype(np.float64)
    pers_auc = safe_auc(yt_ref, pers_pred)
    print(f"  persistence             AUC {pers_auc:.4f}   n={len(yt_ref)}")

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
    print(f"\n  [!] Single station, ~10 months, {len(seeds)}-seed ensemble -- treat as a first look, "
         "not a settled result (report.md 6.6).")


if __name__ == "__main__":
    main()
