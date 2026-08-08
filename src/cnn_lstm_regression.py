"""
Dual-channel CNN + LSTM/self-attention magnitude regression.

Same task as `cnn_regression.py` (magnitude from a 3-second window, plus the
two physical predictors log_snr and log_distance) but with a second input
channel: the raw standardized waveform, run through an LSTM + multi-head
self-attention branch, fused with the 2D image branch exactly as
`cnn_lstm.py`'s `DualChannelRiskNet` does for the (unrelated) catalog-risk
task. This is not a replication of a specific paper -- investigation found
no paper in this project's literature folder that combines a spectrogram
image branch with a raw-waveform LSTM branch for magnitude (the closest,
Shen et al. 2025, is pure 1D CNN+Bi-LSTM with no image channel at all) -- it
is this project's own dual-channel architecture (Wang & Zhao 2025's
1D2D-EDL, already used for detection) retargeted at a task it has not been
tried on.

Consumes `seismic-cli generate-regression-dataset --dual`: the SAME
event-disjoint split, magnitude label, and log_snr/distance_km predictors
`cnn_regression.py` uses, but with `{seq, img}` tensors (SpectrogramDualEncoder
or RamDualEncoder) instead of a single encoded image -- `--dual` reuses the
detection pipeline's existing encoders unchanged, so the two datasets are
directly comparable except for the extra seq channel.

Floors and the single-channel comparison are what settle whether the LSTM
branch is worth having, not assumed:
  * predict-the-mean and ridge(log_snr, log_distance) -- ported unchanged
    from `cnn_regression.py`, same metric space (magnitude itself is not
    log-transformed; only distance is, matching `cnn_regression.py` exactly
    to avoid the metric-space mismatches report.md 13.5 documents).
  * `cnn_regression.py` run on the SAME event split, single-channel
    spectrogram+aux -- the existing result this architecture has to beat,
    not just the floors.
`--channels` ablates 1D/2D/aux individually, same convention as
`cnn_lstm.py`/`cnn_lstm_forecast.py`, so a win can be attributed to a
specific branch rather than assumed to come from the LSTM.

This is reported as a SINGLE-SEED result, explicitly, matching how
report.md's own magnitude-classification section (7) was reported ("at a
single seed and threshold, not yet re-verified") -- this project has
reversed close-margin verdicts on a single seed before (report.md 6.6), so a
close margin here should be re-seeded before being treated as final, not
read as an established result off one run.

Usage:
    python cnn_lstm_regression.py --dataset-dir ../../data_downloader/data/dataset_magclass_dual_3s
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cnn_lstm import CNNBranch, LSTMAttentionBranch
from cnn_regression import AUX_COLUMNS, regression_metrics, report_baselines
from training import seed_everything


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class DualMagnitudeDataset(Dataset):
    """
    Loads the {seq, img} tensors written by
    `seismic-cli generate-regression-dataset --dual`, plus the manifest's
    magnitude/log_snr/log_distance columns -- same standardization convention
    as `cnn_regression.py`'s MagnitudeDataset (TRAIN-only aux stats, NaN ->
    0 after standardization).
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, aux_stats=None):
        self.rows = manifest.reset_index(drop=True)
        self.root = Path(root)

        aux = self.rows[AUX_COLUMNS].to_numpy(dtype=np.float64)
        if aux_stats is None:
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(aux, axis=0)
                sd = np.nanstd(aux, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            aux_stats = (mu, sd)
        self.aux_mu, self.aux_sd = aux_stats
        aux = (aux - self.aux_mu) / self.aux_sd
        self.aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.targets = self.rows["magnitude"].to_numpy(dtype=np.float32)

    def aux_stats(self):
        return (self.aux_mu, self.aux_sd)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        d = torch.load(self.root / row["split"] / row["filename"], weights_only=True)
        return (d["seq"].float(), d["img"].float(),
                torch.from_numpy(self.aux[idx]),
                torch.tensor(self.targets[idx], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model -- same branches as cnn_lstm.py, regression head
# ---------------------------------------------------------------------------

class DualChannelRegressionNet(nn.Module):
    """Same 1D/2D/aux architecture as `cnn_lstm.py`'s DualChannelRiskNet /
    `cnn_lstm_forecast.py`'s DualChannelForecastNet, with a bare linear
    regression head (no activation, matching `RegressionSeismicCNN`)."""

    def __init__(self, seq_dim, img_channels, aux_dim, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all"):
        super().__init__()
        self.channels = channels
        self.use_1d = channels in ("all", "1d", "1d+aux")
        self.use_2d = channels in ("all", "2d", "2d+aux")
        self.use_aux = channels in ("all", "aux", "1d+aux", "2d+aux")
        if not (self.use_1d or self.use_2d or self.use_aux):
            raise ValueError(f"--channels {channels} disables every branch")

        if self.use_1d:
            self.b1 = LSTMAttentionBranch(seq_dim, hidden=hidden, dropout=dropout)
            self.p1 = nn.Linear(self.b1.out_dim, fusion_dim)
        if self.use_2d:
            self.b2 = CNNBranch(img_channels, dropout=dropout)
            self.p2 = nn.Linear(self.b2.out_dim, fusion_dim)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

        head_in = (fusion_dim if (self.use_1d or self.use_2d) else 0) + \
                 (aux_dim if self.use_aux else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, seq, img, aux):
        feats = []
        fused = None
        if self.use_1d:
            fused = self.w1 * self.p1(self.b1(seq))
        if self.use_2d:
            f2 = self.w2 * self.p2(self.b2(img))
            fused = f2 if fused is None else fused + f2
        if fused is not None:
            feats.append(fused)
        if self.use_aux:
            feats.append(aux)
        return self.head(torch.cat(feats, dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dual-channel CNN+LSTM magnitude regression (spectrogram or RAM 2D + raw-waveform 1D).")
    p.add_argument("--dataset-dir", required=True,
                  help="Directory from `seismic-cli generate-regression-dataset --dual`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm_regression")
    p.add_argument("--channels", default="all",
                  choices=["all", "1d", "2d", "aux", "1d+aux", "2d+aux"],
                  help="Ablation switch: which branches to enable.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--huber-delta", type=float, default=0.0,
                  help="If > 0 use SmoothL1 with this beta instead of plain L1 (MAE), "
                       "matching cnn_regression.py's convention.")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    for col in ("magnitude", "log_snr"):
        if col not in manifest.columns:
            raise ValueError(f"manifest.csv is missing '{col}'. Regenerate with "
                             f"`seismic-cli generate-regression-dataset --dual`.")
    if "distance_km" not in manifest.columns:
        manifest["distance_km"] = np.nan
    manifest["log_distance"] = np.log(manifest["distance_km"].clip(lower=1.0))

    parts = {}
    for split in ("train", "val", "test"):
        sub = manifest[manifest.split == split]
        if sub.empty:
            raise ValueError(f"Split '{split}' is empty in the manifest.")
        parts[split] = sub

    train_ds = DualMagnitudeDataset(parts["train"], root)
    stats = train_ds.aux_stats()
    val_ds = DualMagnitudeDataset(parts["val"], root, aux_stats=stats)
    test_ds = DualMagnitudeDataset(parts["test"], root, aux_stats=stats)

    sample_seq, sample_img, sample_aux, _ = train_ds[0]

    print("=" * 64)
    print(f"Dual-channel magnitude regression | channels='{args.channels}'")
    print(f"  seq {tuple(sample_seq.shape)} | img {tuple(sample_img.shape)} "
         f"| aux ({sample_aux.numel()},) ({', '.join(AUX_COLUMNS)})")
    for s in ("train", "val", "test"):
        d = parts[s]
        print(f"  {s:5s}: n={len(d):5d}  M {d.magnitude.min():.1f}-{d.magnitude.max():.1f} "
             f"(mean {d.magnitude.mean():.2f})  events={d.event_id.nunique()}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelRegressionNet(sample_seq.shape[-1], sample_img.shape[0], sample_aux.numel(),
                                     hidden=args.hidden, fusion_dim=args.fusion_dim,
                                     dropout=args.dropout, channels=args.channels).to(device)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    criterion = (nn.SmoothL1Loss(beta=args.huber_delta) if args.huber_delta > 0
                else nn.L1Loss())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_cnnlstm_regression.pth")
    best_val_mae, no_improve = float("inf"), 0

    def evaluate(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for seq, img, aux, y in loader:
                out = model(seq.to(device), img.to(device), aux.to(device))
                preds.extend(out.cpu().tolist())
                trues.extend(y.tolist())
        return np.array(trues), np.array(preds)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for seq, img, aux, y in train_loader:
            seq, img, aux, y = seq.to(device), img.to(device), aux.to(device), y.to(device)
            loss = criterion(model(seq, img, aux), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            running += loss.item() * y.size(0)
        scheduler.step()

        yv, pv = evaluate(val_loader)
        vm = regression_metrics(yv, pv)
        print(f"Epoch {epoch+1}/{args.epochs} | train {running/len(train_ds):.4f} "
             f"| val MAE {vm['MAE']:.4f}  RMSE {vm['RMSE']:.4f}  R2 {vm['R2']:+.4f}")
        if vm["MAE"] < best_val_mae:
            best_val_mae, no_improve = vm["MAE"], 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val MAE {best_val_mae:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val MAE flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, yp = evaluate(test_loader)
    tm = regression_metrics(yt, yp)

    ridge_mae = report_baselines(train_ds, test_ds)
    print(f"\n--- Dual-channel model (channels='{args.channels}') ---")
    print(f"  MAE {tm['MAE']:.3f}  RMSE {tm['RMSE']:.3f}  R2 {tm['R2']:+.3f}")
    if ridge_mae is not None:
        delta = ridge_mae - tm["MAE"]
        verdict = ("adds information beyond amplitude+distance"
                  if delta > 0.01 else
                  "NO measurable gain over amplitude+distance alone")
        print(f"  vs ridge baseline: {delta:+.3f} MAE  ->  {verdict}")

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        print(f"\n  learned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}")

    print(f"\n  residual std {np.std(yt - yp):.3f} magnitude units over "
         f"M {yt.min():.1f}-{yt.max():.1f}")
    print("\n  [!] Single-seed result -- re-seed before treating a close margin as final "
         "(report.md 6.6).")


if __name__ == "__main__":
    main()
