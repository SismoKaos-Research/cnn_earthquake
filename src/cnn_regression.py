"""
Magnitude regression from encoded seismic windows.

Consumes a dataset built by `seismic-cli generate-regression-dataset`: encoded
windows (spectrogram .pt tensors or RAM .png images) plus a manifest carrying,
per window, the source magnitude and two physical predictors -- log SNR
against the station's noise floor, and epicentral distance.

Design follows report.md 8.2/8.3. Local magnitude is essentially log peak
amplitude with a distance correction, but the RAM transform is exactly
scale-invariant, so amplitude cannot reach the network through the image at
all. Feeding `log_snr` and `log(distance)` alongside the encoded window gives
the model a direct path to the dominant term; the image contributes shape,
frequency content and coda structure on top.

Because of that, the script always reports two reference points next to the
CNN, so a headline MAE cannot be mistaken for a result:

  * predict-the-mean  -- the floor any model must beat
  * ridge on the two scalars alone -- effectively a fitted local-magnitude
    relation. If the CNN does not beat this, the encoded window is
    contributing nothing beyond amplitude and distance, which is the specific
    thing worth knowing before investing further.

Usage:
    python cnn_regression.py --dataset-dir dataset_reg_60s --window-seconds 60
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

from training import ImprovedSeismicCNN, seed_everything

AUX_COLUMNS = ["log_snr", "log_distance"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MagnitudeDataset(Dataset):
    """
    Manifest-driven: rows carry the label and auxiliary scalars, so the encoded
    window on disk stays exactly what the generator wrote.

    Auxiliary features are standardized with statistics computed on TRAIN only
    and passed in for val/test -- fitting them per-split would leak the test
    distribution into its own normalization.
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, aux_stats=None, transform=None):
        self.rows = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

        aux = self.rows[AUX_COLUMNS].to_numpy(dtype=np.float64)
        # NaNs (e.g. no station catalog -> no distance) become 0 after
        # standardization, i.e. "no information", rather than poisoning the batch.
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
        path = self.root / row["split"] / row["filename"]
        if path.suffix == ".pt":
            x = torch.load(path, weights_only=True).float()
        else:
            from PIL import Image
            from torchvision import transforms
            x = transforms.functional.to_tensor(Image.open(path).convert("RGB"))
        if self.transform is not None:
            x = self.transform(x)
        return (x,
                torch.from_numpy(self.aux[idx]),
                torch.tensor(self.targets[idx], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RegressionSeismicCNN(nn.Module):
    """
    Shared CNN trunk, then the pooled features are concatenated with the
    auxiliary scalars before the regression head.

    `use_aux=False` reproduces an image-only model, which is the honest
    ablation for "does the encoded window carry magnitude information at all".
    """

    def __init__(self, dropout1=0.4, dropout2=0.2, hidden_dim=128,
                 num_stages=4, in_channels=3, n_aux=len(AUX_COLUMNS), use_aux=True):
        super().__init__()
        backbone = ImprovedSeismicCNN(dropout1=dropout1, dropout2=dropout2,
                                      hidden_dim=hidden_dim, num_stages=num_stages,
                                      in_channels=in_channels)
        self.in_conv = backbone.in_conv
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.global_pool = backbone.global_pool
        feat_dim = 256 if num_stages >= 4 else 128

        self.use_aux = use_aux
        self.n_aux = n_aux if use_aux else 0
        self.head = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(feat_dim + self.n_aux, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, aux=None):
        x = self.in_conv(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.global_pool(x), 1)
        if self.use_aux:
            x = torch.cat([x, aux], dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Metrics / baselines
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def report_baselines(train_ds, test_ds):
    """Predict-the-mean, and ridge on the auxiliary scalars alone."""
    y_tr, y_te = train_ds.targets, test_ds.targets
    print("\n--- Reference points (test set) ---")
    m = regression_metrics(y_te, np.full_like(y_te, y_tr.mean()))
    print(f"  predict-the-mean      MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:+.3f}")

    try:
        ridge = Ridge(alpha=1.0).fit(train_ds.aux, y_tr)
        m2 = regression_metrics(y_te, ridge.predict(test_ds.aux))
        print(f"  ridge(log_snr, log_dist)  MAE {m2['MAE']:.3f}  RMSE {m2['RMSE']:.3f}  R2 {m2['R2']:+.3f}"
              "   <- a fitted local-magnitude relation; the CNN must beat this")
        return m2["MAE"]
    except Exception as e:
        print(f"  [WARN] ridge baseline failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Magnitude regression on encoded seismic windows.")
    p.add_argument("--dataset-dir", type=str, required=True,
                   help="Directory produced by `seismic-cli generate-regression-dataset`.")
    p.add_argument("--save-dir", type=str, default="trained_model_regression")
    p.add_argument("--window-seconds", type=float, default=None,
                   help="Selects the short preset at <= 12s (smaller model, harder regularization).")
    p.add_argument("--no-aux", action="store_true",
                   help="Ablation: train on the encoded window only, dropping log_snr/log_distance.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout1", type=float, default=None)
    p.add_argument("--dropout2", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-stages", type=int, default=None, choices=[3, 4])
    p.add_argument("--huber-delta", type=float, default=0.0,
                   help="If > 0 use SmoothL1 with this beta instead of plain L1 (MAE).")
    args = p.parse_args()

    short = args.window_seconds is not None and args.window_seconds <= 12.0
    defaults = dict(batch_size=64 if short else 128, num_epochs=80 if short else 100,
                    patience=10, lr=2e-4 if short else 1e-4,
                    weight_decay=3e-2 if short else 1e-2,
                    dropout1=0.5 if short else 0.4, dropout2=0.3 if short else 0.2,
                    hidden_dim=64 if short else 128, num_stages=3 if short else 4)
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.preset_name = "short" if short else "long"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    for col in ("magnitude", "log_snr"):
        if col not in manifest.columns:
            raise ValueError(f"manifest.csv is missing '{col}'. Regenerate it with "
                             f"`seismic-cli generate-regression-dataset`.")
    if "distance_km" not in manifest.columns:
        manifest["distance_km"] = np.nan
    manifest["log_distance"] = np.log(manifest["distance_km"].clip(lower=1.0))

    parts = {}
    for split in ("train", "val", "test"):
        sub = manifest[manifest.split == split]
        if sub.empty:
            raise ValueError(f"Split '{split}' is empty in the manifest.")
        parts[split] = sub

    train_ds = MagnitudeDataset(parts["train"], root)
    stats = train_ds.aux_stats()          # TRAIN-only statistics, reused below
    val_ds = MagnitudeDataset(parts["val"], root, aux_stats=stats)
    test_ds = MagnitudeDataset(parts["test"], root, aux_stats=stats)

    sample_x, sample_aux, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    print("=" * 64)
    print(f"Magnitude regression | preset '{args.preset_name}' | aux={'off' if args.no_aux else 'on'}")
    print(f"  input {tuple(sample_x.shape)} | aux dim {sample_aux.numel()} "
          f"({', '.join(AUX_COLUMNS)})")
    for s in ("train", "val", "test"):
        d = parts[s]
        print(f"  {s:5s}: n={len(d):5d}  M {d.magnitude.min():.1f}-{d.magnitude.max():.1f} "
              f"(mean {d.magnitude.mean():.2f})  events={d.event_id.nunique()}")
    n_dist = int(manifest.distance_km.notna().sum())
    if n_dist == 0:
        print("  [note] distance_km is absent -- rerun generation with --station-catalog "
              "to add it; magnitude and distance are confounded without it.")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionSeismicCNN(dropout1=args.dropout1, dropout2=args.dropout2,
                                 hidden_dim=args.hidden_dim, num_stages=args.num_stages,
                                 in_channels=in_channels, n_aux=len(AUX_COLUMNS),
                                 use_aux=not args.no_aux).to(device)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # L1 matches how magnitude error is judged and resists the heavy tail of
    # large events far better than MSE.
    criterion = (nn.SmoothL1Loss(beta=args.huber_delta) if args.huber_delta > 0
                 else nn.L1Loss())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_regression_model.pth")
    best_val_mae = float("inf")
    epochs_no_improve = 0

    def evaluate(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, aux, y in loader:
                x, aux = x.to(device), aux.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = model(x, aux)
                preds.extend(out.float().squeeze(1).cpu().tolist())
                trues.extend(y.tolist())
        return np.array(trues), np.array(preds)

    for epoch in range(args.num_epochs):
        model.train()
        running = 0.0
        for x, aux, y in train_loader:
            x, aux, y = x.to(device), aux.to(device), y.to(device).unsqueeze(1)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x, aux), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            running += loss.item() * x.size(0)
        scheduler.step()

        yt, yp = evaluate(val_loader)
        vm = regression_metrics(yt, yp)
        print(f"Epoch {epoch+1}/{args.num_epochs} | train {running/len(train_ds):.4f} "
              f"| val MAE {vm['MAE']:.4f}  RMSE {vm['RMSE']:.4f}  R2 {vm['R2']:+.4f}")

        if vm["MAE"] < best_val_mae:
            best_val_mae = vm["MAE"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val MAE {best_val_mae:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: val MAE flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, yp = evaluate(test_loader)
    tm = regression_metrics(yt, yp)

    ridge_mae = report_baselines(train_ds, test_ds)
    print(f"\n--- CNN ({'image only' if args.no_aux else 'image + aux'}) ---")
    print(f"  MAE {tm['MAE']:.3f}  RMSE {tm['RMSE']:.3f}  R2 {tm['R2']:+.3f}")
    if ridge_mae is not None:
        delta = ridge_mae - tm["MAE"]
        verdict = ("the encoded window adds information beyond amplitude+distance"
                   if delta > 0.01 else
                   "NO measurable gain over amplitude+distance alone -- the encoding "
                   "is not contributing")
        print(f"  vs ridge baseline: {delta:+.3f} MAE  ->  {verdict}")
    print(f"\n  residual std {np.std(yt - yp):.3f} magnitude units over "
          f"M {yt.min():.1f}-{yt.max():.1f}")


if __name__ == "__main__":
    main()
