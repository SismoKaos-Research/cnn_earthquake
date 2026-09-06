"""
Magnitude-class (binary) classification from encoded seismic windows.

Consumes a dataset built by `seismic-cli generate-regression-dataset`: encoded
windows (spectrogram .pt tensors or RAM .png images) plus a manifest carrying,
per window, the source magnitude and two physical predictors -- log SNR
against the station's noise floor, and epicentral distance. That pipeline was
built for continuous magnitude regression (`cnn_regression.py`) and never
run; this script reuses it unchanged and only relabels the target, since the
manifest already stores the continuous magnitude value this thresholds.

The question this asks is narrower than magnitude regression: not "what is
the magnitude" but "is this a small or a larger-than-small event," classified
directly from a single short window rather than from catalog statistics
about surrounding events (that is a separate, later question -- see
report.md 10.7). `--mag-threshold` (default 2.5) turns the continuous label
into a binary one; it sits between the data's natural median (M2.3, an almost
perfectly balanced split) and the more standard M3.0 "light earthquake" cut
(which leaves only 14% of windows in the positive class) -- close to
balanced without being a value fitted to this specific catalog.

`RegressionSeismicCNN` (imported unchanged from `cnn_regression.py`) already
ends in a single un-squashed `nn.Linear(hidden_dim, 1)`, architecturally
identical to a binary classification logit head -- no model change needed,
only the loss (`BCEWithLogitsLoss` in place of `L1Loss`) and the reported
metrics.

As in `cnn_regression.py`, a headline number is never reported alone:

  * predict-the-majority-class -- the floor any model must beat
  * logistic regression on the two scalars alone -- effectively a fitted
    local-magnitude-threshold relation. If the CNN does not beat this, the
    encoded window is contributing nothing beyond amplitude and distance.

Usage:
    python cnn_magclass.py --dataset-dir dataset_magclass_3s --window-seconds 3

Not imported by anything else -- standalone script.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset

from sismokaos.magnitude.cnn_regression import AUX_COLUMNS, RegressionSeismicCNN
from sismokaos.training import seed_everything

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MagnitudeClassDataset(Dataset):
    """
    Identical to `cnn_regression.MagnitudeDataset` except the target is
    `magnitude >= mag_threshold` instead of the raw continuous value -- same
    manifest-driven loading, same TRAIN-only aux standardization.
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, mag_threshold: float,
                 aux_stats=None, transform=None):
        """Loads manifest rows and fits (or reuses) aux normalization stats.

        Args:
            manifest: Manifest rows for one split, with 'split', 'filename',
                'magnitude', and `AUX_COLUMNS` columns.
            root: Dataset root directory (contains a subdirectory per
                split).
            mag_threshold: Magnitude >= this is the positive class.
            aux_stats: Optional (mean, std) tuple to standardize aux with;
                if None, fit from this split's own data (the train split
                should pass None; val/test must reuse the train split's
                stats).
            transform: Optional callable applied to the loaded window
                tensor before it is returned.
        """
        self.rows = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

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
        self.magnitudes = self.rows["magnitude"].to_numpy(dtype=np.float32)
        self.targets = (self.magnitudes >= mag_threshold).astype(np.float32)

    def aux_stats(self):
        """Returns the (mean, std) tuple this dataset standardized aux with."""
        return (self.aux_mu, self.aux_sd)

    def __len__(self):
        """Returns the number of rows in this split."""
        return len(self.rows)

    def __getitem__(self, idx):
        """Returns one (window, aux, target) sample.

        Loads a .pt spectrogram tensor directly, or a .png RAM image
        converted to a tensor, depending on `filename`'s suffix.

        Args:
            idx: Row index into this split.

        Returns:
            Tuple of (float32 window tensor, float32 aux tensor, float32
            scalar binary target tensor).
        """
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
# Baselines
# ---------------------------------------------------------------------------

def report_baselines(train_ds, test_ds):
    """Predict-the-majority-class, and logistic regression on the auxiliary
    scalars alone.

    Args:
        train_ds: Training-split `MagnitudeClassDataset`.
        test_ds: Test-split `MagnitudeClassDataset`.

    Returns:
        The logistic baseline's test AUC (float), or None if fitting it
        failed.
    """
    y_tr, y_te = train_ds.targets, test_ds.targets
    print("\n--- Reference points (test set) ---")
    majority = 1.0 if y_tr.mean() >= 0.5 else 0.0
    maj_preds = np.full_like(y_te, majority)
    print(f"  predict-the-majority-class ({int(majority)})  "
          f"acc {accuracy_score(y_te, maj_preds) * 100:.2f}%")

    try:
        clf = LogisticRegression(max_iter=1000).fit(train_ds.aux, y_tr)
        probs = clf.predict_proba(test_ds.aux)[:, 1]
        preds = (probs > 0.5).astype(np.float32)
        auc = roc_auc_score(y_te, probs)
        acc = accuracy_score(y_te, preds) * 100
        mcc = matthews_corrcoef(y_te, preds)
        print(f"  logistic(log_snr, log_dist)  acc {acc:.2f}%  auc {auc:.4f}  mcc {mcc:+.4f}"
              "   <- a fitted amplitude/distance threshold; the CNN must beat this")
        return auc
    except Exception as e:
        print(f"  [WARN] logistic baseline failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments and fills in short/long preset defaults.

    Any tunable left at its default (None) is filled from a "short"
    (`--window-seconds <= 12`) or "long" preset, matching `training.py`'s
    PRESETS convention but inlined here rather than sharing that module's
    `build_arg_parser`/`resolve_preset` (this script's dataset/model differ
    enough that reusing them would need as much overriding as writing the
    handful of lines directly).

    Returns:
        argparse.Namespace with the script's CLI options, every tunable
        resolved to a concrete value, plus `preset_name` ("short" or
        "long").
    """
    p = argparse.ArgumentParser(description="Magnitude-class (binary) classification "
                                            "on encoded seismic windows.")
    p.add_argument("--dataset-dir", type=str, required=True,
                   help="Directory produced by `seismic-cli generate-regression-dataset`.")
    p.add_argument("--save-dir", type=str, default="trained_model_magclass")
    p.add_argument("--mag-threshold", type=float, default=2.5,
                   help="Magnitude >= this is the positive class.")
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
    """Loads the encoded-window dataset, trains `RegressionSeismicCNN` as a
    binary classifier, and reports test accuracy/AUC/MCC against the
    majority-class and logistic-regression floors.

    Raises:
        ValueError: If the manifest is missing 'magnitude' or 'log_snr', or
            if any split ("train"/"val"/"test") is empty.
    """
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

    train_ds = MagnitudeClassDataset(parts["train"], root, args.mag_threshold)
    stats = train_ds.aux_stats()          # TRAIN-only statistics, reused below
    val_ds = MagnitudeClassDataset(parts["val"], root, args.mag_threshold, aux_stats=stats)
    test_ds = MagnitudeClassDataset(parts["test"], root, args.mag_threshold, aux_stats=stats)

    sample_x, sample_aux, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    print("=" * 64)
    print(f"Magnitude-class (>= M{args.mag_threshold}) | preset '{args.preset_name}' "
          f"| aux={'off' if args.no_aux else 'on'}")
    print(f"  input {tuple(sample_x.shape)} | aux dim {sample_aux.numel()} "
          f"({', '.join(AUX_COLUMNS)})")
    for s, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        d = parts[s]
        pos = int(ds.targets.sum())
        print(f"  {s:5s}: n={len(d):5d}  M {d.magnitude.min():.1f}-{d.magnitude.max():.1f} "
              f"(mean {d.magnitude.mean():.2f})  events={d.event_id.nunique()}  "
              f"positive={pos} ({100 * pos / len(d):.1f}%)")
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_magclass_model.pth")
    best_val_auc = -1.0
    epochs_no_improve = 0

    def evaluate(loader):
        """Runs the model over `loader` and collects true labels/probabilities.

        Args:
            loader: DataLoader yielding (x, aux, y) batches.

        Returns:
            Tuple of (y_true, y_prob) arrays.
        """
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for x, aux, y in loader:
                x, aux = x.to(device), aux.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = model(x, aux)
                probs.extend(torch.sigmoid(out).float().squeeze(1).cpu().tolist())
                trues.extend(y.tolist())
        return np.array(trues), np.array(probs)

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
        preds = (yp > 0.5).astype(np.float32)
        try:
            val_auc = roc_auc_score(yt, yp)
            val_mcc = matthews_corrcoef(yt, preds)
        except ValueError:
            val_auc = val_mcc = 0.0
        val_acc = accuracy_score(yt, preds)
        print(f"Epoch {epoch+1}/{args.num_epochs} | train {running/len(train_ds):.4f} "
              f"| val acc {val_acc:.4f}  auc {val_auc:.4f}  mcc {val_mcc:+.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val AUC {best_val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: val AUC flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, yp = evaluate(test_loader)
    preds = (yp > 0.5).astype(np.float32)
    test_acc = accuracy_score(yt, preds) * 100
    try:
        test_auc = roc_auc_score(yt, yp)
        test_mcc = matthews_corrcoef(yt, preds)
    except ValueError:
        test_auc = test_mcc = float("nan")

    baseline_auc = report_baselines(train_ds, test_ds)
    print(f"\n--- CNN ({'image only' if args.no_aux else 'image + aux'}) ---")
    print(f"  Accuracy {test_acc:.2f}%  AUC {test_auc:.4f}  MCC {test_mcc:+.4f}")
    if baseline_auc is not None:
        delta = test_auc - baseline_auc
        verdict = ("the encoded window adds information beyond amplitude+distance"
                   if delta > 0.01 else
                   "NO measurable gain over amplitude+distance alone -- the encoding "
                   "is not contributing")
        print(f"  vs logistic baseline: {delta:+.4f} AUC  ->  {verdict}")

    cm = confusion_matrix(yt, preds)
    print("\nConfusion Matrix:")
    print(cm)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:")
    print(classification_report(yt, preds, digits=4))


if __name__ == "__main__":
    main()
