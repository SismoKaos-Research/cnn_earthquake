"""
RAM-image classifier with amplitude-scalar auxiliary inputs -- the direct fix
for RAM's diagnosed blind spot.

RAM is exactly scale-invariant (report.md 8.2): RAM(c*x) == RAM(x) for any
positive c, so the image cannot represent absolute amplitude or amplitude-
above-noise, which is exactly the quantity STA/LTA and normalized
spectrograms use and RAM-only classifiers have consistently trailed them on
at short windows. `--baseline` standardization does not fix this (the
invariance makes the image insensitive to WHICH (mu, sigma) it is computed
with) -- the fix has to bypass the image entirely: concatenate the discarded
scalar(s) directly into the classifier, the same pattern `cnn_regression.py`
uses for magnitude (log_snr, log_distance).

`seismic-cli generate-ram-aux-dataset` writes {img, aux} tensors where
aux = [log_snr, log_rms]. `--no-aux` trains the SAME trunk with the aux
concatenation removed, so it is architecturally identical to the plain RAM
classifier -- a clean before/after comparison isolating exactly what the two
extra scalars are worth, rather than confounding it with other architecture
differences.

Usage:
    python cnn_ram_aux.py --dataset-dir dataset_ramaux_6s
    python cnn_ram_aux.py --dataset-dir dataset_ramaux_6s --no-aux   # ablation

Not imported by anything else -- standalone script.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

from seismolib.metrics import majority_class_baseline
from seismolib.model.trunk2d import SETrunk2D
from seismolib.training import seed_everything

AUX_FEATURES = ["log_snr", "log_rms"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class RamAuxTensorDataset(Dataset):
    """
    Loads the {img, aux} tensors written by `seismic-cli generate-ram-aux-dataset`.

    Expects an ImageFolder-style layout: <root>/<class_name>/<name>.pt
    aux is standardized with TRAIN-only stats (same convention as
    cnn_regression.py's aux features), passed in and reused for val/test.
    """

    def __init__(self, root_dir, aux_stats=None):
        """Indexes every .pt sample under `root_dir`'s class subdirectories.

        Args:
            root_dir: Directory with one subdirectory per class, each
                containing .pt files (`seismic-cli generate-ram-aux-dataset`
                output), e.g. `<root>/00_noise/`, `<root>/01_earthquake/`.
            aux_stats: Optional (mean, std) tuple to standardize `aux` with;
                if None, computed from this split's own samples (the train
                split should always pass None; val/test must reuse the
                train split's stats).

        Raises:
            FileNotFoundError: If `root_dir` doesn't exist.
            RuntimeError: If no .pt files are found under `root_dir`.
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root_dir}")

        self.classes = sorted(d.name for d in self.root_dir.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for fpath in sorted(self.root_dir.rglob("*.pt")):
            cls_name = fpath.parent.name
            if cls_name in self.class_to_idx:
                self.samples.append((fpath, self.class_to_idx[cls_name]))
        if not self.samples:
            raise RuntimeError(f"No .pt tensors found under {self.root_dir}")

        if aux_stats is None:
            auxs = [torch.load(fp, weights_only=True)["aux"].numpy() for fp, _ in self.samples]
            A = np.stack(auxs, axis=0)
            aux_stats = (np.mean(A, 0), np.std(A, 0) + 1e-6)
        self.aux_stats = aux_stats

    def sample_shapes(self):
        """Returns (img_shape, aux_shape) of the first sample, as a quick check.

        Returns:
            Tuple of (img tensor shape tuple, aux tensor shape tuple).
        """
        d = torch.load(self.samples[0][0], weights_only=True)
        return tuple(d["img"].shape), tuple(d["aux"].shape)

    def validate_shapes(self, limit=None):
        """Every tensor must share one shape or the default collate throws mid-run.

        Args:
            limit: If given, only checks the first `limit` samples instead
                of the full dataset (cheaper, useful for a quick smoke test).

        Returns:
            Tuple of (img_shape, aux_shape) that every checked sample matched.

        Raises:
            ValueError: If any checked sample's img or aux shape differs
                from the first sample's.
        """
        img_shape, aux_shape = self.sample_shapes()
        paths = self.samples if limit is None else self.samples[:limit]
        for fpath, _ in paths:
            d = torch.load(fpath, weights_only=True)
            i, a = tuple(d["img"].shape), tuple(d["aux"].shape)
            if i != img_shape or a != aux_shape:
                raise ValueError(
                    f"Inconsistent tensor shapes at {fpath.name}: img {i} (expected "
                    f"{img_shape}), aux {a} (expected {aux_shape}). Regenerate with "
                    f"`seismic-cli generate-ram-aux-dataset`.")
        return img_shape, aux_shape

    def __len__(self):
        """Returns the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns one (img, aux, label) sample, with aux standardized.

        Args:
            idx: Index into `self.samples`.

        Returns:
            Tuple of (float32 img tensor, float32 aux tensor, float32
            scalar label tensor).
        """
        fpath, label = self.samples[idx]
        d = torch.load(fpath, weights_only=True)
        am, asd = self.aux_stats
        aux = (d["aux"].numpy() - am) / asd
        return (d["img"].float(), torch.from_numpy(aux).float(),
                torch.tensor(label, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RamAuxCNN(SETrunk2D):
    """
    `model.trunk2d.SETrunk2D`, with the pooled features concatenated with the
    aux vector before the head. `use_aux=False` drops the concatenation
    entirely -- architecturally identical to the plain RAM classifier, for a
    controlled ablation.
    """

    def __init__(self, aux_dim, use_aux=True, dropout1=0.5, dropout2=0.3,
                hidden_dim=64, num_stages=4, in_channels=3):
        """See `SETrunk2D.__init__` (`num_classes=1` always here; `aux_dim`
        is forced to 0 when `use_aux` is False, disabling the aux
        concatenation for a controlled ablation).

        Args:
            aux_dim: Width of the auxiliary scalar vector. Ignored (treated
                as 0) if `use_aux` is False.
            use_aux: If False, disables the aux concatenation entirely --
                architecturally identical to the plain RAM classifier.
            dropout1: See `SETrunk2D.__init__`.
            dropout2: See `SETrunk2D.__init__`.
            hidden_dim: See `SETrunk2D.__init__`.
            num_stages: See `SETrunk2D.__init__`.
            in_channels: See `SETrunk2D.__init__`.
        """
        super().__init__(num_stages=num_stages, in_channels=in_channels,
                         aux_dim=aux_dim if use_aux else 0, num_classes=1,
                         dropout1=dropout1, dropout2=dropout2, hidden_dim=hidden_dim)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="RAM-image classifier with amplitude aux scalars.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-ram-aux-dataset`.")
    p.add_argument("--save-dir", default="trained_model_ram_aux")
    p.add_argument("--no-aux", action="store_true",
                   help="Drop the aux concatenation -- architecture-matched ablation "
                        "against the plain RAM classifier.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--dropout1", type=float, default=0.6)
    p.add_argument("--dropout2", type=float, default=0.4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-stages", type=int, default=3, choices=[3, 4],
                   help="training.py's 'short' preset (3) for short windows like the 6s "
                        "runs used elsewhere in this project; use 4 for 60s-scale windows.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    """Loads the RAM+aux dataset, trains `RamAuxCNN`, and reports test
    accuracy/AUC/MCC plus the majority-class floor."""
    args = parse_args()
    seed_everything(args.seed)
    use_aux = not args.no_aux

    train_ds = RamAuxTensorDataset(f"{args.dataset_dir}/train")
    val_ds = RamAuxTensorDataset(f"{args.dataset_dir}/val", aux_stats=train_ds.aux_stats)
    test_ds = RamAuxTensorDataset(f"{args.dataset_dir}/test", aux_stats=train_ds.aux_stats)

    img_shape, aux_shape = train_ds.validate_shapes()
    for name, ds in (("val", val_ds), ("test", test_ds)):
        i, a = ds.sample_shapes()
        if i != img_shape or a != aux_shape:
            raise ValueError(f"{name} tensors are img {i} aux {a}, but train is "
                             f"img {img_shape} aux {aux_shape}.")

    print("=" * 64)
    print(f"RAM classifier | use_aux={use_aux}")
    print(f"  img {img_shape} | aux {aux_shape} = {AUX_FEATURES}")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        n_pos = sum(1 for _, lbl in ds.samples if lbl == ds.class_to_idx.get("01_earthquake", 1))
        print(f"  {name:5s}: n={len(ds):6d}  earthquake={n_pos}  noise={len(ds) - n_pos}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RamAuxCNN(aux_shape[-1], use_aux=use_aux, dropout1=args.dropout1,
                      dropout2=args.dropout2, hidden_dim=args.hidden_dim,
                      num_stages=args.num_stages, in_channels=img_shape[0]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | parameters: {n_params:,} | train samples: {len(train_ds)} "
          f"({n_params / max(1, len(train_ds)):.2f} params/sample)")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_ram_aux.pth")
    best_auc, no_improve = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = running_loss_raw = 0.0
        for img, aux, labels in train_loader:
            img, aux = img.to(device), aux.to(device)
            labels_raw = labels.unsqueeze(1).to(device)
            labels_smooth = labels_raw * 0.8 + 0.1

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(img, aux)
                loss = criterion(outputs, labels_smooth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item() * labels.size(0)
            with torch.no_grad():
                raw = F.binary_cross_entropy_with_logits(outputs.detach().float(), labels_raw)
            running_loss_raw += raw.item() * labels.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        avg_loss_raw = running_loss_raw / len(train_loader.dataset)

        model.eval()
        v_labels, v_probs, v_preds = [], [], []
        correct = total = 0
        with torch.no_grad():
            for img, aux, labels in val_loader:
                img, aux = img.to(device), aux.to(device)
                labels_dev = labels.unsqueeze(1).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(img, aux)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.50).float()
                correct += (preds == labels_dev).sum().item()
                total += labels.size(0)
                v_labels.extend(labels.tolist())
                v_probs.extend(probs.float().cpu().squeeze(1).tolist())
                v_preds.extend(preds.float().cpu().squeeze(1).tolist())

        val_acc = correct / max(1, total)
        try:
            val_auc = roc_auc_score(v_labels, v_probs)
            val_mcc = matthews_corrcoef(v_labels, v_preds)
        except ValueError:
            val_auc = val_mcc = 0.0
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | loss {avg_loss:.4f} (unsmoothed {avg_loss_raw:.4f}) "
              f"| val acc {val_acc:.4f} auc {val_auc:.4f} mcc {val_mcc:.4f}")

        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val AUC {best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val AUC flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for img, aux, labels in test_loader:
            img, aux = img.to(device), aux.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(img, aux)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.50).float()
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.float().cpu().squeeze(1).tolist())
            all_preds.extend(preds.float().cpu().squeeze(1).tolist())

    all_labels = np.array(all_labels); all_preds = np.array(all_preds)
    print(f"\n--- RAM classifier (use_aux={use_aux}) ---")
    print(f"Final Test Accuracy: {(all_labels == all_preds).mean() * 100:.2f}%")
    try:
        print(f"Final Test AUC:      {roc_auc_score(all_labels, all_probs):.4f}")
        print(f"Final Test MCC:      {matthews_corrcoef(all_labels, all_preds):.4f}")
    except ValueError:
        pass

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    train_labels = np.array([lbl for _, lbl in train_ds.samples])
    maj, maj_acc, maj_bal = majority_class_baseline(train_labels, all_labels)
    print(f"\nmajority-class floor: predicting {maj} always -> "
         f"accuracy {maj_acc:.4f}  balanced {maj_bal:.4f}")


if __name__ == "__main__":
    main()
