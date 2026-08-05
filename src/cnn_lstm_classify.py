"""
Dual-channel CNN + LSTM/self-attention (1D2D-EDL, Wang & Zhao 2025) for
earthquake-vs-noise classification, applied directly on RAM-transformed
waveforms instead of the catalog forecasting task in `cnn_lstm.py`.

    1D channel : LSTM -> multi-head self-attention over the (target_n, 3*d)
                 chunk matrix `seismic-cli generate-dual-dataset` writes --
                 the SAME reshaped samples the RAM image is computed from.
    2D channel : CNN over the (3, target_n, target_n) RAM image.
    fusion     : F = a*F_1d + b*F_2d   (a, b learned, as in the paper)
    head       : single logit, earthquake vs. noise (BCEWithLogitsLoss).

The two branches deliberately see the same underlying data (see
seismic_cli/ram_dual.py's docstring) -- this isn't pairing two independent
sources of information, it's testing whether the sequential view of the
window (1D) adds anything a CNN over its pairwise-angle image (2D) doesn't
already capture, which `--channels 1d`/`2d` ablate directly. `--channels 2d`
alone is architecturally close to the existing CNN-only RAM classifier
(`cnn_train.py`), so it doubles as that baseline's comparison point here.

Training conventions (label smoothing, unsmoothed-loss diagnostic, AMP, val
AUC/MCC, matched 0.5 threshold) match `training.py`, the shared core for the
image-only classifiers -- kept as its own loop rather than forced through
that module, since it assumes a single-tensor model, not two paired inputs.

Usage:
    python cnn_lstm_classify.py --dataset-dir dataset_dual_6s
    python cnn_lstm_classify.py --dataset-dir dataset_dual_6s --channels 2d   # CNN-only ablation
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

from cnn_lstm import CNNBranch, LSTMAttentionBranch
from training import seed_everything


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class RamDualTensorDataset(Dataset):
    """
    Loads the {seq, img} tensors written by `seismic-cli generate-dual-dataset`.

    Expects an ImageFolder-style layout: <root>/<class_name>/<name>.pt
    Class directories sort as "00_noise" < "01_earthquake", so label 1 is
    always earthquake -- matching BCEWithLogitsLoss's positive-class convention.
    """

    def __init__(self, root_dir):
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

    def sample_shapes(self):
        d = torch.load(self.samples[0][0], weights_only=True)
        return tuple(d["seq"].shape), tuple(d["img"].shape)

    def validate_shapes(self, limit=None):
        """
        Every tensor must share one shape or the default collate throws
        mid-run -- see `SeismicTensorDataset.validate_shapes` in
        cnn_from_tensor.py for the spectrogram-side version of this same
        problem (mixed station sampling rates), which `RamDualEncoder`
        avoids the same way (resample to a nominal rate before reshaping).
        """
        seq_shape, img_shape = self.sample_shapes()
        paths = self.samples if limit is None else self.samples[:limit]
        for fpath, _ in paths:
            d = torch.load(fpath, weights_only=True)
            s, i = tuple(d["seq"].shape), tuple(d["img"].shape)
            if s != seq_shape or i != img_shape:
                raise ValueError(
                    f"Inconsistent tensor shapes at {fpath.name}: seq {s} (expected "
                    f"{seq_shape}), img {i} (expected {img_shape}). Regenerate with "
                    f"`seismic-cli generate-dual-dataset`.")
        return seq_shape, img_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        d = torch.load(fpath, weights_only=True)
        return d["seq"].float(), d["img"].float(), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualChannelBinaryNet(nn.Module):
    """Same fusion design as `cnn_lstm.py`'s DualChannelRiskNet, minus the
    auxiliary-scalar branch (there are no catalog-style physical scalars
    here) and with a single-logit binary head instead of a 3-way softmax."""

    def __init__(self, seq_dim, img_channels, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all"):
        super().__init__()
        self.channels = channels
        self.use_1d = channels in ("all", "1d")
        self.use_2d = channels in ("all", "2d")
        if not (self.use_1d or self.use_2d):
            raise ValueError(f"--channels {channels} disables every branch")

        if self.use_1d:
            self.b1 = LSTMAttentionBranch(seq_dim, hidden=hidden, dropout=dropout)
            self.p1 = nn.Linear(self.b1.out_dim, fusion_dim)
        if self.use_2d:
            self.b2 = CNNBranch(img_channels, dropout=dropout)
            self.p2 = nn.Linear(self.b2.out_dim, fusion_dim)
        # Learned fusion weights (a, b in the paper's notation).
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, seq, img):
        fused = None
        if self.use_1d:
            fused = self.w1 * self.p1(self.b1(seq))
        if self.use_2d:
            f2 = self.w2 * self.p2(self.b2(img))
            fused = f2 if fused is None else fused + f2
        return self.head(fused)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dual-channel CNN+LSTM RAM classifier "
                                            "(earthquake vs. noise).")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-dual-dataset`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm_classify")
    p.add_argument("--channels", default="all", choices=["all", "1d", "2d"],
                   help="Ablation switch: 'all' is the full dual-channel model, "
                        "'1d' is LSTM+attention only, '2d' is CNN-only (close to "
                        "the existing image-only classifier's architecture).")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    train_ds = RamDualTensorDataset(f"{args.dataset_dir}/train")
    val_ds = RamDualTensorDataset(f"{args.dataset_dir}/val")
    test_ds = RamDualTensorDataset(f"{args.dataset_dir}/test")

    seq_shape, img_shape = train_ds.validate_shapes()
    for name, ds in (("val", val_ds), ("test", test_ds)):
        s, i = ds.sample_shapes()
        if s != seq_shape or i != img_shape:
            raise ValueError(f"{name} tensors are seq {s} img {i}, but train is "
                             f"seq {seq_shape} img {img_shape}.")

    print("=" * 64)
    print(f"Dual-channel RAM classifier | channels='{args.channels}'")
    print(f"  seq {seq_shape} | img {img_shape}")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        n_pos = sum(1 for _, lbl in ds.samples if lbl == ds.class_to_idx.get("01_earthquake", 1))
        print(f"  {name:5s}: n={len(ds):6d}  earthquake={n_pos}  noise={len(ds) - n_pos}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, dropout=args.dropout,
                                 channels=args.channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | parameters: {n_params:,} | train samples: {len(train_ds)} "
          f"({n_params / max(1, len(train_ds)):.1f} params/sample)")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_cnnlstm_classify.pth")
    best_auc, no_improve = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = running_loss_raw = 0.0
        for seq, img, labels in train_loader:
            seq, img = seq.to(device), img.to(device)
            labels_raw = labels.unsqueeze(1).to(device)
            labels_smooth = labels_raw * 0.8 + 0.1  # 0 -> 0.1, 1 -> 0.9

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(seq, img)
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
        scheduler.step()

        avg_loss = running_loss / len(train_loader.dataset)
        avg_loss_raw = running_loss_raw / len(train_loader.dataset)

        model.eval()
        v_labels, v_probs, v_preds = [], [], []
        correct = total = 0
        with torch.no_grad():
            for seq, img, labels in val_loader:
                seq, img = seq.to(device), img.to(device)
                labels_dev = labels.unsqueeze(1).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(seq, img)
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
        for seq, img, labels in test_loader:
            seq, img = seq.to(device), img.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(seq, img)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.50).float()
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.float().cpu().squeeze(1).tolist())
            all_preds.extend(preds.float().cpu().squeeze(1).tolist())

    all_labels = np.array(all_labels); all_preds = np.array(all_preds)
    print(f"\n--- Dual-channel RAM classifier (channels='{args.channels}') ---")
    print(f"Final Test Accuracy: {(all_labels == all_preds).mean() * 100:.2f}%")
    try:
        print(f"Final Test AUC:      {roc_auc_score(all_labels, all_probs):.4f}")
        print(f"Final Test MCC:      {matthews_corrcoef(all_labels, all_preds):.4f}")
    except ValueError:
        pass

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        print(f"\nlearned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}"
              "\n(relative magnitude indicates which representation the model leaned on)")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    main()
