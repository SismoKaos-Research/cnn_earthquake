"""
Dual-channel CNN + LSTM/self-attention classifier, plus the amplitude aux
branch that fixed the plain RAM classifier (see cnn_ram_aux.py: test AUC
0.836 -> 0.923 from adding [log_snr, log_rms] alone).

This tests whether that same fix helps once the RAM image is only one of two
branches rather than the whole classifier -- the 2D branch here is exactly
as scale-invariant as ever (report.md 8.2), and this gives the model a
direct route to the amplitude information it structurally cannot see,
whether or not the 1D raw-waveform branch already compensates for some of it.

`--channels` ablates any combination, matching `cnn_lstm.py`'s catalog model:
    all      1D + 2D + aux, fused then concatenated with aux
    1d / 2d  single branch alone (same as cnn_lstm_classify.py's ablations)
    aux      the two scalars alone, through a small head -- the floor
    1d+aux / 2d+aux   one branch plus the amplitude fix, without the other

aux concatenates AFTER the 1D/2D fusion, not through it, so `--fusion` (linear
or gate, same as cnn_lstm_classify.py) is an orthogonal choice to the aux
fix -- this isolates each of the two ideas from report.md's 10.6 next-steps
list rather than conflating them.

Usage:
    python cnn_lstm_classify_aux.py --dataset-dir dataset_dualaux_6s
    python cnn_lstm_classify_aux.py --dataset-dir dataset_dualaux_6s --channels 2d+aux
    python cnn_lstm_classify_aux.py --dataset-dir dataset_dualaux_6s --fusion gate

Also imported (not just run standalone): cnn_lstm_stack_aux.py imports
`DualChannelAuxBinaryNet` and `RamDualAuxTensorDataset` from this module,
loading this script's checkpoint (via `--ckpt-1d`/`--ckpt-2d`) as one branch
of a stacked ensemble.
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
from seismolib.model.dual_channel import DualChannelNet
from seismolib.training import seed_everything

AUX_FEATURES = ["log_snr", "log_rms"]
CHANNEL_CHOICES = ["all", "1d", "2d", "aux", "1d+aux", "2d+aux"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class RamDualAuxTensorDataset(Dataset):
    """
    Loads the {seq, img, aux} tensors written by
    `seismic-cli generate-dual-aux-dataset`.

    Expects an ImageFolder-style layout: <root>/<class_name>/<name>.pt
    aux is standardized with TRAIN-only stats, passed in and reused for val/test.
    """

    def __init__(self, root_dir, aux_stats=None):
        """Indexes every .pt sample under `root_dir`'s class subdirectories.

        Args:
            root_dir: Directory with one subdirectory per class, each
                containing .pt files (`seismic-cli generate-dual-aux-dataset`
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
        """Returns (seq_shape, img_shape, aux_shape) of the first sample, as a quick check.

        Returns:
            Tuple of (seq tensor shape tuple, img tensor shape tuple, aux
            tensor shape tuple).
        """
        d = torch.load(self.samples[0][0], weights_only=True)
        return tuple(d["seq"].shape), tuple(d["img"].shape), tuple(d["aux"].shape)

    def validate_shapes(self, limit=None):
        """Every tensor must share one shape or the default collate throws mid-run.

        Args:
            limit: If given, only checks the first `limit` samples instead
                of the full dataset (cheaper, useful for a quick smoke test).

        Returns:
            Tuple of (seq_shape, img_shape, aux_shape) that every checked
            sample matched.

        Raises:
            ValueError: If any checked sample's seq, img, or aux shape
                differs from the first sample's.
        """
        seq_shape, img_shape, aux_shape = self.sample_shapes()
        paths = self.samples if limit is None else self.samples[:limit]
        for fpath, _ in paths:
            d = torch.load(fpath, weights_only=True)
            s, i, a = tuple(d["seq"].shape), tuple(d["img"].shape), tuple(d["aux"].shape)
            if s != seq_shape or i != img_shape or a != aux_shape:
                raise ValueError(
                    f"Inconsistent tensor shapes at {fpath.name}: seq {s} img {i} aux {a} "
                    f"(expected seq {seq_shape} img {img_shape} aux {aux_shape}). Regenerate "
                    f"with `seismic-cli generate-dual-aux-dataset`.")
        return seq_shape, img_shape, aux_shape

    def __len__(self):
        """Returns the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns one (seq, img, aux, label) sample, with aux standardized.

        Args:
            idx: Index into `self.samples`.

        Returns:
            Tuple of (float32 seq tensor, float32 img tensor, float32 aux
            tensor, float32 scalar label tensor).
        """
        fpath, label = self.samples[idx]
        d = torch.load(fpath, weights_only=True)
        am, asd = self.aux_stats
        aux = (d["aux"].numpy() - am) / asd
        return (d["seq"].float(), d["img"].float(), torch.from_numpy(aux).float(),
                torch.tensor(label, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualChannelAuxBinaryNet(DualChannelNet):
    """`cnn_lstm_classify.DualChannelBinaryNet` plus an aux branch,
    concatenated after the 1D/2D fusion -- matching the pattern
    `cnn_lstm.DualChannelRiskNet` already uses for the catalog model."""

    def __init__(self, seq_dim, img_channels, aux_dim, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", fusion="linear", lstm_layers=1, lstm_heads=4):
        """See `DualChannelNet.__init__` (`n_classes=1`, `squeeze_output=False`
        always here -- the caller squeezes/unsqueezes as needed to match
        `BCEWithLogitsLoss`'s (batch, 1) convention)."""
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         fusion=fusion, lstm_layers=lstm_layers, lstm_heads=lstm_heads,
                         n_classes=1, squeeze_output=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Dual-channel CNN+LSTM RAM classifier + amplitude aux.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-dual-aux-dataset`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm_aux")
    p.add_argument("--channels", default="all", choices=CHANNEL_CHOICES)
    p.add_argument("--fusion", default="linear", choices=["linear", "gate"],
                   help="linear: paper's a*F1+b*F2. gate: per-example gate "
                        "(cnn_lstm.GatedFusion). Only affects channel combos where "
                        "both 1D and 2D are active (all, not 1d+aux/2d+aux).")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--lstm-layers", type=int, default=1,
                   help="LSTMAttentionBranch's LSTM depth. Never swept before this run.")
    p.add_argument("--lstm-heads", type=int, default=4,
                   help="LSTMAttentionBranch's MultiheadAttention head count. Must divide "
                        "2*hidden (the bidirectional LSTM's output width). Never swept before "
                        "this run.")
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    """Loads the dual-tensor+aux dataset, trains `DualChannelAuxBinaryNet`,
    and reports test accuracy/AUC/MCC plus the majority-class floor."""
    args = parse_args()
    seed_everything(args.seed)

    train_ds = RamDualAuxTensorDataset(f"{args.dataset_dir}/train")
    val_ds = RamDualAuxTensorDataset(f"{args.dataset_dir}/val", aux_stats=train_ds.aux_stats)
    test_ds = RamDualAuxTensorDataset(f"{args.dataset_dir}/test", aux_stats=train_ds.aux_stats)

    seq_shape, img_shape, aux_shape = train_ds.validate_shapes()
    for name, ds in (("val", val_ds), ("test", test_ds)):
        s, i, a = ds.sample_shapes()
        if s != seq_shape or i != img_shape or a != aux_shape:
            raise ValueError(f"{name} tensors are seq {s} img {i} aux {a}, but train is "
                             f"seq {seq_shape} img {img_shape} aux {aux_shape}.")

    print("=" * 64)
    print(f"Dual-channel RAM classifier + aux | channels='{args.channels}'")
    print(f"  seq {seq_shape} | img {img_shape} | aux {aux_shape} = {AUX_FEATURES}")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        n_pos = sum(1 for _, lbl in ds.samples if lbl == ds.class_to_idx.get("01_earthquake", 1))
        print(f"  {name:5s}: n={len(ds):6d}  earthquake={n_pos}  noise={len(ds) - n_pos}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelAuxBinaryNet(seq_shape[-1], img_shape[0], aux_shape[-1], hidden=args.hidden,
                                    fusion_dim=args.fusion_dim, dropout=args.dropout,
                                    channels=args.channels, fusion=args.fusion,
                                    lstm_layers=args.lstm_layers, lstm_heads=args.lstm_heads).to(device)
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
    save_path = os.path.join(args.save_dir, "best_cnnlstm_aux.pth")
    best_auc, no_improve = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = running_loss_raw = 0.0
        for seq, img, aux, labels in train_loader:
            seq, img, aux = seq.to(device), img.to(device), aux.to(device)
            labels_raw = labels.unsqueeze(1).to(device)
            labels_smooth = labels_raw * 0.8 + 0.1

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(seq, img, aux)
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
            for seq, img, aux, labels in val_loader:
                seq, img, aux = seq.to(device), img.to(device), aux.to(device)
                labels_dev = labels.unsqueeze(1).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(seq, img, aux)
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
    all_labels, all_probs, all_preds, all_gates = [], [], [], []
    with torch.no_grad():
        for seq, img, aux, labels in test_loader:
            seq, img, aux = seq.to(device), img.to(device), aux.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(seq, img, aux)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.50).float()
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.float().cpu().squeeze(1).tolist())
            all_preds.extend(preds.float().cpu().squeeze(1).tolist())
            if getattr(model, "last_gate", None) is not None:
                all_gates.extend(model.last_gate.float().cpu().squeeze(1).tolist())

    all_labels = np.array(all_labels); all_preds = np.array(all_preds)
    print(f"\n--- Dual-channel RAM classifier + aux (channels='{args.channels}', fusion='{args.fusion}') ---")
    print(f"Final Test Accuracy: {(all_labels == all_preds).mean() * 100:.2f}%")
    try:
        print(f"Final Test AUC:      {roc_auc_score(all_labels, all_probs):.4f}")
        print(f"Final Test MCC:      {matthews_corrcoef(all_labels, all_preds):.4f}")
    except ValueError:
        pass

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        if args.fusion == "gate":
            g = np.array(all_gates)
            correct = (all_labels == all_preds)
            print(f"\ngate g (1 favors 1D, 0 favors 2D): mean {g.mean():.3f}  std {g.std():.3f}"
                  f"\n  earthquake windows: mean g {g[all_labels==1].mean():.3f}"
                  f"\n  noise windows:      mean g {g[all_labels==0].mean():.3f}"
                  f"\n  correct predictions: mean g {g[correct].mean():.3f}"
                  f"\n  wrong predictions:   mean g {g[~correct].mean() if (~correct).any() else float('nan'):.3f}")
        else:
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

    train_labels = np.array([lbl for _, lbl in train_ds.samples])
    maj, maj_acc, maj_bal = majority_class_baseline(train_labels, all_labels)
    print(f"\nmajority-class floor: predicting {maj} always -> "
         f"accuracy {maj_acc:.4f}  balanced {maj_bal:.4f}")


if __name__ == "__main__":
    main()
