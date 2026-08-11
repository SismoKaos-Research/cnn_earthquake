"""
Shared model + training core for the seismic classifiers.

Not a runnable script -- imported only. Both entry points use this:
`cnn_train.py` (RAM PNG images via ImageFolder) and `cnn_from_tensor.py`
(spectrogram .pt tensors) import `ImprovedSeismicCNN`, `PRESETS`,
`build_arg_parser`, `resolve_preset`, `print_config`, and `run_training`.
`seed_everything` is imported by nearly every other training script in
src/. `cnn_run.py` unpickles a full saved `ImprovedSeismicCNN` object by
its qualified class path (`training.ImprovedSeismicCNN`), so that class
must stay defined at this name and module path.

Keeping the loop in one place is deliberate -- the two entry-point scripts
had drifted apart, so fixes landed in one and not the other (the val/test
threshold mismatch, label-smoothing diagnostics, AUC checkpointing, CPU
support, seeding). Anything added here reaches both.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader

from model.blocks import ResBlock, SEBlock  # re-exported for existing `from training import ResBlock` callers
from model.trunk2d import SETrunk2D

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class ImprovedSeismicCNN(SETrunk2D):
    """
    ResNet-style CNN with SE blocks. Global average pooling makes it agnostic
    to input resolution, so the same architecture takes 64x64 RAM images and
    non-square spectrograms (e.g. 129x94) without modification.

    num_stages=4 keeps the original layer1..layer4 state-dict keys, so
    existing checkpoints load unchanged. Stays defined at this name/module
    path (rather than moving to model/trunk2d.py) because cnn_run.py unpickles
    a full saved model object by qualified class path.
    """
    def __init__(self, dropout1=0.5, dropout2=0.3, hidden_dim=64, num_stages=4, in_channels=3):
        """Initializes the trunk with no auxiliary input and a single-logit output.

        Args:
            dropout1: Dropout before the classifier's hidden layer.
            dropout2: Dropout before the classifier's output layer.
            hidden_dim: Width of the classifier's hidden layer.
            num_stages: 3 or 4 residual stages (see `SETrunk2D.__init__`).
            in_channels: Number of input image channels.
        """
        super().__init__(num_stages=num_stages, in_channels=in_channels, aux_dim=0,
                         num_classes=1, dropout1=dropout1, dropout2=dropout2,
                         hidden_dim=hidden_dim)

    def forward(self, x):
        """Runs the trunk and classifier head.

        Args:
            x: Input image batch, shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, 1) -- a raw logit, no activation applied.
        """
        return super().forward(x)


# --------------------------------------------------------------------------
# Window-length presets
# --------------------------------------------------------------------------
#
# Long windows tolerate the full-capacity network; short windows overfit it
# within ~10 epochs. The short preset shrinks the model, regularizes harder,
# and checkpoints on val AUC (ranking) rather than val loss, which degrades
# earlier as calibration drifts.

PRESETS = {
    "long": dict(batch_size=128, num_epochs=100, patience=7, lr=1e-4,
                 weight_decay=1e-2, dropout1=0.5, dropout2=0.3, hidden_dim=64,
                 num_stages=4, monitor="loss", scheduler="plateau", random_erasing=0.0),
    "short": dict(batch_size=64, num_epochs=80, patience=10, lr=2e-4,
                  weight_decay=3e-2, dropout1=0.6, dropout2=0.4, hidden_dim=32,
                  num_stages=3, monitor="auc", scheduler="cosine", random_erasing=0.25),
}
SHORT_WINDOW_THRESHOLD_SEC = 12.0

_TUNABLES = ["batch_size", "num_epochs", "patience", "lr", "weight_decay",
             "dropout1", "dropout2", "hidden_dim", "num_stages", "monitor",
             "scheduler", "random_erasing"]


def build_arg_parser(description: str, default_dataset_dir: str = "./dataset",
                     default_save_dir: str = "trained_model") -> argparse.ArgumentParser:
    """Builds the CLI argument parser shared by cnn_train.py and cnn_from_tensor.py.

    Args:
        description: Program description shown in `--help`.
        default_dataset_dir: Default value for `--dataset-dir`.
        default_save_dir: Default value for `--save-dir`.

    Returns:
        An `argparse.ArgumentParser` with `--dataset-dir`, `--save-dir`,
        `--window-seconds`, `--seed`, `--num-workers`, and one flag per
        tunable in `_TUNABLES` (each defaulting to None, meaning "take it
        from the preset resolved by `resolve_preset`").
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset-dir", type=str, default=default_dataset_dir,
                   help="Directory containing train/val/test subfolders.")
    p.add_argument("--save-dir", type=str, default=default_save_dir,
                   help="Directory to save checkpoints into.")
    p.add_argument("--window-seconds", type=float, default=None,
                   help="Window length of the dataset. Selects the 'short' preset when "
                        f"<= {SHORT_WINDOW_THRESHOLD_SEC:.0f}s, else 'long'. Omit for the long "
                        "preset. Any explicitly passed flag overrides its preset value.")
    p.add_argument("--seed", type=int, default=42, help="Seed for python/numpy/torch.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    # Tunables: None means "take it from the preset".
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stopping patience in epochs without improvement of the monitored metric.")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout1", type=float, default=None)
    p.add_argument("--dropout2", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="Width of the classifier's hidden layer.")
    p.add_argument("--num-stages", type=int, default=None, choices=[3, 4],
                   help="Residual stages (3 = ~0.3M params, 4 = ~1.25M).")
    p.add_argument("--monitor", type=str, default=None, choices=["loss", "auc"],
                   help="Metric used for checkpointing and early stopping.")
    p.add_argument("--scheduler", type=str, default=None, choices=["plateau", "cosine"])
    p.add_argument("--random-erasing", type=float, default=None,
                   help="RandomErasing probability on training inputs (0 disables).")
    return p


def resolve_preset(args):
    """Fills in any unset tunable from the "short" or "long" preset.

    Args:
        args: Parsed args from a parser built by `build_arg_parser`. Mutated
            in place: each tunable in `_TUNABLES` that is currently None is
            set from the resolved preset, and `args.preset_name` is added.

    Returns:
        The same `args` object, for convenient chaining.
    """
    name = "long"
    if args.window_seconds is not None and args.window_seconds <= SHORT_WINDOW_THRESHOLD_SEC:
        name = "short"
    for key, value in PRESETS[name].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.preset_name = name
    return args


def print_config(args, extra=None):
    """Prints the resolved preset name and every tunable's final value.

    Args:
        args: Args already passed through `resolve_preset`.
        extra: Optional dict of additional key/value pairs to print
            alongside the tunables (e.g. dataset-derived info).
    """
    print("=" * 60)
    print(f"Preset: '{args.preset_name}'"
          + (f" (window_seconds={args.window_seconds})" if args.window_seconds
             else " (no --window-seconds given)"))
    for key in _TUNABLES + ["seed"]:
        print(f"  {key:15s} = {getattr(args, key)}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:15s} = {v}")
    print("=" * 60)


def seed_everything(seed: int):
    """Seeds Python's random, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value applied to all three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Training / evaluation

def run_training(args, train_dataset, val_dataset, test_dataset, in_channels=3):
    """Runs the full train/validate/test cycle shared by both entry points.

    Trains an `ImprovedSeismicCNN` with AdamW + AMP + gradient clipping,
    checkpointing on `args.monitor` ("loss" or "auc") with early stopping on
    `args.patience`, then reloads the best checkpoint and reports final test
    accuracy/AUC/MCC, a confusion matrix, and a full classification report.
    Also saves the full model object (not just the state dict) to
    `<save_dir>/full_model.pth` for scripts that need to unpickle it directly.

    Args:
        args: Parsed, preset-resolved args (see `build_arg_parser`,
            `resolve_preset`) providing seed, batch_size, num_workers,
            dropout1/dropout2/hidden_dim/num_stages, lr, weight_decay,
            scheduler, num_epochs, monitor, patience, and save_dir.
        train_dataset: Training dataset yielding (image, label) pairs.
        val_dataset: Validation dataset, same format.
        test_dataset: Test dataset, same format.
        in_channels: Number of input image channels (3 for RGB RAM images,
            1 for single-channel spectrograms).

    Returns:
        None. Prints per-epoch and final metrics; saves
        `<save_dir>/best_seismic_model.pth` (state dict) and
        `<save_dir>/full_model.pth` (full pickled model) as a side effect.
    """
    seed_everything(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSeismicCNN(dropout1=args.dropout1, dropout2=args.dropout2,
                               hidden_dim=args.hidden_dim, num_stages=args.num_stages,
                               in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Model parameters: {n_params:,} | Train samples: {len(train_dataset)} "
          f"({n_params / max(1, len(train_dataset)):.0f} params/sample)")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_seismic_model.pth")
    best_metric = float("inf") if args.monitor == "loss" else float("-inf")
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_loss_raw = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels_raw = labels.float().unsqueeze(1).to(device)
            # Label smoothing (0 -> 0.1, 1 -> 0.9) caps rewarded confidence.
            labels = labels_raw * 0.8 + 0.1

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_train_loss += loss.item() * inputs.size(0)
            # Diagnostic only: BCE against TRUE labels, so it is comparable to
            # val loss. Smoothed loss floors at H(0.1) ~ 0.325 nats, which makes
            # the raw train/val gap look larger than it is. No effect on training.
            with torch.no_grad():
                raw = F.binary_cross_entropy_with_logits(outputs.detach().float(), labels_raw)
            running_train_loss_raw += raw.item() * inputs.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        avg_train_loss_raw = running_train_loss_raw / len(train_loader.dataset)

        # --- validation ---
        model.eval()
        running_val_loss = 0.0
        correct = total = 0
        v_labels, v_probs, v_preds = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.50).float()   # same rule as the final test eval
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                v_labels.extend(labels.cpu().squeeze(1).tolist())
                v_probs.extend(probs.float().cpu().squeeze(1).tolist())
                v_preds.extend(preds.float().cpu().squeeze(1).tolist())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = correct / max(1, total)

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        try:
            val_auc = roc_auc_score(v_labels, v_probs)
            val_mcc = matthews_corrcoef(v_labels, v_preds)
        except ValueError:
            val_auc = val_mcc = 0.0

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (unsmoothed: {avg_train_loss_raw:.4f}) "
              f"| Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_accuracy:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")

        current = avg_val_loss if args.monitor == "loss" else val_auc
        improved = current < best_metric if args.monitor == "loss" else current > best_metric
        if improved:
            best_metric = current
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => Best model saved to {save_path}! (val {args.monitor} = {current:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  => No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: val {args.monitor} hasn't improved in {args.patience} epochs.")
            break

    torch.save(model, os.path.join(args.save_dir, "full_model.pth"))

    # --- final test evaluation on the best checkpoint ---
    print("\nRunning Final Evaluation on Test Set...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    correct = total = 0
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.50).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().squeeze(1).tolist())
            all_probs.extend(probs.float().cpu().squeeze(1).tolist())
            all_preds.extend(preds.float().cpu().squeeze(1).tolist())

    print(f"Final Test Accuracy: {correct / max(1, total) * 100:.2f}%")
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
