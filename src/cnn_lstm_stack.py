"""
Late-fusion stacking on FROZEN 1d-only and 2d-only checkpoints.

The dual-channel model's built-in fusion is `a*F1 + b*F2`, two GLOBAL scalars
learned jointly with both branches. On this task it has consistently
underperformed the best single branch trained alone: with a RAM image as the
2D channel, 1d-only (test AUC 0.922) beat the fused model (0.914); with a
spectrogram, 2d-only (0.979) beat the fused model (0.965). A fixed, global
blend can't suppress the weaker branch on the specific examples where it's
wrong, and joint training may let a noisy branch drag down the stronger one's
own representation.

This tests a different question: given the SAME two branches, already
trained to their own optimum SEPARATELY (no joint training, no compromise),
can a small combiner on top of their frozen outputs do better than either
alone? Concretely: freeze the 1d-only and 2d-only checkpoints, collect their
pre-sigmoid logits on val and test, and fit logistic regression -- one
weight per branch plus a bias, in LOGIT space -- on val logits only (never
seen by either frozen model's own training), then evaluate on test.

If stacking beats both single branches: the problem was specifically the
joint-training fusion mechanism (fixable with a better architecture, e.g. a
per-example gate). If stacking does NOT beat the best single branch either:
the two branches are likely largely redundant rather than complementary, and
there may be a low ceiling on what any fusion of them can achieve here.

Usage:
    python cnn_lstm_stack.py --dataset-dir dataset_specdual_6s \\
        --ckpt-1d .../specdual_1d/best_cnnlstm_classify.pth \\
        --ckpt-2d .../specdual_2d/best_cnnlstm_classify.pth

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader

from cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset


def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Stack frozen 1d-only and 2d-only checkpoints.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt-1d", required=True, help="Checkpoint from `--channels 1d` training.")
    p.add_argument("--ckpt-2d", required=True, help="Checkpoint from `--channels 2d` training.")
    p.add_argument("--hidden", type=int, default=48, help="Must match the checkpoints' training run.")
    p.add_argument("--fusion-dim", type=int, default=96, help="Must match the checkpoints' training run.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def load_frozen(ckpt_path, seq_dim, img_channels, args, channels, device):
    """Rebuilds a `DualChannelBinaryNet` and loads a frozen checkpoint into it.

    Args:
        ckpt_path: Path to a state-dict checkpoint from
            `cnn_lstm_classify.py` training (e.g. `--channels 1d` or
            `--channels 2d`).
        seq_dim: Per-step feature width of the 1D sequence input.
        img_channels: Number of channels of the 2D image input.
        args: Parsed CLI args (uses hidden, fusion_dim -- must match the
            checkpoint's training run).
        channels: Which branch the checkpoint was trained with ("1d" or
            "2d").
        device: torch device to load the model onto.

    Returns:
        The reconstructed model in eval mode with every parameter's
        `requires_grad` set to False.
    """
    model = DualChannelBinaryNet(seq_dim, img_channels, hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, dropout=0.0,
                                 channels=channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def collect_logits(model, loader, device):
    """Runs a frozen model over `loader` and collects its raw logits and true labels.

    Args:
        model: Frozen model in eval mode, called as `model(seq, img)`.
        loader: DataLoader yielding (seq, img, label) batches.
        device: torch device to run inference on.

    Returns:
        Tuple of (logits, labels), each a float/int numpy array of the
        same length as `loader`'s dataset.
    """
    logits, labels = [], []
    with torch.no_grad():
        for seq, img, lbl in loader:
            seq, img = seq.to(device), img.to(device)
            out = model(seq, img)
            logits.extend(out.float().cpu().squeeze(1).tolist())
            labels.extend(lbl.tolist())
    return np.array(logits), np.array(labels)


def report(name, logits, labels):
    """Prints accuracy/AUC/MCC for one set of logits at the 0.5 threshold.

    Args:
        name: Label printed for this row.
        logits: Raw model logits.
        labels: True binary labels, same length as `logits`.

    Returns:
        Tuple of (accuracy, auc, mcc) floats.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    auc = roc_auc_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    print(f"  {name:28s} acc {acc:.4f}   auc {auc:.4f}   mcc {mcc:+.4f}")
    return acc, auc, mcc


def main():
    """Loads two frozen single-branch checkpoints, stacks them with a
    logistic regression combiner fit on val logits, and reports test
    metrics for each branch alone, a naive average, and the stacked
    combiner."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = RamDualTensorDataset(f"{args.dataset_dir}/val")
    test_ds = RamDualTensorDataset(f"{args.dataset_dir}/test")
    seq_shape, img_shape = val_ds.sample_shapes()

    model_1d = load_frozen(args.ckpt_1d, seq_shape[-1], img_shape[0], args, "1d", device)
    model_2d = load_frozen(args.ckpt_2d, seq_shape[-1], img_shape[0], args, "2d", device)

    dl = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)

    print("Collecting frozen-model logits on val/test...")
    val_1d, val_y = collect_logits(model_1d, dl(val_ds), device)
    val_2d, _ = collect_logits(model_2d, dl(val_ds), device)
    test_1d, test_y = collect_logits(model_1d, dl(test_ds), device)
    test_2d, _ = collect_logits(model_2d, dl(test_ds), device)

    print(f"\n--- Reference points (test set, n={len(test_y)}) ---")
    report("1d-only alone", test_1d, test_y)
    report("2d-only alone", test_2d, test_y)
    report("naive average (logits)", (test_1d + test_2d) / 2.0, test_y)

    # Fit on VAL logits only -- never seen by either frozen model's training,
    # so this is a legitimate held-out fit for the combiner (the standard
    # stacked-generalization protocol), evaluated on the still-untouched test set.
    X_val = np.stack([val_1d, val_2d], axis=1)
    combiner = LogisticRegression()
    combiner.fit(X_val, val_y)
    w1, w2 = combiner.coef_[0]
    bias = combiner.intercept_[0]
    print(f"\n[stack] combiner (fit on val logits): "
          f"{w1:.3f}*logit_1d + {w2:.3f}*logit_2d + {bias:+.3f}")

    X_test = np.stack([test_1d, test_2d], axis=1)
    stack_probs = combiner.predict_proba(X_test)[:, 1]
    stack_preds = combiner.predict(X_test)
    acc = float((stack_preds == test_y).mean())
    auc = roc_auc_score(test_y, stack_probs)
    mcc = matthews_corrcoef(test_y, stack_preds)

    print("\n--- Stacked (logistic regression on frozen logits) ---")
    print(f"  accuracy {acc:.4f} | AUC {auc:.4f} | MCC {mcc:+.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(test_y, stack_preds))
    print("\n" + classification_report(test_y, stack_preds, digits=4))


if __name__ == "__main__":
    main()
