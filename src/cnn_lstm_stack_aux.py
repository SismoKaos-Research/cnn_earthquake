"""
Late-fusion stacking on FROZEN amplitude-augmented single-branch checkpoints
(`--channels 1d+aux` / `--channels 2d+aux`).

Mirrors `cnn_lstm_stack.py` exactly (see that file's docstring for the full
rationale on why stacking is being tried at all: the paper's fixed a*F1+b*F2
fusion has repeatedly underperformed the best single branch on this task).
Kept as a separate file rather than extending `cnn_lstm_stack.py` in place,
matching the `cnn_lstm_classify.py` -> `cnn_lstm_classify_aux.py` pairing
convention already used elsewhere in this project: `cnn_lstm_stack.py`'s
already-reported numbers (report.md 10.4) stay untouched by this addition.

This tests whether stacking the amplitude-augmented branches (report.md
10.5.8: `1d+aux` on RAM data reaches 0.9501 AUC alone, close to the full
fused model's 0.9514) does any better than either alone, the same question
`cnn_lstm_stack.py` asked of the plain (non-aux) branches.

Usage:
    python cnn_lstm_stack_aux.py --dataset-dir dataset_specdualaux_6s \\
        --ckpt-1d .../specdualaux_1daux/best_cnnlstm_aux.pth \\
        --ckpt-2d .../specdualaux_2daux/best_cnnlstm_aux.pth

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader

from cnn_lstm_classify_aux import DualChannelAuxBinaryNet, RamDualAuxTensorDataset


def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Stack frozen 1d+aux and 2d+aux checkpoints.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt-1d", required=True, help="Checkpoint from `--channels 1d+aux` training.")
    p.add_argument("--ckpt-2d", required=True, help="Checkpoint from `--channels 2d+aux` training.")
    p.add_argument("--channels-1d", default="1d+aux", choices=["1d", "1d+aux"])
    p.add_argument("--channels-2d", default="2d+aux", choices=["2d", "2d+aux"])
    p.add_argument("--hidden", type=int, default=48, help="Must match the checkpoints' training run.")
    p.add_argument("--fusion-dim", type=int, default=96, help="Must match the checkpoints' training run.")
    p.add_argument("--lstm-layers", type=int, default=1, help="Must match the 1d+aux checkpoint's training run.")
    p.add_argument("--lstm-heads", type=int, default=4, help="Must match the 1d+aux checkpoint's training run.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def load_frozen(ckpt_path, seq_dim, img_channels, aux_dim, args, channels, device):
    """Rebuilds a `DualChannelAuxBinaryNet` and loads a frozen checkpoint into it.

    Args:
        ckpt_path: Path to a state-dict checkpoint from
            `cnn_lstm_classify_aux.py` training (e.g. `--channels 1d+aux`
            or `--channels 2d+aux`).
        seq_dim: Per-step feature width of the 1D sequence input.
        img_channels: Number of channels of the 2D image input.
        aux_dim: Width of the auxiliary scalar vector.
        args: Parsed CLI args (uses hidden, fusion_dim, lstm_layers,
            lstm_heads -- must match the checkpoint's training run).
        channels: Which branches the checkpoint was trained with (e.g.
            "1d+aux" or "2d+aux").
        device: torch device to load the model onto.

    Returns:
        The reconstructed model in eval mode with every parameter's
        `requires_grad` set to False.
    """
    model = DualChannelAuxBinaryNet(seq_dim, img_channels, aux_dim, hidden=args.hidden,
                                    fusion_dim=args.fusion_dim, dropout=0.0,
                                    channels=channels, lstm_layers=args.lstm_layers,
                                    lstm_heads=args.lstm_heads).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def collect_logits(model, loader, device):
    """Runs a frozen model over `loader` and collects its raw logits and true labels.

    Args:
        model: Frozen model in eval mode, called as `model(seq, img, aux)`.
        loader: DataLoader yielding (seq, img, aux, label) batches.
        device: torch device to run inference on.

    Returns:
        Tuple of (logits, labels), each a float/int numpy array of the
        same length as `loader`'s dataset.
    """
    logits, labels = [], []
    with torch.no_grad():
        for seq, img, aux, lbl in loader:
            seq, img, aux = seq.to(device), img.to(device), aux.to(device)
            out = model(seq, img, aux)
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
    """Loads two frozen aux-branch checkpoints, stacks them with a logistic
    regression combiner fit on val logits, and reports test metrics for
    each branch alone, a naive average, and the stacked combiner."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Aux is standardized with TRAIN-only stats (RamDualAuxTensorDataset's
    # convention, matching cnn_lstm_classify_aux.py's own main()) -- the
    # frozen checkpoints were trained against those exact stats, so val/test
    # must reuse them rather than (as a naive stacking-only script might do,
    # since it never otherwise needs a train split) computing fresh stats
    # from val. Using the wrong stats would silently shift every aux value
    # the frozen models see and degrade their logits without any error.
    train_ds = RamDualAuxTensorDataset(f"{args.dataset_dir}/train")
    val_ds = RamDualAuxTensorDataset(f"{args.dataset_dir}/val", aux_stats=train_ds.aux_stats)
    test_ds = RamDualAuxTensorDataset(f"{args.dataset_dir}/test", aux_stats=train_ds.aux_stats)
    seq_shape, img_shape, aux_shape = train_ds.sample_shapes()

    model_1d = load_frozen(args.ckpt_1d, seq_shape[-1], img_shape[0], aux_shape[-1],
                           args, args.channels_1d, device)
    model_2d = load_frozen(args.ckpt_2d, seq_shape[-1], img_shape[0], aux_shape[-1],
                           args, args.channels_2d, device)

    dl = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)

    print("Collecting frozen-model logits on val/test...")
    val_1d, val_y = collect_logits(model_1d, dl(val_ds), device)
    val_2d, _ = collect_logits(model_2d, dl(val_ds), device)
    test_1d, test_y = collect_logits(model_1d, dl(test_ds), device)
    test_2d, _ = collect_logits(model_2d, dl(test_ds), device)

    print(f"\n--- Reference points (test set, n={len(test_y)}) ---")
    report(f"{args.channels_1d} alone", test_1d, test_y)
    report(f"{args.channels_2d} alone", test_2d, test_y)
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
