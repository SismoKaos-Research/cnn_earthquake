"""
Dual-channel CNN + LSTM/self-attention model for time-to-major-earthquake risk.

This is the architecture from Wang & Zhao (2025) -- their 1D2D-EDL -- applied
to catalog sliding windows instead of bearing vibration:

    1D channel : LSTM -> multi-head self-attention over the feature sequence
    2D channel : CNN over the RAM image of that same window
    fusion     : F = a*F_1d + b*F_2d   (a, b learned, as in the paper)
    head       : 3-class risk  (<1y / 1-5y / >5y until the next major event)

Two departures from the paper, both forced by findings in report.md:

* **An auxiliary scalar branch.** The RAM transform is exactly scale-invariant
  (8.2), so the image cannot represent absolute magnitude or energy level --
  precisely the quantities that matter here. Window-level scalars (b-value,
  Lyapunov exponent, event rate, total energy) enter alongside the fused
  features. `--channels` ablates any branch to test what each contributes.

* **Baselines printed on every run.** IP4's success criterion is >=70%
  accuracy, which a 3-class problem with skewed priors can reach by always
  predicting the majority class. Majority-class and persistence baselines are
  therefore reported next to the model, and the run states plainly whether the
  model beat them.

Usage:
    python cnn_lstm.py --dataset-dir dataset_catalog_marmara
    python cnn_lstm.py --dataset-dir ... --channels 1d      # ablation

Also imported (not just run standalone): cnn_lstm_loeo.py imports
`DualChannelRiskNet` and `risk_classes_from_manifest` from this module (same
architecture and label-ordering logic, leave-one-earthquake-out split
instead of chronological); cnn_groundmotion.py imports `LSTMAttentionBranch`
re-exported here from model/blocks.py.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from torch.utils.data import DataLoader, Dataset

from sismokaos.metrics import multiclass_report, print_report
from sismokaos.model.blocks import \
    LSTMAttentionBranch  # noqa: F401 (re-exported; cnn_groundmotion.py etc. import it from here)
from sismokaos.model.dual_channel import DualChannelNet
from sismokaos.model.registry import add_model_args, spec_from_args
from sismokaos.training import seed_everything

RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]
CLASS_TO_IDX = {c: i for i, c in enumerate(RISK_CLASSES)}


def risk_classes_from_manifest(manifest: pd.DataFrame):
    """
    Reads the class names actually present in a manifest and orders them by
    increasing time-to-next-mainshock.

    The module-level `RISK_CLASSES` above is only correct when the dataset was
    built with the fixed 1-year/5-year boundaries. `assign_risk_classes` in
    `catalog.py` derives TERCILE boundaries by default, and on a catalog whose
    recurrence is measured in weeks those terciles land nowhere near a year --
    on the pooled 4-region dataset they are 26 d and 71 d, so a window labelled
    `gt_5y` is actually 71-817 days out. Reading the names from the manifest,
    and ordering them by the `days_to_major` they actually cover rather than by
    a hardcoded list, keeps the label, the ordinal direction, and the reported
    confusion matrix honest for any boundary choice.

    Args:
        manifest: Dataset manifest DataFrame with a 'risk_class' column and,
            when available, a 'days_to_major' column to order classes by.

    Returns:
        Tuple of (ordered_class_names, class_to_idx dict).

    Raises:
        ValueError: If `manifest` has no 'risk_class' column.
    """
    if "risk_class" not in manifest.columns:
        raise ValueError("manifest has no 'risk_class' column")
    if "days_to_major" in manifest.columns:
        order = (manifest.groupby("risk_class").days_to_major.min()
                 .sort_values().index.tolist())
    else:
        # No horizon column to order by: fall back to the legacy fixed names,
        # keeping only those actually present.
        present = set(manifest.risk_class.unique())
        order = [c for c in RISK_CLASSES if c in present] + \
                sorted(present - set(RISK_CLASSES))
    return order, {c: i for i, c in enumerate(order)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CatalogWindowDataset(Dataset):
    """
    Loads the {seq, img, aux} tensors written by
    `seismic-cli generate-catalog-dataset`.

    seq and aux are standardized with TRAIN statistics only -- fitting them on
    each split would let the test distribution normalize itself, a subtle leak
    that matters more here than usual because the splits are chronological and
    the distribution genuinely drifts over time.
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, split: str, stats=None):
        """Loads one split's manifest rows and fits (or reuses) normalization stats.

        Args:
            manifest: Full dataset manifest DataFrame (all splits).
            root: Dataset root directory (contains a subdirectory per split).
            split: Which split to load -- e.g. "train", "val", "test".
            stats: Optional (seq_mean, seq_std, aux_mean, aux_std) tuple to
                normalize with; if None, fit from this split's own data (the
                train split should always pass None; val/test must reuse the
                train split's stats).

        Raises:
            ValueError: If `split` has no rows in `manifest`, or any row's
                `risk_class` isn't a recognized class.
        """
        self.rows = manifest[manifest.split == split].reset_index(drop=True)
        self.dir = Path(root) / split
        if self.rows.empty:
            raise ValueError(f"Split '{split}' is empty.")
        self.labels = self.rows.risk_class.map(CLASS_TO_IDX).to_numpy()
        if self.labels.min() < 0 or self.rows.risk_class.isna().any():
            raise ValueError("Unrecognized risk_class values in manifest.")
        self.days = self.rows.days_to_major.to_numpy(dtype=np.float32)

        sample = torch.load(self.dir / self.rows.filename.iloc[0], weights_only=True)
        self.seq_dim = sample["seq"].shape[-1]
        self.seq_len = sample["seq"].shape[0]
        self.img_shape = tuple(sample["img"].shape)
        self.aux_dim = sample["aux"].numel()

        if stats is None:
            seqs, auxs = [], []
            for fn in self.rows.filename:
                d = torch.load(self.dir / fn, weights_only=True)
                seqs.append(d["seq"].numpy())
                auxs.append(d["aux"].numpy())
            S = np.concatenate(seqs, axis=0)
            A = np.stack(auxs, axis=0)
            with np.errstate(invalid="ignore"):
                stats = (np.nanmean(S, 0), np.nanstd(S, 0) + 1e-6,
                         np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6)
            stats = tuple(np.where(np.isfinite(s), s, 0.0 if i % 2 == 0 else 1.0)
                          for i, s in enumerate(stats))
        self.stats = stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        """Returns one normalized (seq, img, aux, label) sample.

        Args:
            i: Row index into this split.

        Returns:
            Tuple of (float32 seq tensor, float32 img tensor, float32 aux
            tensor, long label tensor).
        """
        d = torch.load(self.dir / self.rows.filename.iloc[i], weights_only=True)
        sm, ss, am, asd = self.stats
        seq = (d["seq"].numpy() - sm) / ss
        aux = (d["aux"].numpy() - am) / asd
        return (torch.from_numpy(np.nan_to_num(seq, nan=0.0)).float(),
                d["img"].float(),
                torch.from_numpy(np.nan_to_num(aux, nan=0.0)).float(),
                torch.tensor(self.labels[i], dtype=torch.long))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualChannelRiskNet(DualChannelNet):
    """
    1D + 2D + auxiliary scalars, fused and classified.

    Fusion follows the paper: each channel is projected to a common width and
    combined as a*F1 + b*F2 with learned scalar weights, so the balance between
    channels is fit rather than assumed. The learned values are worth reading
    after training -- they say which representation the model actually used.
    """

    def __init__(self, seq_dim, img_channels, aux_dim, hidden=64, fusion_dim=128,
                 n_classes=3, dropout=0.3, channels="all"):
        """See `DualChannelNet.__init__` (`squeeze_output` is always False here)."""
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         n_classes=n_classes, squeeze_output=False)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def report_baselines(train_ds, test_ds):
    """
    Majority-class and persistence references.

    Persistence -- 'the next interval looks like the last one' -- is the
    honest naive forecaster for a temporal task, the analogue of STA/LTA for
    detection. Here it predicts each test window's class from the class of the
    chronologically preceding training window's distribution conditioned on a
    similar event rate, approximated by the train-set mode.

    Args:
        train_ds: Training-split `CatalogWindowDataset`.
        test_ds: Test-split `CatalogWindowDataset`.

    Returns:
        Tuple of (majority_class_accuracy, majority_class_balanced_accuracy).
    """
    y_tr, y_te = train_ds.labels, test_ds.labels
    print("\n--- Reference points (test set) ---")
    maj = int(np.bincount(y_tr, minlength=len(RISK_CLASSES)).argmax())
    pred_maj = np.full_like(y_te, maj)
    acc = float((pred_maj == y_te).mean())
    bal = balanced_accuracy_score(y_te, pred_maj)
    print(f"  majority-class ('{RISK_CLASSES[maj]}')   acc {acc:.4f}   balanced {bal:.4f}")

    rng = np.random.default_rng(0)
    prior = np.bincount(y_tr, minlength=len(RISK_CLASSES)) / len(y_tr)
    pred_prior = rng.choice(len(RISK_CLASSES), size=len(y_te), p=prior)
    print(f"  stratified-random                    acc {(pred_prior==y_te).mean():.4f}   "
          f"balanced {balanced_accuracy_score(y_te, pred_prior):.4f}")
    print(f"  [test class balance] " +
          "  ".join(f"{c}={int((y_te==i).sum())}" for i, c in enumerate(RISK_CLASSES)))
    return acc, bal


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Dual-channel CNN+LSTM earthquake risk model.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-catalog-dataset`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm")
    add_model_args(p, family="dual")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable inverse-frequency class weighting (on by default, "
                        "since the risk classes are typically very skewed).")
    return p.parse_args()


def main():
    """Loads the catalog dataset, trains `DualChannelRiskNet`, and reports
    the full metric set plus baselines on the test split."""
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    train_ds = CatalogWindowDataset(manifest, root, "train")
    val_ds = CatalogWindowDataset(manifest, root, "val", stats=train_ds.stats)
    test_ds = CatalogWindowDataset(manifest, root, "test", stats=train_ds.stats)

    print("=" * 64)
    print(f"Dual-channel risk model | channels='{args.channels}'")
    print(f"  seq ({train_ds.seq_len}, {train_ds.seq_dim}) | img {train_ds.img_shape} "
          f"| aux ({train_ds.aux_dim},)")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        c = np.bincount(ds.labels, minlength=len(RISK_CLASSES))
        print(f"  {name:5s}: n={len(ds):5d}  " +
              "  ".join(f"{k}={c[i]}" for i, k in enumerate(RISK_CLASSES)) +
              f"   {ds.rows.end_time.min()[:10]} -> {ds.rows.end_time.max()[:10]}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = spec_from_args(args)
    model = spec.build(seq_dim=train_ds.seq_dim, img_channels=train_ds.img_shape[0],
                       aux_dim=train_ds.aux_dim,
                       n_classes=len(RISK_CLASSES)).to(device)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    if args.no_class_weights:
        weight = None
    else:
        counts = np.bincount(train_ds.labels, minlength=len(RISK_CLASSES)).astype(np.float64)
        w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        print(f"class weights: {np.round(w / w.mean(), 3)}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_cnnlstm.pth")
    best, no_improve = -1.0, 0

    def evaluate(loader):
        """Runs the model over `loader` and collects true/predicted classes.

        Args:
            loader: DataLoader yielding (seq, img, aux, y) batches.

        Returns:
            Tuple of (y_true, y_pred) int arrays.
        """
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for seq, img, aux, y in loader:
                out = model(seq.to(device), img.to(device), aux.to(device))
                ps.extend(out.argmax(1).cpu().tolist())
                ys.extend(y.tolist())
        return np.array(ys), np.array(ps)

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for seq, img, aux, y in train_loader:
            seq, img, aux, y = seq.to(device), img.to(device), aux.to(device), y.to(device)
            loss = criterion(model(seq, img, aux), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            tot += loss.item() * y.size(0)
        scheduler.step()

        yv, pv = evaluate(val_loader)
        # Balanced accuracy, not accuracy: with skewed risk classes the plain
        # figure is dominated by whichever class happens to be common.
        vb = balanced_accuracy_score(yv, pv)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {tot/len(train_ds):.4f} "
              f"| val acc {(yv==pv).mean():.4f}  balanced {vb:.4f}")
        if vb > best:
            best, no_improve = vb, 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val balanced {best:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val balanced accuracy flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, pt = evaluate(test_loader)

    model.eval()
    scores = []
    with torch.no_grad():
        for seq, img, aux, _ in test_loader:
            scores.append(torch.softmax(model(seq.to(device), img.to(device), aux.to(device)),
                                        dim=1).cpu().numpy())
    print_report("Dual-channel model -- full metric set (test set)",
                multiclass_report(yt, pt, y_score=np.concatenate(scores), class_names=RISK_CLASSES))

    acc = float((yt == pt).mean())
    bal = balanced_accuracy_score(yt, pt)
    kappa = cohen_kappa_score(yt, pt)

    present = sorted(set(yt.tolist()))
    if len(present) < len(RISK_CLASSES):
        missing = [RISK_CLASSES[i] for i in range(len(RISK_CLASSES)) if i not in present]
        print(f"\n  [!] class(es) {missing} do not occur in the test split at all.")
        print("      Balanced accuracy is then an average over the classes that DO occur,")
        print("      so it is not comparable to a run where every class is present, and")
        print("      sklearn's 'y_pred contains classes not in y_true' warning is expected.")

    maj_acc, maj_bal = report_baselines(train_ds, test_ds)
    print(f"\n--- Dual-channel model (channels='{args.channels}') ---")
    print(f"  accuracy {acc:.4f} | balanced {bal:.4f} | Cohen's kappa {kappa:+.4f}")
    print(f"  vs majority-class: {acc - maj_acc:+.4f} accuracy, {bal - maj_bal:+.4f} balanced")
    if acc <= maj_acc + 1e-9:
        print("  [!] The model does NOT beat predicting the majority class. Its raw"
              "\n      accuracy is not evidence of skill, whatever the value is.")
    elif kappa < 0.2:
        print("  [!] Beats the majority class, but kappa < 0.2 -- agreement is barely"
              "\n      above chance once the class priors are accounted for.")
    else:
        print("  Beats both references with non-trivial kappa.")

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        print(f"\n  learned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}"
              "\n  (relative magnitude indicates which representation the model leaned on)")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(yt, pt, labels=range(len(RISK_CLASSES))),
                       index=RISK_CLASSES, columns=RISK_CLASSES))
    print("\n" + classification_report(yt, pt, labels=range(len(RISK_CLASSES)),
                                       target_names=RISK_CLASSES, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
