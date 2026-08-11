"""
Leave-one-event-out (LOEO) cross-validation for the dual-channel risk model.

With `generate-catalog-dataset --split-mode loeo`, every window is written
into a single flat "all" split instead of one chronological train/val/test
cut. This script forms the folds itself: for each distinct target event (one
fold per (region, target_time) pair), all of that event's windows become the
test set and every other window becomes the training pool, with a small
stratified slice of the pool held out as an inner validation set for early
stopping.

Why LOEO instead of a single chronological split: even after pooling several
fault zones (`--region`, repeatable, in `generate-catalog-dataset`), a region
can still have too few independent target events for ONE 70/15/15 time cut to
give a stable test set -- whichever handful of events happen to land after the
cut decide the whole result. LOEO lets every event serve as the test set once,
so the reported metrics summarize performance over ALL of them rather than
whichever few were unlucky enough to be "test" under one arbitrary cut.

The trade-off, stated plainly: LOEO trains on windows from events that occur
AFTER the held-out event as well as before it, so it evaluates whether the
representation generalizes across different mainshocks, not whether a
deployed model could have used only past data to predict a held-out event in
real time. It is a model/feature-quality check, not a backtest. The single
chronological split in `cnn_lstm.py` remains the honest backtest for "what
would deployment have looked like."

Usage:
    python cnn_lstm_loeo.py --dataset-dir dataset_catalog_pooled

Not imported by anything else -- standalone script.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cnn_lstm import DualChannelRiskNet, risk_classes_from_manifest
from training import seed_everything

# Populated from the manifest in main(); see `risk_classes_from_manifest`.
# Hardcoding these was a real defect: the tercile boundaries `catalog.py`
# derives by default put `gt_5y` at 71-817 DAYS on the pooled dataset, so
# every printed class name was wrong by more than an order of magnitude.
RISK_CLASSES: list = []
CLASS_TO_IDX: dict = {}


class InMemoryWindowDataset(Dataset):
    """
    Indexes into arrays already loaded once for the whole pooled dataset
    (`preload_all`), standardized with the stats passed in.

    Re-reading every window's .pt file from disk for every one of ~200 LOEO
    folds (each needing its own train/val/test standardization pass) is the
    dominant cost at this dataset's size (~80 MB total) -- loading it all
    into RAM once and slicing by position removes nearly all of that I/O.
    """

    def __init__(self, seq: np.ndarray, img: np.ndarray, aux: np.ndarray,
                labels: np.ndarray, idx: np.ndarray, stats):
        """Slices and standardizes one fold's split from the preloaded arrays.

        Args:
            seq: Full pooled seq array, shape (n_total, seq_len, seq_dim)
                (see `preload_all`).
            img: Full pooled img array, shape (n_total, img_channels,
                height, width).
            aux: Full pooled aux array, shape (n_total, aux_dim).
            labels: Full pooled int label array, shape (n_total,).
            idx: Row indices, into the arrays above, making up this split.
            stats: (seq_mean, seq_std, aux_mean, aux_std) tuple to
                standardize seq/aux with (see `fit_stats`).
        """
        sm, ss, am, asd = stats
        self.seq = np.nan_to_num((seq[idx] - sm) / ss, nan=0.0).astype(np.float32)
        self.img = img[idx]
        self.aux = np.nan_to_num((aux[idx] - am) / asd, nan=0.0).astype(np.float32)
        self.labels = labels[idx]

    def __len__(self):
        """Returns the number of rows in this split."""
        return len(self.labels)

    def __getitem__(self, i):
        """Returns one (seq, img, aux, label) sample.

        Args:
            i: Index into this split.

        Returns:
            Tuple of (float32 seq tensor, float32 img tensor, float32 aux
            tensor, long label tensor).
        """
        return (torch.from_numpy(self.seq[i]), torch.from_numpy(self.img[i]),
                torch.from_numpy(self.aux[i]), torch.tensor(self.labels[i], dtype=torch.long))


def preload_all(root: Path, manifest: pd.DataFrame):
    """Loads every window's {seq, img, aux} tensors into pooled arrays once.

    Args:
        root: Directory containing the window .pt files named in
            `manifest.filename` (the dataset's "all" subdirectory).
        manifest: Full pooled dataset manifest, with 'filename' and
            'risk_class' columns.

    Returns:
        Tuple of (seq, img, aux, labels): `seq` float32 array shape
        (n, seq_len, seq_dim), `img` float32 array shape (n, img_channels,
        height, width), `aux` float32 array shape (n, aux_dim), `labels`
        int array shape (n,) via the module-level `CLASS_TO_IDX`.

    Raises:
        SystemExit: If any row's `risk_class` isn't a recognized class
            (i.e. not in `CLASS_TO_IDX`).
    """
    seqs, imgs, auxs = [], [], []
    for fn in manifest.filename:
        d = torch.load(root / fn, weights_only=True)
        seqs.append(d["seq"].numpy())
        imgs.append(d["img"].numpy())
        auxs.append(d["aux"].numpy())
    seq = np.stack(seqs, axis=0).astype(np.float32)
    img = np.stack(imgs, axis=0).astype(np.float32)
    aux = np.stack(auxs, axis=0).astype(np.float32)
    labels = manifest.risk_class.map(CLASS_TO_IDX).to_numpy()
    if labels.min() < 0 or manifest.risk_class.isna().any():
        raise SystemExit("Unrecognized risk_class values in manifest.")
    return seq, img, aux, labels


def fit_stats(seq: np.ndarray, aux: np.ndarray, idx: np.ndarray):
    """Fits NaN-safe per-feature (mean, std) normalization stats over a subset.

    Args:
        seq: Full pooled seq array, shape (n_total, seq_len, seq_dim).
        aux: Full pooled aux array, shape (n_total, aux_dim).
        idx: Row indices, into `seq`/`aux`, to fit stats from (typically a
            fold's inner-training rows).

    Returns:
        Tuple of (seq_mean, seq_std, aux_mean, aux_std) arrays, NaN/zero-std
        entries replaced with 0.0 (mean) or 1.0 (std).
    """
    S = seq[idx].reshape(-1, seq.shape[-1])
    A = aux[idx]
    with np.errstate(invalid="ignore"):
        stats = (np.nanmean(S, 0), np.nanstd(S, 0) + 1e-6,
                 np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6)
    return tuple(np.where(np.isfinite(s), s, 0.0 if i % 2 == 0 else 1.0)
                for i, s in enumerate(stats))


def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Leave-one-event-out CV for the dual-channel risk model.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `generate-catalog-dataset --split-mode loeo` "
                        "(the one containing an 'all' subfolder).")
    p.add_argument("--channels", default="all",
                   choices=["all", "1d", "2d", "aux", "1d+aux", "2d+aux"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--inner-val-frac", type=float, default=0.15,
                   help="Fraction of each fold's training pool held out for early stopping.")
    p.add_argument("--min-fold-test", type=int, default=1,
                   help="Skip target events with fewer than this many windows.")
    p.add_argument("--max-folds", type=int, default=None,
                   help="Cap the number of folds run -- useful for a quick smoke test.")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-epoch validation balanced accuracy for every fold.")
    return p.parse_args()


def run_fold(seq: np.ndarray, img: np.ndarray, aux: np.ndarray, labels_all: np.ndarray,
            test_idx: np.ndarray, train_pool_idx: np.ndarray, args, device, fold_seed: int):
    """Trains and evaluates one LOEO fold: one held-out event as test,
    everything else as the training pool (with a stratified inner-val slice
    for early stopping).

    Args:
        seq: Full pooled seq array, shape (n_total, seq_len, seq_dim).
        img: Full pooled img array, shape (n_total, img_channels, height,
            width).
        aux: Full pooled aux array, shape (n_total, aux_dim).
        labels_all: Full pooled int label array, shape (n_total,).
        test_idx: Row indices making up this fold's held-out event.
        train_pool_idx: Row indices making up this fold's training pool
            (every window not in `test_idx`).
        args: Parsed CLI args (uses inner_val_frac, hidden, fusion_dim,
            dropout, channels, batch_size, num_workers, no_class_weights,
            lr, weight_decay, epochs, patience, verbose).
        device: torch device to train on.
        fold_seed: Random seed for this fold's inner split, init, and
            shuffling.

    Returns:
        Tuple of (y_true, y_pred, majority_class_pred) arrays for the test
        split, from the best (by inner-val balanced accuracy) epoch's
        weights.
    """
    seed_everything(fold_seed)

    pool_labels = labels_all[train_pool_idx]
    counts = np.bincount(pool_labels, minlength=len(RISK_CLASSES))
    strat = pool_labels if counts.min() >= 2 else None
    if len(train_pool_idx) < 10:
        inner_train_idx, inner_val_idx = train_pool_idx, train_pool_idx
    else:
        inner_train_idx, inner_val_idx = train_test_split(
            train_pool_idx, test_size=args.inner_val_frac, random_state=fold_seed, stratify=strat)

    stats = fit_stats(seq, aux, inner_train_idx)
    train_ds = InMemoryWindowDataset(seq, img, aux, labels_all, inner_train_idx, stats)
    val_ds = InMemoryWindowDataset(seq, img, aux, labels_all, inner_val_idx, stats)
    test_ds = InMemoryWindowDataset(seq, img, aux, labels_all, test_idx, stats)

    model = DualChannelRiskNet(seq.shape[-1], img.shape[1], aux.shape[-1],
                               hidden=args.hidden, fusion_dim=args.fusion_dim,
                               dropout=args.dropout, channels=args.channels).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    if args.no_class_weights:
        weight = None
    else:
        c = np.bincount(train_ds.labels, minlength=len(RISK_CLASSES)).astype(np.float64)
        w = np.where(c > 0, c.sum() / np.maximum(c, 1), 0.0)
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    best_state, best_bal, no_improve = None, -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        for seq, img, aux, y in train_loader:
            seq, img, aux, y = seq.to(device), img.to(device), aux.to(device), y.to(device)
            loss = criterion(model(seq, img, aux), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()

        yv, pv = evaluate(val_loader)
        vb = balanced_accuracy_score(yv, pv) if len(set(yv.tolist())) > 1 else float((yv == pv).mean())
        if args.verbose:
            print(f"    epoch {epoch+1}/{args.epochs} val_balanced {vb:.4f}")
        if vb > best_bal:
            best_bal, no_improve = vb, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, pt = evaluate(test_loader)

    # Majority-class reference from THIS fold's own training pool -- the same
    # baseline cnn_lstm.py reports, computed per fold so it is exposed to the
    # same train/test class-balance shift as the model is, every fold.
    maj = int(np.bincount(train_ds.labels, minlength=len(RISK_CLASSES)).argmax())
    maj_pred = np.full_like(yt, maj)
    return yt, pt, maj_pred


def main():
    """Forms one LOEO fold per (region, target_time) event, trains and
    evaluates each, and reports the pooled result against chance and the
    per-fold majority-class floor.

    Side effect: rebinds the module-level `RISK_CLASSES`/`CLASS_TO_IDX` from
    the manifest (via `risk_classes_from_manifest`), since the class names
    and their time-ordering depend on how the dataset's boundaries were
    generated.

    Raises:
        SystemExit: If the manifest has no region/target_time columns, if
            any row's `risk_class` isn't recognized (via `preload_all`), or
            if fewer than 3 usable folds result from `--min-fold-test`/
            `--max-folds`.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset_dir)
    manifest = pd.read_csv(dataset_dir / "manifest.csv")
    root = dataset_dir / "all"       # window tensors live here; manifest.csv is one level up
    if "region" not in manifest.columns or "target_time" not in manifest.columns:
        raise SystemExit("This manifest has no region/target_time columns -- regenerate with "
                          "a current `generate-catalog-dataset --split-mode loeo` run.")
    manifest["target_time"] = pd.to_datetime(manifest.target_time)

    global RISK_CLASSES, CLASS_TO_IDX
    RISK_CLASSES, CLASS_TO_IDX = risk_classes_from_manifest(manifest)
    if "days_to_major" in manifest.columns:
        spans = manifest.groupby("risk_class").days_to_major.agg(["min", "max"])
        print("[classes] ordered by actual time-to-next-mainshock:")
        for c in RISK_CLASSES:
            print(f"            {c:12s} {spans.loc[c, 'min']:8.1f} - {spans.loc[c, 'max']:8.1f} days")

    groups = [(k, g) for k, g in manifest.groupby(["region", "target_time"])
             if len(g) >= args.min_fold_test]
    if args.max_folds:
        groups = groups[:args.max_folds]
    if len(groups) < 3:
        raise SystemExit(f"Only {len(groups)} usable fold(s) -- not enough for a meaningful CV. "
                         f"Pool more regions or lower --min-fold-test.")

    print(f"\n[preload] reading {len(manifest)} window tensors into memory once "
          f"(shared across all {len(groups)} folds)...")
    seq, img, aux, labels_all = preload_all(root, manifest)
    print(f"[preload] seq {seq.shape}  img {img.shape}  aux {aux.shape}")

    print("=" * 64)
    print(f"Leave-one-event-out CV | channels='{args.channels}' | {len(groups)} folds")
    print(f"Device: {device}")
    print("=" * 64)

    all_pos = np.arange(len(manifest))
    all_yt, all_pt, all_maj = [], [], []
    fold_rows = []
    for i, ((region, target_time), test_rows) in enumerate(groups):
        test_idx = test_rows.index.to_numpy()
        train_pool_idx = np.setdiff1d(all_pos, test_idx, assume_unique=True)
        yt, pt, maj_pred = run_fold(seq, img, aux, labels_all, test_idx, train_pool_idx,
                                    args, device, fold_seed=args.seed + i)
        all_yt.append(yt); all_pt.append(pt); all_maj.append(maj_pred)
        acc = float((yt == pt).mean())
        maj_acc = float((yt == maj_pred).mean())
        true_classes = ",".join(sorted({RISK_CLASSES[c] for c in yt}))
        print(f"[{i+1:3d}/{len(groups)}] {region:10s} {pd.Timestamp(target_time).date()}  "
              f"n_test={len(test_rows):3d}  true={true_classes:20s}  "
              f"acc={acc:.2f}  (fold majority-baseline {maj_acc:.2f})")
        fold_rows.append((region, target_time, len(test_rows), acc, maj_acc))

    yt = np.concatenate(all_yt)
    pt = np.concatenate(all_pt)
    maj = np.concatenate(all_maj)

    acc = float((yt == pt).mean())
    bal = balanced_accuracy_score(yt, pt)
    kappa = cohen_kappa_score(yt, pt)
    maj_acc = float((yt == maj).mean())
    maj_bal = balanced_accuracy_score(yt, maj)

    chance = 1.0 / len(RISK_CLASSES)

    print("\n" + "=" * 64)
    print(f"POOLED RESULT across {len(groups)} folds / {len(yt)} held-out test windows")
    print("=" * 64)
    print(f"  chance ({len(RISK_CLASSES)} classes)                       acc {chance:.4f}"
          f"   <- THE floor that matters")
    print(f"  majority-class (per-fold train mode)  acc {maj_acc:.4f}   balanced {maj_bal:.4f}")
    print(f"  dual-channel model                    acc {acc:.4f}   balanced {bal:.4f}   "
          f"kappa {kappa:+.4f}")
    print(f"  vs chance:         {acc - chance:+.4f} accuracy")
    print(f"  vs majority-class: {acc - maj_acc:+.4f} accuracy, {bal - maj_bal:+.4f} balanced")

    # Do NOT quote the majority-class figure as the floor when the dataset is
    # class-balanced. Under leave-one-event-out, removing a fold tips the
    # remaining counts away from that fold's own dominant class, so the
    # per-fold "majority" is systematically the class the fold has LEAST of --
    # measured at 2/264 folds matching the fold's true mode and 175/264
    # matching its rarest class on the pooled 4-region dataset. Beating an
    # anti-predictive baseline is evidence of nothing.
    if maj_acc < chance - 0.05:
        print(f"\n  [!] The majority-class figure ({maj_acc:.4f}) is far BELOW chance "
              f"({chance:.4f}) -- an artifact of balanced classes under LOEO, not a")
        print("      baseline worth beating. Compare against chance instead.")
    if acc <= chance + 1e-9:
        print("  [!] Does NOT beat chance. The model has learned nothing usable.")
    elif kappa < 0.2:
        print("  [!] Beats chance, but kappa < 0.2 -- agreement is barely above chance.")
    else:
        print("  Beats chance with non-trivial kappa.")

    print("\nConfusion matrix (rows = true, cols = predicted), pooled over all folds:")
    print(pd.DataFrame(confusion_matrix(yt, pt, labels=range(len(RISK_CLASSES))),
                       index=RISK_CLASSES, columns=RISK_CLASSES))
    print("\n" + classification_report(yt, pt, labels=range(len(RISK_CLASSES)),
                                       target_names=RISK_CLASSES, digits=4, zero_division=0))

    fdf = pd.DataFrame(fold_rows, columns=["region", "target_time", "n_test", "acc", "majority_acc"])
    print(f"\nPer-fold accuracy: mean {fdf.acc.mean():.3f}  std {fdf.acc.std():.3f}  "
          f"(vs per-fold majority-baseline mean {fdf.majority_acc.mean():.3f})")
    print("\nPer-region breakdown:")
    for region, g in fdf.groupby("region"):
        print(f"  {region:10s} folds={len(g):3d}  mean acc {g.acc.mean():.3f}  "
              f"(majority-baseline {g.majority_acc.mean():.3f})")


if __name__ == "__main__":
    main()
