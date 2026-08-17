"""
CNN+LSTM 3-class monthly chunk classifier: does a M>=threshold AEGEAN event
occur INSIDE this chunk ("event"), shortly AFTER it ("event_after"), or
neither ("none")?

Distinct from every other script in this "current work" family in two ways:

1. Three classes, not binary. Every other forecast/proximity script here
   lumps "the window itself contains the event" and "the event is coming
   soon but isn't in the window yet" into one "positive" class -- but those
   are different questions with (plausibly) different waveform signatures:
   the first can be answered just by detecting the event's own seismic
   energy in-window (nothing to do with precursors), the second is the
   actually-interesting precursor question. Splitting them out means a
   model that's just re-detecting in-window events can't get credit for
   "forecasting skill" it doesn't have.

2. Per-DAY sequence steps, not per-hour. `raw_cnn_lstm_forecast.py`'s
   RawCNNLSTM embeds each HOUR via a CNN, then LSTMs over up to `--seq-hours`
   of those hourly embeddings -- a 720-step LSTM sequence for a month, which
   is both slow and a lot of redundant near-duplicate steps (adjacent hours
   of background noise look alike). This script's `DayCNNLSTM3Class` embeds
   each DAY (24h of raw waveform concatenated) via the same `RawWaveformEncoder`
   architecture, then LSTMs over ~30 daily embeddings for a month -- a much
   shorter, richer-per-step sequence, motivated by "does aggregating to
   daily resolution reveal something hourly/weekly granularity didn't."

Usage:
    python cnn_lstm_daily_3class.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --chunk-days 30 --horizon-days 30 --cv-folds 2

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from feature_lstm_forecast import load_aegean_events, walk_forward_splits
from seismolib.metrics import majority_class_baseline, multiclass_report, print_report
from seismolib.model.blocks import LSTMAttentionBranch
from raw_cnn_lstm_forecast import (RawWaveformEncoder, load_hourly_raw,
                                   load_hourly_raw_consolidated)
from seismolib.training import seed_everything

CLASS_NAMES = ["none", "event", "event_after"]
HOURS_PER_DAY = 24


def build_3class_daily_chunks(hour_index, major_times, chunk_days: int, horizon_days: float):
    """Divides the timeline into non-overlapping `chunk_days`-day blocks and
    3-way labels each.

    Args:
        hour_index: DatetimeIndex of hour starts, one per raw hour.
        major_times: Sorted array of qualifying event times.
        chunk_days: Number of days per chunk.
        horizon_days: How far past the chunk's end to look for the
            "event_after" class when the chunk itself contains no event.

    Returns:
        Tuple of (chunk_start_hour_indices int array, labels int64 array in
        {0=none, 1=event, 2=event_after}, chunk_hours int). A trailing
        partial chunk is dropped.
    """
    chunk_hours = chunk_days * HOURS_PER_DAY
    n = len(hour_index)
    n_chunks = n // chunk_hours
    starts = np.arange(n_chunks) * chunk_hours
    t = hour_index.to_numpy()
    horizon = np.timedelta64(int(horizon_days * 24), "h")
    labels = np.zeros(n_chunks, dtype=np.int64)
    for i, s in enumerate(starts):
        chunk_start_t, chunk_end_t = t[s], t[s + chunk_hours - 1]
        within = major_times[(major_times >= chunk_start_t) & (major_times <= chunk_end_t)]
        if len(within):
            labels[i] = 1  # event
        else:
            after = major_times[(major_times > chunk_end_t) & (major_times <= chunk_end_t + horizon)]
            labels[i] = 2 if len(after) else 0  # event_after / none
    return starts, labels, chunk_hours


class DailyChunkDataset(Dataset):
    """One sample per chunk -- a (chunk_days, 3, HOURS_PER_DAY*hour_samples)
    sequence, one step per day, each day's `HOURS_PER_DAY` hours of raw
    waveform concatenated along time. Per-channel z-normalized; val/test
    reuse the train set's stats."""

    def __init__(self, raw: np.ndarray, labels: np.ndarray, starts: np.ndarray,
                chunk_hours: int, indices: np.ndarray, stats=None):
        """Builds the dataset.

        Args:
            raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
            labels: Per-chunk 3-class labels, shape (n_chunks,).
            starts: Chunk start hour-indices, shape (n_chunks,).
            chunk_hours: Number of consecutive hours per chunk
                (chunk_days * HOURS_PER_DAY).
            indices: Chunk indices this split uses.
            stats: Optional (mean, std) tuple, each shape (3, 1), to
                normalize with; if None, computed from this split's first 5
                chunks.
        """
        self.raw = raw
        self.labels = labels
        self.starts = starts
        self.chunk_hours = chunk_hours
        self.n_days = chunk_hours // HOURS_PER_DAY
        self.indices = indices
        if stats is None:
            sub = np.concatenate([np.asarray(raw[starts[c]:starts[c] + chunk_hours])
                                  for c in indices[:5]], axis=0)
            mu = sub.mean(axis=(0, 2), keepdims=True)
            sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
            stats = (mu[0], sd[0])
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (daily_seq, label) for one chunk.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (chunk_days, 3,
            HOURS_PER_DAY*hour_samples), int64 scalar tensor class label).
        """
        c = self.indices[idx]
        s = self.starts[c]
        seq = np.asarray(self.raw[s:s + self.chunk_hours])  # (chunk_hours, 3, hour_samples)
        mu, sd = self.stats
        seq = (seq - mu[None]) / sd[None]
        hour_samples = seq.shape[-1]
        seq = seq.reshape(self.n_days, HOURS_PER_DAY, 3, hour_samples)
        x = np.ascontiguousarray(seq.transpose(0, 2, 1, 3).reshape(self.n_days, 3, HOURS_PER_DAY * hour_samples))
        return torch.from_numpy(x).float(), torch.tensor(self.labels[c], dtype=torch.long)


class DayCNNLSTM3Class(nn.Module):
    """`RawWaveformEncoder` (same CNN as the other raw-waveform scripts,
    here embedding one DAY at a time instead of one hour) -> LSTM+attention
    over the resulting daily-embedding sequence -> 3-class head."""

    def __init__(self, cnn_out=32, hidden=16, dropout=0.4, n_classes=3):
        """Initializes the day-encoder, LSTM/attention branch, and head.

        Args:
            cnn_out: Width of the CNN's per-day embedding.
            hidden: LSTM hidden size (per direction) and head hidden width.
            dropout: Dropout used throughout.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.encoder = RawWaveformEncoder(out_dim=cnn_out, dropout=dropout)
        self.branch = LSTMAttentionBranch(cnn_out, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.branch.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.branch.out_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, seq):
        """Classifies one batch of chunks.

        Args:
            seq: Input batch, shape (batch, chunk_days, 3, day_samples).

        Returns:
            Tensor of shape (batch, n_classes), raw logits.
        """
        b, t = seq.shape[:2]
        emb = self.encoder(seq.reshape(b * t, *seq.shape[2:])).reshape(b, t, -1)
        return self.head(self.branch(emb))


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--chunk-days", type=int, default=30)
    p.add_argument("--horizon-days", type=float, default=30.0,
                  help="How far past a chunk's end to look for the event_after class "
                       "when the chunk itself contains no event. Defaults to --chunk-days.")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--embargo-chunks", type=int, default=1)
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    return p.parse_args()


def train_one_seed(args, seed, raw, labels, starts, chunk_hours, train_idx, val_idx, test_idx, device):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args.
        seed: Random seed for init/shuffling.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-chunk 3-class labels, shape (n_chunks,).
        starts: Chunk start hour-indices, shape (n_chunks,).
        chunk_hours: Number of consecutive hours per chunk.
        train_idx: Chunk indices for the training split.
        val_idx: Chunk indices for the validation split.
        test_idx: Chunk indices for the test split.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_pred, y_prob) arrays for the test split, from
        the best (by val balanced accuracy) epoch's weights.
    """
    seed_everything(seed)
    train_ds = DailyChunkDataset(raw, labels, starts, chunk_hours, train_idx)
    val_ds = DailyChunkDataset(raw, labels, starts, chunk_hours, val_idx, stats=train_ds.stats)
    test_ds = DailyChunkDataset(raw, labels, starts, chunk_hours, test_idx, stats=train_ds.stats)

    model = DayCNNLSTM3Class(cnn_out=args.cnn_out, hidden=args.hidden, dropout=args.dropout).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    class_counts = np.bincount(labels[train_idx], minlength=3)
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, preds, probs, losses = [], [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                losses.append(criterion(logits, y).item() * y.size(0))
                p = torch.softmax(logits, dim=-1)
                preds.extend(p.argmax(-1).cpu().tolist())
                probs.extend(p.cpu().tolist())
                ys.extend(y.cpu().tolist())
        return (np.array(ys, dtype=np.int64), np.array(preds, dtype=np.int64),
               np.array(probs), sum(losses) / max(len(ys), 1))

    best = -1.0
    no_improve, best_state = 0, None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()

        yv, predv, _, val_loss = evaluate(val_loader)
        val_bal_acc = balanced_accuracy_score(yv, predv) if len(np.unique(yv)) > 1 else 0.0
        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val balanced_acc {val_bal_acc:.4f} "
             f"val loss {val_loss:.4f}")
        improved = val_bal_acc > best
        if improved:
            best, no_improve = val_bal_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, predt, probt, _ = evaluate(test_loader)
    test_bal_acc = balanced_accuracy_score(yt, predt) if len(np.unique(yt)) > 1 else float("nan")
    print(f"  [seed {seed}] test balanced_acc {test_bal_acc:.4f}")
    return yt, predt, probt


def run_fold(fold_label, args, raw, labels, starts, chunk_hours, train_idx, val_idx, test_idx,
            seeds, device):
    """Trains the seed ensemble on one split and reports it.

    Args:
        fold_label: Header string printed above this fold's report.
        args: Parsed CLI args, forwarded to `train_one_seed`.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-chunk 3-class labels, shape (n_chunks,).
        starts: Chunk start hour-indices, shape (n_chunks,).
        chunk_hours: Number of consecutive hours per chunk.
        train_idx: Chunk indices for the training split.
        val_idx: Chunk indices for the validation split.
        test_idx: Chunk indices for the test split.
        seeds: List of random seeds to train and ensemble.
        device: torch device to train on.

    Returns:
        Tuple of (ensemble_balanced_acc, floor_balanced_acc, report_dict),
        or None if the split is too thin (fewer than 6 train or 3 test
        chunks) to mean anything.
    """
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological, chunks): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            counts = np.bincount(labels[idx], minlength=3)
            print(f"    {name:5s}: " + " ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(3)))

    if len(train_idx) < 6 or len(test_idx) < 3:
        print("[ERROR] Not enough chunks for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_probs = []
    yt_ref = None
    for seed in seeds:
        yt, _, probt = train_one_seed(args, seed, raw, labels, starts, chunk_hours,
                                      train_idx, val_idx, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_probs.append(probt)

    ensemble_probs = np.mean(per_seed_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(-1)

    print("\n--- Floor (test set) ---")
    maj, floor_acc, floor_bal_acc = majority_class_baseline(labels[train_idx], yt_ref)
    print(f"  majority-class ({CLASS_NAMES[maj]})   accuracy {floor_acc:.4f}   "
         f"balanced_accuracy {floor_bal_acc:.4f}   n={len(yt_ref)}")

    print(f"\n--- Day-CNN+LSTM 3-class ---")
    ensemble_bal_acc = balanced_accuracy_score(yt_ref, ensemble_preds) if len(np.unique(yt_ref)) > 1 else float("nan")
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   "
         f"balanced_accuracy {ensemble_bal_acc:.4f}   n={len(yt_ref)}")

    report = multiclass_report(yt_ref, ensemble_preds, y_score=ensemble_probs, class_names=CLASS_NAMES)
    print_report(f"Day-CNN+LSTM 3-class ({fold_label}, test set)", report)
    return ensemble_bal_acc, floor_bal_acc, report


def main():
    """Loads the raw waveform archive/catalog, builds 3-class daily chunks,
    and runs the fold sweep."""
    args = parse_args()

    print("Loading raw preprocessed waveform and building 3-class daily chunks...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    starts, labels, chunk_hours = build_3class_daily_chunks(
        hour_index, major_times, args.chunk_days, args.horizon_days)
    n_chunks = len(starts)
    counts = np.bincount(labels, minlength=3)
    print(f"  {n_chunks} chunks of {args.chunk_days}d each -- "
         + " ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(3)))

    valid_indices = np.arange(n_chunks)

    if args.cv_folds <= 1:
        i_train = int(n_chunks * args.train_frac)
        i_val = int(n_chunks * (args.train_frac + args.val_frac))
        folds = [(valid_indices[:i_train],
                 valid_indices[i_train + args.embargo_chunks:i_val],
                 valid_indices[i_val + args.embargo_chunks:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_indices, args.cv_folds, embargo=args.embargo_chunks)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, raw, labels, starts, chunk_hours,
                          train_idx, val_idx, test_idx, seeds, device)
        if result is not None:
            results.append(result)

    if args.cv_folds > 1 and results:
        accs = np.array([r[0] for r in results])
        floors = np.array([r[1] for r in results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(results)}/{args.cv_folds} folds)\n{'=' * 64}")
        print(f"  ensemble balanced_acc per fold: {[f'{a:.4f}' for a in accs]}")
        print(f"  ensemble balanced_acc:  mean {np.nanmean(accs):.4f}  std {np.nanstd(accs):.4f}")
        print(f"  floor balanced_acc:     mean {floors.mean():.4f}  std {floors.std():.4f}")
        print(f"  beats its own fold's floor in {int((accs > floors).sum())}/{len(results)} folds")


if __name__ == "__main__":
    main()
