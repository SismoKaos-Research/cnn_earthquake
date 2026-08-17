"""
CNN-only classifier over non-overlapping multi-day chunks: does an
M>=threshold AEGEAN event occur in the `--horizon-days` right after this
chunk ends?

Sibling to cnn_proximity_classify.py, same CNN-only philosophy (no LSTM, no
multi-hour sequence bridging via recurrence), but a different way of using
the "no LSTM means no window-overlap constraint" freedom: instead of one
sample per hour, this chunks the timeline into non-overlapping
`--chunk-hours` blocks (default 168h = 1 week) and gives the CNN the WHOLE
chunk's raw waveform concatenated along time as a single input, asking a
single forward-looking question per chunk rather than a dense per-hour
label. Fewer, much richer samples (~100 week-chunks across the archive vs.
~17000 hours) -- trades sample count for per-sample context length, the
opposite trade the sequence-window scripts make.

This is a real memory/compute departure from every other script here: one
week at 5Hz is 168*18000 = 3,024,000 samples/channel in a single training
example, run through the same RawWaveformEncoder (4 stride-4 Conv1d blocks,
ending in AdaptiveAvgPool1d(1) so it accepts any length) the other raw-
waveform scripts use for single hours. Expect to need a very small
--batch-size; a one-batch memory probe is recommended before a real run --
see this repo's session notes / ask before assuming a given --batch-size
fits on a given GPU.

Usage:
    python cnn_chunk_forecast.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --chunk-hours 168 --horizon-days 7 --batch-size 2 --cv-folds 2

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader, Dataset

from cnn_proximity_classify import CNNProximityClassifier
from feature_lstm_forecast import load_aegean_events, safe_auc, walk_forward_splits
from seismolib.metrics import binary_report, print_report
from raw_cnn_lstm_forecast import load_hourly_raw, load_hourly_raw_consolidated
from seismolib.training import seed_everything


def build_chunks(hour_index, major_times, chunk_hours: int, horizon_days: float):
    """Divides the hourly timeline into non-overlapping chunks and labels each.

    Args:
        hour_index: DatetimeIndex of hour starts, one per raw hour.
        major_times: Sorted array of qualifying event times.
        chunk_hours: Number of consecutive hours per chunk.
        horizon_days: An event in `(chunk_end, chunk_end + horizon_days]`
            labels that chunk positive.

    Returns:
        Tuple of (chunk_start_hour_indices int array, labels int64 array),
        one entry per complete chunk (a trailing partial chunk is dropped).
    """
    n = len(hour_index)
    n_chunks = n // chunk_hours
    starts = np.arange(n_chunks) * chunk_hours
    t = hour_index.to_numpy()
    horizon = np.timedelta64(int(horizon_days * 24), "h")
    labels = np.zeros(n_chunks, dtype=np.int64)
    for i, s in enumerate(starts):
        chunk_end_t = t[s + chunk_hours - 1]
        fut = major_times[(major_times > chunk_end_t) & (major_times <= chunk_end_t + horizon)]
        labels[i] = int(len(fut) > 0)
    return starts, labels


class ChunkRawDataset(Dataset):
    """One sample per chunk -- that chunk's `chunk_hours` of raw waveform,
    concatenated along time into one (3, chunk_hours*hour_samples) input.
    Per-channel z-normalized; val/test reuse the train set's stats."""

    def __init__(self, raw: np.ndarray, labels: np.ndarray, starts: np.ndarray,
                chunk_hours: int, indices: np.ndarray, stats=None):
        """Builds the dataset.

        Args:
            raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
            labels: Per-chunk binary labels, shape (n_chunks,).
            starts: Chunk start hour-indices, shape (n_chunks,).
            chunk_hours: Number of consecutive hours per chunk.
            indices: Chunk indices (into `starts`/`labels`) this split uses.
            stats: Optional (mean, std) tuple, each shape (3, 1), to
                normalize with; if None, computed from this split's first
                10 chunks (each chunk is already `chunk_hours` hours wide,
                so 10 chunks is already a substantial sample).
        """
        self.raw = raw
        self.labels = labels
        self.starts = starts
        self.chunk_hours = chunk_hours
        self.indices = indices
        if stats is None:
            sub = np.concatenate([np.asarray(raw[starts[c]:starts[c] + chunk_hours])
                                  for c in indices[:10]], axis=0)
            mu = sub.mean(axis=(0, 2), keepdims=True)
            sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
            stats = (mu[0], sd[0])
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (waveform, label) for one chunk.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (3, chunk_hours*hour_samples),
            float32 scalar tensor label).
        """
        c = self.indices[idx]
        s = self.starts[c]
        seq = np.asarray(self.raw[s:s + self.chunk_hours])  # (chunk_hours, 3, hour_samples)
        mu, sd = self.stats
        seq = (seq - mu[None]) / sd[None]
        # (chunk_hours, 3, hour_samples) -> (3, chunk_hours*hour_samples), time-concatenated
        x = np.ascontiguousarray(seq.transpose(1, 0, 2).reshape(3, -1))
        return torch.from_numpy(x).float(), torch.tensor(self.labels[c], dtype=torch.float32)


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--chunk-hours", type=int, default=168, help="168 = 1 week.")
    p.add_argument("--horizon-days", type=float, default=7.0,
                  help="An event in (chunk_end, chunk_end+horizon_days] labels the chunk "
                       "positive. Defaults to matching --chunk-hours (i.e. 'does the "
                       "immediately-following chunk contain an event').")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2,
                  help="Small by default -- one chunk is chunk_hours*hour_samples samples "
                       "per channel (3M+ for a 5Hz week), far larger than any single-hour "
                       "input elsewhere in this project. Probe memory before raising this.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--embargo-chunks", type=int, default=1,
                  help="Chunks of gap enforced between chronological CV blocks -- defensive "
                       "against adjacent-chunk boundary autocorrelation.")
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    p.add_argument("--balanced-sampling", action="store_true")
    return p.parse_args()


def train_one_seed(args, seed, raw, labels, starts, train_idx, val_idx, test_idx, device):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args.
        seed: Random seed for init/shuffling.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-chunk binary labels, shape (n_chunks,).
        starts: Chunk start hour-indices, shape (n_chunks,).
        train_idx: Chunk indices for the training split.
        val_idx: Chunk indices for the validation split.
        test_idx: Chunk indices for the test split.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_score) arrays for the test split, from the
        best (by val AUC) epoch's weights.
    """
    seed_everything(seed)
    train_ds = ChunkRawDataset(raw, labels, starts, args.chunk_hours, train_idx)
    val_ds = ChunkRawDataset(raw, labels, starts, args.chunk_hours, val_idx, stats=train_ds.stats)
    test_ds = ChunkRawDataset(raw, labels, starts, args.chunk_hours, test_idx, stats=train_ds.stats)

    model = CNNProximityClassifier(cnn_out=args.cnn_out, dropout=args.dropout).to(device)

    if args.balanced_sampling:
        train_labels_arr = labels[train_idx]
        class_counts = np.bincount(train_labels_arr, minlength=2)
        sample_weights = (1.0 / np.maximum(class_counts, 1))[train_labels_arr]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.balanced_sampling:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos = labels[train_idx].mean()
        pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    yv0, _, _ = evaluate(val_loader)
    use_loss_fallback = len(np.unique(yv0)) < 2
    if use_loss_fallback:
        print(f"  [seed {seed}] val split is single-class -- checkpointing on train AUC instead.")

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

        yv, sv, val_loss = evaluate(val_loader)
        val_auc = safe_auc(yv, sv)
        if use_loss_fallback:
            ytr, tr_scores, _ = evaluate(train_loader)
            metric = safe_auc(ytr, tr_scores)
        else:
            metric = val_auc
        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val AUC {val_auc:.4f} val loss {val_loss:.4f}"
             + (f" train AUC {metric:.4f}" if use_loss_fallback else ""))
        improved = metric > best
        if improved:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = evaluate(test_loader)
    print(f"  [seed {seed}] test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def run_fold(fold_label, args, raw, labels, starts, train_idx, val_idx, test_idx, seeds, device):
    """Trains the seed ensemble on one split and reports it.

    Args:
        fold_label: Header string printed above this fold's report.
        args: Parsed CLI args, forwarded to `train_one_seed`.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-chunk binary labels, shape (n_chunks,).
        starts: Chunk start hour-indices, shape (n_chunks,).
        train_idx: Chunk indices for the training split.
        val_idx: Chunk indices for the validation split.
        test_idx: Chunk indices for the test split.
        seeds: List of random seeds to train and ensemble.
        device: torch device to train on.

    Returns:
        Tuple of (ensemble_auc, floor_auc, report_dict), or None if the
        split is too thin (fewer than 6 train or 3 test chunks) to mean
        anything.
    """
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological, chunks): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")

    if len(train_idx) < 6 or len(test_idx) < 3:
        print("[ERROR] Not enough chunks for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw, labels, starts, train_idx, val_idx, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    ensemble_score = np.mean(per_seed_scores, axis=0)

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    single_class = len(np.unique(yt_ref)) < 2
    base_brier = float("nan") if single_class else float(brier_score_loss(yt_ref, base_pred))
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   Brier {base_brier:.4f}   n={len(yt_ref)}")

    print(f"\n--- CNN-only chunk forecaster ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc)
    report = binary_report(yt_ref, ensemble_score)
    bss = (float("nan") if (single_class or not np.isfinite(base_brier) or base_brier == 0)
          else 1.0 - report["brier"] / base_brier)
    report["brier_skill_score_vs_base_rate"] = bss
    print_report(f"CNN-only chunk forecaster ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def main():
    """Loads the raw waveform archive/catalog, chunks it, and runs the fold sweep."""
    args = parse_args()

    print("Loading raw preprocessed waveform and building chunk labels...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    starts, labels = build_chunks(hour_index, major_times, args.chunk_hours, args.horizon_days)
    n_chunks = len(starts)
    print(f"  {n_chunks} chunks of {args.chunk_hours}h each, positive rate "
         f"(event within {args.horizon_days:.0f}d after chunk end): {labels.mean():.3f}")

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
        result = run_fold(fold_label, args, raw, labels, starts, train_idx, val_idx, test_idx,
                          seeds, device)
        if result is not None:
            results.append(result)

    if args.cv_folds > 1 and results:
        aucs = np.array([r[0] for r in results])
        floors = np.array([r[1] for r in results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(results)}/{args.cv_folds} folds)\n{'=' * 64}")
        print(f"  ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
        print(f"  ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}")
        print(f"  floor AUC:     mean {floors.mean():.4f}  std {floors.std():.4f}")
        print(f"  beats its own fold's floor in {int((aucs > floors).sum())}/{len(results)} folds")


if __name__ == "__main__":
    main()
