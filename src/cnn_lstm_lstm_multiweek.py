"""
Hierarchical CNN -> LSTM(within-week) -> LSTM(across-weeks) 3-class classifier,
pooling BOTH stations (BODT + DAT) for more independent samples than either
alone can give at week-chunk granularity.

"Idea #5" from this session's brainstorming: keep the day-CNN + within-week
LSTM design from cnn_lstm_daily_3class.py (each day embedded by
RawWaveformEncoder, an LSTMAttentionBranch pools 7 days into one week
embedding), but add a SECOND LSTMAttentionBranch on top that sees a
chronological sequence of `--n-weeks` consecutive week-embeddings, so the
model can track how the situation is changing across weeks rather than
judging each week in isolation. The label (event/event_after/none, same
3-class target as cnn_lstm_daily_3class.py) is for the LAST week in each
sequence -- the preceding weeks are context only.

Sample count is the reason this wasn't just built directly on one station:
disjoint week-sequences would leave ~20 examples per station (the same wall
that made the daily-3class and pre-event experiments uninterpretable). Two
fixes stacked: (1) pool both stations -- BODT and DAT are separate physical
recordings, so their week-chunks are built independently and never span
across a station boundary, but the resulting training examples are pooled;
(2) use OVERLAPPING multi-week sequences (sliding by 1 week, not disjoint
blocks of --n-weeks), which is the dense-vs-curated tradeoff this session
kept running into -- more samples, but adjacent sequences now share
--n-weeks-1 of their weeks, so an embargo (in week-chunk units) is enforced
between train/val/test to prevent that overlap crossing the split boundary.

Usage:
    python cnn_lstm_lstm_multiweek.py \\
        --data-roots ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated,../../Sismokaos/feature-extract/data/aegean_dat_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --n-weeks 4 --horizon-days 7 --cv-folds 2

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from cnn_lstm_daily_3class import CLASS_NAMES, HOURS_PER_DAY, build_3class_daily_chunks
from feature_lstm_forecast import load_aegean_events, truncate_to_reliable_catalog_end
from metrics import majority_class_baseline, multiclass_report, print_report
from model.blocks import LSTMAttentionBranch
from raw_cnn_lstm_forecast import RawWaveformEncoder, load_hourly_raw_consolidated
from training import seed_everything

WEEK_DAYS = 7


def build_multiweek_sequences(hour_index, major_times, n_weeks: int, horizon_days: float):
    """Builds overlapping n_weeks-long sequences of week-chunks for one station.

    Args:
        hour_index: DatetimeIndex of hour starts, one per raw hour.
        major_times: Sorted array of qualifying event times.
        n_weeks: Number of consecutive week-chunks per sequence.
        horizon_days: Forwarded to `build_3class_daily_chunks` for the
            event_after class.

    Returns:
        Tuple of (week_starts, seq_start_week_indices, seq_labels,
        chunk_hours) -- `week_starts[i]` is hour i's week-chunk's start
        index into the raw array; `seq_start_week_indices[j]` is the first
        week-chunk index of sequence j (spanning weeks
        [j, j+n_weeks)); `seq_labels[j]` is the LAST week-chunk's own
        3-class label.
    """
    week_starts, week_labels, chunk_hours = build_3class_daily_chunks(
        hour_index, major_times, WEEK_DAYS, horizon_days)
    n_seq = len(week_starts) - n_weeks + 1
    if n_seq <= 0:
        return week_starts, np.array([], dtype=np.int64), np.array([], dtype=np.int64), chunk_hours
    seq_start_idx = np.arange(n_seq)
    seq_labels = week_labels[n_weeks - 1:]
    return week_starts, seq_start_idx, seq_labels, chunk_hours


class MultiStationMultiWeekDataset(Dataset):
    """One sample per (station, sequence) pair -- `n_weeks` consecutive
    week-chunks' raw waveform, reshaped to (n_weeks, WEEK_DAYS, 3,
    HOURS_PER_DAY*hour_samples). Per-channel z-normalized per station (each
    station's own stats -- site response/instrument noise differs by
    station, so pooling with one global normalization would let the model
    key off which station a sample came from instead of the waveform
    content)."""

    def __init__(self, raws, labels, week_starts, chunk_hours, n_weeks,
                station_seq_pairs, stats_per_station=None):
        """Builds the dataset.

        Args:
            raws: List of hourly raw waveform arrays, one per station.
            labels: List of per-week-chunk labels, one array per station
                (from `build_multiweek_sequences`'s week-level label, though
                only the sequence-level label is actually used here -- kept
                for interface symmetry, unused).
            week_starts: List of per-station week-chunk start hour-indices.
            chunk_hours: Hours per week-chunk (same across stations).
            n_weeks: Number of consecutive week-chunks per sequence.
            station_seq_pairs: List of (station_idx, seq_start_week_idx,
                label) tuples this split uses.
            stats_per_station: Optional list of (mean, std) tuples, one per
                station; if None, computed from this split's first 3
                sequences per station.
        """
        self.raws = raws
        self.week_starts = week_starts
        self.chunk_hours = chunk_hours
        self.n_weeks = n_weeks
        self.pairs = station_seq_pairs
        if stats_per_station is None:
            stats_per_station = []
            for s in range(len(raws)):
                own = [p for p in station_seq_pairs if p[0] == s][:3]
                if not own:
                    stats_per_station.append((np.zeros((3, 1)), np.ones((3, 1))))
                    continue
                chunks = []
                for st, seq_i, _ in own:
                    ws = self.week_starts[st][seq_i]
                    chunks.append(np.asarray(self.raws[st][ws:ws + self.chunk_hours]))
                sub = np.concatenate(chunks, axis=0)
                mu = sub.mean(axis=(0, 2), keepdims=True)
                sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
                stats_per_station.append((mu[0], sd[0]))
        self.stats_per_station = stats_per_station

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Returns (multiweek_seq, label) for one (station, sequence) pair.

        Args:
            idx: Index into `self.pairs`.

        Returns:
            Tuple of (float32 tensor shape (n_weeks, WEEK_DAYS, 3,
            HOURS_PER_DAY*hour_samples), int64 scalar tensor class label).
        """
        station, seq_start_week, label = self.pairs[idx]
        ws = self.week_starts[station][seq_start_week:seq_start_week + self.n_weeks]
        raw = self.raws[station]
        chunks = np.stack([np.asarray(raw[s:s + self.chunk_hours]) for s in ws], axis=0)
        mu, sd = self.stats_per_station[station]
        chunks = (chunks - mu[None, None]) / sd[None, None]
        hour_samples = chunks.shape[-1]
        chunks = chunks.reshape(self.n_weeks, WEEK_DAYS, HOURS_PER_DAY, 3, hour_samples)
        x = np.ascontiguousarray(
            chunks.transpose(0, 1, 3, 2, 4).reshape(self.n_weeks, WEEK_DAYS, 3, HOURS_PER_DAY * hour_samples))
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long)


class HierarchicalCNNLSTMLSTM(nn.Module):
    """`RawWaveformEncoder` (day embedding) -> `LSTMAttentionBranch`
    (within-week, 7 days -> 1 week embedding) -> `LSTMAttentionBranch`
    (across-weeks, n_weeks embeddings -> 1 pooled embedding) -> 3-class
    head."""

    def __init__(self, cnn_out=32, week_hidden=16, seq_hidden=16, dropout=0.4, n_classes=3):
        """Initializes the day-encoder and the two LSTM/attention branches.

        Args:
            cnn_out: Width of the CNN's per-day embedding.
            week_hidden: Within-week LSTM hidden size (per direction).
            seq_hidden: Across-weeks LSTM hidden size (per direction).
            dropout: Dropout used throughout.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.encoder = RawWaveformEncoder(out_dim=cnn_out, dropout=dropout)
        self.week_branch = LSTMAttentionBranch(cnn_out, hidden=week_hidden, dropout=dropout)
        self.seq_branch = LSTMAttentionBranch(self.week_branch.out_dim, hidden=seq_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.seq_branch.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.seq_branch.out_dim, seq_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_hidden, n_classes),
        )

    def forward(self, x):
        """Classifies one batch of multi-week sequences.

        Args:
            x: Input batch, shape (batch, n_weeks, WEEK_DAYS, 3, day_samples).

        Returns:
            Tensor of shape (batch, n_classes), raw logits.
        """
        b, w, d = x.shape[:3]
        day_emb = self.encoder(x.reshape(b * w * d, *x.shape[3:])).reshape(b * w, d, -1)
        week_emb = self.week_branch(day_emb).reshape(b, w, -1)
        seq_emb = self.seq_branch(week_emb)
        return self.head(seq_emb)


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-roots", required=True, help="Comma-separated consolidated archive dirs, one per station.")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--n-weeks", type=int, default=4)
    p.add_argument("--horizon-days", type=float, default=7.0)
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--week-hidden", type=int, default=16)
    p.add_argument("--seq-hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--embargo-weeks", type=int, default=None,
                  help="Weeks of gap between train/val/test within each station -- defaults "
                       "to --n-weeks (the exact amount needed to eliminate all overlap between "
                       "adjacent sliding sequences crossing the split boundary).")
    return p.parse_args()


def train_one_seed(args, seed, raws, week_starts, chunk_hours, train_pairs, val_pairs, test_pairs, device):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args.
        seed: Random seed for init/shuffling.
        raws: List of hourly raw waveform arrays, one per station.
        week_starts: List of per-station week-chunk start hour-indices.
        chunk_hours: Hours per week-chunk.
        train_pairs: List of (station, seq_start_week, label) for training.
        val_pairs: Same, for validation.
        test_pairs: Same, for test.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_pred, y_prob) arrays for the test split, from
        the best (by val balanced accuracy) epoch's weights.
    """
    seed_everything(seed)
    train_ds = MultiStationMultiWeekDataset(raws, None, week_starts, chunk_hours, args.n_weeks, train_pairs)
    val_ds = MultiStationMultiWeekDataset(raws, None, week_starts, chunk_hours, args.n_weeks, val_pairs,
                                          stats_per_station=train_ds.stats_per_station)
    test_ds = MultiStationMultiWeekDataset(raws, None, week_starts, chunk_hours, args.n_weeks, test_pairs,
                                           stats_per_station=train_ds.stats_per_station)

    model = HierarchicalCNNLSTMLSTM(cnn_out=args.cnn_out, week_hidden=args.week_hidden,
                                    seq_hidden=args.seq_hidden, dropout=args.dropout).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    train_labels_arr = np.array([p[2] for p in train_pairs])
    class_counts = np.bincount(train_labels_arr, minlength=3)
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


def main():
    """Loads both stations, builds pooled overlapping multi-week sequences,
    and trains/evaluates the hierarchical model."""
    args = parse_args()
    data_roots = args.data_roots.split(",")
    embargo_weeks = args.n_weeks if args.embargo_weeks is None else args.embargo_weeks

    major_times = load_aegean_events(args.catalog_path, args.threshold)

    raws, week_starts_list = [], []
    train_pairs, val_pairs, test_pairs = [], [], []
    chunk_hours = None
    for s, root in enumerate(data_roots):
        print(f"Loading station {s}: {root}")
        hour_index, raw = load_hourly_raw_consolidated(root)
        hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                            buffer_days=args.horizon_days)
        week_starts, seq_start_idx, seq_labels, chunk_hours = build_multiweek_sequences(
            hour_index, major_times, args.n_weeks, args.horizon_days)
        raws.append(raw)
        week_starts_list.append(week_starts)
        n_seq = len(seq_start_idx)
        counts = np.bincount(seq_labels, minlength=3) if n_seq else [0, 0, 0]
        print(f"  {len(week_starts)} week-chunks, {n_seq} overlapping {args.n_weeks}-week sequences -- "
             + " ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(3)))

        i_train = int(n_seq * args.train_frac)
        i_val = int(n_seq * (args.train_frac + args.val_frac))
        for j in range(0, i_train):
            train_pairs.append((s, seq_start_idx[j], int(seq_labels[j])))
        for j in range(i_train + embargo_weeks, i_val):
            val_pairs.append((s, seq_start_idx[j], int(seq_labels[j])))
        for j in range(i_val + embargo_weeks, n_seq):
            test_pairs.append((s, seq_start_idx[j], int(seq_labels[j])))

    print(f"\nPooled across {len(data_roots)} station(s): train={len(train_pairs)} "
         f"val={len(val_pairs)} test={len(test_pairs)}")
    for name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        if pairs:
            counts = np.bincount([p[2] for p in pairs], minlength=3)
            print(f"  {name:5s}: " + " ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(3)))

    if len(train_pairs) < 10 or len(test_pairs) < 5:
        print("[ERROR] Not enough pooled sequences for a meaningful split.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(x) for x in args.ensemble_seeds.split(",")]

    per_seed_probs = []
    yt_ref = None
    for seed in seeds:
        yt, _, probt = train_one_seed(args, seed, raws, week_starts_list, chunk_hours,
                                      train_pairs, val_pairs, test_pairs, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_probs.append(probt)

    ensemble_probs = np.mean(per_seed_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(-1)

    print("\n--- Floor (test set) ---")
    train_labels_arr = np.array([p[2] for p in train_pairs])
    maj, floor_acc, floor_bal_acc = majority_class_baseline(train_labels_arr, yt_ref)
    print(f"  majority-class ({CLASS_NAMES[maj]})   accuracy {floor_acc:.4f}   "
         f"balanced_accuracy {floor_bal_acc:.4f}   n={len(yt_ref)}")

    print(f"\n--- Hierarchical CNN-LSTM-LSTM (pooled stations) ---")
    ensemble_bal_acc = (balanced_accuracy_score(yt_ref, ensemble_preds)
                        if len(np.unique(yt_ref)) > 1 else float("nan"))
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   "
         f"balanced_accuracy {ensemble_bal_acc:.4f}   n={len(yt_ref)}")

    report = multiclass_report(yt_ref, ensemble_preds, y_score=ensemble_probs, class_names=CLASS_NAMES)
    print_report("Hierarchical CNN-LSTM-LSTM (pooled stations, test set)", report)


if __name__ == "__main__":
    main()
