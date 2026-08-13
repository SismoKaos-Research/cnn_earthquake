"""
Same setup as feature_lstm_forecast.py (KO.GEDZ, May 2024-Feb 2025, M>=4.5
dense forecast target, 24h sequences, same split, same 3-seed ensemble) but
the per-hour input is a small 1D CNN over the raw waveform (3, 18000)
instead of the hand-crafted feature vector. Reads directly from
Sismokaos-featureExtract's preprocessor.preprocess output
(data/aegean_2024_2025/YYYY_MM_DD/*.npy), not the feature CSV.

Usage:
    python raw_cnn_lstm_forecast.py \\
        --data-root ../../Sismokaos-featureExtract/data/aegean_2024_2025 \\
        --catalog-path ../../data_downloader/catalogs/data_large.csv

Also imported (not just run standalone): raw100hz_cnn_lstm_forecast.py
imports `load_hourly_raw`, `load_hourly_raw_consolidated`, `parse_args`, and
`run_horizon` from this module -- it is the same pipeline at native sample
rate, differing only in the per-step encoder and HOUR_SAMPLES (`run_horizon`
is called with its own `model_cls` so training reuses this module's fold
loop with a different network).
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from feature_lstm_forecast import (days_since_prev_major, label_hours,
                                   load_aegean_events, print_split_diagnostics,
                                   safe_auc, truncate_to_reliable_catalog_end,
                                   walk_forward_splits)
from metrics import binary_report, print_report
from sklearn.metrics import brier_score_loss
from model.sequence import SequenceHeadNet
from training import seed_everything

_DATE_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
HOUR_SAMPLES = 18000  # 3600s * 5Hz


def load_hourly_raw(data_root: str, hour_samples: int = HOUR_SAMPLES, max_days: int = None):
    """Loads every hour's raw waveform .npy file into one in-RAM array.

    Gaps (usually a couple percent per hour) are linearly interpolated, then
    anything left over is zeroed.

    `hour_samples` defaults to this module's 5Hz-decimated HOUR_SAMPLES, but
    accepts any per-hour sample count -- e.g. a sibling script reading
    scripts/gap_only_preprocess.py's native-rate (100Hz, unfiltered) .npy
    output passes hour_samples=3600*100 instead. The whole return value is
    one in-RAM array (no memory-mapping), which is fine at 5Hz -- the full
    multi-year archive is a few GB -- but native rate is 20x that; at full
    archive length it doesn't fit in RAM on a machine this size. `max_days`
    caps how many (chronologically first) date directories get loaded, so a
    native-rate run can stay within memory instead of OOMing partway through
    loading -- leave it None for the 5Hz case, where the full archive is fine.

    Args:
        data_root: Sismokaos-featureExtract preprocessed dir containing
            YYYY_MM_DD/*.npy struct-array files (E/N/Z fields).
        hour_samples: Expected samples per hour per channel.
        max_days: Cap on the number of (chronologically first) date
            directories to load. None loads the full archive.

    Returns:
        Tuple of (DatetimeIndex of hour starts, float32 array shape
        (n_hours, 3, hour_samples)).
    """
    root = Path(data_root)
    date_dirs = sorted(d for d in root.iterdir() if d.is_dir() and _DATE_DIR_RE.match(d.name))
    if max_days is not None:
        date_dirs = date_dirs[:max_days]

    # First pass resolves (hour, path) pairs only -- no array data loaded
    # yet, so this stays cheap even for a huge archive.
    entries = []
    for date_dir in date_dirs:
        for npy_path in sorted(date_dir.glob("*.npy")):
            parts = npy_path.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                hour_dt = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
            except ValueError:
                continue
            entries.append((hour_dt, npy_path))
    entries.sort(key=lambda e: e[0])
    hour_index = pd.DatetimeIndex([e[0] for e in entries])

    # Pre-allocated once at final size and filled in place, rather than
    # accumulating a Python list of every hour's array and np.stack-ing it at
    # the end -- that held the list AND the stacked copy in memory at once
    # (plus a float64-to-float32 cast that briefly doubled it again), which
    # is harmless at 5Hz (a few GB total) but OOM'd a native-rate (100Hz,
    # ~20x the samples/hour) run on a 14GB machine even with max_days capping
    # the day count.
    raw = np.empty((len(entries), 3, hour_samples), dtype=np.float32)
    for h, (_, npy_path) in enumerate(entries):
        struct = np.load(npy_path)
        for c, comp in enumerate(("E", "N", "Z")):
            x = (struct[comp].astype(np.float32) if comp in struct.dtype.names
                else np.full(len(struct), np.nan, dtype=np.float32))
            if len(x) != hour_samples:
                fixed = np.full(hour_samples, np.nan, dtype=np.float32)
                n = min(hour_samples, len(x))
                fixed[:n] = x[:n]
                x = fixed
            nan = np.isnan(x)
            if nan.any() and (~nan).sum() > 3:
                x[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), x[~nan])
            raw[h, c] = np.nan_to_num(x, nan=0.0)
    return hour_index, raw


def load_hourly_raw_consolidated(consolidated_dir: str, mmap: bool = True):
    """Loads a directory built by consolidate_hourly_raw.py.

    Reads raw.npy (one pre-interpolated (n_hours, 3, hour_samples) float32
    array) + hours.npy (int64 epoch-second timestamps, same order).
    `mmap=True` (default) opens raw.npy with mmap_mode='r' so only the hours
    a batch actually indexes get pulled into RAM -- RawSeqDataset's slicing +
    arithmetic already materializes exactly the touched slice, unchanged, so
    this is a drop-in replacement for load_hourly_raw's (hour_index, raw)
    return -- except the whole archive no longer needs to fit in RAM at
    once, which native-rate archives can exceed on a small machine.

    Args:
        consolidated_dir: Directory containing raw.npy and hours.npy, as
            written by consolidate_hourly_raw.py.
        mmap: If True, opens raw.npy memory-mapped rather than loading it
            fully into RAM.

    Returns:
        Tuple of (DatetimeIndex of hour starts, float32 array shape
        (n_hours, 3, hour_samples), memmap-backed if `mmap`).
    """
    d = Path(consolidated_dir)
    hours = np.load(d / "hours.npy")
    hour_index = pd.DatetimeIndex(pd.to_datetime(hours, unit="s"))
    raw = np.load(d / "raw.npy", mmap_mode="r" if mmap else None)
    return hour_index, raw


class RawSeqDataset(Dataset):
    """Windows of `seq_hours` consecutive hourly raw waveforms, per-channel
    z-normalized. Normalization stats are fit on (a sample of) this
    dataset's own windows unless `stats` is passed in (val/test must reuse
    the train set's stats to avoid leaking each split's own distribution
    into its inputs)."""

    def __init__(self, raw: np.ndarray, labels: np.ndarray, seq_hours: int, indices: np.ndarray,
                stats=None):
        """Builds the windowed dataset.

        Args:
            raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples)
                (in-RAM or memmap -- see `load_hourly_raw_consolidated`).
            labels: Per-hour binary labels, shape (n_hours,).
            seq_hours: Number of consecutive hours per window.
            indices: End-index (inclusive) of each window, into `raw`.
            stats: Optional (mean, std) tuple, each shape (3, 1), to
                normalize with; if None, computed from the first 50 of this
                dataset's own windows (a full pass over a native-rate
                archive would be expensive, and 50 windows is enough to
                estimate per-channel scale).
        """
        self.raw = raw
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
        if stats is None:
            # Sample windows spread across the whole split rather than its first 50.
            # The opening hours of an archive are a narrow, unrepresentative slice
            # (station conditions and seasonal noise drift), and the estimate matters
            # most exactly where this dataset gets used hardest -- a cross-station run
            # fits these stats on one station and applies them to another, so a
            # 50-window estimate is a thin basis for that transfer.
            stat_idx = indices[np.linspace(0, len(indices) - 1, min(500, len(indices))).astype(int)]
            sub = np.concatenate([raw[max(0, i - seq_hours + 1):i + 1] for i in stat_idx], axis=0)
            mu = sub.mean(axis=(0, 2), keepdims=True)
            sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
            stats = (mu[0], sd[0])  # (3,1) each
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (seq, label) for one window.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (seq_hours, 3, hour_samples),
            float32 scalar tensor label).
        """
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        seq = self.raw[start:end + 1]  # (T, 3, HOUR_SAMPLES)
        mu, sd = self.stats
        seq = (seq - mu[None]) / sd[None]
        return (torch.from_numpy(seq).float(),
               torch.tensor(self.labels[end], dtype=torch.float32))


class RawWaveformEncoder(nn.Module):
    """1D CNN that embeds one hour's raw 3-component waveform."""

    def __init__(self, out_dim=32, dropout=0.3):
        """Initializes the 4-stage strided 1D CNN.

        Args:
            out_dim: Width of the embedding this encoder produces per hour.
            dropout: Dropout used after each conv block.
        """
        super().__init__()

        def block(cin, cout, k, s):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                                 nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(dropout))

        self.net = nn.Sequential(
            block(3, 16, 7, 4),
            block(16, 32, 5, 4),
            block(32, 32, 5, 4),
            block(32, out_dim, 3, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        """Embeds one batch of hourly waveforms.

        Args:
            x: Input batch, shape (batch, 3, hour_samples).

        Returns:
            Tensor of shape (batch, out_dim).
        """
        return self.net(x).squeeze(-1)


class RawCNNLSTM(SequenceHeadNet):
    """`SequenceHeadNet` with a `RawWaveformEncoder` embedding each hour's
    raw waveform before the LSTM branch sees it."""

    def __init__(self, cnn_out=32, hidden=16, dropout=0.5):
        """See `SequenceHeadNet.__init__`; `encoder` is always a
        `RawWaveformEncoder(out_dim=cnn_out, dropout=dropout)` here.

        Args:
            cnn_out: Width of the per-hour CNN embedding (`feat_dim` for the
                LSTM branch).
            hidden: LSTM hidden size (per direction) and head hidden width.
            dropout: Dropout used throughout the encoder, branch, and head.
        """
        super().__init__(cnn_out, hidden=hidden, dropout=dropout,
                         encoder=RawWaveformEncoder(out_dim=cnn_out, dropout=dropout))


def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Raw-waveform CNN-LSTM forecaster (vs. hand features).")
    p.add_argument("--data-root", required=True,
                  help="Sismokaos-featureExtract preprocessed dir (data/<EARTHQUAKE_NAME>).")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--stations", nargs="+", default=["BODT", "DAT"], metavar="NAME",
                  help="Stations whose distance --max-station-dist-km is measured from "
                       "(nearest one wins). Names index STATION_COORDS.")
    p.add_argument("--max-station-dist-km", type=float, default=None,
                  help="Keep only events within this many km of the nearest --stations "
                       "entry. Off by default (whole AEGEAN_BBOX). Motivation: over the "
                       "archive window only 1 M>=4.5 event falls within 100 km of BODT "
                       "against 34 across the bbox, so most labelled events sit 200-400 km "
                       "from the instruments -- the leading explanation for the waveform "
                       "branch's failures. 150 is a reasonable starting cap (66 M>=3.5 "
                       "events within 150 km of BODT/DAT, vs 34 M>=4.5 bbox-wide).")
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=None,
                  help="Comma-separated horizon-day values (e.g. 7,14,30,60,90) to sweep in "
                       "one run instead of a single --horizon-days. Each horizon gets its own "
                       "labels and its own full fold loop (same splits reused across horizons, "
                       "since the split itself doesn't depend on labels), plus a cross-horizon "
                       "summary at the end. Overrides --horizon-days when given.")
    p.add_argument("--seq-hours", type=int, default=24)
    p.add_argument("--max-days", type=int, default=None,
                  help="Cap the archive to the first N (chronologically) preprocessed days "
                       "before loading. Mostly superseded by --consolidated (which doesn't need "
                       "the archive to fit in RAM at all), but still useful for a quick, cheap run.")
    p.add_argument("--consolidated", action="store_true",
                  help="--data-root points at a directory built by consolidate_hourly_raw.py "
                       "(raw.npy + hours.npy) instead of the raw YYYY_MM_DD/*.npy tree. Loads via "
                       "mmap -- only the hours a batch actually touches get pulled into RAM, so "
                       "the full archive no longer needs to fit in memory at once. Ignores "
                       "--max-days (the consolidated array is already fully assembled).")
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1,
                  help="1 (default) keeps the single --train-frac/--val-frac chronological "
                       "split. >1 switches to expanding-window walk-forward CV with that many "
                       "folds instead (ignores --train-frac/--val-frac), so one swarm can't "
                       "dominate the whole reported result by sitting in the one test window.")
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD",
                  help="1-indexed fold number(s) to skip entirely (no training, no report) "
                       "when --cv-folds > 1, e.g. --skip 1 2 trains/reports only fold 3. "
                       "Useful once you already know a fold's test block is degenerate "
                       "(single-class -- see the split diagnostics) and don't want to spend "
                       "training time on it again.")
    p.add_argument("--shift-diagnostic", action="store_true",
                  help="Print an RBF-kernel MMD^2 + permutation-test significance check "
                       "between each fold's train and test waveform windows (RMS, kurtosis, "
                       "low/high-band PSD ratio per channel). Quantifies the same swarm/quiet "
                       "distribution shift the positive-rate skew warning flags, but from the "
                       "input waveform's own statistics rather than through the label. Off by "
                       "default -- adds real per-fold cost (samples + Welch PSD + a permutation "
                       "test over up to 150 windows/side).")
    p.add_argument("--balanced-folds", action="store_true",
                  help="When --cv-folds > 1, place walk-forward block boundaries by equal "
                       "positive-label mass instead of equal hour count, using the first "
                       "--horizons value (or --horizon-days) as the reference label set for "
                       "boundary placement -- other horizons in a --horizons sweep reuse the "
                       "same boundaries. Fixes a swarm sitting entirely inside one block (e.g. "
                       "three consecutive 0.9+ positive-rate blocks landing wholly inside one "
                       "fold's test split) at the cost of blocks no longer being equal-width "
                       "in wall-clock time.")
    return p.parse_args()


def train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device, model_cls=None):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args (uses seq_hours, cnn_out, hidden, dropout,
            batch_size, lr, weight_decay, epochs, patience).
        seed: Random seed for init/shuffling.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-hour binary labels for this horizon, shape (n_hours,).
        train_idx: Window end-indices for the training split.
        val_idx: Window end-indices for the validation split.
        test_idx: Window end-indices for the test split.
        device: torch device to train on.
        model_cls: Model class to instantiate, called as
            `model_cls(cnn_out=..., hidden=..., dropout=...)`. Defaults to
            `RawCNNLSTM`; raw100hz_cnn_lstm_forecast.py passes `NativeCNNLSTM`
            here via `run_horizon`/`run_fold`.

    Returns:
        Tuple of (y_true, y_score) arrays for the test split, from the
        best (by checkpoint-selection metric) epoch's weights.
    """
    seed_everything(seed)
    train_ds = RawSeqDataset(raw, labels, args.seq_hours, train_idx)
    val_ds = RawSeqDataset(raw, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = RawSeqDataset(raw, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model_cls = model_cls or RawCNNLSTM
    model = model_cls(cnn_out=args.cnn_out, hidden=args.hidden, dropout=args.dropout).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=2)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = labels[train_idx].mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for seq, y in loader:
                seq, y = seq.to(device), y.to(device)
                logit = model(seq)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    yv0, _, _ = evaluate(val_loader)
    use_loss_fallback = len(np.unique(yv0)) < 2
    if use_loss_fallback:
        print(f"  [seed {seed}] val split is single-class -- val loss is minimized by always "
             "predicting that one class, which rewards collapse rather than discrimination. "
             "Checkpointing on train AUC instead (the split guaranteed to have both classes).")

    best = -1.0
    no_improve, best_state = 0, None
    for epoch in range(args.epochs):
        model.train()
        for seq, y in train_loader:
            seq, y = seq.to(device), y.to(device)
            loss = criterion(model(seq), y)
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
        improved = metric > best
        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val AUC {val_auc:.4f} val loss {val_loss:.4f}"
             + (f" train AUC {metric:.4f}" if use_loss_fallback else ""))
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


def waveform_shift_mmd(raw, idx_a, idx_b, seq_hours, fs, n_sample=150, n_permute=300, rng_seed=0):
    """Quantifies the distribution shift between two groups of raw-waveform
    windows via RBF-kernel MMD^2 on simple per-channel engineered features,
    with a permutation-test significance threshold -- the same swarm/quiet
    shift `print_split_diagnostics`'s positive-rate skew warning flags, but
    measured directly on the input waveform's own statistics rather than
    through the label (so it can't be fooled by a horizon choice that merely
    changes how the *same* underlying shift gets binarized).

    Per window (pooled across its `seq_hours` hours), computes 3 features per
    channel -- RMS, excess kurtosis, and log(low-band/high-band Welch PSD
    power, split at fs/4) -- then estimates MMD^2 between the two groups'
    (jointly standardized) feature vectors using a median-heuristic RBF
    kernel, and a null distribution via label permutation (Gretton et al.
    2012) to judge significance.

    Args:
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        idx_a: Window end-indices for the first group (e.g. train).
        idx_b: Window end-indices for the second group (e.g. test).
        seq_hours: Number of consecutive hours per window.
        fs: Sample rate in Hz of `raw`'s last axis (5.0 for the decimated
            archive, 100.0 for the native-rate one).
        n_sample: Max windows sampled per group -- full-archive feature
            extraction is expensive, this bounds the cost.
        n_permute: Number of label permutations for the null distribution.
        rng_seed: Random seed for sampling and permutation.

    Returns:
        Dict with "mmd2" (point estimate), "null_95pct" (significance
        threshold), "significant_shift" (mmd2 > null_95pct), "n_a", "n_b".
    """
    from scipy.signal import welch
    from scipy.stats import kurtosis

    rng = np.random.default_rng(rng_seed)

    def window_features(end):
        start = end - seq_hours + 1
        seq = np.asarray(raw[start:end + 1])  # (T, 3, hour_samples)
        feats = []
        for c in range(seq.shape[1]):
            x = seq[:, c, :].reshape(-1).astype(np.float64)
            feats.append(np.sqrt(np.mean(x ** 2)))
            feats.append(kurtosis(x, fisher=True, bias=False))
            f, pxx = welch(x, fs=fs, nperseg=min(4096, len(x)))
            low, high = pxx[f < fs / 4].sum(), pxx[f >= fs / 4].sum()
            feats.append(np.log((low + 1e-12) / (high + 1e-12)))
        return np.array(feats)

    def sample_group(idx):
        chosen = idx if len(idx) <= n_sample else rng.choice(idx, size=n_sample, replace=False)
        return np.array([window_features(e) for e in chosen])

    Xa, Xb = sample_group(idx_a), sample_group(idx_b)
    pooled = np.concatenate([Xa, Xb])
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    Xa, Xb, pooled = (Xa - mu) / sd, (Xb - mu) / sd, (pooled - mu) / sd

    def rbf_mmd2(A, B, gamma):
        Kaa = np.exp(-gamma * ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
        Kbb = np.exp(-gamma * ((B[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        Kab = np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        na, nb = len(A), len(B)
        return ((Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
               + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1)) - 2 * Kab.mean())

    pooled_dists = ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / max(np.median(pooled_dists[pooled_dists > 0]), 1e-12)

    point = rbf_mmd2(Xa, Xb, gamma)
    na = len(Xa)
    null = np.array([rbf_mmd2(pooled[p[:na]], pooled[p[na:]], gamma)
                     for p in (rng.permutation(len(pooled)) for _ in range(n_permute))])
    null_95 = float(np.percentile(null, 95))
    return {"mmd2": float(point), "null_95pct": null_95, "significant_shift": point > null_95,
           "n_a": na, "n_b": len(Xb)}


def run_fold(fold_label, args, raw, labels, dsp, hour_index, train_idx, val_idx, test_idx,
            seeds, device, model_cls=None, horizon_days=None):
    """Trains the seed ensemble on one split and reports it.

    `horizon_days` defaults to args.horizon_days -- a caller sweeping
    multiple horizons passes each one explicitly, since `labels` and the
    persistence floor both need to agree on which horizon produced them.

    Args:
        fold_label: Header string printed above this fold's report.
        args: Parsed CLI args, forwarded to `train_one_seed`.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-hour binary labels for this horizon, shape (n_hours,).
        dsp: Days-since-previous-major-event array, shape (n_hours,), for
            the persistence floor.
        hour_index: DatetimeIndex of hour starts, for split diagnostics.
        train_idx: Window end-indices for the training split.
        val_idx: Window end-indices for the validation split.
        test_idx: Window end-indices for the test split.
        seeds: List of random seeds to train and ensemble.
        device: torch device to train on.
        model_cls: Model class forwarded to `train_one_seed`.
        horizon_days: Forecast horizon in days; defaults to
            `args.horizon_days`.

    Returns:
        Tuple of (ensemble_auc, floor_auc, report_dict), or None if the
        split is too thin (fewer than 10 train or 5 test windows) to mean
        anything.
    """
    horizon_days = args.horizon_days if horizon_days is None else horizon_days
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx)

    if getattr(args, "shift_diagnostic", False) and len(train_idx) and len(test_idx):
        fs = raw.shape[-1] / 3600.0
        shift = waveform_shift_mmd(raw, train_idx, test_idx, args.seq_hours, fs)
        flag = "[!] SIGNIFICANT" if shift["significant_shift"] else "not significant"
        print(f"  [shift-diagnostic] train vs test waveform MMD^2={shift['mmd2']:.4f} "
             f"(null 95th pct {shift['null_95pct']:.4f}, n={shift['n_a']}/{shift['n_b']}) -- {flag}")

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device,
                                model_cls=model_cls)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    ensemble_score = np.mean(per_seed_scores, axis=0)

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(yt_ref)}")
    pers_dsp = dsp[test_idx]
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= horizon_days).astype(int)).astype(np.float64)
    pers_auc = safe_auc(yt_ref, pers_pred)
    single_class = len(np.unique(yt_ref)) < 2
    pers_brier = float("nan") if single_class else float(brier_score_loss(yt_ref, pers_pred))
    print(f"  persistence             AUC {pers_auc:.4f}   Brier {pers_brier:.4f}   n={len(yt_ref)}")

    print(f"\n--- Raw-waveform CNN-LSTM ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, pers_auc)
    report = binary_report(yt_ref, ensemble_score)
    # Brier Skill Score vs. the persistence floor: how much the model reduces
    # squared calibration error relative to "predict same as last hour", the
    # probabilistic-calibration counterpart to the AUC-vs-floor comparison
    # above (recommended by both the DL-forecasting-evaluation and
    # ML-pitfalls literature reviewed alongside this fix, over plain AUC
    # which can look fine while badly miscalibrated).
    bss = (float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
          else 1.0 - report["brier"] / pers_brier)
    report["brier_skill_score_vs_persistence"] = bss
    print_report(f"Raw-waveform CNN-LSTM ensemble ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def run_horizon(horizon_days, args, hour_index, raw, major_times, dsp, folds, fold_labels,
                skip, seeds, device, model_cls=None):
    """Labels the archive for one horizon and runs the full fold loop.

    Splits are the same across every horizon -- the split doesn't depend on
    labels, only on window count -- so folds are computed once by the caller
    and reused here.

    Args:
        horizon_days: Forecast horizon in days.
        args: Parsed CLI args, forwarded to `run_fold`.
        hour_index: DatetimeIndex of hour starts.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        major_times: Sorted array of qualifying event times.
        dsp: Days-since-previous-major-event array, shape (n_hours,).
        folds: List of (train_idx, val_idx, test_idx) tuples, one per fold.
        fold_labels: Header string per fold, same length as `folds`.
        skip: Set of 1-indexed fold numbers to skip.
        seeds: List of random seeds to train and ensemble, per fold.
        device: torch device to train on.
        model_cls: Model class forwarded to `run_fold`/`train_one_seed`.

    Returns:
        List of per-fold (ensemble_auc, floor_auc, report) results (skipped
        or too-thin folds omitted), same shape `run_fold`'s callers already
        handle.
    """
    print(f"\n{'#' * 64}\n# horizon = {horizon_days:.0f} days\n{'#' * 64}")
    labels = label_hours(hour_index, major_times, horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, raw, labels, dsp, hour_index,
                          train_idx, val_idx, test_idx, seeds, device,
                          model_cls=model_cls, horizon_days=horizon_days)
        if result is not None:
            results.append(result)

    if args.cv_folds > 1 and results:
        aucs = np.array([r[0] for r in results])
        floors = np.array([r[1] for r in results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary ({len(results)}/{args.cv_folds} folds, "
             f"horizon={horizon_days:.0f}d)\n{'=' * 64}")
        print(f"  ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
        print(f"  ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}"
             f"  ({np.isnan(aucs).sum()} fold(s) undefined -- single-class test set)"
             if np.isnan(aucs).any() else
             f"  ensemble AUC:  mean {aucs.mean():.4f}  std {aucs.std():.4f}")
        print(f"  floor AUC:     mean {floors.mean():.4f}  std {floors.std():.4f}")
        beats = (aucs > floors + 1e-9).sum()
        print(f"  beats its own fold's floor in {beats}/{len(results)} folds")

    return results


def main():
    """Loads the raw waveform archive/catalog, builds hourly labels, and
    runs the fold/horizon sweep."""
    args = parse_args()

    print("Loading raw preprocessed waveform and building hourly labels...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold,
                                     stations=args.stations,
                                     max_dist_km=args.max_station_dist_km)
    if args.max_station_dist_km:
        print(f"  [distance cap] events restricted to <= {args.max_station_dist_km:.0f} km "
             f"from nearest of {args.stations}")
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    max_horizon = max([float(h) for h in args.horizons.split(",")] if args.horizons
                      else [args.horizon_days])
    hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                        buffer_days=max_horizon)

    dsp = days_since_prev_major(hour_index, major_times)

    n = len(hour_index)
    valid_end_indices = np.arange(args.seq_hours - 1, n)

    horizons = ([float(h) for h in args.horizons.split(",")] if args.horizons
               else [args.horizon_days])

    # A window ending at index e covers raw hours [e-seq_hours+1, e], so with no
    # gap the first val/test window right after a split boundary shares up to
    # seq_hours-1 hours of *input* with the last training window. embargo removes
    # that overlap (see walk_forward_splits' docstring).
    #
    # The label additionally looks horizon_days FORWARD, so a seq_hours-1 gap alone
    # still lets the last ~horizon_days of each block carry labels determined by
    # events inside the next block (train labels encoding what happens in val, val
    # labels encoding what happens in test). The horizon term below closes that
    # overlapping-label leak (Lopez de Prado, Advances in Financial Machine
    # Learning, Ch. 7). These folds are built once and reused across every horizon
    # in a --horizons sweep, so size the gap for the LARGEST horizon or the shorter
    # ones would silently re-open the leak for the longer ones.
    embargo = args.seq_hours - 1 + int(round(max(horizons) * 24))

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train], valid_end_indices[i_train + embargo:i_val],
                 valid_end_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    elif args.balanced_folds:
        ref_labels = label_hours(hour_index, major_times, horizons[0])
        print(f"  [balanced-folds] placing block boundaries by positive-label mass at "
             f"horizon={horizons[0]:.0f}d (reused for every horizon in this run)")
        folds = walk_forward_splits(valid_end_indices, args.cv_folds,
                                    labels=ref_labels[valid_end_indices], embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]
    else:
        folds = walk_forward_splits(valid_end_indices, args.cv_folds, embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    per_horizon = {}
    for horizon_days in horizons:
        per_horizon[horizon_days] = run_horizon(horizon_days, args, hour_index, raw, major_times,
                                                dsp, folds, fold_labels, skip, seeds, device)

    if len(horizons) > 1:
        print(f"\n{'#' * 64}\n# Cross-horizon summary\n{'#' * 64}")
        print(f"  {'horizon (d)':>12s}  {'ensemble AUC (mean)':>20s}  {'floor AUC (mean)':>16s}  folds")
        for horizon_days in horizons:
            results = per_horizon[horizon_days]
            if not results:
                print(f"  {horizon_days:12.0f}  {'(no folds ran)':>20s}")
                continue
            aucs = np.array([r[0] for r in results])
            floors = np.array([r[1] for r in results])
            print(f"  {horizon_days:12.0f}  {np.nanmean(aucs):20.4f}  {floors.mean():16.4f}  {len(results)}")


if __name__ == "__main__":
    main()
