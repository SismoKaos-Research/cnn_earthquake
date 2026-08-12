"""
LSTM forecaster trained directly on Sismokaos-featureExtract's hand-crafted
continuous features (STA/LTA, Hjorth, permutation entropy, spectral
centroid, cross-axis correlation -- computed on real KO.GEDZ continuous
data, Aegean zone), instead of the catalog-derived sequences
cnn_lstm_forecast.py uses. Same validated target as that script: will a
M >= threshold event occur in this zone within horizon_days? -- reusing the
target definition, not the catalog features, so this is a fair test of
whether the hand-crafted continuous features carry the same (or different)
forecasting signal.

Pipeline:
  1. Load the combined features CSV (Sismokaos-featureExtract/results/GEDZ/*.csv).
  2. Reconstruct each row's approximate absolute UTC time from Pencere_ID
     (date + hour + within-file window index; ~1 window / 50s resolution).
  3. Aggregate to HOURLY mean feature vectors -- the label doesn't move
     faster than daily, and raw 50s-resolution would make the LSTM's
     sequence length intractable for a 100-day span (~172k raw steps vs.
     ~2.4k hourly steps).
  4. Label each hour: will a M>=threshold AEGEAN event occur within the next
     horizon_days (dense target, no declustering -- same choice
     catalog.py's build_dense_windows makes and the same reason: a
     clustered sequence is signal, not noise to define away).
  5. Sliding windows of `seq_hours` consecutive hourly vectors, chronological
     train/val/test split (no shuffling across time).
  6. Train an LSTM + attention (reuses cnn_lstm.py's LSTMAttentionBranch)
     binary classifier; report against base-rate and persistence floors,
     matching cnn_lstm_forecast.py's convention exactly so the two are
     directly comparable.

Usage:
    python feature_lstm_forecast.py \\
        --features-csv ../../Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv

Also imported (not just run standalone): raw_cnn_lstm_forecast.py and
raw100hz_cnn_lstm_forecast.py both import `days_since_prev_major`,
`label_hours`, `print_split_diagnostics`, `walk_forward_splits` (and
`safe_auc`, re-exported below) from this module rather than duplicating
them, since the raw-waveform forecasters share the same hourly-label /
walk-forward-split machinery as this hand-feature one.
"""

import argparse
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from metrics import binary_report, print_report, safe_auc  # noqa: F401 (safe_auc re-exported; raw_cnn_lstm_forecast.py imports it from here)
from model.sequence import SequenceHeadNet
from training import seed_everything

AEGEAN_BBOX = (36.0, 40.0, 25.0, 30.0)  # lat0, lat1, lon0, lon1 -- matches forecast.py's FAULT_ZONES


def parse_hour_start(pencere_id: str):
    """Parses a Sismokaos-featureExtract window ID into its containing hour.

    e.g. '2024_11_15_00_w01' -> datetime(2024,11,15,0,0,0). Ignores the small
    (<=150s) stitching offset from PREV_LEN carry-over -- negligible against
    a 30-day forecast horizon.

    Args:
        pencere_id: Window ID string, format 'YYYY_MM_DD_HH_wNN'.

    Returns:
        Tuple of (hour_start datetime, window index int), or (None, None) if
        `pencere_id` doesn't match the expected format.
    """
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_w(\d+)", pencere_id)
    if not m:
        return None, None
    y, mo, d, h, w = m.groups()
    return datetime(int(y), int(mo), int(d), int(h)), int(w)


def load_hourly_features(features_csv: str) -> pd.DataFrame:
    """Loads the combined features file and aggregates it to hourly means.

    Args:
        features_csv: Path to a Sismokaos-featureExtract combined features
            file (one row per ~50s window, column 'Pencere_ID' identifying
            it) -- either the original CSV format or the structured-.npy
            format the pipeline now writes (same columns, loaded via
            `np.load` instead of `pd.read_csv` when the path ends in
            `.npy`).

    Returns:
        DataFrame indexed by hour-start datetime, one row per hour, columns
        = the mean of every feature column across that hour's windows.
    """
    if str(features_csv).endswith(".npy"):
        df = pd.DataFrame.from_records(np.load(features_csv, allow_pickle=False))
    else:
        df = pd.read_csv(features_csv)
    parsed = df["Pencere_ID"].apply(parse_hour_start)
    df["hour_start"] = [p[0] for p in parsed]
    df = df.dropna(subset=["hour_start"])
    feature_cols = [c for c in df.columns if c not in ("Pencere_ID", "Zaman_Dk", "hour_start")]
    hourly = df.groupby("hour_start")[feature_cols].mean().sort_index()
    return hourly


def load_aegean_events(catalog_path: str, min_magnitude: float = 4.5) -> np.ndarray:
    """Loads catalog events within the Aegean bounding box at or above a magnitude.

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include.

    Returns:
        Sorted array of numpy datetime64 event times.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    return np.sort(aegean.dt.to_numpy())


def load_aegean_events_with_magnitude(catalog_path: str, min_magnitude: float = 3.0):
    """Loads catalog events (times AND magnitudes) within the Aegean bounding box.

    Companion to `load_aegean_events`, which only returns times -- this is
    for magnitude-derived catalog features (mean magnitude, b-value, energy
    release, magnitude deficit) that need a lower completeness threshold
    than the M>=4.5 "major event" set used for labels/persistence, since
    b-value estimation needs more data points than the rare large events
    alone provide (16,724 M>=3.0 Aegean events vs. 261 M>=4.5 ones).

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include (completeness
            threshold for the returned "background" catalog).

    Returns:
        Tuple of (times, magnitudes) -- times is a sorted array of numpy
        datetime64 event times, magnitudes is the matching float64 array,
        same order.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()].sort_values("dt")
    return aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64)


def truncate_to_reliable_catalog_end(hour_index: pd.DatetimeIndex, raw: np.ndarray,
                                     major_times: np.ndarray, buffer_days: float = 0):
    """Drops hours past the point where the catalog can no longer reliably
    inform a forward-looking label.

    Catalogs lag real time -- `major_times[-1]` is the last event anyone's
    gotten around to cataloging, not the last real event. Hours near the
    raw archive's end can have their forward-looking label search window
    reach past `major_times[-1]` into a gap that's empty because nothing's
    been entered yet, not because it was quiet -- a right-censoring
    artifact that inflates apparent AUC for large horizons on whichever CV
    block happens to land at the archive's tail (caught in this project by
    a fold's AUC climbing suspiciously in lockstep with horizon length).

    Args:
        hour_index: DatetimeIndex of hour starts.
        raw: Hourly array (waveform or features), shape (n_hours, ...),
            first axis aligned with `hour_index`.
        major_times: Sorted array of qualifying event times.
        buffer_days: Extra safety margin subtracted from the last event's
            time -- pass the largest horizon this run will evaluate, so no
            eligible hour's forward window can reach past the catalog's
            actual end.

    Returns:
        Tuple of (truncated hour_index, truncated raw). A no-op if the
        cutoff falls after the archive's own end already.
    """
    cutoff = major_times[-1] - np.timedelta64(int(buffer_days * 24), "h")
    n_keep = int((hour_index.to_numpy() <= cutoff).sum())
    if n_keep < len(hour_index):
        print(f"  [!] catalog's last event is {major_times[-1]} -- truncating archive from "
             f"{hour_index[-1]} to {hour_index[n_keep - 1]} ({len(hour_index) - n_keep} hours "
             f"dropped, buffer={buffer_days:.0f}d) to avoid right-censoring the forward-looking "
             f"label near the archive's end.")
    return hour_index[:n_keep], raw[:n_keep]


def label_hours(hourly_index: pd.DatetimeIndex, major_times: np.ndarray, horizon_days: float) -> np.ndarray:
    """Labels each hour with whether a qualifying event occurs within the horizon.

    Args:
        hourly_index: Hour-start timestamps to label, one per sample.
        major_times: Sorted array of qualifying event times (see
            `load_aegean_events`).
        horizon_days: Forecast horizon in days. An hour is labeled positive
            iff a qualifying event occurs in `(hour, hour + horizon_days]`.

    Returns:
        int64 array of 0/1 labels, same length as `hourly_index`.
    """
    horizon = np.timedelta64(int(horizon_days), "D")
    t = hourly_index.to_numpy()
    labels = np.zeros(len(t), dtype=np.int64)
    for i, ti in enumerate(t):
        fut = major_times[(major_times > ti) & (major_times <= ti + horizon)]
        labels[i] = int(len(fut) > 0)
    return labels


def days_since_prev_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days elapsed since the previous qualifying event, per hour.

    Feeds `metrics.persistence_baseline`, which predicts positive iff this
    value is within the forecast horizon.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        major_times: Sorted array of qualifying event times (see
            `load_aegean_events`).

    Returns:
        float64 array, same length as `hourly_index`; NaN where no prior
        qualifying event exists.
    """
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        prev = major_times[major_times < ti]
        if len(prev):
            out[i] = (ti - prev[-1]) / np.timedelta64(1, "D")
    return out


def days_until_next_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days until the next qualifying event, per hour.

    Symmetric counterpart to `days_since_prev_major`. Used by
    `cnn_proximity_classify.py` to build a "is this hour close to a
    qualifying event, looking either direction in time" label -- a regime
    classification (was there an event nearby) rather than a forecast (will
    one occur), so looking backward is fair game, not leakage: the label
    isn't claiming to predict anything unknown at the time.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        major_times: Sorted array of qualifying event times (see
            `load_aegean_events`).

    Returns:
        float64 array, same length as `hourly_index`; NaN where no future
        qualifying event exists (e.g. the last event in the catalog).
    """
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        nxt = major_times[major_times >= ti]
        if len(nxt):
            out[i] = (nxt[0] - ti) / np.timedelta64(1, "D")
    return out


def print_split_diagnostics(hourly_index: pd.DatetimeIndex, labels: np.ndarray,
                            train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                            n_blocks: int = 10, skew_ratio: float = 1.5) -> None:
    """Positive rate over time in `n_blocks` equal-width windows, so a swarm
    or quiet period concentrated in one split is visible rather than hiding
    behind a single split-level average. Chronological splits on this kind of
    non-stationary hourly-hazard label are prone to exactly this: a train/val
    positive rate near the long-run average and a test positive rate that
    reflects whatever happened to be active right at the end of the record.

    If train and test positive rates differ by more than `skew_ratio`x,
    prints an explicit warning -- the model's test AUC on a heavily skewed
    split isn't automatically wrong, but it isn't automatically "the model
    generalized" either. The base-rate/persistence floors printed alongside
    it see the identical skew, so they are the correct thing to compare
    against, not 0.5.
    """
    n = len(hourly_index)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    split_of = np.full(n, "", dtype=object)
    split_of[train_idx] = "train"
    split_of[val_idx] = "val"
    split_of[test_idx] = "test"

    print("\n  positive rate over time (equal-width blocks):")
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        block_splits = split_of[lo:hi]
        present = block_splits[block_splits != ""]
        dominant = pd.Series(present).mode().iloc[0] if len(present) else "-"
        print(f"    {hourly_index[lo].date()} .. {hourly_index[hi - 1].date()}  "
             f"pos rate {labels[lo:hi].mean():.3f}  n={hi - lo:4d}  split~{dominant}")

    rates = {name: labels[idx].mean() for name, idx in
            (("train", train_idx), ("val", val_idx), ("test", test_idx)) if len(idx)}
    if "train" in rates and "test" in rates and rates["train"] > 0:
        ratio = rates["test"] / rates["train"]
        if ratio > skew_ratio or ratio < 1 / skew_ratio:
            print(f"\n  [!] test positive rate ({rates['test']:.3f}) is {ratio:.2f}x train's "
                 f"({rates['train']:.3f}) -- likely a swarm or quiet period concentrated in "
                 "one split rather than the model generalizing. Compare AUC against the "
                 "base-rate/persistence floors below (same skew), not against 0.5.")


def walk_forward_splits(valid_end_indices: np.ndarray, n_folds: int, labels: np.ndarray = None,
                        embargo: int = 0):
    """Expanding-window walk-forward splits, so one arbitrary swarm can only
    dominate a single fold's test set instead of the entire reported result.

    Divides the eligible index range into n_folds+2 contiguous blocks. Fold k
    trains on blocks 0..k, validates on block k+1, tests on block k+2 --
    training data only ever grows forward in time and every fold's test
    block is strictly later than everything it trained or validated on, so
    this stays a legitimate past-predicts-future evaluation (unlike
    shuffling weeks across splits, which would let a model train on data
    chronologically after some of what it's tested on).

    Args:
        valid_end_indices: Eligible window end-indices, chronologically
            sorted.
        n_folds: Number of walk-forward folds.
        labels: Optional per-`valid_end_indices` 0/1 label array (same
            length and order as `valid_end_indices`, e.g.
            `label_hours(...)[valid_end_indices]`). When given, block
            boundaries are placed so each block holds an equal share of
            total positive-label mass instead of an equal share of hours --
            plain equal-width blocks let a single sustained swarm fill one
            block almost entirely (e.g. three consecutive 0.9+ positive-rate
            blocks landing wholly inside one fold's test split), which is
            what produced a below-both-floors fold result. When `None`
            (default) or the label array is entirely one class, falls back
            to the original equal-width behavior.
        embargo: Minimum index gap enforced between consecutive blocks (0
            disables it). A window ending at index `e` covers raw hours
            `[e-seq_hours+1, e]`, so with no gap the first val/test window
            right after a block boundary shares up to `seq_hours-1` hours of
            *input* with the last training window -- the model is scored on
            a sample whose input nearly duplicates one it trained on
            directly. Callers should pass `seq_hours - 1` (the exact value
            that eliminates all window-input overlap across every block
            boundary, train->val and val->test alike) once they know the
            sequence length used to build windows from these indices.
    """
    n_blocks = n_folds + 2
    if labels is None:
        edges = np.linspace(0, len(valid_end_indices), n_blocks + 1).astype(int)
    else:
        cum = np.concatenate([[0], np.cumsum(labels)])
        total = cum[-1]
        if total == 0:
            edges = np.linspace(0, len(valid_end_indices), n_blocks + 1).astype(int)
        else:
            targets = np.linspace(0, total, n_blocks + 1)
            edges = np.searchsorted(cum, targets)
            edges[0], edges[-1] = 0, len(valid_end_indices)
            edges = np.maximum.accumulate(edges)
    blocks = [valid_end_indices[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    if embargo > 0:
        for i in range(1, n_blocks):
            if len(blocks[i - 1]) == 0:
                continue
            cutoff = blocks[i - 1][-1] + embargo
            blocks[i] = blocks[i][blocks[i] > cutoff]
    return [(np.concatenate(blocks[:k + 1]), blocks[k + 1], blocks[k + 2])
           for k in range(n_folds)]


class HourlySeqDataset(Dataset):
    """Windows of `seq_hours` consecutive hourly feature vectors, z-normalized.

    Normalization stats are fit on this dataset's own windows unless `stats`
    is passed in (val/test sets must reuse the train set's stats -- fitting
    separately would leak each split's own distribution into its inputs).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_hours: int,
                indices: np.ndarray, stats=None):
        """Builds the windowed dataset.

        Args:
            features: Hourly feature matrix, shape (n_hours, n_features).
            labels: Per-hour binary labels, shape (n_hours,).
            seq_hours: Number of consecutive hours per window.
            indices: End-index (inclusive) of each window, into `features`.
            stats: Optional (mean, std) tuple to normalize with; if None,
                computed from this dataset's own windows (NaN-safe, with std
                floored to avoid divide-by-zero on constant features).
        """
        self.features = features
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices  # end-index (inclusive) of each window, into `features`
        if stats is None:
            train_feats = np.concatenate([features[max(0, i - seq_hours + 1):i + 1] for i in indices], axis=0)
            with np.errstate(invalid="ignore"):
                mu, sd = np.nanmean(train_feats, axis=0), np.nanstd(train_feats, axis=0) + 1e-6
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            stats = (mu, sd)
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (seq, label) for one window.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (seq_hours, n_features),
            float32 scalar tensor label).
        """
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        seq = self.features[start:end + 1].copy()
        mu, sd = self.stats
        seq = (seq - mu) / sd
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return (torch.from_numpy(seq).float(),
               torch.tensor(self.labels[end], dtype=torch.float32))

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="LSTM forecaster on hand-crafted continuous features.")
    p.add_argument("--features-csv", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=None,
                  help="Comma-separated horizon-day values (e.g. 7,14,30,60,90) to sweep in "
                       "one run instead of a single --horizon-days. Each horizon gets its own "
                       "labels and its own full fold loop (same splits reused across horizons, "
                       "since the split itself doesn't depend on labels), plus a cross-horizon "
                       "summary at the end. Overrides --horizon-days when given.")
    p.add_argument("--seq-hours", type=int, default=168, help="7 days of hourly context per sample.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44",
                  help="Comma-separated seeds; trains one model per seed and reports both "
                       "per-seed AUC and the ensemble (mean-probability) AUC.")
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
    return p.parse_args()


class ForecastLSTM(SequenceHeadNet):
    """`SequenceHeadNet` with no per-step encoder -- the hourly feature
    vectors are fed to the LSTM branch directly (unlike the raw-waveform
    forecasters, which encode each hour's raw samples first)."""

    def __init__(self, feat_dim, hidden=64, dropout=0.3):
        """See `SequenceHeadNet.__init__` (`encoder` is always None here)."""
        super().__init__(feat_dim, hidden=hidden, dropout=dropout)


def train_one_seed(args, seed, feature_cols, features, labels,
                   train_idx, val_idx, test_idx, device):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args (uses seq_hours, hidden, dropout, batch_size,
            lr, weight_decay, epochs, patience).
        seed: Random seed for init/shuffling.
        feature_cols: List of feature column names (used only for width).
        features: Hourly feature matrix, shape (n_hours, n_features).
        labels: Per-hour binary labels for this horizon, shape (n_hours,).
        train_idx: Window end-indices for the training split.
        val_idx: Window end-indices for the validation split.
        test_idx: Window end-indices for the test split.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_score) arrays for the test split, from the
        best (by checkpoint-selection metric) epoch's weights.
    """
    seed_everything(seed)
    train_ds = HourlySeqDataset(features, labels, args.seq_hours, train_idx)
    val_ds = HourlySeqDataset(features, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = HourlySeqDataset(features, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model = ForecastLSTM(len(feature_cols), hidden=args.hidden, dropout=args.dropout).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh)
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

    # Checkpoint selection prefers val AUC, but this task's val split can be a
    # single-class stretch (e.g. entirely inside a sustained swarm's 30-day
    # lookahead) where AUC is undefined for every epoch, not just some --
    # detected once up front rather than falling back per-epoch, since a
    # date-range split's class composition doesn't change epoch to epoch.
    # Val LOSS is not a safe fallback: on a single-class val set it's
    # minimized by always predicting that one class (e.g. all-negative val ->
    # loss keeps dropping as the model pushes its logit toward -inf for
    # everything, with zero discriminative learning required), which rewards
    # collapse rather than penalizing it. Checkpointing on train AUC instead
    # -- the one split guaranteed to have both classes.
    yv0, _, _ = evaluate(val_loader)
    use_loss_fallback = len(np.unique(yv0)) < 2
    if use_loss_fallback:
        print(f"  [seed {seed}] val split is single-class (positive rate {yv0.mean():.3f}) -- "
             "checkpointing on train AUC instead of val loss.")

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


def run_fold(fold_label, args, feature_cols, features, labels, dsp, hourly_index,
            train_idx, val_idx, test_idx, seeds, device, horizon_days=None):
    """Trains the seed ensemble on one split and reports it. Returns
    (ensemble_auc, floor_auc, report_dict), or None if the split is too thin
    to mean anything -- callers decide whether that's fatal (a single split)
    or just a fold to skip (walk-forward CV). `horizon_days` defaults to
    args.horizon_days -- a caller sweeping multiple horizons passes each one
    explicitly, since `labels` and the persistence floor both need to agree
    on which horizon produced them."""
    horizon_days = args.horizon_days if horizon_days is None else horizon_days
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hourly_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split -- need more days of features.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, feature_cols, features, labels,
                                train_idx, val_idx, test_idx, device)
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
    print(f"  persistence             AUC {pers_auc:.4f}   n={len(yt_ref)}")

    print(f"\n--- Hand-feature LSTM ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, pers_auc)
    if ensemble_auc <= floor + 1e-9:
        print("\n  [!] Ensemble does NOT clear max(chance, persistence) -- not evidence of forecasting skill.")
    else:
        print(f"\n  Ensemble beats max(chance, persistence) by {ensemble_auc - floor:+.4f} AUC.")

    report = binary_report(yt_ref, ensemble_score)
    print_report(f"Hand-feature LSTM ensemble ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def run_horizon(horizon_days, args, hourly_index, feature_cols, features, major_times, dsp,
                folds, fold_labels, skip, seeds, device):
    """Labels the archive for one horizon and runs the full fold loop (same
    splits as every other horizon -- the split doesn't depend on labels, only
    on window count -- so folds are computed once by the caller and reused
    here). Returns the list of per-fold (ensemble_auc, floor_auc, report)
    results, same shape run_fold's callers already handle."""
    print(f"\n{'#' * 64}\n# horizon = {horizon_days:.0f} days\n{'#' * 64}")
    labels = label_hours(hourly_index, major_times, horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, feature_cols, features, labels, dsp, hourly_index,
                          train_idx, val_idx, test_idx, seeds, device, horizon_days=horizon_days)
        if result is not None:
            results.append(result)

    if args.cv_folds <= 1:
        if results:
            print(f"\n  [!] Single station, ~10 months, {len(seeds)}-seed ensemble -- treat as a first look, "
                 "not a settled result (report.md 6.6).")
    elif results:
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
    """Loads features/catalog, builds hourly labels, and runs the fold/horizon sweep."""
    args = parse_args()

    print("=" * 64)
    print("Loading hand-crafted continuous features and building hourly labels...")
    hourly = load_hourly_features(args.features_csv)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hourly)} hourly feature vectors, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    dsp = days_since_prev_major(hourly.index, major_times)
    feature_cols = list(hourly.columns)
    features = hourly[feature_cols].to_numpy(dtype=np.float64)

    n = len(hourly)
    valid_end_indices = np.arange(args.seq_hours - 1, n)

    # A window ending at index e covers [e-seq_hours+1, e], so with no gap the
    # first val/test window right after a split boundary shares up to
    # seq_hours-1 hours of *input* with the last training window. embargo
    # removes that overlap (see walk_forward_splits' docstring).
    embargo = args.seq_hours - 1

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train], valid_end_indices[i_train + embargo:i_val],
                 valid_end_indices[i_val + embargo:])]
        fold_labels = ["single split"]
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

    horizons = ([float(h) for h in args.horizons.split(",")] if args.horizons
               else [args.horizon_days])

    per_horizon = {}
    for horizon_days in horizons:
        per_horizon[horizon_days] = run_horizon(horizon_days, args, hourly.index, feature_cols,
                                                features, major_times, dsp, folds, fold_labels,
                                                skip, seeds, device)

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
