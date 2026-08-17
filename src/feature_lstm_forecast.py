"""
LSTM forecaster trained directly on Sismokaos-featureExtract's hand-crafted
continuous features (STA/LTA, Hjorth, permutation entropy, spectral
centroid, cross-axis correlation -- computed on real KO.GEDZ continuous
data, Aegean zone).

Features:
- Robust absolute-time parsing via Unix Epoch (Zaman_Dk) for gap preservation.
- Walk-forward CV for non-stationary timeline evaluation.
- Dual-logging (Terminal + File).
- Real-time epoch tracking.

Usage:
    python feature_lstm_forecast.py \
        --features-csv ../../Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy \
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \
        --cv-folds 5
"""

import argparse
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from metrics import binary_report, print_report, safe_auc  # noqa: F401
from model.sequence import SequenceHeadNet
from training import seed_everything

AEGEAN_BBOX = (36.0, 40.0, 25.0, 30.0)  # lat0, lat1, lon0, lon1

STATION_COORDS = {"BODT": (37.0622, 27.3103), "DAT": (36.7308, 27.5767)}


class DualLogger:
    """Intercepts sys.stdout to print to both the terminal and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    r = 6371.0
    la1, lo1 = np.radians(lat0), np.radians(lon0)
    la2, lo2 = np.radians(lats), np.radians(lons)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def station_distance_mask(lats: np.ndarray, lons: np.ndarray, stations, max_dist_km: float):
    """Boolean mask for events within `max_dist_km` of the NEAREST named station."""
    if not max_dist_km or max_dist_km <= 0:
        return np.ones(len(lats), dtype=bool)
    best = None
    for s in stations:
        lat, lon = STATION_COORDS[s] if isinstance(s, str) else s
        d = haversine_km(lat, lon, lats, lons)
        best = d if best is None else np.minimum(best, d)
    return best <= max_dist_km


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
    
    Relies on Zaman_Dk representing absolute UTC minutes since the 1970 Unix Epoch
    (exported by the updated Rust engine). This guarantees time alignment even 
    if the sensor went offline for days at a time.
    """
    if str(features_csv).endswith(".npy"):
        df = pd.DataFrame.from_records(np.load(features_csv, allow_pickle=False))
    else:
        df = pd.read_csv(features_csv)

    # Vectorized absolute time assignment using Zaman_Dk minutes
    exact_times = pd.to_datetime(df["Zaman_Dk"], unit="m")
    
    # .copy() prevents Pandas fragmentation warnings before assigning the new column
    # "h" is lowercase to comply with Pandas 2.2+ frequency alias deprecations
    df = df.copy().assign(hour_start=exact_times.dt.floor("h"))
    
    feature_cols = [c for c in df.columns if c not in ("Pencere_ID", "Zaman_Dk", "hour_start", "index")]
    hourly = df.groupby("hour_start")[feature_cols].mean().sort_index()
    return hourly


def load_aegean_events(catalog_path: str, min_magnitude: float = 4.5,
                       stations=None, max_dist_km: float = None) -> np.ndarray:
    """Loads catalog events within the Aegean bounding box at or above a magnitude."""
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                            aegean.Longitude.to_numpy(),
                                            stations, max_dist_km)]
    return np.sort(aegean.dt.to_numpy())


def load_aegean_events_with_magnitude(catalog_path: str, min_magnitude: float = 3.0,
                                      stations=None, max_dist_km: float = None):
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
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                              aegean.Longitude.to_numpy(),
                                              stations, max_dist_km)]
    return aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64)


def load_aegean_events_with_location(catalog_path: str, min_magnitude: float = 3.0,
                                     stations=None, max_dist_km: float = None):
    """Loads catalog events (times, magnitudes, AND lat/lon) within the Aegean bbox.

    Companion to `load_aegean_events_with_magnitude`, adding coordinates for
    features that need event location -- nearest-neighbour distance
    (Zaliapin & Ben-Zion) and spatial Shannon entropy, both from Convertito
    et al. 2024 (Sci. Rep. 14:2964).

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include (completeness
            threshold for the returned "background" catalog).

    Returns:
        Tuple of (times, magnitudes, lats, lons), all sorted by time, same order.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()].sort_values("dt")
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                              aegean.Longitude.to_numpy(),
                                              stations, max_dist_km)]
    return (aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64),
           aegean.Latitude.to_numpy(dtype=np.float64), aegean.Longitude.to_numpy(dtype=np.float64))


def truncate_to_reliable_catalog_end(hour_index: pd.DatetimeIndex, raw: np.ndarray,
                                     major_times: np.ndarray, buffer_days: float = 0):
    """Drops hours past the point where the catalog can no longer reliably inform labels."""
    cutoff = major_times[-1] - np.timedelta64(int(buffer_days * 24), "h")
    n_keep = int((hour_index.to_numpy() <= cutoff).sum())
    if n_keep < len(hour_index):
        print(f"  [!] catalog's last event is {major_times[-1]} -- truncating archive from "
               f"{hour_index[-1]} to {hour_index[n_keep - 1]} ({len(hour_index) - n_keep} hours "
               f"dropped, buffer={buffer_days:.0f}d) to avoid right-censoring the forward-looking "
               f"label near the archive's end.")
    return hour_index[:n_keep], raw[:n_keep]


def count_events_in_window(hourly_index: pd.DatetimeIndex, times: np.ndarray,
                           window_days: float, forward: bool) -> np.ndarray:
    """Counts events in a trailing or leading window around each hour."""
    t = hourly_index.to_numpy()
    w = np.timedelta64(int(round(window_days * 24)), "h")
    if forward:
        return (np.searchsorted(times, t + w, side="right")
                - np.searchsorted(times, t, side="right")).astype(np.int64)
    return (np.searchsorted(times, t, side="right")
            - np.searchsorted(times, t - w, side="right")).astype(np.int64)


def label_hours_rate_change(hourly_index: pd.DatetimeIndex, rate_times: np.ndarray,
                            horizon_days: float, baseline_days: float = None):
    """Labels each hour with whether seismicity RATE will increase ("variant B").

    A different forecasting target from `label_hours`: instead of "does one
    rare M>=threshold event occur in the next horizon" (whose positive class,
    at M>=4.5, is driven by a handful of events per fold -- 4 in fold 1 --
    making the effective sample size far smaller than the hour count
    suggests), this asks "will the next window contain MORE events than the
    trailing window did".

    That is a rate/acceleration forecast, which is what ETAS-family models and
    CSEP evaluation actually target, and it uses a much lower magnitude
    threshold (typically M>=3.0), so the label is driven by ~10^3 events
    instead of ~10^1. It is also the quantity Convertito et al. 2024's
    beta-statistic measures -- but as the target itself rather than as a mask
    on a rare-event label (`label_hours_beta_precursor`), which is what made
    that earlier attempt fail.

    Note the natural baseline here is strongly ANTI-correlated: during an
    aftershock sequence a high trailing rate predicts a DECREASE (Omori
    decay). Score any model against `rate_persistence_auc`, not against 0.5.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        rate_times: Sorted array of event times defining the rate (e.g.
            M>=3.0 events -- a much lower threshold than the label-defining
            `major_times` used by `label_hours`).
        horizon_days: Length of the forward window being forecast.
        baseline_days: Length of the trailing comparison window. Defaults to
            `horizon_days` (a like-for-like comparison, so the label is a
            clean "up or down" with no window-length bias).

    Returns:
        Tuple of (labels, forward_counts, trailing_counts) -- labels is an
        int64 0/1 array (1 = rate increases), the counts are returned so
        callers can build the persistence floor and report diagnostics
        without recomputing them.
    """
    if baseline_days is None:
        baseline_days = horizon_days
    fwd = count_events_in_window(hourly_index, rate_times, horizon_days, forward=True)
    bwd = count_events_in_window(hourly_index, rate_times, baseline_days, forward=False)
    return (fwd > bwd).astype(np.int64), fwd, bwd


def rate_persistence_auc(labels: np.ndarray, trailing_counts: np.ndarray) -> float:
    """AUC of the trivial trailing-rate baseline for `label_hours_rate_change`.

    The trailing count is a legitimate backward-looking predictor, but its
    relationship to a rate-INCREASE label is inverted (Omori decay: busy now
    implies calmer next). Reports the achievable baseline as
    `max(auc, 1 - auc)`, since a forecaster free to choose the sign of a
    known-anti-correlated predictor gets the flipped value for free -- so
    that, not 0.5, is the bar a model has to clear.

    Args:
        labels: 0/1 rate-increase labels.
        trailing_counts: Trailing-window event counts, same length.

    Returns:
        Baseline AUC in [0.5, 1.0], or NaN if `labels` is single-class.
    """
    auc = safe_auc(labels, trailing_counts.astype(np.float64))
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")


def label_hours(hourly_index: pd.DatetimeIndex, major_times: np.ndarray, horizon_days: float) -> np.ndarray:
    """Labels each hour with whether a qualifying event occurs within the horizon."""
    horizon = np.timedelta64(int(horizon_days), "D")
    t = hourly_index.to_numpy()
    labels = np.zeros(len(t), dtype=np.int64)
    for i, ti in enumerate(t):
        fut = major_times[(major_times > ti) & (major_times <= ti + horizon)]
        labels[i] = int(len(fut) > 0)
    return labels


def compute_beta_statistic(hourly_index: pd.DatetimeIndex, bg_times: np.ndarray,
                           recent_days: float = 7.0, baseline_days: float = 30.0) -> np.ndarray:
    """Backward-looking seismicity-rate-acceleration z-score (no leakage).

    The beta-statistic (Reasenberg & Simpson 1992; Matthews & Reasenberg) is
    the test Convertito et al. 2024 (Sci. Rep. 14:2964) use to identify
    precursor windows: compare the observed event count in a recent window
    against what a constant-rate Poisson process (calibrated on the
    immediately preceding baseline window) would predict.

        beta = (n_recent - E[n_recent]) / sqrt(E[n_recent])
        E[n_recent] = (n_baseline / baseline_days) * recent_days

    Large positive beta means the rate has significantly accelerated
    relative to the recent past; beta near/below 0 means no acceleration
    (or a slowdown). Uses the same lower-completeness-threshold background
    catalog as the other rate/regularity features (more data points than
    the rare M>=threshold events alone give).

    Args:
        hourly_index: Hour-start timestamps to score, one per sample.
        bg_times: Sorted array of lower-threshold "background" event times
            (see `load_aegean_events_with_magnitude`).
        recent_days: Length of the recent window being tested.
        baseline_days: Length of the preceding baseline window the recent
            rate is compared against.

    Returns:
        float64 array of beta values, same length as `hourly_index`; 0.0
        where the baseline window has no events (no rate to compare against).
    """
    t = hourly_index.to_numpy()
    recent_td = np.timedelta64(int(round(recent_days * 24)), "h")
    baseline_td = np.timedelta64(int(round(baseline_days * 24)), "h")
    beta = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        n_recent = np.sum((bg_times <= ti) & (bg_times > ti - recent_td))
        n_baseline = np.sum((bg_times <= ti - recent_td) & (bg_times > ti - recent_td - baseline_td))
        expected = (n_baseline / baseline_days) * recent_days
        beta[i] = (n_recent - expected) / np.sqrt(expected) if expected > 0 else 0.0
    return beta


def label_hours_beta_precursor(hourly_index: pd.DatetimeIndex, major_times: np.ndarray,
                               horizon_days: float, beta: np.ndarray,
                               beta_threshold: float = 1.645) -> np.ndarray:
    """Convertito et al. 2024-style precursor labeling.

    Narrows `label_hours`' "M>=threshold within horizon_days" positive
    class to hours that are ALSO showing a statistically significant
    seismicity-rate acceleration (see `compute_beta_statistic`) -- i.e.
    genuinely accelerating toward the event, not merely close to it in
    calendar time. Hours within the horizon of a qualifying event but
    without significant acceleration revert to label 0.

    `beta_threshold=1.645` is a one-sided z-critical value at alpha=0.05 --
    our own choice, standing in for the paper's per-sequence empirically
    tuned threshold (the paper doesn't give one universal number).

    Args:
        hourly_index: Hour-start timestamps to label, one per sample.
        major_times: Sorted array of qualifying event times.
        horizon_days: Forecast horizon in days (same semantics as `label_hours`).
        beta: Per-hour beta-statistic array from `compute_beta_statistic`.
        beta_threshold: Minimum beta to count as "significant acceleration".

    Returns:
        int64 array of 0/1 labels, same length as `hourly_index`.
    """
    base = label_hours(hourly_index, major_times, horizon_days)
    return (base & (beta > beta_threshold)).astype(np.int64)


def days_since_prev_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days elapsed since the previous qualifying event, per hour."""
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
    """Positive rate over time in `n_blocks` equal-width windows."""
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
    """Expanding-window walk-forward splits."""
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
    """Windows of `seq_hours` consecutive hourly feature vectors, z-normalized."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_hours: int,
                 indices: np.ndarray, stats=None):
        self.features = features
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
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
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        seq = self.features[start:end + 1].copy()
        mu, sd = self.stats
        seq = (seq - mu) / sd
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return (torch.from_numpy(seq).float(),
                torch.tensor(self.labels[end], dtype=torch.float32))


def parse_args():
    p = argparse.ArgumentParser(description="LSTM forecaster on hand-crafted continuous features.")
    p.add_argument("--features-csv", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--horizon-days", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=None)
    p.add_argument("--seq-hours", type=int, default=168)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    return p.parse_args()


class ForecastLSTM(SequenceHeadNet):
    """`SequenceHeadNet` with no per-step encoder."""
    def __init__(self, feat_dim, hidden=64, dropout=0.3):
        super().__init__(feat_dim, hidden=hidden, dropout=dropout)


def train_one_seed(args, seed, feature_cols, features, labels,
                   train_idx, val_idx, test_idx, device):
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
            
        print(f"  [seed {seed}] ForecastLSTM epoch {epoch+1:02d}/{args.epochs} "
               f"val AUC {val_auc:.4f} val loss {val_loss:.4f}"
               + (f" train AUC {metric:.4f}" if use_loss_fallback else ""))
               
        improved = metric > best
        if improved:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [seed {seed}] Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = evaluate(test_loader)
    print(f"  [seed {seed}] test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def run_fold(fold_label, args, feature_cols, features, labels, dsp, hourly_index,
             train_idx, val_idx, test_idx, seeds, device, horizon_days=None):
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

    print(f"\nTraining ForecastLSTM ({len(seeds)} seed(s)): {seeds}")
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
    print(f"  persistence            AUC {pers_auc:.4f}   n={len(yt_ref)}")

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
            print(f"\n  [!] Single station, ~10 months, {len(seeds)}-seed ensemble -- treat as a first look.")
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
    args = parse_args()
    
    # Setup Dual Logging
    log_filename = f"lstm_forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("=" * 64)
    print(f"Logging initialized. Saving all terminal output to: {log_filename}")
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

    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
