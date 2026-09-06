"""Hourly raw-waveform loading and the 1D CNN encoder over it.

The archive is stored as per-day directories of per-hour arrays; these
loaders assemble them into the (hours, channels, samples) block the
sequence models consume, with a memory-mapped path for the consolidated
form."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

_DATE_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")


HOUR_SAMPLES = 36000  # 3600s * 10Hz. The 5Hz archives use 18000; pass
                      # hour_samples explicitly when reading those.


def load_raw_f32(path):
    """Memory-maps the flat f32 output of `sismokaos-cli preprocess`.

    Returns `(samples, metadata)` where `samples` is a read-only
    `(n_samples, 3)` memmap in E/N/Z order and `metadata` is the sidecar dict.

    This replaces the Parquet -> `parquet_to_memory.py` -> `.dat` round trip:
    the CLI now writes this layout directly, so there is nothing to convert.
    Nothing is read into RAM here -- the OS pages in only what gets indexed.

    Args:
        path: Path to the `.f32` file. Its `.f32.json` sidecar must sit beside
            it; the sidecar carries the sample rate and the segment table.

    Raises:
        FileNotFoundError: If either the data file or its sidecar is missing.
        ValueError: If the file length disagrees with the sidecar's sample
            count, which means one of the two is stale.
    """
    path = Path(path)
    meta_path = path.with_suffix(".f32.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Sidecar {meta_path} not found. The .f32 file alone does not carry "
            f"its sample rate or segment table and cannot be timed without it.")
    meta = json.loads(meta_path.read_text())

    a = np.memmap(path, dtype="<f4", mode="r").reshape(-1, 3)
    if a.shape[0] != meta["samples"]:
        raise ValueError(
            f"{path} holds {a.shape[0]} samples but {meta_path.name} claims "
            f"{meta['samples']}; one of them is stale.")
    return a, meta


def raw_f32_times(meta, index=None):
    """Absolute epoch seconds for samples of a `load_raw_f32` array.

    The time axis is **piecewise** linear: the CLI restarts its clock at every
    data gap, so a single start time plus `i/fs` would place every sample after
    the first gap at the wrong moment. This walks the sidecar's segment table
    instead.

    Args:
        meta: The metadata dict from `load_raw_f32`.
        index: Sample indices to time. Defaults to every sample.

    Returns:
        float64 array of Unix epoch seconds, same shape as `index`.
    """
    fs = meta["fs"]
    segs = meta["segments"]
    if index is None:
        index = np.arange(meta["samples"])
    index = np.asarray(index)

    offsets = np.array([s["sample_offset"] for s in segs])
    epochs = np.array([s["start_epoch"] for s in segs])
    # Which segment each index falls in: the last one starting at or before it.
    which = np.searchsorted(offsets, index, side="right") - 1
    which = np.clip(which, 0, len(segs) - 1)
    return epochs[which] + (index - offsets[which]) / fs


def raw_f32_to_hours(path, hour_samples=HOUR_SAMPLES):
    """Reshapes a flat `.f32` stream into `(hours, 3, hour_samples)`.

    Trailing samples that do not fill a whole hour are dropped rather than
    zero-padded, so every returned hour is real data.

    This is a view over the memmap wherever the shape allows, so it stays out
    of RAM.
    """
    a, meta = load_raw_f32(path)
    n_hours = a.shape[0] // hour_samples
    if n_hours == 0:
        raise ValueError(
            f"{path} holds {a.shape[0]} samples, fewer than one {hour_samples}-sample hour.")
    trimmed = a[: n_hours * hour_samples]
    # (hours, samples, channels) -> (hours, channels, samples)
    return trimmed.reshape(n_hours, hour_samples, 3).transpose(0, 2, 1), meta


def load_really_long_csv(csv_path: str, chunksize: int = 100_000) -> pd.DataFrame:
    """Loads a really long CSV file in memory-efficient chunks.

    Designed for massive Rust feature outputs or raw exports where loading the 
    entire file at once would risk an Out-Of-Memory (OOM) crash.
    """
    print(f"  [streaming] Reading massive CSV in chunks of {chunksize:,} rows: {csv_path}")
    chunks = []
    
    # Iterate through the CSV in chunks to keep RAM usage strictly bounded
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        # Vectorized absolute time assignment using Zaman_Dk minutes (Unix Epoch)
        if "Zaman_Dk" in chunk.columns:
            exact_times = pd.to_datetime(chunk["Zaman_Dk"], unit="m")
            chunk = chunk.copy().assign(hour_start=exact_times.dt.floor("h"))
        chunks.append(chunk)
        
    df = pd.concat(chunks, ignore_index=True)
    
    # If hour_start was successfully created, aggregate to hourly means
    if "hour_start" in df.columns:
        feature_cols = [c for c in df.columns if c not in ("Pencere_ID", "Zaman_Dk", "hour_start", "index")]
        hourly = df.groupby("hour_start")[feature_cols].mean().sort_index()
        return hourly
    return df


def load_hourly_raw(data_root: str, hour_samples: int = HOUR_SAMPLES, max_days: int = None):
    """Loads every hour's raw waveform .npy file into one in-RAM array."""
    root = Path(data_root)
    
    # If the user passed a really long CSV instead of a directory, delegate to CSV loader
    if root.is_file() or str(data_root).endswith(".csv"):
        return load_really_long_csv(str(data_root))

    date_dirs = sorted(d for d in root.iterdir() if d.is_dir() and _DATE_DIR_RE.match(d.name))
    if max_days is not None:
        date_dirs = date_dirs[:max_days]

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
    """Loads a directory built by consolidate_hourly_raw.py."""
    d = Path(consolidated_dir)
    hours = np.load(d / "hours.npy")
    hour_index = pd.DatetimeIndex(pd.to_datetime(hours, unit="s"))
    raw = np.load(d / "raw.npy", mmap_mode="r" if mmap else None)
    return hour_index, raw


class RawSeqDataset(Dataset):
    """Windows of `seq_hours` consecutive hourly raw waveforms or features."""

    def __init__(self, raw, labels: np.ndarray, seq_hours: int, indices: np.ndarray, stats=None):
        self.raw = raw
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
        
        # Support both 3D raw arrays (n_hours, 3, samples) and 2D feature DataFrames/arrays (n_hours, n_features)
        self.is_tabular = isinstance(raw, pd.DataFrame) or (isinstance(raw, np.ndarray) and raw.ndim == 2)
        
        if stats is None:
            stat_idx = indices[np.linspace(0, len(indices) - 1, min(500, len(indices))).astype(int)]
            if self.is_tabular:
                sub = np.concatenate([raw.iloc[max(0, i - seq_hours + 1):i + 1].to_numpy() for i in stat_idx], axis=0)
                mu, sd = np.nanmean(sub, axis=0), np.nanstd(sub, axis=0) + 1e-6
                mu = np.where(np.isfinite(mu), mu, 0.0)
                sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
                stats = (mu, sd)
            else:
                sub = np.concatenate([raw[max(0, i - seq_hours + 1):i + 1] for i in stat_idx], axis=0)
                mu = sub.mean(axis=(0, 2), keepdims=True)
                sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
                stats = (mu[0], sd[0])
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        
        if self.is_tabular:
            seq = self.raw.iloc[start:end + 1].copy().to_numpy() if isinstance(self.raw, pd.DataFrame) else self.raw[start:end + 1].copy()
            mu, sd = self.stats
            seq = (seq - mu) / sd
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            seq = self.raw[start:end + 1]
            mu, sd = self.stats
            seq = (seq - mu[None]) / sd[None]
            
        return (torch.from_numpy(seq).float(),
                torch.tensor(self.labels[end], dtype=torch.float32))


class RawWaveformEncoder(nn.Module):
    """1D CNN that embeds one hour's raw 3-component waveform."""

    def __init__(self, out_dim=32, dropout=0.3):
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
        return self.net(x).squeeze(-1)


class NativeWaveformEncoder(nn.Module):
    """1D CNN over one hour's native-rate (100Hz, unfiltered) raw waveform.

    raw_cnn_lstm_forecast.py's RawWaveformEncoder reaches its pre-pool
    resolution (18000 -> ~70) in 4 stride-4 blocks; naively reusing those
    same strides on a 20x longer (100Hz) input would leave ~1400 samples
    before pooling and cost ~20x the compute in the first conv layer alone.
    Adds a stage and front-loads a bigger first stride (8 instead of 4) so
    the input is cut down early, landing at a comparable ~75 pre-pool
    resolution instead of paying for 20x the length all the way through."""

    def __init__(self, out_dim=32, dropout=0.3):
        """Initializes the 5-stage strided 1D CNN.

        Args:
            out_dim: Width of the embedding this encoder produces per hour.
            dropout: Dropout used after each conv block.
        """
        super().__init__()

        def block(cin, cout, k, s):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                                 nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(dropout))

        self.net = nn.Sequential(
            block(3, 16, 15, 8),
            block(16, 32, 9, 6),
            block(32, 32, 5, 5),
            block(32, 48, 5, 5),
            block(48, out_dim, 3, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        """Embeds one batch of hourly native-rate waveforms.

        Args:
            x: Input batch, shape (batch, 3, NATIVE_HOUR_SAMPLES).

        Returns:
            Tensor of shape (batch, out_dim).
        """
        return self.net(x).squeeze(-1)
