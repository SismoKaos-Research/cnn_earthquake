"""One entry point for everything `sismokaos-cli` writes.

The Rust preprocessor emits two things, and they need different handling:

- **Engineered features** (`run`) -- a Parquet table, one row per window.
- **Decimated waveform** (`preprocess`) -- a flat little-endian `f32` stream
  plus a `.f32.json` sidecar.

`RustData.open()` loads either or both and, when both are present, aligns them
onto one hourly index so row `i` of the feature table and hour `i` of the
waveform are the same hour. Getting that alignment wrong is silent -- the
shapes still match, the model still trains, the labels are just attached to
the wrong data -- so it is done here once rather than at each call site.

Two subtleties this class exists to absorb:

**Time is stored differently in the two files.** The feature Parquet carries
`Zaman_Dk`, minutes since the Unix epoch, at the *end* of each window. The
waveform sidecar carries a segment table because the CLI restarts its clock at
every data gap, so `start_epoch + i/fs` is only valid within a segment.

**Hours are not guaranteed contiguous.** Real archives have gaps. Alignment is
by timestamp, never by position, and hours present in one source but not the
other are dropped from both with a count reported.

Usage:

    from seismolib.rust_io import RustData

    d = RustData.open(features="aegean_bodt_features.parquet")
    d = RustData.open(raw="aegean_bodt_preprocessed.f32")
    d = RustData.open(features=..., raw=...)     # aligned on the shared hours

    d.hour_index      # pandas DatetimeIndex, one entry per usable hour
    d.features        # (hours, n_features) float32, or None
    d.waveform        # (hours, 3, hour_samples) float32 memmap, or None
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class RustData:
    """Aligned view over the CLI's feature table and/or waveform stream.

    Construct with `RustData.open`, not directly.

    Attributes:
        hour_index: `DatetimeIndex` of the hours held, ascending and unique.
        features: `(hours, n_features)` float32 array, or None.
        feature_names: Column names matching `features`, or None.
        waveform: `(hours, 3, hour_samples)` float32 memmap, or None.
        fs: Waveform sample rate, or None.
        meta: The waveform sidecar dict, or None.
    """

    def __init__(self, hour_index, features, feature_names, waveform, fs, meta):
        self.hour_index = hour_index
        self.features = features
        self.feature_names = feature_names
        self.waveform = waveform
        self.fs = fs
        self.meta = meta

    # -- loading ---------------------------------------------------------

    @classmethod
    def open(cls, features=None, raw=None, columns=None, hour_samples=None):
        """Loads one or both outputs and aligns them.

        Args:
            features: Path to a feature Parquet. Accepts both the CLI's own
                output (a `Zaman_Dk` column) and a table already carrying a
                `DatetimeIndex`.
            raw: Path to a `.f32` stream. Its `.f32.json` sidecar must sit
                beside it.
            columns: Feature columns to keep. Defaults to every numeric column
                that is not a time or id key.
            hour_samples: Samples per hour in the waveform. Defaults to
                `round(fs * 3600)` from the sidecar, which is right unless the
                stream was written with a different convention.

        Returns:
            A `RustData`.

        Raises:
            ValueError: If neither source is given, or if they share no hours.
        """
        if features is None and raw is None:
            raise ValueError("Give at least one of features= or raw=.")

        feat_df = cls._load_features(features, columns) if features else None
        wave, w_hours, fs, meta = (
            cls._load_waveform(raw, hour_samples) if raw else (None, None, None, None))

        if feat_df is not None and wave is not None:
            shared = feat_df.index.intersection(w_hours)
            if len(shared) == 0:
                raise ValueError(
                    f"Feature hours ({feat_df.index[0]} .. {feat_df.index[-1]}) and "
                    f"waveform hours ({w_hours[0]} .. {w_hours[-1]}) do not overlap.")
            dropped_f = len(feat_df.index) - len(shared)
            dropped_w = len(w_hours) - len(shared)
            if dropped_f or dropped_w:
                print(f"  [rust_io] aligned on {len(shared)} shared hours "
                      f"(dropped {dropped_f} feature-only, {dropped_w} waveform-only)")
            # Positional lookup, so the memmap is sliced rather than copied.
            w_pos = pd.Series(np.arange(len(w_hours)), index=w_hours).loc[shared].to_numpy()
            feat_df = feat_df.loc[shared]
            wave = _HourView(wave, w_pos)
            hour_index = shared
        elif feat_df is not None:
            hour_index = feat_df.index
        else:
            hour_index = w_hours

        return cls(
            hour_index=hour_index,
            features=(feat_df.to_numpy(dtype=np.float32) if feat_df is not None else None),
            feature_names=(list(feat_df.columns) if feat_df is not None else None),
            waveform=wave,
            fs=fs,
            meta=meta,
        )

    @staticmethod
    def _load_features(path, columns):
        """Feature Parquet -> DataFrame indexed by hour, ascending and unique."""
        df = pd.read_parquet(path)

        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        elif "Zaman_Dk" in df.columns:
            # Minutes since the Unix epoch, at the END of each window.
            idx = pd.to_datetime(df["Zaman_Dk"], unit="m")
        else:
            raise ValueError(
                f"{path} has neither a DatetimeIndex nor a Zaman_Dk column, so its "
                f"rows cannot be placed in time.")

        drop = {"Pencere_ID", "Zaman_Dk", "index", "hour_start"}
        keep = columns if columns else [
            c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing requested columns: {missing}")

        out = df[keep].copy()
        out.index = idx.floor("h")
        # A window per hour is the assumption downstream; if the CLI ran with a
        # sub-hourly step, collapse to the last window in each hour rather than
        # silently emitting duplicate hours.
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    @staticmethod
    def _load_waveform(path, hour_samples):
        """`.f32` + sidecar -> `(hours, 3, hour_samples)` memmap and its hours."""
        path = Path(path)
        meta_path = path.with_suffix(".f32.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. The .f32 stream does not carry its own "
                f"sample rate or segment table and cannot be timed without it.")
        meta = json.loads(meta_path.read_text())
        fs = float(meta["fs"])
        hour_samples = int(hour_samples or round(fs * 3600))

        flat = np.memmap(path, dtype="<f4", mode="r").reshape(-1, 3)
        if flat.shape[0] != meta["samples"]:
            raise ValueError(
                f"{path} holds {flat.shape[0]} samples but the sidecar claims "
                f"{meta['samples']}; one of them is stale.")

        n_hours = flat.shape[0] // hour_samples
        if n_hours == 0:
            raise ValueError(
                f"{path} holds {flat.shape[0]} samples, under one "
                f"{hour_samples}-sample hour.")

        # (hours, samples, ch) -> (hours, ch, samples); a view, still a memmap.
        block = flat[:n_hours * hour_samples].reshape(n_hours, hour_samples, 3)
        wave = block.transpose(0, 2, 1)

        starts = _segment_times(meta, np.arange(n_hours) * hour_samples)
        hours = pd.to_datetime(starts, unit="s").floor("h")
        hours = pd.DatetimeIndex(hours)

        # A gap can put two blocks in the same clock hour; keep the first and
        # drop the rest rather than emitting a duplicated index.
        dup = hours.duplicated(keep="first")
        if dup.any():
            print(f"  [rust_io] dropped {int(dup.sum())} waveform block(s) that "
                  f"landed in an already-used hour (data gaps)")
            wave = _HourView(wave, np.flatnonzero(~dup))
            hours = hours[~dup]

        order = np.argsort(hours.values)
        if not (np.diff(order) == 1).all():
            wave = _HourView(wave, order)
            hours = hours[order]

        return wave, hours, fs, meta


def _segment_times(meta, sample_indices):
    """Epoch seconds for sample indices, honouring the sidecar's segments.

    The time axis is piecewise linear -- the CLI restarts its clock at each
    data gap -- so this resolves each index against the last segment beginning
    at or before it, rather than assuming one global start.
    """
    segs = meta["segments"]
    offsets = np.array([s["sample_offset"] for s in segs], dtype=np.int64)
    epochs = np.array([s["start_epoch"] for s in segs], dtype=np.float64)
    which = np.clip(np.searchsorted(offsets, sample_indices, side="right") - 1,
                    0, len(segs) - 1)
    return epochs[which] + (sample_indices - offsets[which]) / float(meta["fs"])


class _HourView:
    """Lazy row selection over a memmap.

    `wave[rows]` with a fancy index would materialise the whole selection in
    RAM, which for a multi-year archive is tens of gigabytes. This keeps the
    memmap and translates indices on access, so only the hours a DataLoader
    actually touches are ever paged in.
    """

    __slots__ = ("_base", "_rows")

    def __init__(self, base, rows):
        # Collapse nested views so repeated alignment does not build a chain.
        if isinstance(base, _HourView):
            rows = base._rows[rows]
            base = base._base
        self._base = base
        self._rows = np.asarray(rows, dtype=np.int64)

    def __len__(self):
        return len(self._rows)

    @property
    def shape(self):
        return (len(self._rows),) + tuple(self._base.shape[1:])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _HourView(self._base, self._rows[key])
        if isinstance(key, (int, np.integer)):
            return self._base[self._rows[key]]
        return np.stack([self._base[r] for r in self._rows[np.asarray(key)]])

    def __array__(self, dtype=None):
        out = np.stack([self._base[r] for r in self._rows])
        return out.astype(dtype) if dtype else out
