"""Sliding-window dataset pairing hourly catalog features with raw waveform.

Feeds `forecasting/gru_cnn.py`. One sample is `seq_len` consecutive hours of
catalog features (and optionally the raw waveform for those same hours),
labelled with whether a qualifying event follows the window's **last** hour
within the forecast horizon.

**Samples are addressed by their END index**, not their start. Every split in
this repo is expressed as a set of valid end indices -- that is what
`sismokaos.splits.walk_forward_splits` returns and what the embargo is measured
in -- so indexing the same way lets a fold be built by handing this class its
own index array, with no arithmetic in between to get wrong.

**Normalisation is fitted by the caller, never here.** An earlier version took
the mean and standard deviation over every row it was given, including the
rows that would become validation and test. That is the same leak that
invalidated an AUC of 0.6670 elsewhere in this project: the scaler carries
information about the future into training. `fit_normalizer` now runs on the
training slice alone and the result is passed to every fold.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# The three catalog features that survived LightGBM RFE. Kept as a named
# constant because the model's `cat_dim` has to agree with it.
OPTIMIZED_CATALOG_FEATURES = [
    "cat_moment_7d",
    "cat_rate_30d",
    "cat_rate_90d",
]


def fit_normalizer(cat_array, train_end_indices, seq_len):
    """Mean and standard deviation over the training rows only.

    Args:
        cat_array: `(hours, n_features)` float array, unnormalised.
        train_end_indices: End indices of the training windows.
        seq_len: Window length, so the rows those windows actually read are
            included -- a window ending at `e` reads back to `e - seq_len + 1`.

    Returns:
        `(mean, std)`, each `(n_features,)` float32. `std` is floored away from
        zero so a constant feature cannot produce infinities.
    """
    if len(train_end_indices) == 0:
        raise ValueError("Cannot fit a normalizer on an empty training split.")
    lo = max(0, int(np.min(train_end_indices)) - seq_len + 1)
    hi = int(np.max(train_end_indices)) + 1
    rows = cat_array[lo:hi]
    mean = rows.mean(axis=0).astype(np.float32)
    std = (rows.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


class SeismicFusionDataset(Dataset):
    """Windows of catalog features, optionally paired with raw waveform.

    Args:
        cat_array: `(hours, n_features)` float array of catalog features,
            **unnormalised**; this class applies `mean`/`std`.
        labels: `(hours,)` binary array, already forward-looking -- `labels[i]`
            means "an event follows hour `i` within the horizon".
        end_indices: Which windows this dataset holds, by end index. Any index
            below `seq_len - 1` is dropped, since the window would run off the
            start of the array.
        seq_len: Hours per window.
        mean: Per-feature means from `fit_normalizer`.
        std: Per-feature standard deviations from `fit_normalizer`.
        waveform: Optional `(hours, 3, hour_samples)` array, typically a
            memmap from `sismokaos.waveform.raw_f32_to_hours`. Indexed lazily
            per item so the archive is never pulled into RAM.

    Yields:
        `(cat_seq, wave_seq, label)`. `wave_seq` is an empty tensor when no
        waveform was supplied -- the default `collate_fn` needs a tensor, not
        `None`, and the model ignores it when `use_waveform=False`.
    """

    def __init__(self, cat_array, labels, end_indices, seq_len, mean, std,
                 waveform=None):
        self.seq_len = int(seq_len)
        normalized = (np.asarray(cat_array, dtype=np.float32) - mean) / std
        self.cat = torch.from_numpy(np.ascontiguousarray(normalized))
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32)

        end_indices = np.asarray(end_indices, dtype=np.int64)
        keep = end_indices >= self.seq_len - 1
        if keep.sum() < len(end_indices):
            dropped = len(end_indices) - int(keep.sum())
            print(f"    [dataset] dropped {dropped} window(s) that would start "
                  f"before hour 0")
        self.end_indices = end_indices[keep]

        self.waveform = waveform
        if waveform is not None and len(waveform) < len(self.cat):
            raise ValueError(
                f"waveform covers {len(waveform)} hours but the catalog covers "
                f"{len(self.cat)}; they must be aligned hour-for-hour.")

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, i):
        end = int(self.end_indices[i])
        start = end - self.seq_len + 1
        cat_seq = self.cat[start:end + 1]

        if self.waveform is None:
            wave_seq = torch.empty(0)
        else:
            block = np.asarray(self.waveform[start:end + 1], dtype=np.float32)
            wave_seq = torch.from_numpy(block)

        return cat_seq, wave_seq, self.labels[end]

    def positive_rate(self):
        """Fraction of this split's windows that are positive.

        Used for the loss's `pos_weight`. Computed from the windows actually
        held here, so a fold cannot inherit the whole dataset's balance.
        """
        if len(self.end_indices) == 0:
            return 0.0
        return float(self.labels[self.end_indices].mean())
