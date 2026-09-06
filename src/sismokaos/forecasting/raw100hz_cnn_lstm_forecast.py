"""Same setup as raw_cnn_lstm_forecast.py (KO station, M>=4.5 dense forecast
target, walk-forward CV) but reads the gap-only preprocessing output (the producing script is not in
this repo and could not be located in 2026-09; treat the input format as the
contract, not that path)
instead of the real pipeline's: gaps filled/masked, no bandpass filter, no
5Hz decimation -- native rate (100Hz), 20x more samples per hour.

raw_cnn_lstm_forecast.py already tests whether a CNN learns a better
representation than hand-crafted features; this asks the same question one
level down -- does it also do better learning its own frequency selectivity
from close-to-raw signal than being handed a fixed 0.1-2Hz/5Hz choice made
ahead of time? Everything except the per-hour encoder and the input's sample
count is identical to (and imported from) raw_cnn_lstm_forecast.py, so a
result here is a fair apples-to-apples comparison against that script's.

Usage:
    python raw100hz_cnn_lstm_forecast.py \\
        --data-root ../../Sismokaos-featureExtract/data/aegean_bodt_2024_2026_gaponly \\
        --catalog-path ../../data_downloader/catalogs/catalog_current.csv
"""

import numpy as np
import torch
import torch.nn as nn

from sismokaos.forecasting.raw_cnn_lstm_forecast import parse_args, run_horizon
from sismokaos.catalog import (days_since_prev_major, label_hours,
                               load_aegean_events)
from sismokaos.model.sequence import SequenceHeadNet
# NativeWaveformEncoder moved next to its sibling RawWaveformEncoder in
# sismokaos.waveform, so sismokaos.model.registry can build
# `--model sequence-head --model-branch native-100hz` without importing this
# trainer. Re-exported so existing callers of this module keep working.
from sismokaos.waveform import NativeWaveformEncoder  # noqa: F401
from sismokaos.splits import walk_forward_splits
from sismokaos.waveform import load_hourly_raw, load_hourly_raw_consolidated

NATIVE_HOUR_SAMPLES = 3600 * 100  # 3600s * 100Hz, vs. raw_cnn_lstm_forecast.py's 3600*5Hz


class NativeCNNLSTM(SequenceHeadNet):
    """`SequenceHeadNet` with a `NativeWaveformEncoder` embedding each
    hour's native-rate raw waveform before the LSTM branch sees it."""

    def __init__(self, cnn_out=32, hidden=16, dropout=0.5):
        """See `SequenceHeadNet.__init__`; `encoder` is always a
        `NativeWaveformEncoder(out_dim=cnn_out, dropout=dropout)` here.

        Args:
            cnn_out: Width of the per-hour CNN embedding (`feat_dim` for the
                LSTM branch).
            hidden: LSTM hidden size (per direction) and head hidden width.
            dropout: Dropout used throughout the encoder, branch, and head.
        """
        super().__init__(cnn_out, hidden=hidden, dropout=dropout,
                         encoder=NativeWaveformEncoder(out_dim=cnn_out, dropout=dropout))


def main():
    """Loads the native-rate waveform archive/catalog, builds hourly labels,
    and runs the fold/horizon sweep using `NativeCNNLSTM`."""
    args = parse_args()

    print("Loading native-rate (gap-only) waveform and building hourly labels...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, hour_samples=NATIVE_HOUR_SAMPLES,
                                          max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    dsp = days_since_prev_major(hour_index, major_times)

    n = len(hour_index)
    valid_end_indices = np.arange(args.seq_hours - 1, n)

    horizons = ([float(h) for h in args.horizons.split(",")] if args.horizons
               else [args.horizon_days])

    # A window ending at index e covers raw hours [e-seq_hours+1, e], so with no
    # gap the first val/test window right after a split boundary shares up to
    # seq_hours-1 hours of *input* with the last training window. embargo removes
    # that overlap (see walk_forward_splits' docstring).
    embargo = args.seq_hours - 1

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
                                                dsp, folds, fold_labels, skip, seeds, device,
                                                model_cls=NativeCNNLSTM)

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
