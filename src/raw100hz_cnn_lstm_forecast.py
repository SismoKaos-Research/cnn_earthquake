"""Same setup as raw_cnn_lstm_forecast.py (KO station, M>=4.5 dense forecast
target, walk-forward CV) but reads scripts/gap_only_preprocess.py's output
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
        --catalog-path ../../data_downloader/catalogs/data_large.csv
"""

import numpy as np
import torch
import torch.nn as nn

from seismolib.catalog import days_since_prev_major, label_hours, load_aegean_events
from seismolib.splits import walk_forward_splits
from seismolib.model.sequence import SequenceHeadNet
from raw_cnn_lstm_forecast import parse_args, run_horizon
from seismolib.waveform import load_hourly_raw, load_hourly_raw_consolidated

NATIVE_HOUR_SAMPLES = 3600 * 100  # 3600s * 100Hz, vs. raw_cnn_lstm_forecast.py's 3600*5Hz


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
