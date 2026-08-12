"""
CNN+LSTM classifier over CURATED chunks: reuses the exact architecture and
training loop from raw_cnn_lstm_forecast.py (RawCNNLSTM, train_one_seed,
run_fold, RawSeqDataset -- all imported unchanged), but replaces its dense
"every hour is a training example" sampling with a deliberately balanced,
non-overlapping selection of two anchor types:

  - "pre-event" chunks: the `--seq-hours` hours immediately before a
    qualifying event (one per event, deduplicated so overlapping events
    collapse to a single window -- a swarm's several events within
    `seq_hours` of each other only contribute one training example, not one
    per event).
  - "not pre-event" chunks: randomly sampled elsewhere, kept at least
    `--seq-hours` away from every pre-event window (and from each other),
    same count as positives -- balanced 1:1 by construction, unlike the
    dense scripts where positive rate is whatever the raw catalog gives you.

This is the same underlying question raw_cnn_lstm_forecast.py already asks
(will an event occur soon after this window), just resampled: fewer, curated
examples instead of every hour, motivated by this session's finding that raw
dense sampling produces wildly different train/test positive rates across
walk-forward folds (a real, ~7-month sustained swarm dominating some test
blocks). Curated 1:1 sampling can't fix a swarm being temporally clustered,
but it does mean every fold sees roughly the same class balance regardless
of where its boundary falls, since positive and negative *counts* -- not
just their underlying calendar rate -- are controlled directly.

`--seq-hours` sets the chunk length -- 720 = 1 month, 168 = 1 week, whatever.
Every other flag is raw_cnn_lstm_forecast.py's own parse_args(), reused
unchanged (this script imports it directly rather than redefining it).

Usage:
    python cnn_lstm_pre_event_classify.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --seq-hours 720 --cv-folds 2

Not imported by anything else -- standalone script.
"""

import numpy as np
import torch

from feature_lstm_forecast import (days_since_prev_major, load_aegean_events,
                                   walk_forward_splits)
from raw_cnn_lstm_forecast import (load_hourly_raw, load_hourly_raw_consolidated,
                                   parse_args, run_fold)


def build_curated_anchors(hour_index, major_times, chunk_hours, seed=0):
    """Selects a balanced, non-overlapping set of pre-event / not-pre-event
    window end-indices.

    Args:
        hour_index: DatetimeIndex of hour starts, one per raw hour.
        major_times: Sorted array of qualifying event times.
        chunk_hours: Window length in hours (`--seq-hours`).
        seed: Random seed for negative sampling.

    Returns:
        Tuple of (end_indices int array sorted ascending, n_positive int) --
        `end_indices[:n_positive]` sorted-mixed-in are positives; use the
        returned full-length `labels` array (see `main`) to tell which is
        which, this function only picks the anchors.
    """
    t = hour_index.to_numpy()
    n = len(t)

    event_hour_idx = np.unique(np.searchsorted(t, major_times))
    event_hour_idx = event_hour_idx[(event_hour_idx >= chunk_hours) & (event_hour_idx < n)]

    pos_ends = []
    last = -10**9
    for e in sorted(event_hour_idx):
        if e - last >= chunk_hours:
            pos_ends.append(int(e) - 1)
            last = e
    pos_ends = np.array(sorted(pos_ends))

    excluded = np.zeros(n, dtype=bool)
    for pe in pos_ends:
        lo = max(0, pe - chunk_hours + 1 - chunk_hours)
        hi = min(n, pe + chunk_hours + 1)
        excluded[lo:hi] = True

    rng = np.random.default_rng(seed)
    candidates = [e for e in range(chunk_hours - 1, n) if not excluded[e]]
    rng.shuffle(candidates)

    neg_ends = []
    for c in candidates:
        if len(neg_ends) >= len(pos_ends):
            break
        if all(abs(c - x) >= chunk_hours for x in neg_ends):
            neg_ends.append(c)
    neg_ends = np.array(sorted(neg_ends))

    return pos_ends, neg_ends


def main():
    """Loads the raw waveform archive/catalog, builds curated pre-event
    anchors, and runs the fold sweep using raw_cnn_lstm_forecast.py's
    unmodified training/reporting machinery."""
    args = parse_args()

    print("Loading raw preprocessed waveform and building curated pre-event anchors...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    dsp = days_since_prev_major(hour_index, major_times)

    pos_ends, neg_ends = build_curated_anchors(hour_index, major_times, args.seq_hours)
    print(f"  {len(pos_ends)} pre-event chunks (deduplicated, non-overlapping), "
         f"{len(neg_ends)} not-pre-event chunks (balanced, {args.seq_hours}h each)")
    if len(pos_ends) < 10:
        print(f"[ERROR] Only {len(pos_ends)} distinct pre-event windows at seq-hours="
             f"{args.seq_hours} -- too few events are that far apart. Use a shorter --seq-hours.")
        return

    n = len(hour_index)
    labels = np.zeros(n, dtype=np.int64)
    labels[pos_ends] = 1
    valid_end_indices = np.sort(np.concatenate([pos_ends, neg_ends]))

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train], valid_end_indices[i_train:i_val],
                 valid_end_indices[i_val:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_end_indices, args.cv_folds)
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
        result = run_fold(fold_label, args, raw, labels, dsp, hour_index,
                          train_idx, val_idx, test_idx, seeds, device)
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
