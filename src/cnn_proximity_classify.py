"""
CNN-only classifier: is this hour close (in either time direction) to an
M>=threshold AEGEAN event -- a regime-recognition task, not a forecast.

Every other script in this "current work" family (feature_lstm_forecast.py,
raw_cnn_lstm_forecast.py, raw100hz_cnn_lstm_forecast.py) asks the CNN+LSTM to
forecast a *future* outcome from a sequence of hours, which inherits every
non-stationarity/leakage problem discussed in this project's investigation:
train and test regimes can differ, and dense causal labeling burns hours of
walk-forward CV on a swarm-driven, hard-to-generalize target. This script
asks a narrower, easier-to-answer question first: does the raw waveform of a
*single* hour, on its own -- no LSTM, no multi-hour sequence, no forecasting
into the unknown -- carry any signal about whether that hour sits near a
qualifying event at all (before OR after it)? If a plain CNN can't pick that
up, a forecasting LSTM built on the same per-hour CNN embedding has little
reason to either; the CNN branch is architecturally identical to
RawWaveformEncoder in raw_cnn_lstm_forecast.py, just without the LSTM riding
on top and without the forward-only causal constraint on the label.

Motivated directly by this session's MMD distribution-shift diagnostic
(waveform_shift_mmd in raw_cnn_lstm_forecast.py), which found no significant
difference between label=1 and label=0 hours using hand-picked features (RMS,
kurtosis, PSD band ratio) -- even at maximum contrast, across the whole
archive. A trained CNN classifier is a stronger test of the same question,
since it learns its own features instead of the ones picked for the MMD
check.

Label: `label_proximity` marks an hour positive iff it is within
`--close-days` of a qualifying event *either* looking back (via
`days_since_prev_major`) or forward (via the new `days_until_next_major`).
Looking backward AND forward is deliberate and is not leakage here: this
script never claims to forecast anything unknown at prediction time, only to
recognize "this hour sits inside an active window" -- a fair pattern-
recognition target, unlike the forecast scripts' "will an event occur next."

No sequence windowing -- each hour is one independent training example (its
own raw (3, hour_samples) waveform), so the window-input-overlap leakage the
other scripts' `--embargo` fixes doesn't apply the same way here. A modest
`--embargo-hours` is still applied between chronological CV blocks as a
defensive measure against adjacent-hour near-duplicate leakage (immediately
adjacent hours of continuous background recording are highly autocorrelated
even without explicit windowing).

Usage:
    python cnn_proximity_classify.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --close-days 7 --cv-folds 2

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader, Dataset

from feature_lstm_forecast import (days_since_prev_major, days_until_next_major,
                                   label_hours, load_aegean_events, print_split_diagnostics,
                                   safe_auc, walk_forward_splits)
from metrics import binary_report, print_report
from raw_cnn_lstm_forecast import (RawWaveformEncoder, load_hourly_raw,
                                   load_hourly_raw_consolidated)
from training import seed_everything


def label_proximity(dsp: np.ndarray, dun: np.ndarray, close_days: float) -> np.ndarray:
    """Labels each hour 1 iff a qualifying event is within `close_days`, either direction.

    Args:
        dsp: Days since the previous qualifying event (NaN if none exists
            before this hour) -- see `days_since_prev_major`.
        dun: Days until the next qualifying event (NaN if none exists after
            this hour) -- see `days_until_next_major`.
        close_days: Distance threshold in days.

    Returns:
        int64 array of 0/1 labels, same length as `dsp`/`dun`.
    """
    dist = np.fmin(np.nan_to_num(dsp, nan=np.inf), np.nan_to_num(dun, nan=np.inf))
    return (dist <= close_days).astype(np.int64)


class HourlyRawDataset(Dataset):
    """One independent sample per hour -- that hour's raw (3, hour_samples)
    waveform, no multi-hour windowing. Per-channel z-normalized; val/test
    must reuse the train set's stats (see `RawSeqDataset`'s docstring in
    raw_cnn_lstm_forecast.py for why)."""

    def __init__(self, raw: np.ndarray, labels: np.ndarray, indices: np.ndarray, stats=None):
        """Builds the dataset.

        Args:
            raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
            labels: Per-hour binary labels, shape (n_hours,).
            indices: Hour indices this split uses.
            stats: Optional (mean, std) tuple, each shape (3, 1), to
                normalize with; if None, computed from this split's first 50
                hours.
        """
        self.raw = raw
        self.labels = labels
        self.indices = indices
        if stats is None:
            sub = np.stack([np.asarray(raw[h]) for h in indices[:50]], axis=0)
            mu = sub.mean(axis=(0, 2), keepdims=True)
            sd = sub.std(axis=(0, 2), keepdims=True) + 1e-6
            stats = (mu[0], sd[0])
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (waveform, label) for one hour.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (3, hour_samples), float32
            scalar tensor label).
        """
        h = self.indices[idx]
        x = np.asarray(self.raw[h])
        mu, sd = self.stats
        x = (x - mu) / sd
        return torch.from_numpy(x).float(), torch.tensor(self.labels[h], dtype=torch.float32)


class CNNProximityClassifier(nn.Module):
    """`RawWaveformEncoder` (same CNN as the forecast scripts' per-hour
    branch) plus a linear head -- no LSTM, no sequence."""

    def __init__(self, cnn_out=32, dropout=0.3):
        """Initializes the encoder + classification head.

        Args:
            cnn_out: Width of the CNN's per-hour embedding.
            dropout: Dropout used inside the CNN encoder.
        """
        super().__init__()
        self.encoder = RawWaveformEncoder(out_dim=cnn_out, dropout=dropout)
        self.head = nn.Linear(cnn_out, 1)

    def forward(self, x):
        """Classifies one batch of hourly waveforms.

        Args:
            x: Input batch, shape (batch, 3, hour_samples).

        Returns:
            Tensor of shape (batch,), raw logits.
        """
        return self.head(self.encoder(x)).squeeze(-1)


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--close-days", type=float, default=7.0,
                  help="An hour is labeled positive iff a qualifying event occurs within "
                       "this many days, looking either direction in time. Ignored when "
                       "--horizons is given.")
    p.add_argument("--horizons", type=str, default=None,
                  help="Comma-separated forward-only forecast horizon-day values (e.g. "
                       "7,14,30,60,120) to sweep instead of the symmetric --close-days "
                       "regime label. Each horizon uses `label_hours` (same forward-only "
                       "'will a qualifying event occur within N days' target as the other "
                       "forecast scripts), so this switches the task from regime "
                       "classification back to forecasting -- but still with the CNN-only, "
                       "single-hour, no-LSTM architecture, to see whether horizon choice "
                       "alone (independent of the LSTM/sequence machinery) finds a horizon "
                       "where the label is learnable.")
    p.add_argument("--balanced-sampling", action="store_true",
                  help="Use a WeightedRandomSampler over the training split so each batch "
                       "sees roughly equal positive/negative mass, instead of plain shuffled "
                       "sampling. Safe to do freely here since each hour is one independent "
                       "sample (no window overlap between adjacent training examples, unlike "
                       "the LSTM sequence datasets) -- resampling can't leak one hour's data "
                       "into another's input the way it could if samples were multi-hour "
                       "windows.")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--embargo-hours", type=int, default=6,
                  help="Hours of gap enforced between chronological CV blocks -- defensive "
                       "against adjacent-hour near-duplicate leakage even without explicit "
                       "sequence windowing (continuous background recording is highly "
                       "autocorrelated hour-to-hour).")
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    return p.parse_args()


def train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args.
        seed: Random seed for init/shuffling.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-hour binary proximity labels, shape (n_hours,).
        train_idx: Hour indices for the training split.
        val_idx: Hour indices for the validation split.
        test_idx: Hour indices for the test split.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_score) arrays for the test split, from the
        best (by val AUC) epoch's weights.
    """
    seed_everything(seed)
    train_ds = HourlyRawDataset(raw, labels, train_idx)
    val_ds = HourlyRawDataset(raw, labels, val_idx, stats=train_ds.stats)
    test_ds = HourlyRawDataset(raw, labels, test_idx, stats=train_ds.stats)

    model = CNNProximityClassifier(cnn_out=args.cnn_out, dropout=args.dropout).to(device)

    if args.balanced_sampling:
        train_labels_arr = labels[train_idx]
        class_counts = np.bincount(train_labels_arr, minlength=2)
        sample_weights = (1.0 / np.maximum(class_counts, 1))[train_labels_arr]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dl = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader, test_loader = dl(val_ds), dl(test_ds)

    # WeightedRandomSampler already balances what the model sees each batch --
    # pos_weight on top of that would double-correct and bias toward positive.
    if args.balanced_sampling:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos = labels[train_idx].mean()
        pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    yv0, _, _ = evaluate(val_loader)
    use_loss_fallback = len(np.unique(yv0)) < 2
    if use_loss_fallback:
        print(f"  [seed {seed}] val split is single-class -- checkpointing on train AUC instead.")

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


def run_fold(fold_label, args, raw, labels, dsp, hour_index, train_idx, val_idx, test_idx,
            seeds, device, dist_days=None, forecast_mode=False):
    """Trains the seed ensemble on one split and reports it.

    Args:
        fold_label: Header string printed above this fold's report.
        args: Parsed CLI args, forwarded to `train_one_seed`.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        labels: Per-hour binary labels, shape (n_hours,).
        dsp: Days-since-previous-major-event array, for the recency/persistence floor.
        hour_index: DatetimeIndex of hour starts, for split diagnostics.
        train_idx: Hour indices for the training split.
        val_idx: Hour indices for the validation split.
        test_idx: Hour indices for the test split.
        seeds: List of random seeds to train and ensemble.
        device: torch device to train on.
        dist_days: Distance/horizon threshold in days used for the
            recency/persistence floor. Defaults to `args.close_days`.
        forecast_mode: If True, labels `dsp`'s floor "persistence" (matching
            the other forecast scripts' terminology, since in forward-only
            mode this floor *is* persistence) instead of "recency".

    Returns:
        Tuple of (ensemble_auc, floor_auc, report_dict), or None if the
        split is too thin (fewer than 10 train or 5 test hours) to mean
        anything.
    """
    dist_days = args.close_days if dist_days is None else dist_days
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw, labels, train_idx, val_idx, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    ensemble_score = np.mean(per_seed_scores, axis=0)

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(yt_ref)}")
    # In proximity mode this is a "recency" floor: predict close iff within dist_days of the
    # PAST event alone -- fair, since dsp uses only backward-looking (already-known)
    # information, unlike the label itself which also looks forward. In forecast_mode this
    # exact same computation IS the persistence floor the other forecast scripts use.
    rec_dsp = dsp[test_idx]
    rec_pred = np.where(np.isnan(rec_dsp), 0, (rec_dsp <= dist_days).astype(int)).astype(np.float64)
    rec_auc = safe_auc(yt_ref, rec_pred)
    single_class = len(np.unique(yt_ref)) < 2
    rec_brier = float("nan") if single_class else float(brier_score_loss(yt_ref, rec_pred))
    floor_name = "persistence            " if forecast_mode else "recency (backward-only)"
    print(f"  {floor_name} AUC {rec_auc:.4f}   Brier {rec_brier:.4f}   n={len(yt_ref)}")

    print(f"\n--- CNN-only proximity classifier ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, rec_auc)
    report = binary_report(yt_ref, ensemble_score)
    bss = (float("nan") if (single_class or not np.isfinite(rec_brier) or rec_brier == 0)
          else 1.0 - report["brier"] / rec_brier)
    report["brier_skill_score_vs_recency"] = bss
    print_report(f"CNN-only proximity classifier ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def main():
    """Loads the raw waveform archive/catalog, builds proximity labels, and
    runs the fold sweep."""
    args = parse_args()

    print("Loading raw preprocessed waveform and building proximity labels...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog")

    dsp = days_since_prev_major(hour_index, major_times)
    dun = days_until_next_major(hour_index, major_times)

    forecast_mode = args.horizons is not None
    if forecast_mode:
        horizon_vals = [float(h) for h in args.horizons.split(",")]
        label_configs = []
        for h in horizon_vals:
            lab = label_hours(hour_index, major_times, h)
            print(f"  horizon={h:.0f}d hourly positive rate: {lab.mean():.3f}")
            label_configs.append((f"forecast horizon={h:.0f}d", lab, h))
    else:
        labels = label_proximity(dsp, dun, args.close_days)
        print(f"  hourly positive rate (within {args.close_days:.0f}d, either direction): {labels.mean():.3f}")
        label_configs = [(f"proximity close={args.close_days:.0f}d", labels, args.close_days)]

    n = len(hour_index)
    valid_indices = np.arange(n)

    if args.cv_folds <= 1:
        i_train = int(n * args.train_frac)
        i_val = int(n * (args.train_frac + args.val_frac))
        folds = [(valid_indices[:i_train], valid_indices[i_train + args.embargo_hours:i_val],
                 valid_indices[i_val + args.embargo_hours:])]
        fold_labels = ["single split"]
    else:
        folds = walk_forward_splits(valid_indices, args.cv_folds, embargo=args.embargo_hours)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    per_config_results = {}
    for config_label, labels, dist_days in label_configs:
        print(f"\n{'#' * 64}\n# {config_label}\n{'#' * 64}")
        results = []
        for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
            if k in skip:
                continue
            result = run_fold(fold_label, args, raw, labels, dsp, hour_index,
                              train_idx, val_idx, test_idx, seeds, device,
                              dist_days=dist_days, forecast_mode=forecast_mode)
            if result is not None:
                results.append(result)
        per_config_results[config_label] = results

        if args.cv_folds > 1 and results:
            aucs = np.array([r[0] for r in results])
            floors = np.array([r[1] for r in results])
            print(f"\n{'=' * 64}\nWalk-forward CV summary ({config_label}, {len(results)}/{args.cv_folds} "
                 f"folds)\n{'=' * 64}")
            print(f"  ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
            print(f"  ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}")
            print(f"  floor AUC:     mean {floors.mean():.4f}  std {floors.std():.4f}")
            print(f"  beats its own fold's floor in {int((aucs > floors).sum())}/{len(results)} folds")

    if len(label_configs) > 1:
        print(f"\n{'#' * 64}\n# Cross-horizon summary\n{'#' * 64}")
        print(f"  {'config':>28s}  {'ensemble AUC (mean)':>20s}  {'floor AUC (mean)':>16s}  folds")
        for config_label, results in per_config_results.items():
            if not results:
                print(f"  {config_label:>28s}  {'(no folds ran)':>20s}")
                continue
            aucs = np.array([r[0] for r in results])
            floors = np.array([r[1] for r in results])
            print(f"  {config_label:>28s}  {np.nanmean(aucs):20.4f}  {floors.mean():16.4f}  {len(results)}")


if __name__ == "__main__":
    main()
