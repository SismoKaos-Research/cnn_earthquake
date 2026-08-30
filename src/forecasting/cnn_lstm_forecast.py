"""
Dual-channel CNN + LSTM/self-attention model, retargeted onto the validated
dense per-zone forecasting target: will a M >= threshold event occur in this
fault zone within horizon_days?

**Why this script exists instead of editing `cnn_lstm.py` in place.**
`cnn_lstm.py`/`cnn_lstm_loeo.py` were built for the ABANDONED three-class
"days until the next major earthquake" target, which measured at chance
(kappa -0.028) -- see `report.md`'s catalog work. A dense reformulation of
the target ("will M >= threshold occur within horizon_days?", no
declustering, every window has a label) was later validated to carry real
signal in 2 of 4 fault zones, but only ever under a logistic-regression /
gradient-boosting scalar model (the now-retired `forecast_eval.py` and
friends) -- the dual-channel network built for exactly this task was never
retried against it. This script does that: same architecture (LSTM+attention
1D branch, CNN 2D branch over the RAM image, auxiliary physical scalars),
binary head instead of 3-class, trained and evaluated against the dataset
`seismic-cli generate-catalog-forecast-dataset` produces.

**Why `cnn_lstm_loeo.py` isn't retargeted too.** LOEO forms one fold per
distinct (region, target_time) pair -- it needs a single discrete event each
window is "about". The dense target has no such event: every window's label
is its own horizon-bounded outcome, so there is nothing to hold out one of.
The single chronological split this script uses is the only evaluation mode
that applies to a dense target, which is also why `forecast.py` (the
retired scalar version) never had a LOEO variant either.

**Floors, printed on every run** (this project's standing rule -- IP4's 70%
target is reachable by a model that has learned nothing, on a task this
imbalanced):
  * base rate  -- always predict the training period's majority class.
  * persistence -- predict positive iff a qualifying event occurred in the
    PREVIOUS horizon_days (available directly as the `days_since_prev_major`
    aux feature). This is the honest domain floor: earthquakes cluster, so
    "it happened recently, predict another" is free.

**Evaluation is block-level, not raw per-window**, via
`data_downloader/seismic_cli/forecast.py`'s `build_blocks`: consecutive
windows overlap 11-46x (stride 8 of a 64-event window), so per-window AUC
overstates confidence in its own value by up to ~7x in standard error. Block
evaluation needs the sibling repo importable; pass --data-downloader-root if
it isn't at the default relative location, or block-level results are simply
skipped and only per-window numbers are printed.

Usage:
    python cnn_lstm_forecast.py \\
        --dataset-dir ../../data_downloader/data/dataset_catalog_forecast \\
        --catalog-path ../../data_downloader/catalogs/catalog_current.csv

Also imported (not just run standalone): catalog_forecast_predict.py imports
`AUX_FEATURES`, `DenseWindowDataset`, and `DualChannelForecastNet` from this
module (same feature list, dataset loader, and architecture used to produce
predictions from a trained checkpoint), plus `safe_auc` -- which this module
does not itself define, only re-exports via its `from metrics import ...`
below.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Dataset

from seismolib.catalog import days_since_prev_major
from seismolib.metrics import (binary_report, print_report, regression_report,
                               safe_auc, safe_mcc)
from seismolib.model.dual_channel import DualChannelDualHeadNet
from seismolib.training import seed_everything

AUX_FEATURES = ["log_duration_days", "log_rate", "log_rate_recent", "rate_accel",
               "mean_mag", "max_mag", "mag_std", "log_total_energy",
               "log_energy_recent_frac", "b_value", "days_since_prev_major"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class DenseWindowDataset(Dataset):
    """
    Loads the {seq, img, aux} tensors written by
    `seismic-cli generate-catalog-forecast-dataset`.

    seq/aux are standardized with TRAIN statistics only, NaNs ignored when
    computing the stats (days_since_prev_major is NaN when no qualifying
    event precedes a window) and zero-filled after standardization -- same
    convention `cnn_lstm.py`'s CatalogWindowDataset uses, so the two datasets'
    numbers stay comparable.
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, split: str, stats=None):
        """Loads one split's manifest rows and fits (or reuses) normalization stats.

        Args:
            manifest: Full dataset manifest DataFrame (all splits).
            root: Dataset root directory (contains a subdirectory per split).
            split: Which split to load -- e.g. "train", "val", "test".
            stats: Optional (seq_mean, seq_std, aux_mean, aux_std) tuple to
                normalize with; if None, fit from this split's own data (the
                train split should always pass None; val/test must reuse the
                train split's stats).

        Raises:
            ValueError: If `split` has no rows in `manifest`.
        """
        self.rows = manifest[manifest.split == split].reset_index(drop=True)
        self.dir = Path(root) / split
        if self.rows.empty:
            raise ValueError(f"Split '{split}' is empty.")
        self.labels = self.rows.label.to_numpy(dtype=np.int64)
        self.days_since_prev_major = self.rows.days_since_prev_major.to_numpy(dtype=np.float64)
        # Magnitude of the next qualifying event, NaN where label == 0 (kept
        # as a raw array like days_since_prev_major -- it's a target, not a
        # standardized model input, so the masked loss handles the NaNs).
        self.next_magnitude = self.rows.next_magnitude.to_numpy(dtype=np.float64)

        sample = torch.load(self.dir / self.rows.filename.iloc[0], weights_only=True)
        self.seq_dim = sample["seq"].shape[-1]
        self.seq_len = sample["seq"].shape[0]
        self.img_shape = tuple(sample["img"].shape)
        self.aux_dim = sample["aux"].numel()

        if stats is None:
            seqs, auxs = [], []
            for fn in self.rows.filename:
                d = torch.load(self.dir / fn, weights_only=True)
                seqs.append(d["seq"].numpy())
                auxs.append(d["aux"].numpy())
            S = np.concatenate(seqs, axis=0)
            A = np.stack(auxs, axis=0)
            with np.errstate(invalid="ignore"):
                stats = (np.nanmean(S, 0), np.nanstd(S, 0) + 1e-6,
                        np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6)
            stats = tuple(np.where(np.isfinite(s), s, 0.0 if i % 2 == 0 else 1.0)
                         for i, s in enumerate(stats))
        self.stats = stats

    def __len__(self):
        """Returns the number of rows in this split."""
        return len(self.rows)

    def standardized_aux_matrix(self) -> np.ndarray:
        """Loads and standardizes aux for every row in this split -- used
        only for the magnitude head's ridge floor (a one-time, end-of-run
        computation), not the hot training path, which loads per-item.

        Returns:
            float64 array, shape (n_rows, aux_dim), NaN-filled entries
            replaced with 0.0.
        """
        am, asd = self.stats[2], self.stats[3]
        A = np.stack([torch.load(self.dir / fn, weights_only=True)["aux"].numpy()
                     for fn in self.rows.filename], axis=0)
        return np.nan_to_num((A - am) / asd, nan=0.0)

    def __getitem__(self, i):
        """Returns one normalized (seq, img, aux, label, next_magnitude) sample.

        Args:
            i: Row index into this split.

        Returns:
            Tuple of (float32 seq tensor, float32 img tensor, float32 aux
            tensor, float32 scalar label tensor, float32 scalar
            next_magnitude tensor -- NaN where `label` is 0).
        """
        d = torch.load(self.dir / self.rows.filename.iloc[i], weights_only=True)
        sm, ss, am, asd = self.stats
        seq = (d["seq"].numpy() - sm) / ss
        aux = (d["aux"].numpy() - am) / asd
        return (torch.from_numpy(np.nan_to_num(seq, nan=0.0)).float(),
               d["img"].float(),
               torch.from_numpy(np.nan_to_num(aux, nan=0.0)).float(),
               torch.tensor(self.labels[i], dtype=torch.float32),
               torch.tensor(self.next_magnitude[i], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model -- same branches as cnn_lstm.py, binary head
# ---------------------------------------------------------------------------

class DualChannelForecastNet(DualChannelDualHeadNet):
    """Same 1D/2D/aux architecture as `cnn_lstm.py`'s DualChannelRiskNet, with
    a shared trunk feeding two heads: `binary_out` (will M>=threshold occur
    within horizon_days -- the original, validated task) and `magnitude_out`
    (point estimate of that event's magnitude, meaningful only when the
    binary answer is yes). A shared trunk rather than two separate networks:
    most of the representation-learning is already proven for "does
    something happen," and "how big" conditional on "something happens" is
    related enough to share it, not a good reason to duplicate the backbone."""

    def __init__(self, seq_dim, img_channels, aux_dim, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all"):
        """See `DualChannelDualHeadNet.__init__`."""
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels)


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

def persistence_prediction(days_since_prev_major: np.ndarray, horizon_days: float) -> np.ndarray:
    """Predicts positive iff a qualifying event occurred in the PREVIOUS
    horizon_days. NaN (no prior qualifying event on record) predicts negative --
    there is nothing to persist from.

    Args:
        days_since_prev_major: Days since the previous qualifying event, per
            window; NaN where none precedes it.
        horizon_days: Forecast horizon in days.

    Returns:
        int64 array of 0/1 predictions, same length as
        `days_since_prev_major`.
    """
    d = days_since_prev_major
    return np.where(np.isnan(d), 0, (d <= horizon_days).astype(int)).astype(np.int64)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over rows where target is not NaN (i.e. label==1
    windows) only. Returns 0 (no gradient contribution) for a batch with no
    such rows, rather than NaN -- a real edge case per-zone even though rare
    pooled, given batch_size=64 against this task's ~0.59 pooled positive
    rate.

    Args:
        pred: Predicted magnitudes, shape (batch,).
        target: True magnitudes, shape (batch,), NaN where label == 0.

    Returns:
        Scalar tensor: the mean absolute error over non-NaN rows, or 0.0
        (no gradient contribution) if every row in the batch is NaN.
    """
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return (pred[mask] - target[mask]).abs().mean()


def report_row(name: str, y: np.ndarray, score: np.ndarray) -> dict:
    """Prints and returns one row's AUC/accuracy/MCC.

    Args:
        name: Label printed for this row (e.g. a zone or floor name).
        y: True binary labels.
        score: Predicted positive-class probability (or 0/1 prediction).

    Returns:
        Dict with keys "name", "auc", "acc", "mcc", "n".
    """
    pred = (score >= 0.5).astype(np.int64)
    auc, acc, mcc = safe_auc(y, score), float((pred == y).mean()), safe_mcc(y, pred)
    print(f"  {name:28s} AUC {auc:.4f}   acc {acc:.4f}   MCC {mcc:+.4f}   n={len(y)}")
    return {"name": name, "auc": auc, "acc": acc, "mcc": mcc, "n": len(y)}


# ---------------------------------------------------------------------------
# Block-level evaluation (optional -- needs the sibling repo importable)
# ---------------------------------------------------------------------------

def try_block_eval(test_rows: pd.DataFrame, scores: np.ndarray, catalog_path: str,
                   data_downloader_root: str, horizon_days: float, threshold: float) -> None:
    """Re-scores the test predictions at the honest (block) sample size using
    `seismic_cli.forecast.build_blocks` -- the same partitioning
    `catalog_report.md` used, so this number is directly comparable to the
    retired scalar forecaster's. Skipped with a clear message, not a crash,
    if the sibling repo isn't importable from here.

    Args:
        test_rows: Test-split manifest rows, with 'end_time' and 'region'
            columns.
        scores: Predicted positive-class probability, aligned with
            `test_rows`.
        catalog_path: Path to the earthquake catalog CSV.
        data_downloader_root: Path to the sibling data_downloader repo
            (must contain a `seismic_cli/` package to import from).
        horizon_days: Forecast horizon in days (block width).
        threshold: Minimum magnitude for a catalog event to qualify.

    Returns:
        None. Prints per-zone block-level AUC/accuracy, or a skip message if
        the sibling repo isn't importable or a zone has no usable blocks.
    """
    root = Path(data_downloader_root).resolve()
    if not (root / "seismic_cli").is_dir():
        print(f"\n[block-eval] skipped: {root} has no seismic_cli/ -- pass "
             f"--data-downloader-root to enable.")
        return
    sys.path.insert(0, str(root))
    try:
        from seismic_cli.forecast import (FAULT_ZONES, build_blocks,
                                          load_catalog)
    except ImportError as e:
        print(f"\n[block-eval] skipped: could not import seismic_cli.forecast ({e}).")
        return

    d = test_rows.copy()
    d["end_time"] = pd.to_datetime(d["end_time"])
    d["score"] = scores
    cat = load_catalog(catalog_path, min_magnitude=threshold)

    print(f"\n--- Block-level evaluation ({horizon_days:.0f}-day disjoint blocks) ---")
    print("  (honest sample size -- consecutive windows overlap 11-46x, so the")
    print("   per-window numbers above overstate confidence in themselves)")
    for zone in sorted(d.region.unique()):
        if zone not in FAULT_ZONES:
            print(f"  {zone:9s} skipped: not one of forecast.FAULT_ZONES")
            continue
        la0, la1, lo0, lo1 = FAULT_ZONES[zone]
        zcat = cat[cat.lat.between(la0, la1) & cat.lon.between(lo0, lo1)]
        if zcat.empty:
            print(f"  {zone:9s} skipped: no catalog events in this bbox")
            continue
        major_times = np.sort(zcat.time.to_numpy().astype("datetime64[ns]"))
        catalog_end = zcat.time.max()

        blocks = build_blocks(d, zone, horizon_days, catalog_end, major_times)
        if blocks.empty:
            print(f"  {zone:9s} 0 blocks (too little test-period data)")
            continue
        g = d[d.region == zone].sort_values("end_time").reset_index(drop=True)
        blocks["score"] = g["score"].to_numpy()[blocks.fc_index.to_numpy()]
        auc = safe_auc(blocks.label.to_numpy(), blocks.score.to_numpy())
        pred = (blocks.score.to_numpy() >= 0.5).astype(np.int64)
        acc = float((pred == blocks.label.to_numpy()).mean())
        print(f"  {zone:9s} n_blocks={len(blocks):4d}  positive rate "
             f"{blocks.label.mean():.3f}  AUC {auc:.4f}  acc {acc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(
        description="Dual-channel CNN+LSTM, retargeted onto the dense per-zone forecast target.")
    p.add_argument("--dataset-dir", required=True,
                  help="Directory from `seismic-cli generate-catalog-forecast-dataset`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm_forecast")
    p.add_argument("--channels", default="all",
                  choices=["all", "1d", "2d", "aux", "1d+aux", "2d+aux"],
                  help="Ablation switch: which branches to enable.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--horizon-days", type=float, default=30.0,
                  help="Must match --horizon-days used at dataset-generation time.")
    p.add_argument("--threshold", type=float, default=4.5,
                  help="Must match --threshold used at dataset-generation time.")
    p.add_argument("--mag-loss-weight", type=float, default=1.0,
                  help="Weight on the magnitude head's masked L1 loss relative to the "
                       "binary head's BCE loss.")
    p.add_argument("--catalog-path", default=None,
                  help="Earthquake catalog CSV, for block-level evaluation. "
                       "Skipped if not given.")
    p.add_argument("--data-downloader-root", default="../../data_downloader",
                  help="Path to the sibling data_downloader repo, for block-level eval.")
    return p.parse_args()


def main():
    """Loads the dense-window dataset, trains `DualChannelForecastNet`, and
    reports the binary and magnitude heads against their floors, per-zone
    and (optionally) block-level."""
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    train_ds = DenseWindowDataset(manifest, root, "train")
    val_ds = DenseWindowDataset(manifest, root, "val", stats=train_ds.stats)
    test_ds = DenseWindowDataset(manifest, root, "test", stats=train_ds.stats)

    print("=" * 64)
    print(f"Dual-channel forecast model | channels='{args.channels}' | "
         f"M>={args.threshold} within {args.horizon_days:.0f}d")
    print(f"  seq ({train_ds.seq_len}, {train_ds.seq_dim}) | img {train_ds.img_shape} "
         f"| aux ({train_ds.aux_dim},)")
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        print(f"  {name:5s}: n={len(ds):5d}  positive rate {ds.labels.mean():.3f}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelForecastNet(train_ds.seq_dim, train_ds.img_shape[0], train_ds.aux_dim,
                                   hidden=args.hidden, fusion_dim=args.fusion_dim,
                                   dropout=args.dropout, channels=args.channels).to(device)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = train_ds.labels.mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    print(f"pos_weight (train): {pos_weight.item():.3f}  (train positive rate {pos:.3f})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_cnnlstm_forecast.pth")
    best, no_improve = -1.0, 0

    def evaluate(loader):
        """Runs the model over `loader` and collects labels/scores/magnitudes.

        Args:
            loader: DataLoader yielding (seq, img, aux, y, mag) batches.

        Returns:
            Tuple of (y_true int64 array, y_score array, magnitude_true
            array, magnitude_pred array).
        """
        model.eval()
        ys, ss, mag_true, mag_pred = [], [], [], []
        with torch.no_grad():
            for seq, img, aux, y, mag in loader:
                logit, mag_out = model(seq.to(device), img.to(device), aux.to(device))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.tolist())
                mag_true.extend(mag.tolist())
                mag_pred.extend(mag_out.cpu().tolist())
        return (np.array(ys, dtype=np.int64), np.array(ss),
               np.array(mag_true), np.array(mag_pred))

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for seq, img, aux, y, mag in train_loader:
            seq, img, aux, y, mag = (seq.to(device), img.to(device), aux.to(device),
                                     y.to(device), mag.to(device))
            logit, mag_out = model(seq, img, aux)
            loss = criterion(logit, y) + args.mag_loss_weight * masked_l1_loss(mag_out, mag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            tot += loss.item() * y.size(0)
        scheduler.step()

        yv, sv, _, _ = evaluate(val_loader)
        val_auc = safe_auc(yv, sv)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {tot/len(train_ds):.4f} | val AUC {val_auc:.4f}")
        if val_auc > best:
            best, no_improve = val_auc, 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val AUC {best:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val AUC flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, st, mag_true, mag_pred = evaluate(test_loader)

    print("\n--- Floors (test set) ---")
    base_pred = np.full_like(yt, int(round(pos)))
    report_row("base-rate (majority)", yt, base_pred.astype(np.float64))
    pers_pred = persistence_prediction(test_ds.days_since_prev_major, args.horizon_days)
    report_row("persistence", yt, pers_pred.astype(np.float64))

    print(f"\n--- Dual-channel forecast model (channels='{args.channels}') ---")
    model_row = report_row("model", yt, st)
    print_report("Dual-channel forecast model -- full binary metric set (test set)",
                binary_report(yt, st))
    valid_mag = ~np.isnan(mag_true)
    if valid_mag.any():
        print_report("Magnitude head (positive windows only)",
                    regression_report(mag_true[valid_mag], mag_pred[valid_mag]))

    print("\n--- Per zone (test set, pooled window level) ---")
    test_rows = test_ds.rows.copy()
    for zone in sorted(test_rows.region.unique()):
        m = (test_rows.region == zone).to_numpy()
        report_row(zone, yt[m], st[m])

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        print(f"\n  learned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}")

    if model_row["auc"] <= max(0.5, safe_auc(yt, pers_pred.astype(np.float64))) + 1e-9:
        print("\n  [!] The model does NOT clear max(chance, persistence). Its AUC is not")
        print("      evidence of forecasting skill, whatever the value is -- report it as such.")
    else:
        print("\n  Beats max(chance, persistence) at the pooled window level (see block-level")
        print("  numbers below for the honest sample size before trusting this).")

    if args.catalog_path:
        try_block_eval(test_rows, st, args.catalog_path, args.data_downloader_root,
                       args.horizon_days, args.threshold)
    else:
        print("\n[block-eval] skipped: pass --catalog-path to enable "
             "(consecutive windows overlap 11-46x; per-window AUC above overstates confidence).")

    # -------------------------------------------------------------------
    # Magnitude head (test set, POSITIVE windows only -- next_magnitude is
    # only defined where an event actually occurs within the horizon).
    # -------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("--- Magnitude head (test set, positive windows only) ---")
    test_pos = test_ds.labels.astype(bool)
    n_pos = int(test_pos.sum())
    if n_pos < 10:
        print(f"  only {n_pos} positive test windows -- too few to report a floor comparison")
    else:
        y_mag = mag_true[test_pos]
        p_mag = mag_pred[test_pos]
        mae_model = float(np.mean(np.abs(y_mag - p_mag)))

        train_pos = train_ds.labels.astype(bool)
        train_mag = train_ds.next_magnitude[train_pos]
        mae_mean = float(np.mean(np.abs(y_mag - train_mag.mean())))
        print(f"  predict-the-mean (train positives)   MAE {mae_mean:.3f}   n={n_pos}")

        floor_cols = [AUX_FEATURES.index(c) for c in ("max_mag", "mean_mag", "b_value", "log_rate")]
        train_aux = train_ds.standardized_aux_matrix()[train_pos][:, floor_cols]
        test_aux = test_ds.standardized_aux_matrix()[test_pos][:, floor_cols]
        try:
            ridge = Ridge(alpha=1.0).fit(train_aux, train_mag)
            mae_ridge = float(np.mean(np.abs(y_mag - ridge.predict(test_aux))))
            print(f"  ridge(max_mag, mean_mag, b_value, log_rate)  MAE {mae_ridge:.3f}   n={n_pos}"
                 "   <- fitted statistical relation; the model must beat this")
        except Exception as e:
            mae_ridge = None
            print(f"  [WARN] ridge floor failed: {e}")

        print(f"\n  model                                 MAE {mae_model:.3f}   n={n_pos}")

        floors = [m for m in (mae_mean, mae_ridge) if m is not None]
        if floors and mae_model < min(floors):
            print(f"\n  Beats both floors by {min(floors) - mae_model:+.3f} MAE.")
        elif floors:
            print(f"\n  [!] Does NOT beat {'both floors' if len(floors) > 1 else 'the floor'} "
                 "-- the magnitude head is not adding information beyond the simpler baseline(s).")
        print("\n  [!] Single-seed result -- re-seed before treating a close margin as final "
             "(report.md 6.6).")


if __name__ == "__main__":
    main()
