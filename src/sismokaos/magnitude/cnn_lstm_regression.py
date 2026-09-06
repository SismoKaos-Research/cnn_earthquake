"""
Dual-channel CNN + LSTM/self-attention magnitude regression.

Same task as `cnn_regression.py` (magnitude from a 3-second window, plus the
two physical predictors log_snr and log_distance) but with a second input
channel: the raw standardized waveform, run through an LSTM + multi-head
self-attention branch, fused with the 2D image branch exactly as
`cnn_lstm.py`'s `DualChannelRiskNet` does for the (unrelated) catalog-risk
task. This is not a replication of a specific paper -- investigation found
no paper in this project's literature folder that combines a spectrogram
image branch with a raw-waveform LSTM branch for magnitude (the closest,
Shen et al. 2025, is pure 1D CNN+Bi-LSTM with no image channel at all) -- it
is this project's own dual-channel architecture (Wang & Zhao 2025's
1D2D-EDL, already used for detection) retargeted at a task it has not been
tried on.

Consumes `seismic-cli generate-regression-dataset --dual`: the SAME
event-disjoint split, magnitude label, and log_snr/distance_km predictors
`cnn_regression.py` uses, but with `{seq, img}` tensors (SpectrogramDualEncoder
or RamDualEncoder) instead of a single encoded image -- `--dual` reuses the
detection pipeline's existing encoders unchanged, so the two datasets are
directly comparable except for the extra seq channel.

Floors and the single-channel comparison are what settle whether the LSTM
branch is worth having, not assumed:
  * predict-the-mean and ridge(log_snr, log_distance) -- ported unchanged
    from `cnn_regression.py`, same metric space (magnitude itself is not
    log-transformed; only distance is, matching `cnn_regression.py` exactly
    to avoid the metric-space mismatches report.md 13.5 documents).
  * `cnn_regression.py` run on the SAME event split, single-channel
    spectrogram+aux -- the existing result this architecture has to beat,
    not just the floors.
`--channels` ablates 1D/2D/aux individually, same convention as
`cnn_lstm.py`/`cnn_lstm_forecast.py`, so a win can be attributed to a
specific branch rather than assumed to come from the LSTM.

This is reported as a SINGLE-SEED result, explicitly, matching how
report.md's own magnitude-classification section (7) was reported ("at a
single seed and threshold, not yet re-verified") -- this project has
reversed close-margin verdicts on a single seed before (report.md 6.6), so a
close margin here should be re-seeded before being treated as final, not
read as an established result off one run.

Usage:
    python cnn_lstm_regression.py --dataset-dir ../seismic_cli/data/dataset_magclass_dual_3s

Not imported by anything else -- standalone script.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sismokaos.model.registry import add_model_args, spec_from_args
from sismokaos.runlog import RunLog
from sismokaos.magnitude.cnn_regression import (AUX_COLUMNS, detect_aux_columns,
                                      regression_metrics, report_baselines)
from sismokaos.metrics import (predict_mean_baseline, print_report,
                               regression_report)
from sismokaos.model.dual_channel import DualChannelNet
from sismokaos.training import seed_everything

# ---------------------------------------------------------------------------
# Splits -- mirrors cnn_groundmotion.py's respilt/report_split (report.md
# 13.8), applied here to check whether this task's event-disjoint headline
# numbers (7.5-7.8) are inflated by site memorisation: 175/181 (3s) and
# 148/152 (6s) stations appear in more than one split under the generator's
# default event-disjoint grouping.
# ---------------------------------------------------------------------------

def resplit(d, how, seed=42, ratios=(0.70, 0.15, 0.15), detector_manifest=None):
    """Re-partitions rows without moving any tensor on disk. The manifest's
    original `split` names the DIRECTORY a tensor lives in, so it is
    preserved as `file_split`; only the LOGICAL split (used for train/val/
    test grouping below) changes.

    `how`:
      event   -- the generator's default (unchanged): events are disjoint,
                 so the magnitude label cannot leak, but most stations are
                 shared, so site response can be learned and reused.
      station -- stations are disjoint, so site response cannot leak, but
                 an earthquake recorded at a train station and a test
                 station now shares its source term (magnitude) across the
                 split -- regression.py's own docstring calls this usually
                 the worse leak for a regression target.
      both    -- station-disjoint, then every val/test row whose event also
                 appears in train is DROPPED. Neither term can leak. Costs
                 rows; the count dropped is reported, not hidden.
      detector -- like "both", but the station partition is COPIED from a
                 detector manifest instead of being drawn here. Required
                 when this regressor is the second stage of a cascade: if
                 the two stages partition stations independently, the
                 detector's test stations land in the regressor's training
                 set (measured at 77% on the pre-existing datasets) and an
                 end-to-end evaluation silently leaks. Stations absent from
                 the detector manifest belong to no detector split, so they
                 cannot leak and are assigned to train.

    Args:
        d: Manifest DataFrame with 'split', 'station_key', and 'event_id'
            columns.
        how: Grouping to use -- "event" (unchanged), "station", or "both".
        seed: Seed for the station shuffle used by "station"/"both".
        ratios: Target (train, val, test) row-count fractions for the
            station partition.
        detector_manifest: Path to the detector's manifest.csv. Required for
            `how="detector"`, ignored otherwise.

    Returns:
        A copy of `d` with 'file_split' added (the original directory-based
        split) and, for "station"/"both", 'split' reassigned to the new
        station-disjoint grouping (rows dropped for "both").
    """
    d = d.copy()
    if "file_split" not in d:
        d["file_split"] = d["split"]
    if how == "event":
        return d

    if how == "detector":
        if detector_manifest is None:
            raise ValueError("--split-by detector needs --detector-manifest.")
        det = pd.read_csv(detector_manifest)
        assign = dict(zip(det.station_key, det.split))
        unseen = sorted(set(d.station_key) - set(assign))
        # A station the detector never used sits in none of its splits, so it
        # cannot carry information between the two stages. Train is the safe
        # home for it.
        for st in unseen:
            assign[st] = "train"
        if unseen:
            print(f"[split] {len(unseen)} station(s) absent from the detector manifest "
                  f"-> assigned to train")
        d["split"] = d.station_key.map(assign)
        train_events = set(d.loc[d.split == "train", "event_id"])
        clash = (d.split != "train") & d.event_id.isin(train_events)
        print(f"[split] detector-aligned: dropping {int(clash.sum())} val/test rows whose "
              f"event also appears in train")
        return d[~clash].copy()

    rng = random.Random(seed)
    stations = sorted(set(d.station_key))
    rng.shuffle(stations)
    size = d.station_key.value_counts().to_dict()
    total = len(d)
    targets = {s: r * total for s, r in zip(("train", "val", "test"), ratios)}
    running = {s: 0 for s in targets}
    assign = {}
    for st in stations:
        best = max(targets, key=lambda s: (targets[s] - running[s]) / max(targets[s], 1.0))
        assign[st] = best
        running[best] += size[st]
    d["split"] = d.station_key.map(assign)

    if how == "both":
        train_events = set(d.loc[d.split == "train", "event_id"])
        clash = (d.split != "train") & d.event_id.isin(train_events)
        print(f"[split] doubly-disjoint: dropping {int(clash.sum())} val/test rows whose "
              f"event also appears in train")
        d = d[~clash].copy()
    return d


def report_split(d, how):
    """Prints the row counts and any train/test event or station overlap.

    Args:
        d: Manifest DataFrame after `resplit`, with 'split', 'event_id',
            and 'station_key' columns.
        how: Grouping name to print (as returned by `resplit`'s `how` arg).
    """
    tr, te = d[d.split == "train"], d[d.split == "test"]
    shared_ev = len(set(tr.event_id) & set(te.event_id))
    shared_st = len(set(tr.station_key) & set(te.station_key))
    print(f"[split] grouping='{how}'  train {len(tr)}  val {int((d.split=='val').sum())}  "
          f"test {len(te)}")
    print(f"[split]   events shared train/test : {shared_ev}"
          f"   ({'LEAK: source term' if shared_ev else 'clean'})")
    print(f"[split]   stations shared          : {shared_st}"
          f"   ({'LEAK: site response' if shared_st else 'clean'})")
    print(f"[split]   test stations unseen in train: "
          f"{len(set(te.station_key) - set(tr.station_key))}/{te.station_key.nunique()}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class DualMagnitudeDataset(Dataset):
    """
    Loads the {seq, img} tensors written by
    `seismic-cli generate-regression-dataset --dual`, plus the manifest's
    magnitude/log_snr/log_distance columns -- same standardization convention
    as `cnn_regression.py`'s MagnitudeDataset (TRAIN-only aux stats, NaN ->
    0 after standardization).
    """

    def __init__(self, manifest: pd.DataFrame, root: Path, aux_stats=None):
        """Loads manifest rows and fits (or reuses) aux normalization stats.

        Args:
            manifest: Manifest rows for one split, with 'file_split',
                'filename', 'magnitude', and `AUX_COLUMNS` columns.
            root: Dataset root directory (contains a subdirectory per
                file_split).
            aux_stats: Optional (mean, std) tuple to standardize aux with;
                if None, fit from this split's own data (the train split
                should pass None; val/test must reuse the train split's
                stats).
        """
        self.rows = manifest.reset_index(drop=True)
        self.root = Path(root)

        aux = self.rows[AUX_COLUMNS].to_numpy(dtype=np.float64)
        if aux_stats is None:
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(aux, axis=0)
                sd = np.nanstd(aux, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            aux_stats = (mu, sd)
        self.aux_mu, self.aux_sd = aux_stats
        aux = (aux - self.aux_mu) / self.aux_sd
        self.aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.targets = self.rows["magnitude"].to_numpy(dtype=np.float32)

    def aux_stats(self):
        """Returns the (mean, std) tuple this dataset standardized aux with."""
        return (self.aux_mu, self.aux_sd)

    def __len__(self):
        """Returns the number of rows in this split."""
        return len(self.rows)

    def __getitem__(self, idx):
        """Returns one normalized (seq, img, aux, magnitude) sample.

        Args:
            idx: Row index into this split.

        Returns:
            Tuple of (float32 seq tensor, float32 img tensor, float32 aux
            tensor, float32 scalar magnitude tensor).
        """
        row = self.rows.iloc[idx]
        d = torch.load(self.root / row["file_split"] / row["filename"], weights_only=True)
        return (d["seq"].float(), d["img"].float(),
                torch.from_numpy(self.aux[idx]),
                torch.tensor(self.targets[idx], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model -- same branches as cnn_lstm.py, regression head
# ---------------------------------------------------------------------------

class DualChannelRegressionNet(DualChannelNet):
    """Same 1D/2D/aux architecture as `cnn_lstm.py`'s DualChannelRiskNet /
    `cnn_lstm_forecast.py`'s DualChannelForecastNet, with a bare linear
    regression head (no activation, matching `RegressionSeismicCNN`)."""

    def __init__(self, seq_dim, img_channels, aux_dim, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all"):
        """See `DualChannelNet.__init__` (`n_classes=1`, `squeeze_output=True`
        always here)."""
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         n_classes=1, squeeze_output=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(
        description="Dual-channel CNN+LSTM magnitude regression (spectrogram or RAM 2D + raw-waveform 1D).")
    p.add_argument("--dataset-dir", required=True,
                  help="Directory from `seismic-cli generate-regression-dataset --dual`.")
    p.add_argument("--save-dir", default="trained_model_cnnlstm_regression")
    # The regressor's own flags were --channels/--hidden/--fusion-dim/--dropout,
    # and its defaults are the architecture's, so nothing here moves. What it
    # gains is --fusion and --model-branch: DualChannelNet has had a gated
    # fusion and a convolutional 1D front end all along, and only the detection
    # scripts ever exposed them. Magnitude has never been tried with either.
    add_model_args(p, family="dual")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--huber-delta", type=float, default=0.0,
                  help="If > 0 use SmoothL1 with this beta instead of plain L1 (MAE), "
                       "matching cnn_regression.py's convention.")
    p.add_argument("--split-by", default="auto",
                  choices=["auto", "event", "station", "both", "detector"],
                  help="event: the generator's own split, unchanged (default). station: "
                       "re-partition in memory so stations are disjoint (site response "
                       "cannot leak, but a shared event can). both: station-disjoint AND "
                       "event-disjoint -- drops val/test rows whose event also appears in "
                       "train (report.md 13.8's doubly-disjoint check, applied here). "
                       "detector: like 'both', but the station partition is copied from "
                       "--detector-manifest, so this regressor can be stage 2 of a cascade "
                       "without the detector's test stations leaking into its training set.")
    p.add_argument("--detector-manifest", default=None,
                  help="Path to the detector dataset's manifest.csv. Required for "
                       "--split-by detector.")
    p.add_argument("--oversample-alpha", type=float, default=0.0,
                   help="TT-SAM magnitude oversampling (Chen et al. 2026 3.2.1): "
                        "rows with M>=--oversample-min are repeated "
                        "int(alpha**(M-1) - 1) extra times during TRAINING only, "
                        "so larger events are seen more often. 0.0 (default) "
                        "disables it and reproduces every published number here. "
                        "1.25 is the published factor. Val and test are never "
                        "resampled -- doing so would change what the number means.")
    p.add_argument("--oversample-min", type=float, default=4.0,
                   help="magnitude at or above which oversampling applies")
    p.add_argument("--seed-split", type=int, default=42,
                  help="Seed for the station partition (--split-by station/both). "
                       "Independent of --seed (model init/shuffle).")
    return p.parse_args()


def main():
    """Loads the dual-tensor magnitude dataset, trains
    `DualChannelRegressionNet`, and reports test MAE/RMSE/R2 against the
    ridge baseline.

    Side effect: rebinds the module-level `AUX_COLUMNS` to the columns
    actually detected in the manifest (`detect_aux_columns`), since the set
    of available auxiliary predictors can vary by dataset.

    Raises:
        ValueError: If the manifest is missing 'magnitude' or any detected
            aux column, or if any split ("train"/"val"/"test") is empty.
    """
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")
    if "magnitude" not in manifest.columns:
        raise ValueError("manifest.csv is missing 'magnitude'. Regenerate with "
                         "`seismic-cli generate-regression-dataset --dual`.")
    if "distance_km" not in manifest.columns:
        manifest["distance_km"] = np.nan
    manifest["log_distance"] = np.log(manifest["distance_km"].clip(lower=1.0))

    global AUX_COLUMNS
    AUX_COLUMNS = detect_aux_columns(manifest)
    missing = [c for c in AUX_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest.csv is missing {missing}. Regenerate with "
                         f"`seismic-cli generate-regression-dataset --dual`.")

    if args.split_by == "auto":
        # Station-disjoint whenever the corpus can support it. A single-station
        # corpus cannot -- and every magnitude figure this project has published
        # came from one, with the split diagnostics printing "LEAK: site
        # response" underneath. Making the honest protocol the default, and the
        # single-station case say so out loud, is the difference between "a
        # magnitude estimator" and "this station's magnitude estimator".
        n_st = manifest.station_key.nunique()
        args.split_by = "both" if n_st > 1 else "event"
        print(f"[split] --split-by auto -> {args.split_by!r} "
              f"({n_st} station(s) in the corpus)")
        if n_st == 1:
            print("[split]   a one-station corpus cannot be split station-disjoint. "
                  "Whatever this run reports is THIS STATION's estimator, and does "
                  "not transfer -- say so wherever the number is quoted.")

    manifest = resplit(manifest, args.split_by, seed=args.seed_split,
                       detector_manifest=args.detector_manifest)
    report_split(manifest, args.split_by)

    # Provenance is opened BEFORE training, so a run that dies mid-epoch still
    # leaves a record of what was attempted rather than none at all.
    runlog = RunLog(f"magnitude/cnn_lstm_regression", args.save_dir, vars(args))
    runlog.note(model=spec_from_args(args).to_dict())

    parts = {}
    for split in ("train", "val", "test"):
        sub = manifest[manifest.split == split]
        if sub.empty:
            raise ValueError(f"Split '{split}' is empty in the manifest.")
        parts[split] = sub

    # Oversampling is applied to the TRAIN rows only, and the aux statistics are
    # fitted BEFORE replication -- fitting them after would let the duplicated
    # large events shift the normalization, changing the input distribution
    # rather than only the sampling frequency.
    train_rows = parts["train"]
    stats = DualMagnitudeDataset(train_rows, root).aux_stats()
    if args.oversample_alpha > 1.0:
        m = train_rows.magnitude.to_numpy()
        reps = np.where(m >= args.oversample_min,
                        (args.oversample_alpha ** (m - 1.0) - 1.0).astype(int), 0)
        reps = np.clip(reps, 0, None)
        extra = train_rows.loc[np.repeat(train_rows.index.to_numpy(), reps)]
        before = len(train_rows)
        train_rows = pd.concat([train_rows, extra], ignore_index=True)
        big = (train_rows.magnitude >= args.oversample_min).mean()
        print(f"[oversample] alpha={args.oversample_alpha} min M{args.oversample_min:g}: "
              f"{before:,} -> {len(train_rows):,} train rows; "
              f"M>={args.oversample_min:g} share "
              f"{(parts['train'].magnitude >= args.oversample_min).mean():.3%} -> {big:.3%}")

    train_ds = DualMagnitudeDataset(train_rows, root, aux_stats=stats)

    # Validation is resampled the SAME way. Early stopping selects on val MAE,
    # and an un-resampled val set is dominated by the small events oversampling
    # exists to de-emphasise -- so leaving it alone makes the selection
    # criterion fight the training objective and picks the checkpoint that is
    # best at exactly what was being down-weighted. Measured: oversampling
    # train only made the compression WORSE (slope 0.642 -> 0.507).
    val_rows = parts["val"]
    if args.oversample_alpha > 1.0:
        mv = val_rows.magnitude.to_numpy()
        rv = np.clip(np.where(mv >= args.oversample_min,
                              (args.oversample_alpha ** (mv - 1.0) - 1.0).astype(int),
                              0), 0, None)
        val_rows = pd.concat([val_rows,
                              val_rows.loc[np.repeat(val_rows.index.to_numpy(), rv)]],
                             ignore_index=True)
        print(f"[oversample] val {len(parts['val']):,} -> {len(val_rows):,} rows "
              f"(model selection now matches the training objective)")
    val_ds = DualMagnitudeDataset(val_rows, root, aux_stats=stats)
    test_ds = DualMagnitudeDataset(parts["test"], root, aux_stats=stats)

    sample_seq, sample_img, sample_aux, _ = train_ds[0]

    print("=" * 64)
    print(f"Dual-channel magnitude regression | channels='{args.channels}'")
    print(f"  seq {tuple(sample_seq.shape)} | img {tuple(sample_img.shape)} "
         f"| aux ({sample_aux.numel()},) ({', '.join(AUX_COLUMNS)})")
    for s in ("train", "val", "test"):
        d = parts[s]
        print(f"  {s:5s}: n={len(d):5d}  M {d.magnitude.min():.1f}-{d.magnitude.max():.1f} "
             f"(mean {d.magnitude.mean():.2f})  events={d.event_id.nunique()}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Registry-built rather than by class name, so --model-branch/--fusion reach
    # the model and the resolved spec can be written beside the weights.
    # DualChannelRegressionNet stays defined above: cascade_eval.py rebuilds it
    # by name to load these checkpoints, and it is the same network.
    spec = spec_from_args(args)
    model = spec.build(seq_dim=sample_seq.shape[-1], img_channels=sample_img.shape[0],
                       aux_dim=sample_aux.numel(), squeeze_output=True).to(device)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    criterion = (nn.SmoothL1Loss(beta=args.huber_delta) if args.huber_delta > 0
                else nn.L1Loss())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    # The protocol travels with the weights, not just the geometry: a
    # checkpoint scored under a different split is scored on a different test
    # set, and the number that comes back is for a question nobody asked.
    spec.save(args.save_dir,
              protocol={"split_by": args.split_by, "seed_split": args.seed_split,
                        "detector_manifest": args.detector_manifest})
    # The checkpoint name must identify the RUN, not just the task. A fixed
    # filename means two runs sharing --save-dir overwrite each other, and the
    # second silently reloads the first's weights at the end of training. That
    # exact bug produced -- and forced the retraction of -- an ensemble AUC of
    # 0.9108 on the detection side (report.md 8.1). Config, split, seed and PID
    # together make collision impossible even for two identical commands.
    # New segments are appended only when they are not the architecture's
    # default, so every checkpoint written before --fusion and --model-branch
    # reached this script keeps the name it has. sismokaos/checkpoints.py
    # records what happened the last time a tag grew for everyone at once.
    arch_tag = "".join(f"_{v}" for v in (spec.branch, spec.params["fusion"])
                       if v not in ("lstm", "linear"))
    run_tag = (f"{args.channels}{arch_tag}_{args.split_by}_seed{args.seed}"
               f"_split{args.seed_split}_pid{os.getpid()}")
    save_path = os.path.join(args.save_dir, f"best_cnnlstm_regression_{run_tag}.pth")
    best_val_mae, no_improve = float("inf"), 0

    def evaluate(loader):
        """Runs the model over `loader` and collects true/predicted magnitudes.

        Args:
            loader: DataLoader yielding (seq, img, aux, y) batches.

        Returns:
            Tuple of (y_true, y_pred) float arrays.
        """
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for seq, img, aux, y in loader:
                out = model(seq.to(device), img.to(device), aux.to(device))
                preds.extend(out.cpu().tolist())
                trues.extend(y.tolist())
        return np.array(trues), np.array(preds)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for seq, img, aux, y in train_loader:
            seq, img, aux, y = seq.to(device), img.to(device), aux.to(device), y.to(device)
            loss = criterion(model(seq, img, aux), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            running += loss.item() * y.size(0)
        scheduler.step()

        yv, pv = evaluate(val_loader)
        vm = regression_metrics(yv, pv)
        print(f"  [seed {args.seed}] epoch {epoch+1}/{args.epochs} val MAE {vm['MAE']:.4f} "
             f"train loss {running/len(train_ds):.4f} RMSE {vm['RMSE']:.4f} R2 {vm['R2']:+.4f}")
        if vm["MAE"] < best_val_mae:
            best_val_mae, no_improve = vm["MAE"], 0
            torch.save(model.state_dict(), save_path)
            print(f"  => saved (val MAE {best_val_mae:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping: val MAE flat for {args.patience} epochs.")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    yt, yp = evaluate(test_loader)
    tm = regression_metrics(yt, yp)
    runlog.note(dataset=str(root), n_train=len(train_ds), n_val=len(val_ds),
                n_test=len(test_ds), aux_columns=AUX_COLUMNS)

    ridge_mae = report_baselines(train_ds, test_ds, aux_names=AUX_COLUMNS)

    # Two floors, in increasing order of difficulty. The constant-mean floor only
    # shows the target has variance; the ridge-on-(amplitude, distance) floor is the
    # physics formula local magnitude is actually built from, and is the one that
    # matters -- beating a constant proves nothing about seismology.
    print("\n--- Floors (test set) ---")
    y_tr = np.asarray([train_ds[i][2] for i in range(len(train_ds))], dtype=np.float64) \
        if not hasattr(train_ds, "targets") else np.asarray(train_ds.targets, dtype=np.float64)
    mean_floor = predict_mean_baseline(y_tr, yt)
    print(f"  constant-mean            MAE {mean_floor['MAE']:.4f}  RMSE {mean_floor['RMSE']:.4f}  "
          f"R2 {mean_floor['R2']:+.4f}")
    if ridge_mae is not None:
        print(f"  ridge(amplitude,distance) MAE {ridge_mae:.4f}   <- the physics floor")
    runlog.finish(metrics=dict(tm, ridge_floor=ridge_mae,
                               mean_floor=mean_floor["MAE"]),
                  checkpoints=[save_path])
    print(f"  provenance -> {runlog.path}")

    print_report(f"Magnitude regressor [channels={args.channels}] (seed {args.seed}, test set)",
                 regression_report(yt, yp))

    print(f"\n  vs constant-mean floor:  {mean_floor['MAE'] - tm['MAE']:+.4f} MAE "
          f"({100 * (mean_floor['MAE'] - tm['MAE']) / mean_floor['MAE']:+.1f}%)")
    if ridge_mae is not None:
        delta = ridge_mae - tm["MAE"]
        verdict = ("adds information beyond amplitude+distance"
                  if delta > 0.01 else
                  "NO measurable gain over amplitude+distance alone")
        print(f"  vs ridge physics floor:  {delta:+.4f} MAE "
              f"({100 * delta / ridge_mae:+.1f}%)  ->  {verdict}")

    if getattr(model, "use_1d", False) and getattr(model, "use_2d", False):
        print(f"\n  learned fusion weights: 1D={model.w1.item():+.3f}  2D={model.w2.item():+.3f}")

    print(f"\n  residual std {np.std(yt - yp):.3f} magnitude units over "
         f"M {yt.min():.1f}-{yt.max():.1f}")
    print("\n  [!] Single-seed result -- re-seed before treating a close margin as final "
         "(report.md 6.6).")


if __name__ == "__main__":
    main()
