"""Trains `gru_cnn.SeismicFusionModel` with walk-forward CV and honest floors.

## Running it

Catalog-only baseline -- start here, it trains in minutes:

    python3 src/forecasting/gru_cnn_train.py \\
        --features combined_features_114d.parquet \\
        --catalog-path ../Sismokaos/data_downloader/catalogs/data_large.csv

Add the waveform branch (needs a `.f32` stream from `sismokaos-cli preprocess`):

    python3 src/forecasting/gru_cnn_train.py \\
        --features combined_features_114d.parquet \\
        --raw-f32 aegean_bodt_preprocessed.f32 \\
        --catalog-path ../Sismokaos/data_downloader/catalogs/data_large.csv

Useful knobs: `--horizon-days` (default 14), `--seq-len` (24), `--cv-folds`
(2), `--ensemble-seeds 42,43,44`, `--epochs`, `--batch-size`.

## What the output means

Per fold you get each seed's test AUC, their spread, and the ensemble; then
the ensemble against two floors. **Read the floors, not the AUC.**

- *base rate* -- predicting the training positive rate for every hour. Always
  0.5 AUC; shown only to make the balance visible.
- *persistence* -- ranking by how recently the last qualifying event occurred.
  This is the bar. Seismicity clusters, so "an event just happened" is a
  genuinely predictive rule that costs nothing to compute, and a model that
  cannot beat it has learned nothing about forecasting. It is reported
  orientation-corrected (`max(auc, 1-auc)`), because a rule that is reliably
  backwards is just as exploitable once flipped.

A single fold beating persistence by a hair is noise. Look at whether the
model clears it on *both* folds and whether the seed spread is smaller than
the margin.

## Why the splits look the way they do

Expanding-window walk-forward, never random. Hours are not independent
observations: overlapping windows share inputs, and a 14-day label horizon
means adjacent hours share their *answer* too. The embargo between splits is
therefore `seq_len - 1 + horizon_days * 24` hours -- input overlap plus the
full forward reach of the label. An embargo covering only the input overlap
was worth 0.14 AUC of leakage elsewhere in this project.

Normalisation is fitted on each fold's training rows alone, for the same
reason.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from features.seismic_fusion_dataset import (OPTIMIZED_CATALOG_FEATURES,
                                             SeismicFusionDataset,
                                             fit_normalizer)
from forecasting.gru_cnn import SeismicFusionModel
from seismolib.catalog import (days_since_prev_major, label_hours,
                               load_aegean_events)
from seismolib.metrics import binary_report, print_report, safe_auc
from seismolib.rust_io import RustData
from seismolib.splits import walk_forward_splits
from seismolib.training import seed_everything


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True,
                   help="Feature Parquet: the CLI's own output, or any table "
                        "with a DatetimeIndex.")
    p.add_argument("--raw-f32", default=None,
                   help="Waveform stream from `sismokaos-cli preprocess`. Its "
                        ".f32.json sidecar must sit beside it. Omit for the "
                        "catalog-only baseline.")
    p.add_argument("--catalog-path", required=True,
                   help="Event catalog CSV; supplies the labels.")
    p.add_argument("--threshold", type=float, default=4.5,
                   help="Minimum magnitude that counts as an event.")
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--seq-len", type=int, default=24, help="Hours per window.")
    p.add_argument("--feature-columns", default=None,
                   help="Comma-separated columns. Defaults to the RFE subset "
                        f"({','.join(OPTIMIZED_CATALOG_FEATURES)}) when present.")
    p.add_argument("--cv-folds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8,
                   help="Stop a seed after this many epochs without a val-AUC "
                        "improvement.")
    p.add_argument("--ensemble-seeds", default="42,43,44",
                   help="Comma-separated seeds, averaged as probabilities.")
    p.add_argument("--min-minority", type=int, default=30,
                   help="Refuse to score a test split whose minority class is "
                        "smaller than this. An AUC over a handful of hours does "
                        "not reproduce.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-dir", default="trained_model_gru_cnn")
    return p.parse_args()


def build_loaders(data, labels, ends, seq_len, args, waveform):
    """One `(train, val, test)` triple of DataLoaders for a fold.

    The normalizer is fitted on the training end-indices only and handed to
    all three splits, so validation and test never inform the scaler.
    """
    train_e, val_e, test_e = ends
    mean, std = fit_normalizer(data, train_e, seq_len)

    def make(idx, shuffle):
        ds = SeismicFusionDataset(data, labels, idx, seq_len, mean, std,
                                  waveform=waveform)
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    train_ds, train_dl = make(train_e, True)
    _, val_dl = make(val_e, False)
    _, test_dl = make(test_e, False)
    return train_ds, train_dl, val_dl, test_dl


@torch.no_grad()
def score(model, loader, device, use_waveform):
    """Returns `(labels, probabilities)` over a loader."""
    model.eval()
    probs, ys = [], []
    for cat_seq, wave_seq, y in loader:
        cat_seq = cat_seq.to(device)
        wave = wave_seq.to(device) if use_waveform else None
        logits = model(cat_seq, wave)
        probs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_one_seed(args, seed, fold_tag, train_ds, train_dl, val_dl, test_dl,
                   cat_dim, device):
    """Trains one seed, returns `(y_test, probs_test, n_params)`.

    Selection is on validation AUC, and the checkpoint name carries the fold,
    the seed and the process id: naming a checkpoint by seed alone let two
    concurrent runs reload each other's weights elsewhere in this project, and
    when the architectures matched it was silent.
    """
    seed_everything(seed)
    use_waveform = args.raw_f32 is not None

    model = SeismicFusionModel(use_waveform=use_waveform, cat_dim=cat_dim).to(device)

    # Weight the positive class by the training split's own balance. Taking it
    # from the full dataset would leak the other folds' event rate.
    pos_rate = train_ds.positive_rate()
    pos_weight = torch.tensor(
        [(1.0 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 2))

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = Path(args.save_dir) / (
        f"gru_cnn_{'fusion' if use_waveform else 'catalog'}_"
        f"{fold_tag}_seed{seed}_pid{os.getpid()}.pth")

    best_auc, best_epoch, no_improve = -1.0, -1, 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for cat_seq, wave_seq, y in train_dl:
            cat_seq = cat_seq.to(device)
            wave = wave_seq.to(device) if use_waveform else None
            y = y.unsqueeze(1).to(device)

            optimizer.zero_grad()
            loss = criterion(model(cat_seq, wave), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()

        yv, pv = score(model, val_dl, device, use_waveform)
        val_auc = safe_auc(yv, pv)
        scheduler.step(val_auc if np.isfinite(val_auc) else 0.0)

        print(f"    [seed {seed}] epoch {epoch + 1}/{args.epochs} "
              f"train loss {total / max(len(train_dl), 1):.4f} "
              f"val AUC {val_auc:.4f}")

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, no_improve = val_auc, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"    [seed {seed}] early stop: no val-AUC gain for "
                      f"{args.patience} epochs")
                break

    if best_epoch < 0:
        print(f"    [seed {seed}] validation AUC was never finite -- the split "
              f"likely holds one class. Scoring the final weights instead.")
    else:
        model.load_state_dict(torch.load(ckpt, weights_only=True))

    yt, pt = score(model, test_dl, device, use_waveform)
    print(f"    [seed {seed}] test AUC {safe_auc(yt, pt):.4f} "
          f"(best val {best_auc:.4f} @ epoch {best_epoch + 1})")
    return yt, pt, model.n_params()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_waveform = args.raw_f32 is not None

    cols = args.feature_columns.split(",") if args.feature_columns else None
    data = RustData.open(features=args.features, raw=args.raw_f32, columns=cols)

    # Prefer the RFE subset when the table has it and nothing was asked for.
    names = data.feature_names
    if cols is None and all(c in names for c in OPTIMIZED_CATALOG_FEATURES):
        keep = [names.index(c) for c in OPTIMIZED_CATALOG_FEATURES]
        feats = data.features[:, keep]
        names = OPTIMIZED_CATALOG_FEATURES
    else:
        feats = data.features

    hours = data.hour_index
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    labels = label_hours(hours, major_times, args.horizon_days)
    dsp = days_since_prev_major(hours, major_times)

    print("=" * 66)
    print(f"GRU/CNN forecaster | {'catalog + waveform' if use_waveform else 'catalog only'}")
    print(f"  hours {len(hours)}  {hours[0]} .. {hours[-1]}")
    print(f"  features {len(names)}: {', '.join(names)}")
    if use_waveform:
        print(f"  waveform {data.waveform.shape} @ {data.fs} Hz")
    print(f"  M>={args.threshold} within {args.horizon_days:g} d | "
          f"positives {labels.sum()}/{len(labels)} ({labels.mean():.3%}) | "
          f"{len(major_times)} catalog events")
    print(f"  seq_len {args.seq_len} h | device {device}")
    print("=" * 66)

    if labels.sum() == 0:
        raise SystemExit(
            "No positive hours: every label is 0. Lower --threshold or raise "
            "--horizon-days; there is nothing to learn here.")

    # Input overlap plus the label's full forward reach.
    embargo = args.seq_len - 1 + int(round(args.horizon_days * 24))
    valid_ends = np.arange(args.seq_len - 1, len(hours))
    folds = walk_forward_splits(valid_ends, args.cv_folds, labels=labels[valid_ends],
                                embargo=embargo)
    print(f"embargo {embargo} h ({args.seq_len - 1} input overlap + "
          f"{int(round(args.horizon_days * 24))} label horizon)\n")

    seeds = [int(s) for s in args.ensemble_seeds.split(",")]
    fold_rows = []

    for k, ends in enumerate(folds, start=1):
        tr, va, te = ends
        n_pos = int(labels[te].sum()) if len(te) else 0
        n_neg = len(te) - n_pos
        print(f"--- fold {k}/{len(folds)} | train {len(tr)} val {len(va)} test {len(te)} "
              f"| test {n_pos} pos / {n_neg} neg")

        # A split whose minority class is a handful of hours cannot produce a
        # meaningful AUC -- a single flipped ranking moves it by a large
        # fraction. Refusing to print a number is more useful than printing one
        # that will not reproduce.
        minority = min(n_pos, n_neg)
        if len(te) == 0 or labels[tr].sum() == 0 or minority < args.min_minority:
            print(f"    skipped: test minority class is {minority} hour(s), under "
                  f"--min-minority {args.min_minority}. With a 14-day horizon on a "
                  f"short archive the label is mostly 1, so most windows are "
                  f"positive; shorten --horizon-days or raise --threshold to get a "
                  f"split worth scoring.\n")
            continue

        train_ds, train_dl, val_dl, test_dl = build_loaders(
            feats, labels, (tr, va, te), args.seq_len, args, data.waveform)

        per_seed, y_ref, n_params = [], None, 0
        for seed in seeds:
            yt, pt, n_params = train_one_seed(
                args, seed, f"fold{k}", train_ds, train_dl, val_dl, test_dl,
                feats.shape[1], device)
            y_ref = yt if y_ref is None else y_ref
            per_seed.append(pt)

        aucs = [safe_auc(y_ref, p) for p in per_seed]
        ens = np.mean(per_seed, axis=0)
        ens_auc = safe_auc(y_ref, ens)

        # Persistence: rank by recency of the last event. Oriented, because a
        # reliably-inverted rule is just as usable once flipped.
        pers = safe_auc(y_ref, -dsp[te], oriented=True)
        floor = max(0.5, pers)

        print(f"\n  model parameters: {n_params:,} | train windows {len(train_ds)} "
              f"({n_params / max(len(train_ds), 1):.1f} params/sample)")
        print(f"  per-seed test AUC: {[f'{a:.4f}' for a in aucs]}")
        print(f"    mean {np.mean(aucs):.4f}  std {np.std(aucs):.4f}  "
              f"spread {max(aucs) - min(aucs):.4f}")
        if any(a < 0.5 for a in aucs):
            print("  !! a seed scored below chance; treat this fold as a failed run, "
                  "not a data point")
        print(f"  base rate (train {labels[tr].mean():.3%}) AUC 0.5000")
        print(f"  persistence (days since last event) AUC {pers:.4f}  <- the bar")
        print(f"  ENSEMBLE {ens_auc:.4f} vs floor {floor:.4f} -> "
              f"{ens_auc - floor:+.4f}\n")

        fold_rows.append((k, ens_auc, floor, np.mean(aucs), np.std(aucs)))

    if not fold_rows:
        raise SystemExit("No fold could be scored.")

    print("=" * 66)
    print(f"{'fold':>5} {'ensemble':>9} {'floor':>8} {'edge':>8} {'seed std':>9}")
    for k, e, f, _m, s in fold_rows:
        print(f"{k:>5} {e:>9.4f} {f:>8.4f} {e - f:>+8.4f} {s:>9.4f}")
    edges = [e - f for _, e, f, _m, _s in fold_rows]
    beat = sum(1 for x in edges if x > 0)
    print(f"\nbeats the persistence floor in {beat}/{len(fold_rows)} fold(s); "
          f"mean edge {np.mean(edges):+.4f}")
    if beat < len(fold_rows):
        print("Not clearing the floor on every fold is the honest headline here.")
    print("=" * 66)


if __name__ == "__main__":
    main()
