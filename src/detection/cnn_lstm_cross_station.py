"""
Cross-station generalization test: train the raw-waveform CNN+LSTM forecaster
entirely on one station's archive, evaluate on a second station's archive
that the model never saw during training.

Every other script here validates via chronological walk-forward CV within
a single station's continuous recording -- a real test of temporal
generalization, but it can't tell us whether the model learned genuine
waveform-forecasting signal or station-specific artifacts (site response,
instrument noise characteristics, local geology) that happen to correlate
with the regional catalog label. Training on BODT and testing on DAT
(Datça) -- a different station recording the same regional (Aegean-zone
catalog-defined) seismicity -- is the same logic as the "lights and
shadows" survey paper's LA-trained/Tokyo-tested check: if performance
survives moving to unseen station, that's real evidence; if it collapses,
today's within-BODT results were partly station-specific noise.

The test archive is normalized using the TRAINING archive's own stats (not
its own), matching how the model would actually be deployed -- you would
not know a new station's distribution in advance.

Usage:
    python cnn_lstm_cross_station.py \\
        --train-data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --test-data-root ../../Sismokaos/feature-extract/data/aegean_dat_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --horizon-days 14

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader

from seismolib.catalog import days_since_prev_major, label_hours, load_aegean_events, truncate_to_reliable_catalog_end
from seismolib.metrics import safe_auc
from seismolib.metrics import binary_report, print_report
from forecasting.raw_cnn_lstm_forecast import RawCNNLSTM
from seismolib.waveform import RawSeqDataset, load_hourly_raw, load_hourly_raw_consolidated
from seismolib.training import seed_everything


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--train-data-root", required=True, help="Station used for training+val.")
    p.add_argument("--test-data-root", required=True, help="Different station, used entirely as test.")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--stations", nargs="+", default=["BODT", "DAT"], metavar="NAME",
                  help="Stations whose distance --max-station-dist-km is measured from "
                       "(nearest one wins). Names index STATION_COORDS. Both stations in a "
                       "cross-station run should normally be listed, so train and test see "
                       "the same event set.")
    p.add_argument("--max-station-dist-km", type=float, default=None,
                  help="Keep only events within this many km of the nearest --stations "
                       "entry. Off by default (whole AEGEAN_BBOX). Only 1 M>=4.5 event in "
                       "the archive window falls within 100 km of BODT against 34 bbox-wide, "
                       "so most labelled events sit far outside plausible sensing range. "
                       "150 pairs well with --threshold 3.5 (66 such events near BODT/DAT).")
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--consolidated", action="store_true", default=True)
    p.add_argument("--seq-hours", type=int, default=24)
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--test-after-train", action="store_true",
                  help="Restrict the test station to hours strictly AFTER the training "
                       "window ends (plus embargo). Without it the two stations span the "
                       "same period and share the same catalog labels, so hour H is in "
                       "train (station A) and test (station B) with an identical label -- "
                       "a model can ride shared seasonal/cultural noise structure from "
                       "noise signature to period to label and appear to transfer with no "
                       "precursor signal. Costs test hours; pair with a lower --train-frac.")
    p.add_argument("--train-frac", type=float, default=0.85,
                  help="Fraction of the training station's timeline used for training; the "
                       "rest is validation (for early stopping). No test split is carved from "
                       "the training station -- the entire test station is the test set.")
    return p.parse_args()


def train_one_seed(args, seed, raw_tr, labels_tr, train_idx, val_idx,
                   raw_te, labels_te, test_idx, device):
    """Trains on the training station, evaluates on the held-out test station.

    Args:
        args: Parsed CLI args.
        seed: Random seed for init/shuffling.
        raw_tr: Training station's hourly raw waveform array.
        labels_tr: Training station's per-hour binary labels.
        train_idx: Window end-indices for the training split (into raw_tr).
        val_idx: Window end-indices for the validation split (into raw_tr).
        raw_te: Test station's hourly raw waveform array.
        labels_te: Test station's per-hour binary labels.
        test_idx: Window end-indices for the test split (into raw_te) --
            every eligible hour, the whole test station is used.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_score) arrays for the test station, from the
        best (by val AUC) epoch's weights.
    """
    seed_everything(seed)
    train_ds = RawSeqDataset(raw_tr, labels_tr, args.seq_hours, train_idx)
    val_ds = RawSeqDataset(raw_tr, labels_tr, args.seq_hours, val_idx, stats=train_ds.stats)
    # test station normalized with the TRAINING station's stats -- deployment-realistic,
    # we would not know the new station's own distribution in advance.
    test_ds = RawSeqDataset(raw_te, labels_te, args.seq_hours, test_idx, stats=train_ds.stats)

    model = RawCNNLSTM(cnn_out=args.cnn_out, hidden=args.hidden, dropout=args.dropout).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=2)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = labels_tr[train_idx].mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for seq, y in loader:
                seq, y = seq.to(device), y.to(device)
                logit = model(seq)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    best = -1.0
    no_improve, best_state = 0, None
    for epoch in range(args.epochs):
        model.train()
        for seq, y in train_loader:
            seq, y = seq.to(device), y.to(device)
            loss = criterion(model(seq), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()

        yv, sv, val_loss = evaluate(val_loader)
        val_auc = safe_auc(yv, sv)
        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val AUC {val_auc:.4f} val loss {val_loss:.4f}")
        improved = val_auc > best
        if improved:
            best, no_improve = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = evaluate(test_loader)
    print(f"  [seed {seed}] test (cross-station) AUC {safe_auc(yt, st):.4f}")
    return yt, st


def main():
    """Loads both stations' archives, trains on one, evaluates on the other."""
    args = parse_args()

    print(f"Loading training station: {args.train_data_root}")
    hour_index_tr, raw_tr = load_hourly_raw_consolidated(args.train_data_root)
    print(f"Loading test station: {args.test_data_root}")
    hour_index_te, raw_te = load_hourly_raw_consolidated(args.test_data_root)

    major_times = load_aegean_events(args.catalog_path, args.threshold,
                                     stations=args.stations,
                                     max_dist_km=args.max_station_dist_km)
    if args.max_station_dist_km:
        print(f"  [distance cap] events restricted to <= {args.max_station_dist_km:.0f} km "
             f"from nearest of {args.stations}")
    print(f"  train: {len(hour_index_tr)} hours {raw_tr.shape}; test: {len(hour_index_te)} hours "
         f"{raw_te.shape}; {len(major_times)} M>={args.threshold} AEGEAN events in the full catalog")

    hour_index_tr, raw_tr = truncate_to_reliable_catalog_end(hour_index_tr, raw_tr, major_times,
                                                              buffer_days=args.horizon_days)
    hour_index_te, raw_te = truncate_to_reliable_catalog_end(hour_index_te, raw_te, major_times,
                                                              buffer_days=args.horizon_days)

    labels_tr = label_hours(hour_index_tr, major_times, args.horizon_days)
    labels_te = label_hours(hour_index_te, major_times, args.horizon_days)
    dsp_te = days_since_prev_major(hour_index_te, major_times)

    n_tr, n_te = len(hour_index_tr), len(hour_index_te)
    # The label looks horizon_days forward, so a seq_hours-1 gap alone leaves the last
    # ~horizon_days of train carrying labels decided by events inside val.
    embargo = args.seq_hours - 1 + int(round(args.horizon_days * 24))
    valid_tr = np.arange(args.seq_hours - 1, n_tr)
    i_split = int(len(valid_tr) * args.train_frac)
    train_idx = valid_tr[:i_split]
    val_idx = valid_tr[i_split + embargo:]

    test_idx = np.arange(args.seq_hours - 1, n_te)  # entire test station
    if args.test_after_train:
        # Station split alone controls for site response and instrument character, but
        # NOT for shared time: both stations span the same window and are labelled from
        # the same catalog, so hour H appears in training (station A, label L(H)) and in
        # test (station B, the SAME L(H)). Ambient noise carries strong seasonal and
        # cultural structure that two stations 44 km apart share, so a model can map
        # noise signature -> period -> label and "transfer" with no precursor signal at
        # all. Holding the test station to hours strictly after training ends removes
        # that path, making this a station AND time holdout.
        train_end = hour_index_tr[train_idx[-1]]
        cutoff = train_end + pd.Timedelta(hours=embargo)
        test_idx = test_idx[hour_index_te[test_idx] > cutoff]
        print(f"  [test-after-train] test station restricted to hours after {cutoff} "
             f"(train ends {train_end}, +{embargo}h embargo)")
        if len(test_idx) < 100:
            raise SystemExit(f"[ERROR] --test-after-train left only {len(test_idx)} test "
                             f"windows. Lower --train-frac so the training window ends "
                             f"earlier, or drop the flag.")

    print(f"\n  train (BODT-side): n={len(train_idx)} positive rate {labels_tr[train_idx].mean():.3f}")
    print(f"  val   (BODT-side): n={len(val_idx)} positive rate {labels_tr[val_idx].mean():.3f}")
    print(f"  test  (DAT, entire station): n={len(test_idx)} positive rate {labels_te[test_idx].mean():.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw_tr, labels_tr, train_idx, val_idx,
                                raw_te, labels_te, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    ensemble_score = np.mean(per_seed_scores, axis=0)

    print("\n--- Floors (test = DAT station) ---")
    pos_tr = labels_tr[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    print(f"  base-rate (BODT train majority)   AUC {base_auc:.4f}   n={len(yt_ref)}")
    pers_dsp = dsp_te[test_idx]
    pers_pred = np.where(np.isnan(pers_dsp), 0, (pers_dsp <= args.horizon_days).astype(int)).astype(np.float64)
    pers_auc = safe_auc(yt_ref, pers_pred)
    single_class = len(np.unique(yt_ref)) < 2
    pers_brier = float("nan") if single_class else float(brier_score_loss(yt_ref, pers_pred))
    print(f"  persistence (DAT's own catalog)   AUC {pers_auc:.4f}   Brier {pers_brier:.4f}   n={len(yt_ref)}")

    print(f"\n--- Cross-station CNN-LSTM (trained BODT, tested DAT) ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    report = binary_report(yt_ref, ensemble_score)
    bss = (float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
          else 1.0 - report["brier"] / pers_brier)
    report["brier_skill_score_vs_persistence"] = bss
    print_report("Cross-station CNN-LSTM ensemble (BODT->DAT, test set)", report)


if __name__ == "__main__":
    main()
