"""Where does the magnitude regressor's error actually live?

The training logs report one MAE per run and nothing else. That is enough to
say the model beats its floor and not enough to say anything operational: a
magnitude estimate is used to decide whether to warn, and the decision is
sensitive to error *at the large end*, where the training set is thinnest.

This scores a trained checkpoint's own test split and breaks the error down by
magnitude, epicentral distance and SNR, against the same ridge(log_snr,
log_distance) floor the training run uses. No training happens here.

**The split is reproduced, not re-drawn.** `resplit` is deterministic given
(how, seed_split, detector_manifest), so passing the same arguments the run used
recovers exactly the rows it tested on. Anything else would score the model on
its own training data -- the failure that cost this project a retraction once
already.

**Aux standardization comes from TRAIN.** `DualMagnitudeDataset` fits (mu, sd)
from whichever split it is handed, so the test split must be given the train
split's stats explicitly; letting it fit its own would leak test statistics into
the input and quietly improve the number.

    python3 scripts/magnitude_error_profile.py \\
        --dataset-dir .../dataset_magreg_catalog_6s \\
        --ckpt trained_model_magreg_grid/best_..._both_seed42_split42_pid58288.pth \\
        --split-by both --seed-split 42 --out mag_profile_p42.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magnitude.cnn_lstm_regression import (DualChannelRegressionNet,
                                           DualMagnitudeDataset, resplit)
from magnitude.cnn_regression import AUX_COLUMNS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split-by", default="both",
                   choices=["event", "station", "both", "detector"])
    p.add_argument("--seed-split", type=int, default=42)
    p.add_argument("--detector-manifest", default=None)
    p.add_argument("--channels", default="2d+aux")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default=None, help="per-row predictions CSV")
    return p.parse_args()


@torch.no_grad()
def predict(model, ds, batch_size, device):
    """Returns predicted magnitude for every row of `ds`, in order."""
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for seq, img, aux, _ in loader:
        out.append(model(seq.to(device), img.to(device),
                         aux.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def band_table(df, col, bins, label, floor_col="ridge_ae"):
    """Prints MAE and the floor's MAE within each bin of `col`."""
    print(f"\n  error by {label}")
    print(f"    {'band':>14}{'n':>8}{'MAE':>9}{'ridge':>9}{'ratio':>8}{'bias':>9}")
    cut = pd.cut(df[col], bins=bins, include_lowest=True)
    for b, g in df.groupby(cut, observed=True):
        if len(g) < 20:
            continue
        mae, rid = g.ae.mean(), g[floor_col].mean()
        print(f"    {str(b):>14}{len(g):>8,}{mae:>9.4f}{rid:>9.4f}"
              f"{mae / max(rid, 1e-9):>8.3f}{g.err.mean():>+9.4f}")


def main():
    """Scores one checkpoint's own test split and profiles the error."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.dataset_dir)

    man = pd.read_csv(root / "manifest.csv")
    # log_distance is derived at load time by the training script, not stored
    # in the manifest; deriving it identically here keeps the aux vector the
    # model sees the same as the one it was trained on.
    man["log_distance"] = np.log(man["distance_km"].clip(lower=1.0))
    man = resplit(man, args.split_by, seed=args.seed_split,
                  detector_manifest=args.detector_manifest)
    tr = man[man.split == "train"]
    te = man[man.split == "test"]
    print(f"[split] how={args.split_by} seed={args.seed_split}  "
          f"train {len(tr):,}  test {len(te):,}")
    shared_ev = len(set(tr.event_id) & set(te.event_id))
    shared_st = len(set(tr.station_key) & set(te.station_key))
    print(f"[split] shared events {shared_ev}, shared stations {shared_st}"
          + ("  (clean)" if shared_ev == 0 else "  ** LEAK **"))
    if shared_ev:
        sys.exit("test split shares events with train -- refusing to report")

    ds_tr = DualMagnitudeDataset(tr, root)
    ds_te = DualMagnitudeDataset(te, root, aux_stats=ds_tr.aux_stats())

    seq, img, aux, _ = ds_te[0]
    model = DualChannelRegressionNet(seq.shape[-1], img.shape[0], aux.numel(),
                                     hidden=args.hidden, fusion_dim=args.fusion_dim,
                                     channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    pred = predict(model, ds_te, args.batch_size, device)
    true = te.magnitude.to_numpy(dtype=np.float64)

    # The floor is refitted on TRAIN and applied to TEST, exactly as the
    # training run does -- a floor fitted on test would be a different, easier
    # baseline and would understate the model's edge.
    ok_tr = tr[AUX_COLUMNS].notna().all(axis=1)
    ridge = Ridge(alpha=1.0).fit(tr.loc[ok_tr, AUX_COLUMNS].to_numpy(),
                                 tr.loc[ok_tr, "magnitude"].to_numpy())
    aux_te = te[AUX_COLUMNS].to_numpy(dtype=np.float64)
    aux_te = np.nan_to_num(aux_te, nan=np.nanmedian(aux_te))
    ridge_pred = ridge.predict(aux_te)

    df = pd.DataFrame({
        "magnitude": true, "pred": pred, "err": pred - true,
        "ae": np.abs(pred - true), "ridge_ae": np.abs(ridge_pred - true),
        "distance_km": te.distance_km.to_numpy(),
        "log_snr": te.log_snr.to_numpy(),
    })

    print(f"\n{'=' * 66}\nMAGNITUDE ERROR PROFILE  ({Path(args.ckpt).name[:40]}...)\n{'=' * 66}")
    print(f"  n={len(df):,}   MAE {df.ae.mean():.4f}   ridge floor {df.ridge_ae.mean():.4f}"
          f"   ratio {df.ae.mean() / df.ridge_ae.mean():.3f}")
    print(f"  bias {df.err.mean():+.4f}   residual std {df.err.std():.4f}"
          f"   max |err| {df.ae.max():.3f}")

    band_table(df, "magnitude", [0, 2, 2.5, 3, 3.5, 4, 10], "magnitude")
    band_table(df, "distance_km", [0, 25, 50, 100, 200, 1e5], "distance (km)")
    band_table(df, "log_snr", 5, "log SNR (equal-width)")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
