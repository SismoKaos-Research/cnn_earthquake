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
from seismolib.model.registry import ModelSpec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split-by", default=None,
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

    # The protocol comes from the checkpoint's own record when it has one.
    # This defaulted to "both" while the trainer defaulted to "event", so
    # scoring a checkpoint without repeating the flag re-derived a DIFFERENT
    # test set than the training run reported on -- same model, same weights, a
    # number for a question nobody asked.
    proto = ModelSpec.load_extra(Path(args.ckpt).parent, "protocol") or {}
    if args.split_by is None:
        args.split_by = proto.get("split_by") or "event"
        src = "model.json" if proto.get("split_by") else "fallback"
        print(f"  [split] --split-by {args.split_by!r} (from {src})")
    elif proto.get("split_by") and proto["split_by"] != args.split_by:
        print(f"  [split] ** you passed --split-by {args.split_by!r} but this "
              f"checkpoint was trained with {proto['split_by']!r}; the test set "
              f"below is NOT the one its training log reported on **")
    if proto.get("seed_split") is not None and args.seed_split != proto["seed_split"]:
        print(f"  [split] ** --seed-split {args.seed_split} != the checkpoint's "
              f"{proto['seed_split']}; a different partition is being scored **")


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
    # The saved spec wins over the flags. --channels defaults to "2d+aux" here
    # and to "all" in the trainer, so scoring a --channels all run without
    # saying so builds a different network; load_state_dict then either raises
    # or, if the shapes happen to line up, quietly reports another model's
    # numbers. A run that wrote `model.json` beside its weights no longer needs
    # its geometry retyped, and a disagreement is named rather than guessed at.
    spec = ModelSpec.load(Path(args.ckpt).parent)
    if spec is None:
        print(f"  [spec] no model.json beside {args.ckpt} -- using the flags as "
              f"given (channels={args.channels}, hidden={args.hidden}, "
              f"fusion_dim={args.fusion_dim}); check they match the run")
        spec = ModelSpec(model="dual-channel", branch="lstm",
                         params={"channels": args.channels, "hidden": args.hidden,
                                 "fusion_dim": args.fusion_dim, "fusion": "linear",
                                 "dropout": 0.3, "lstm_layers": 1, "lstm_heads": 4})
    else:
        print(f"  [spec] {spec.describe()}   (from model.json)")
        asked = {"channels": args.channels, "hidden": args.hidden,
                 "fusion_dim": args.fusion_dim}
        clash = {k: (v, spec.params.get(k)) for k, v in asked.items()
                 if k in spec.params and v != spec.params[k]}
        if clash:
            print("  [spec] flags disagree with the saved spec; the SPEC is used:")
            for k, (was, now) in clash.items():
                print(f"           --{k.replace('_', '-')} {was!r} ignored, {now!r} used")

    model = spec.build(seq_dim=seq.shape[-1], img_channels=img.shape[0],
                       aux_dim=aux.numel(), squeeze_output=True).to(device)
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
