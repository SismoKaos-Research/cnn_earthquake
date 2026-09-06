"""What does the detector actually detect? Recall stratified by magnitude, SNR and distance.

A single ROC-AUC says how well the detector ranks windows; it does not say
which earthquakes it finds. This script answers the operational question --
"recall X% for magnitude >= M within D km" -- which is the form a deployment
or a review needs, and which the cascade inherits directly: a missed event can
never be assigned a magnitude downstream.

No training. The detection manifest carries no source parameters, but every
detection window joins the magnitude dataset's manifest **by filename** (a
100% join for the 6s catalog build), and that manifest carries `magnitude`,
`log_snr` and `distance_km`.

Checkpoints are selected by exact branch tag, not a substring: the save dir
holds every arm of the architecture grid, and a naive `*cnn*` glob would sweep
`cnn-lstm` checkpoints into a `cnn` ensemble.

Usage:
    python3 src/sismokaos/detection/operating_envelope.py \\
        --detector-dir  .../dataset_specdual_catalog_6s_matched_hard \\
        --magnitude-dir .../dataset_magreg_catalog_6s \\
        --ckpt-dir trained_model_branch1d_asinh --branch-1d cnn-lstm
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from sismokaos.detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset
from sismokaos.checkpoints import find_checkpoints


def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Detector recall by magnitude, SNR and distance.")
    p.add_argument("--detector-dir", required=True)
    p.add_argument("--magnitude-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--branch-1d", default="cnn-lstm", choices=["lstm", "cnn", "cnn-lstm"])
    p.add_argument("--channels", default="1d", choices=["all", "1d", "2d"])
    p.add_argument("--fusion", default="linear", choices=["linear", "gate"])
    p.add_argument("--seq-transform", default="asinh", choices=["none", "asinh"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-csv", default=None,
                   help="Optional path to write the per-window scored table.")
    return p.parse_args()


@torch.no_grad()
def score(ckpts, ds, args, device):
    """Probability-averaged ensemble over `ckpts`.

    Returns:
        Tuple of (probs, labels, filenames) aligned to `ds.samples`.
    """
    seq_shape, img_shape = ds.sample_shapes()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    per_ckpt = []
    for c in ckpts:
        model = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                     fusion_dim=args.fusion_dim, channels=args.channels,
                                     fusion=args.fusion, branch1d=args.branch_1d).to(device)
        model.load_state_dict(torch.load(c, weights_only=True))
        model.eval()
        probs = []
        for seq, img, _ in loader:
            out = model(seq.to(device), img.to(device))
            probs.extend(torch.sigmoid(out).float().cpu().squeeze(1).tolist())
        per_ckpt.append(np.asarray(probs))
        print(f"  scored {c.name.split('_seed')[-1]}")
    labels = np.asarray([lbl for _, lbl in ds.samples])
    names = [Path(f).name for f, _ in ds.samples]
    return np.mean(per_ckpt, axis=0), labels, names


def strat(df, col, bins, label, threshold):
    """Prints recall within each bin of `col`."""
    print(f"\n  recall by {label}")
    print(f"    {'bin':>16}{'n':>8}{'detected':>10}{'recall':>9}{'mean p':>9}")
    cut = pd.cut(df[col], bins=bins, include_lowest=True)
    for b, g in df.groupby(cut, observed=True):
        det = int((g.prob > threshold).sum())
        print(f"    {str(b):>16}{len(g):>8}{det:>10}{det / len(g):>9.4f}{g.prob.mean():>9.4f}")


def main():
    """Scores the detector's test events and reports recall by source parameter."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = find_checkpoints(args.ckpt_dir, args.channels, args.fusion, args.branch_1d)
    print(f"[ensemble] {len(ckpts)} checkpoint(s) for "
          f"channels={args.channels} branch-1d={args.branch_1d}")

    ds = RamDualTensorDataset(f"{args.detector_dir}/test", seq_transform=args.seq_transform)
    probs, labels, names = score(ckpts, ds, args, device)
    print(f"\n[check] test ROC-AUC {roc_auc_score(labels, probs):.4f} "
          f"(should match the training log's ensemble figure)")

    mag = pd.read_csv(Path(args.magnitude_dir) / "manifest.csv")
    ev = pd.DataFrame({"filename": names, "prob": probs, "label": labels})
    ev = ev[ev.label == 1].merge(
        mag[["filename", "magnitude", "log_snr", "distance_km"]], on="filename", how="left")

    missing = int(ev.magnitude.isna().sum())
    print(f"[join] {len(ev) - missing}/{len(ev)} event windows carry source parameters"
          + (f"  ({missing} unmatched, excluded)" if missing else ""))
    ev = ev.dropna(subset=["magnitude"])

    det = int((ev.prob > args.threshold).sum())
    print(f"\n{'=' * 62}\nOPERATING ENVELOPE  (threshold {args.threshold})\n{'=' * 62}")
    print(f"  overall recall {det / len(ev):.4f}  ({det}/{len(ev)}, "
          f"{len(ev) - det} missed)")

    strat(ev, "magnitude", [0, 1.5, 2.0, 2.5, 3.0, 3.5, 10], "magnitude", args.threshold)
    strat(ev, "log_snr", 6, "log SNR (equal-width bins)", args.threshold)
    strat(ev, "distance_km", [0, 25, 50, 100, 200, 1e5], "epicentral distance (km)",
          args.threshold)

    miss = ev[ev.prob <= args.threshold]
    print(f"\n  missed events (n={len(miss)}) vs detected (n={det}):")
    for c in ("magnitude", "log_snr", "distance_km"):
        print(f"    {c:<14} missed median {miss[c].median():8.3f}   "
              f"detected median {ev.loc[ev.prob > args.threshold, c].median():8.3f}")

    if args.out_csv:
        ev.to_csv(args.out_csv, index=False)
        print(f"\n  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
