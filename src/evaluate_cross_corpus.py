"""
Evaluates already-trained `cnn_lstm_classify.py` checkpoints on a dataset they
were never trained on -- the cross-corpus test.

`cnn_lstm_classify.py` always trains before it scores, so it cannot answer
"how does the Turkish-trained detector do on STEAD?". This script loads the
checkpoints and only does inference.

**Read the stratified table, not the headline.** STEAD is not merely a
different corpus, it is a harder one: median event M1.09 against the Turkish
corpus's M2.30, 44% of its events below M1.0 (which is below the Turkish
minimum of M2.0 entirely), and epicentral distances to 329 km where the
Turkish download radius stops at ~56 km. Only 7.1% of STEAD matches the
training distribution. A single pooled number therefore mixes "fails to
generalize" with "was asked a harder question", and the two need separating
before anything is claimed. Passing `--manifest` breaks the score out by
magnitude and distance, which is the comparison that actually supports a claim.

**AUC transfers; thresholded metrics do not.** STEAD noise sits ~2x higher on
the amplitude scale than Turkish noise (median `seq` std 0.98 vs 0.47), an
artifact of how each corpus's noise baseline is estimated rather than of the
seismology. Ranking within STEAD is unaffected, so AUC and PR-AUC stay
meaningful; accuracy, MCC and Brier at the trained 0.5 threshold do not, and
are printed only for completeness. Recalibrate on a held-out STEAD split
before quoting them.

Usage:
    python3 evaluate_cross_corpus.py \\
        --checkpoints 'trained_model_cnnlstm_classify/best_*2d*seed*.pth' \\
        --dataset-dir <stead dataset root with 00_noise/ and 01_earthquake/> \\
        --channels 2d --manifest <that dataset>/manifest.csv
"""

import argparse
import csv
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset, \
    trivial_amplitude_floor
from seismolib.metrics import binary_report, print_report, safe_auc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", required=True,
                   help="Glob or comma-separated .pth paths. Multiple are "
                        "probability-averaged, matching how the training script "
                        "reports its ensemble.")
    p.add_argument("--dataset-dir", required=True,
                   help="Root holding 00_noise/ and 01_earthquake/ subdirectories.")
    p.add_argument("--manifest", default=None,
                   help="manifest.csv with magnitude/distance_km, enabling the "
                        "stratified breakdown. Strongly recommended.")
    # These must match the values the checkpoints were trained with; a mismatch
    # surfaces as a state_dict error rather than a silently wrong number.
    p.add_argument("--channels", default="2d", choices=["all", "1d", "2d"])
    p.add_argument("--fusion", default="linear", choices=["linear", "gate"])
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def resolve_checkpoints(spec):
    paths = []
    for part in spec.split(","):
        part = part.strip()
        paths.extend(sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part])
    if not paths:
        raise SystemExit(f"No checkpoints matched: {spec!r}")
    return paths


@torch.no_grad()
def score_dataset(model, loader, device):
    probs, labels = [], []
    model.eval()
    for seq, img, y in loader:
        logits = model(seq.to(device), img.to(device))
        probs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def load_strata(manifest_path, n_expected):
    """filename -> (magnitude, distance_km); missing values become NaN."""
    if not manifest_path:
        return {}
    strata = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            name = row.get("filename")
            if not name:
                continue

            def num(key):
                v = (row.get(key) or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return float("nan")
            strata[name] = (num("magnitude"), num("distance_km"))
    return strata


def stratified_report(y, probs, names, strata):
    """AUC within magnitude and distance bands, each against its own band's noise.

    Noise carries no magnitude, so every band is scored against the FULL noise
    set rather than a slice of it. That keeps each band's AUC on the same
    question -- "can this event be told from noise" -- instead of silently
    changing the negative class between rows.
    """
    if not strata:
        return
    mags = np.array([strata.get(n, (np.nan, np.nan))[0] for n in names])
    dist = np.array([strata.get(n, (np.nan, np.nan))[1] for n in names])
    is_noise = y == 0

    for label, values, bands in (
        ("magnitude", mags, [(0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
                             (2.5, 3.0), (3.0, 99)]),
        ("distance km", dist, [(0, 25), (25, 56), (56, 100), (100, 400)]),
    ):
        print(f"\n  --- by {label} (each band vs the full noise set) ---")
        for lo, hi in bands:
            sel = (y == 1) & (values >= lo) & (values < hi)
            n_ev = int(sel.sum())
            if n_ev < 30:
                print(f"    {lo:>5.1f}-{hi:<5.1f}  n={n_ev:<6d} (too few to score)")
                continue
            mask = sel | is_noise
            auc = safe_auc(y[mask], probs[mask])
            print(f"    {lo:>5.1f}-{hi:<5.1f}  n={n_ev:<6d} AUC {auc:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = resolve_checkpoints(args.checkpoints)

    ds = RamDualTensorDataset(args.dataset_dir)
    seq_shape, img_shape = ds.validate_shapes()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print("=" * 66)
    print(f"CROSS-CORPUS EVALUATION | channels='{args.channels}' fusion='{args.fusion}'")
    print(f"  dataset {args.dataset_dir}")
    print(f"  seq {seq_shape} | img {img_shape} | n = {len(ds)}")
    counts = defaultdict(int)
    for _f, lbl in ds.samples:
        counts[lbl] += 1
    print(f"  class counts: {dict(counts)}")
    print(f"  {len(paths)} checkpoint(s)")
    print("=" * 66)

    per_ckpt, y_ref = [], None
    for path in paths:
        model = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                     fusion_dim=args.fusion_dim, dropout=args.dropout,
                                     channels=args.channels, fusion=args.fusion).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        y, probs = score_dataset(model, loader, device)
        y_ref = y if y_ref is None else y_ref
        per_ckpt.append(probs)
        print(f"  {path.split('/')[-1]:<70s} AUC {safe_auc(y, probs):.4f}")

    aucs = [safe_auc(y_ref, p) for p in per_ckpt]
    print(f"\n  per-checkpoint AUC: {[f'{a:.4f}' for a in aucs]}")
    print(f"    mean {np.mean(aucs):.4f}  std {np.std(aucs):.4f}  "
          f"spread {max(aucs) - min(aucs):.4f}")
    inverted = [(p, a) for p, a in zip(paths, aucs) if a < 0.5]
    for p, a in inverted:
        print(f"  !! {p} scored {a:.4f} -- BELOW CHANCE; excluded results would differ")

    print("\n--- Trivial floors on THIS corpus ---")
    floors = trivial_amplitude_floor(ds, y_ref)
    for name, val in floors.items():
        print(f"  {name:<22s} AUC {val:.4f}   (single scalar, no learning)")
    best_name, best = max(floors.items(), key=lambda kv: kv[1])
    print(f"  -> strongest trivial floor: {best_name} at {best:.4f}")

    ensemble = np.mean(per_ckpt, axis=0)
    report = binary_report(y_ref, ensemble, y_pred=(ensemble > 0.5).astype(float))
    print_report(f"Cross-corpus [{args.channels}/{args.fusion}] "
                 f"({len(paths)}-checkpoint ensemble)", report)
    edge = report["roc_auc"] - best
    print(f"\n  ROC-AUC {report['roc_auc']:.4f} vs {best_name} floor {best:.4f} "
          f"-> {'+' if edge >= 0 else ''}{edge:.4f}   <- the number that matters")
    print("  (accuracy/MCC/Brier above use the 0.5 threshold carried over from "
          "training and are NOT calibrated for this corpus -- compare via AUC.)")

    names = [f.name for f, _ in ds.samples]
    stratified_report(y_ref, ensemble, names, load_strata(args.manifest, len(ds)))


if __name__ == "__main__":
    main()
