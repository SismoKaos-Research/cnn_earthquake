"""How does a detector behave when the negative regime shifts under it?

A deployed station sees the whole noise distribution, not a curated one. So a
number measured against one choice of negatives says little on its own. This
scores the same trained arms against every negative regime built over the
*same* P-only event windows, which isolates negative selection as the only
variable -- verified: all builds share the same 35 test stations and the same
7,908 event windows.

The regimes answer different questions:

  matched   noise amplitude distribution mirrors the events'. Amplitude is
            neutralised where the pool allows, so what remains is waveform
            character. This is a measurement instrument, not a deployment
            distribution.
  band      75th-99th percentile of the pool -- loud noise only. Hard, but it
            puts a floor under negative amplitude that positives lack, which
            on a P-only window makes P(event|amplitude) U-shaped.
  wideband  full pool below the 99th percentile, spread evenly across
            quantiles. Coverage across the range.
  natural   no mining at all; noise follows the pool's own density. This is
            the regime a station actually sees.

**Every regime has its own floor, so raw AUC is not comparable across
columns.** Headroom captured -- (AUC - floor) / (1 - floor) -- is. Both the
monotone floor (ROC-AUC of the scalar) and the non-monotone one (a depth-4
tree on that same scalar, fit on train) are reported, because the two diverge
precisely when a regime has introduced an artifact: on the band-mined P-only
set they differ by +0.1015, on every other dataset in this project by less
than 0.02.

Usage:
    python3 src/sismokaos/detection/negative_regime_transfer.py \\
        --ckpt-dir trained_model_ponly_matched \\
        --datasets matched=/path/to/..._matched band=/path/to/..._hard
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier


from sismokaos.detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset
from sismokaos.checkpoints import find_checkpoints
from sismokaos.metrics import safe_auc

ARMS = [("1d", "1D only"), ("2d", "2D only"), ("all", "fusion")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="name=path pairs; positives must be identical across them")
    p.add_argument("--branch-1d", default="cnn-lstm")
    p.add_argument("--fusion", default="linear")
    p.add_argument("--seq-transform", default="asinh")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def find_ckpts(ckpt_dir, channels, fusion, branch):
    """`find_checkpoints`, but an absent arm is empty rather than an error.

    This script sweeps every arm of a matrix and prints `(none)` for the cells
    a directory does not have, so "no such arm" is an expected outcome here
    rather than a mistake.
    """
    try:
        return find_checkpoints(ckpt_dir, channels, fusion, branch)
    except FileNotFoundError:
        return []


@torch.no_grad()
def score(ckpts, ds, channels, args, device):
    """Probability-averaged ensemble over `ckpts`."""
    seq_shape, img_shape = ds.sample_shapes()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    per_ckpt = []
    for c in ckpts:
        m = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, channels=channels,
                                 fusion=args.fusion, branch1d=args.branch_1d).to(device)
        m.load_state_dict(torch.load(c, weights_only=True))
        m.eval()
        probs = []
        for seq, img, _ in loader:
            probs.extend(torch.sigmoid(m(seq.to(device), img.to(device)))
                         .float().cpu().squeeze(1).tolist())
        per_ckpt.append(np.asarray(probs))
    return np.mean(per_ckpt, axis=0)


def floors(path, seq_transform):
    """Monotone and non-monotone floors from the strongest learning-free scalar.

    The tree is fit on train and scored on test, so the non-monotone figure is
    an honest out-of-sample bar rather than a fit to the thing it measures.
    """
    def stats(split):
        ds = RamDualTensorDataset(f"{path}/{split}", seq_transform="none")
        y = np.asarray([l for _, l in ds.samples])
        am, im = [], []
        for f, _ in ds.samples:
            d = torch.load(f, weights_only=True)
            am.append(float(d["seq"].float().abs().max()))
            im.append(float(d["img"].float().mean()))
        return y, np.asarray(am), np.asarray(im)

    ytr, atr, itr = stats("train")
    yte, ate, ite = stats("test")
    out = {}
    sg = lambda v: np.log1p(np.abs(v)) * np.sign(v)
    for name, tr, te in (("seq abs-max", atr, ate), ("img mean dB", itr, ite)):
        mono = safe_auc(yte, te, oriented=True)
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=0)
        t.fit(sg(tr).reshape(-1, 1), ytr)
        nm = safe_auc(yte, t.predict_proba(sg(te).reshape(-1, 1))[:, 1], oriented=True)
        out[name] = (mono, nm)
    mono = max(v[0] for v in out.values())
    nonmono = max(v[1] for v in out.values())
    return mono, nonmono, out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.datasets]

    print("=" * 78)
    print("FLOORS PER NEGATIVE REGIME  (same positives throughout)")
    print("=" * 78)
    print(f"{'regime':<12}{'monotone':>10}{'non-mono':>10}{'gap':>9}   strongest statistic")
    fl = {}
    for name, path in sets:
        mono, nm, detail = floors(path, args.seq_transform)
        fl[name] = (mono, nm)
        best = max(detail.items(), key=lambda kv: kv[1][0])[0]
        flag = "  <-- ARTIFACT" if nm - mono > 0.02 else ""
        print(f"{name:<12}{mono:>10.4f}{nm:>10.4f}{nm-mono:>+9.4f}   {best}{flag}")

    print()
    print("=" * 78)
    print(f"TRANSFER MATRIX  (models trained on '{Path(args.ckpt_dir).name}')")
    print("=" * 78)
    header = f"{'arm':<10}" + "".join(f"{n:>18}" for n, _ in sets)
    # Recall is deliberately absent: the positives are identical across
    # regimes, so it cannot vary. What the negatives change is false alarms.
    for label, fmt in (("AUC", "{:.4f}"), ("captured", "{:.1%}"),
                       ("false alarms @0.5", "{:d}"), ("precision @0.5", "{:.4f}")):
        print(f"\n-- {label} --")
        print(header)
        for ch, desc in ARMS:
            ckpts = find_ckpts(args.ckpt_dir, ch, args.fusion, args.branch_1d)
            if not ckpts:
                print(f"{desc:<10}" + "".join(f"{'(none)':>18}" for _ in sets))
                continue
            row = f"{desc:<10}"
            for name, path in sets:
                ds = RamDualTensorDataset(f"{path}/test", seq_transform=args.seq_transform)
                y = np.asarray([l for _, l in ds.samples])
                p = score(ckpts, ds, ch, args, device)
                auc = safe_auc(y, p, oriented=False)
                floor = max(fl[name])          # the honest bar of the two
                tp = int((p[y == 1] > args.threshold).sum())
                fp = int((p[y == 0] > args.threshold).sum())
                if label == "AUC":
                    v = fmt.format(auc)
                elif label == "captured":
                    v = fmt.format((auc - floor) / (1 - floor))
                elif label.startswith("false"):
                    v = fmt.format(fp)
                else:
                    v = fmt.format(tp / max(tp + fp, 1))
                row += f"{v:>18}"
            print(row)

    print("\nHeadroom captured uses the higher (honest) of the two floors per regime.")
    print("Raw AUC is NOT comparable across columns -- the floors differ.")


if __name__ == "__main__":
    main()
