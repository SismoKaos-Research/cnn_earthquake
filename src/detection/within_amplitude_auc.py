"""Does the detector read waveform shape, or just loudness with extra steps?

Clearing a conditional amplitude floor does **not** establish that a model uses
anything but amplitude. The floor is the ROC-AUC of a single scalar, which
measures only how well that scalar *ranks*. A model that learned a better-shaped
function of the same scalar -- a threshold at the right place, a non-monotone
response -- would clear it while still being, in substance, an amplitude
detector. On the band-mined P-only set that gap was worth 0.10 AUC, which is how
large "a better-shaped function of the same number" can be.

The way to separate the two is to hold amplitude nearly constant and see whether
discrimination survives. Inside a narrow amplitude bin the scalar carries almost
no information by construction, so any AUC meaningfully above 0.5 has to come
from something else -- and the only other thing in the window is the shape of
the waveform.

**Read the bin widths, not just the AUCs.** Amplitude here is heavy-tailed, so
equal-count deciles are wildly unequal in amplitude *range*: on this corpus
deciles 2-7 span only 1.4-1.7x each, while the top decile spans ~530x. A high
AUC in the top decile proves nothing -- amplitude still varies enormously inside
it. The middle deciles are the evidence. This script prints the range and the
width ratio next to every bin so that distinction cannot be lost.

Class balance varies across bins (roughly 0.23-0.99 here). That is fine: AUC is
invariant to class balance, which is precisely why it is the right statistic for
this test and accuracy would not be.

Usage:
    python3 src/detection/within_amplitude_auc.py \\
        --dataset-dir .../dataset_specdual_ponly_3p4s_matched \\
        --ckpt-dir trained_model_ponly_natural --channels all
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset
from seismolib.metrics import safe_auc

# Bins narrower than this in amplitude ratio are treated as evidence; wider ones
# are reported but flagged, because amplitude is still free to vary inside them.
NARROW_RATIO = 2.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--channels", default="all", choices=["all", "1d", "2d"])
    p.add_argument("--branch-1d", default="cnn-lstm")
    p.add_argument("--fusion", default="linear")
    p.add_argument("--seq-transform", default="asinh")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min-bin", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two views of the same split: the transformed one the model consumes, and
    # the raw one the amplitude statistic must be read from. asinh is monotone
    # so it would not change the binning, but reading amplitude from the raw
    # tensors keeps the reported ranges in station-sigma units.
    ds = RamDualTensorDataset(f"{args.dataset_dir}/test", seq_transform=args.seq_transform)
    raw = RamDualTensorDataset(f"{args.dataset_dir}/test", seq_transform="none")
    y = np.asarray([l for _, l in ds.samples])
    amp = np.asarray([float(torch.load(f, weights_only=True)["seq"].float().abs().max())
                      for f, _ in raw.samples])

    pat = re.compile(rf"_{re.escape(args.channels)}_{re.escape(args.fusion)}"
                     rf"_{re.escape(args.branch_1d)}_")
    ckpts = sorted(p for p in Path(args.ckpt_dir).glob("*.pth") if pat.search(p.name))
    if not ckpts:
        raise SystemExit(f"no checkpoints for {args.channels}/{args.fusion}/{args.branch_1d}")

    seq_shape, img_shape = ds.sample_shapes()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    per = []
    for c in ckpts:
        m = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, channels=args.channels,
                                 fusion=args.fusion, branch1d=args.branch_1d).to(dev)
        m.load_state_dict(torch.load(c, weights_only=True))
        m.eval()
        pr = []
        with torch.no_grad():
            for s, i, _ in loader:
                pr.extend(torch.sigmoid(m(s.to(dev), i.to(dev)))
                          .float().cpu().squeeze(1).tolist())
        per.append(np.asarray(pr))
    p = np.mean(per, axis=0)

    print(f"dataset  {Path(args.dataset_dir).name}")
    print(f"model    {Path(args.ckpt_dir).name}  channels={args.channels}  "
          f"({len(ckpts)} seeds)")
    print(f"pooled AUC {safe_auc(y, p, oriented=False):.4f}   "
          f"amplitude floor {safe_auc(y, amp, oriented=True):.4f}\n")

    print(f"{'bin':>4}{'n':>7}{'P(event)':>10}{'amplitude range':>24}"
          f"{'width':>9}{'AUC within':>12}   evidence?")
    edges = np.percentile(amp, np.linspace(0, 100, args.bins + 1))
    narrow = []
    for k in range(args.bins):
        hi_inclusive = (k == args.bins - 1)
        msk = (amp >= edges[k]) & ((amp <= edges[k + 1]) if hi_inclusive
                                   else (amp < edges[k + 1]))
        if msk.sum() < args.min_bin or len(np.unique(y[msk])) < 2:
            continue
        a = safe_auc(y[msk], p[msk], oriented=False)
        lo, hi = edges[k], edges[k + 1]
        ratio = hi / lo if lo > 0 else float("inf")
        ok = ratio <= NARROW_RATIO
        if ok:
            narrow.append(a)
        print(f"{k+1:>4}{int(msk.sum()):>7}{y[msk].mean():>10.2f}"
              f"{f'{lo:.2f} - {hi:.2f}':>24}"
              f"{('%.1fx' % ratio) if np.isfinite(ratio) else '  inf':>9}"
              f"{a:>12.4f}   {'yes' if ok else 'no (too wide)'}")

    if narrow:
        print(f"\n  median AUC across NARROW bins (<= {NARROW_RATIO:g}x): "
              f"{np.median(narrow):.4f}   over {len(narrow)} bins")
        print("  Amplitude varies by at most "
              f"{NARROW_RATIO:g}x inside these, so this is discrimination the")
        print("  amplitude scalar cannot account for.")
    else:
        print("\n  No sufficiently narrow bins -- increase --bins.")


if __name__ == "__main__":
    main()
