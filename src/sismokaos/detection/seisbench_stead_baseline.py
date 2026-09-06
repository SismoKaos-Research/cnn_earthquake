"""Pretrained EQTransformer on the same STEAD set this project reports in 4.4.

Answers the question a reviewer asks first and this project could not previously
answer: how does a purpose-built, widely used detector score on the identical
evaluation set? Section 4.4 reports 0,9971 for the catalogue-anchored model on
the 27.378-trace matched STEAD split against a 0,9752 amplitude floor. This
script scores EQTransformer on those same traces.

**Window asymmetry, stated up front.** This project's detector sees 600 samples
(6 s). EQTransformer expects 6000 (60 s) and PhaseNet 3001. Padding a 6 s window
up to 60 s would be out of distribution for EQTransformer and would produce a
strawman. So EQTransformer is run on the **full 60 s STEAD trace as intended**,
and scored two ways:

  window : max Detection probability inside the 6 s span this project's
           detector saw -- the like-for-like comparison
  trace  : max Detection probability over all 60 s -- EQTransformer used as it
           is meant to be used, with 10x the context

The `trace` figure is the one that flatters the baseline, and it is the one to
quote when asking "is this project's detector competitive". The context
advantage is real and must be disclosed either way.

**Weight choice matters.** `original` and `stead` weights were TRAINED ON STEAD;
using them here would be testing on training data. The default is `instance`
(INSTANCE, Italy) -- no STEAD overlap, and the closest analogue to this
project's own setup, which trains on Aegean data and tests on STEAD. Pass
--weights stead deliberately if you want the in-corpus ceiling, and label it as
such. Note `scedc` is a poor choice despite being non-STEAD by name: STEAD
draws heavily on Southern California, so overlap is likely.

Usage:
    python3 src/sismokaos/detection/seisbench_stead_baseline.py --weights instance
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

MANIFEST = ("/home/hogib/Projects/Sismokaos/seismic_cli/raw/data/"
            "dataset_stead_matched_6s/manifest.csv")
CHUNKS = {"01_earthquake": "/home/hogib/Projects/Sismokaos/stead_data_process/raw/earthquake/chunk2.hdf5",
          "00_noise": "/home/hogib/Projects/Sismokaos/stead_data_process/raw/noise/chunk1.hdf5"}
WINDOW = 600  # samples this project's detector sees, at 100 Hz


def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Pretrained EQTransformer on matched STEAD.")
    p.add_argument("--weights", default="instance",
                   help="SeisBench weight set. 'original'/'stead' are STEAD-trained "
                        "-- only for a deliberately labelled in-corpus ceiling.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="Debug: score only N traces.")
    p.add_argument("--window-only", action="store_true",
                   help="Zero everything outside the 6 s window before the forward "
                        "pass, so EQTransformer sees the same information this "
                        "project's detector does. Out of distribution for it, so the "
                        "result is a LOWER bound, not a fair score.")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def zero_outside(x, starts):
    """Zeroes everything outside each trace's 6 s window.

    Restricting where the output is *read* does not restrict what the model
    *saw*: EQTransformer runs recurrence and attention over all 6000 samples,
    so a probability inside the window is still computed having seen the S
    arrival and coda outside it. Only masking the input equalises the
    information the two models get. The mask is out of distribution for a model
    trained on continuous traces, so this yields a lower bound.
    """
    m = np.zeros_like(x)
    for i, s in enumerate(starts):
        m[i, :, s:s + WINDOW] = x[i, :, s:s + WINDOW]
    return m


def preprocess(raw):
    """STEAD (n, 6000, 3) in ENZ -> EQTransformer's (n, 3, 6000) in ZNE, peak-normed.

    STEAD stores components as E, N, Z; SeisBench's EQTransformer declares
    component_order 'ZNE', so the axis is reversed rather than transposed. The
    model's own norm is 'peak', reproduced here: demean per component, then
    divide by the largest absolute value across all three.
    """
    x = np.asarray(raw, dtype=np.float32)[:, :, ::-1]      # ENZ -> ZNE
    x = np.transpose(x, (0, 2, 1))                          # (n, 3, 6000)
    x = x - x.mean(axis=2, keepdims=True)
    peak = np.abs(x).max(axis=(1, 2), keepdims=True)
    peak[peak == 0] = 1.0
    return x / peak


@torch.no_grad()
def score(model, args, device):
    """Runs EQTransformer over every manifest trace.

    Returns:
        DataFrame with per-trace window/trace detection scores and the label.
    """
    man = pd.read_csv(MANIFEST)
    if args.limit:
        man = pd.concat([g.head(args.limit // 2) for _, g in man.groupby("cls")])
    rows = []
    for cls, path in CHUNKS.items():
        sub = man[man.cls == cls].reset_index(drop=True)
        print(f"[{cls}] {len(sub)} traces from {Path(path).name}", flush=True)
        with h5py.File(path, "r") as f:
            g = f["data"]
            for i in range(0, len(sub), args.batch_size):
                blk = sub.iloc[i:i + args.batch_size]
                raw = np.stack([g[tn][:] for tn in blk.trace_name])
                xp = preprocess(raw)
                if args.window_only:
                    xp = zero_outside(xp, [int(v) for v in blk.start_sample])
                x = torch.from_numpy(xp).to(device)
                det = model(x)[0].float().cpu().numpy()     # (n, 6000) Detection
                for (_, r), d in zip(blk.iterrows(), det):
                    s = int(r.start_sample)
                    rows.append({
                        "trace_name": r.trace_name,
                        "label": 1 if cls == "01_earthquake" else 0,
                        "p_window": float(d[s:s + WINDOW].max()),
                        "p_trace": float(d.max()),
                    })
                if i % (args.batch_size * 20) == 0:
                    print(f"  {i}/{len(sub)}", flush=True)
    return pd.DataFrame(rows)


def main():
    """Scores EQTransformer and prints both AUCs against section 4.4's figures."""
    args = parse_args()
    import seisbench.models as sbm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.weights in ("original", "stead", "original_nonconservative"):
        print(f"!! WARNING: '{args.weights}' weights were trained on STEAD. This is an "
              f"in-corpus ceiling, NOT a cross-corpus baseline. Label it as such.\n")
    model = sbm.EQTransformer.from_pretrained(args.weights).to(device)
    model.eval()
    print(f"EQTransformer[{args.weights}] on {device}\n")

    df = score(model, args, device)
    auc_w = roc_auc_score(df.label, df.p_window)
    auc_t = roc_auc_score(df.label, df.p_trace)

    print("\n" + "=" * 68)
    print(f"EQTransformer [{args.weights}]  n={len(df)}  "
          f"({int(df.label.sum())} eq / {int((1 - df.label).sum())} noise)")
    print("=" * 68)
    print(f"  ROC-AUC, max Detection in the 6 s window   {auc_w:.4f}")
    print(f"  ROC-AUC, max Detection over the full 60 s  {auc_t:.4f}   <- 10x context")
    print("\n  Section 4.4 reference, same 27.378-trace set:")
    print("    this project, catalogue-anchored           0.9971")
    print("    seq abs-max floor (no learning)            0.9752")
    print(f"\n  vs floor:  EQT window {auc_w - 0.9752:+.4f}   "
          f"EQT trace {auc_t - 0.9752:+.4f}   this project +0.0219")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\n  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
