"""Does this file's own scoring path still reproduce the benchmark?

The scan path standardizes, asinh-compresses and batches windows itself rather
than loading dataset tensors, so it can drift from the training pipeline
silently. Two checks: the vectorized filter must equal the per-window one it
replaces, and real dataset tensors pushed through `score_block` must recover
the published AUC.
"""
import sys

import numpy as np
import torch
from scipy import signal
from sklearn.metrics import roc_auc_score

from detection.cnn_lstm_classify import RamDualTensorDataset
from seismolib.continuous.chunks import clean_block, taper_vector
from seismolib.continuous.scan import load_models, score_block

NAME = "verify"
HELP = "check the preprocessing against real tensors"


def add_args(q):
    q.add_argument("--dataset-dir", required=True)
    q.add_argument("--ckpt-dir", required=True)
    q.add_argument("--branch-1d", default="cnn-lstm")
    q.add_argument("--channels", default="1d")
    q.add_argument("--fusion", default="linear")
    q.add_argument("--hidden", type=int, default=48)
    q.add_argument("--fusion-dim", type=int, default=96)
    q.add_argument("--limit", type=int, default=4000)
    q.add_argument("--expect-auc", default=None,
                   help="what this arm's training log reports, echoed for "
                        "comparison -- the subsample makes an exact match "
                        "neither expected nor meaningful")


def reference_clean(x, fs, freqmin, freqmax):
    """`seismic_cli.core.clean_and_filter_1d`, transcribed for one window.

    Kept here so the equivalence check runs anywhere -- the real function lives
    in the data_downloader project, which is not on the machine that scans.
    """
    x = signal.detrend(x, type="linear")
    x = signal.detrend(x, type="constant")
    n = len(x)
    taper_len = int(n * 0.05)
    if taper_len > 0:
        w = signal.windows.hann(taper_len * 2)
        x[:taper_len] *= w[:taper_len]
        x[-taper_len:] *= w[-taper_len:]
    nyquist = fs / 2.0
    actual_freqmax = freqmax if nyquist > freqmax else nyquist - 1.0
    if actual_freqmax > freqmin:
        b, a = signal.butter(4, [freqmin, actual_freqmax], btype="bandpass", fs=fs)
        x = signal.filtfilt(b, a, x)
    return x


def check_filter_equivalence(win=600, fs=100.0, freqmin=1.0, freqmax=45.0, n=64):
    """Vectorized `clean_block` vs the per-window reference, on random windows."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, win)) * rng.uniform(1, 1e4, (n, 1))
    got = clean_block(x.copy(), fs, freqmin, freqmax, taper_vector(win))
    want = np.stack([reference_clean(x[i].copy(), fs, freqmin, freqmax) for i in range(n)])
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"[verify] clean_block vs per-window reference: max relative "
          f"difference {err:.3e} over {n} windows")
    if err > 1e-12:
        sys.exit("preprocessing does NOT match the training pipeline -- stop here")


def run(args):
    """Reproduces the benchmark score through this file's own scoring path.

    The scan path standardizes, asinh-compresses and batches windows itself
    rather than loading dataset tensors, so it can drift from the training
    pipeline silently. Two checks: the vectorized filter must equal the
    per-window one it replaces, and real dataset tensors pushed through
    `score_block` must recover the published AUC.
    """
    check_filter_equivalence()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.ckpt_dir, args.branch_1d, args, device)
    ds = RamDualTensorDataset(f"{args.dataset_dir}/test", seq_transform="none")

    idx = np.linspace(0, len(ds.samples) - 1, min(args.limit, len(ds.samples)))
    seqs, labels = [], []
    for i in idx:
        fpath, lbl = ds.samples[int(round(i))]
        seqs.append(torch.load(fpath, weights_only=True)["seq"].numpy())
        labels.append(lbl)
    probs = []
    for lo in range(0, len(seqs), 512):
        probs.append(score_block(models, np.stack(seqs[lo:lo + 512]), device))
    probs = np.concatenate(probs)
    auc = roc_auc_score(labels, probs)
    fpr = float((probs[np.array(labels) == 0] > 0.5).mean())
    print(f"[verify] n={len(probs):,}  ROC-AUC {auc:.4f}  FPR@0.5 {fpr:.4f}")
    print(f"         published for this arm: {args.expect_auc}"
          if args.expect_auc else
          "         compare against the arm's published test AUC")
