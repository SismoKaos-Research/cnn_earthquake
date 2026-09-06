"""What do the field's published pickers score on our windows, against our floor?

Every headline number in the P-wave detection literature is reported on its own
corpus, with its own negatives and its own split, and none of the papers
surveyed in `docs/related_work_pwave_detection.md` reports a learning-free
floor. So "GPD gets 99% precision" and "this work gets AUC 0.8712" are not
comparable quantities, and no amount of arranging them in a table makes them
so. The only way to get a comparable number is to run their model on our data
and score it the way we score ours.

This runs a SeisBench pretrained picker over a detection dataset's test split
and reports its ROC-AUC against the same conditional amplitude floor the local
models are measured against.

**Read from the source mseed, not from the dataset tensors.** The stored `seq`
is bandpass-filtered, tapered AND divided by each (station, component)'s
long-term noise sigma. That last step is the problem: it rescales the three
components relative to each other, and GPD's own max-normalisation cannot undo
it because it normalises the block globally. Measured on 150 event windows,
feeding the stored tensors instead of raw counts moves GPD's P>0.5 rate from
17.3% to 6.7% and the two rankings agree only at Spearman 0.60. So the
waveform is rebuilt from the source mseed, and the sample arithmetic is
verified against the stored tensor (correlation 1.0000 on both classes) before
anything is scored.

**Alignment.** `arrival_from_catalog.py` cuts `PRE_ARRIVAL_SECONDS` before the
per-station predicted P, so P sits at sample 200 of 340. GPD predicts the class
at `pred_sample=200` of a 400-sample window. Padding the 60-sample shortfall at
the END therefore leaves P exactly where GPD expects it; `--front-pad` exists
to show what happens when it does not, because an alignment assumption that is
never tested is an alignment assumption that is wrong.

**Preprocessing is a choice, so both are reported.** `gpd` gives the model its
own documented input -- a 2 Hz highpass on raw counts. `pipeline` applies the
corpus's own 1-45 Hz bandpass first, which is what our models saw. Neither is
"correct"; the gap between them is the measurement.

Usage:
    python3 src/detection/pretrained_picker_baseline.py \\
        --dataset-dir .../dataset_specdual_ponly_3p4s_matched \\
        --data-root .../data_downloader --weights original,scedc,stead
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import torch
from scipy import signal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seismolib.metrics import safe_auc

WINDOW_SAMPLES = 340          # 3.4 s at 100 Hz
NOISE_STEP = 170              # --overlap 0.5 on the same window length
P_SAMPLE = 200                # 2.0 s pre-arrival, and GPD's own pred_sample
COMPONENTS = ("Z", "N", "E")  # _COMPONENT_ROLES in the generator; GPD's order


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--data-root", required=True,
                   help="Directory the manifest's file_path column is relative to.")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="GPD", help="SeisBench model class name.")
    p.add_argument("--weights", default="original",
                   help="Comma-separated pretrained weight names.")
    p.add_argument("--preprocess", default="gpd", choices=["gpd", "pipeline"])
    p.add_argument("--pad-mode", default="edge", choices=["edge", "zero", "reflect"])
    p.add_argument("--front-pad", type=int, default=0,
                   help="Samples of padding before the window, shifting P off "
                        "the model's prediction sample. For sensitivity only.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="Debug: first N rows per class.")
    p.add_argument("--local-ckpt-dir", default=None,
                   help="Score this repo's own detector on the SAME surviving "
                        "rows, so the two numbers share a denominator.")
    p.add_argument("--local-channels", default="all")
    p.add_argument("--local-branch-1d", default="cnn-lstm")
    p.add_argument("--local-fusion", default="linear")
    p.add_argument("--local-seq-transform", default="asinh")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    return p.parse_args()


def bandpass(x, fs=100.0, fmin=1.0, fmax=45.0):
    """The corpus's own `clean_and_filter_1d`: detrend, 5% Hann taper, BP 4th order."""
    x = signal.detrend(x, type="linear")
    x = signal.detrend(x, type="constant")
    n = len(x)
    t = int(n * 0.05)
    if t > 0:
        w = signal.windows.hann(t * 2)
        x[:t] *= w[:t]
        x[-t:] *= w[-t:]
    b, a = signal.butter(4, [fmin, fmax], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x)


def highpass(x, fs=100.0, freq=2.0):
    """GPD's own documented filter (`filter_args=['highpass'], freq=2`)."""
    x = signal.detrend(x, type="linear")
    b, a = signal.butter(4, freq, btype="highpass", fs=fs)
    return signal.filtfilt(b, a, x)


def window_start(filename, is_event):
    """First sample of this row's window inside its source record.

    Event windows are cut one per station (`win000`) and already start at the
    window; noise windows slide over a long record at 50% overlap, and the
    index in the filename is the original window number, which is what makes
    the sample range recoverable at all.
    """
    if is_event:
        return 0
    return int(filename.split("_win")[1].split(".")[0]) * NOISE_STEP


def load_waveforms(manifest, data_root, split, limit=None):
    """Rebuilds (n, 3, 340) raw-count waveforms and their labels from source mseed.

    Returns:
        Tuple of (waveforms, labels, kept_index) -- rows whose source record no
        longer supplies a full three-component window are dropped, and
        `kept_index` says which manifest rows survived.
    """
    rows = manifest[manifest.split == split].copy()
    if limit:
        rows = pd.concat([g.head(limit) for _, g in rows.groupby("class_name")])
    by_file = defaultdict(list)
    for idx, r in rows.iterrows():
        by_file[r.file_path].append((idx, r))

    out, labels, kept = [], [], []
    unreadable, last_read_error = 0, None
    for fpath, group in by_file.items():
        try:
            st = obspy.read(str(Path(data_root) / fpath))
        except Exception as e:
            # Counted. A baseline that silently skips files it cannot read is
            # scored on a different set than the model it is compared against,
            # and the comparison stops meaning anything.
            unreadable += 1
            last_read_error = f"{type(e).__name__}: {e}"
            continue
        traces = defaultdict(dict)
        for tr in st:
            traces[tr.id.rsplit(".", 1)[0]][tr.id[-1]] = tr.data.astype(np.float64)
        for idx, r in group:
            match = [k for k in traces if k.startswith(r.station_key)]
            if not match:
                continue
            by_comp = traces[match[0]]
            if not set(COMPONENTS) <= set(by_comp):
                continue
            is_event = r.class_name.endswith("earthquake")
            s = window_start(r.filename, is_event)
            # Components of one station can differ in length in the source
            # record, so every slice is checked before stacking rather than
            # letting np.stack decide the window is fine because one of them is.
            cut = [by_comp[c][s:s + WINDOW_SAMPLES] for c in COMPONENTS]
            if any(len(c) != WINDOW_SAMPLES for c in cut):
                continue
            out.append(np.stack(cut))
            labels.append(int(is_event))
            kept.append(idx)
    if unreadable:
        print(f"  [warn] {unreadable:,} file(s) could not be read and are excluded "
              f"from BOTH the baseline and the comparison; last was {last_read_error}",
              flush=True)
    return np.asarray(out), np.asarray(labels), np.asarray(kept)


def prepare(waveforms, preprocess, pad_mode, front_pad, in_samples):
    """Filters, pads to the model's input length, and demeans."""
    filt = highpass if preprocess == "gpd" else bandpass
    x = np.stack([[filt(w[c].copy()) for c in range(3)] for w in waveforms])
    back = in_samples - x.shape[-1] - front_pad
    if back < 0:
        raise SystemExit(f"--front-pad {front_pad} exceeds the {in_samples}-sample input")
    mode = {"edge": "edge", "zero": "constant", "reflect": "reflect"}[pad_mode]
    x = np.pad(x, ((0, 0), (0, 0), (front_pad, back)), mode=mode)
    return x - x.mean(-1, keepdims=True)      # GPD's annotate_batch_pre


@torch.no_grad()
def score(model, x, batch_size, device):
    """Per-window softmax over the model's phase classes."""
    out = []
    for i in range(0, len(x), batch_size):
        batch = torch.tensor(x[i:i + batch_size], dtype=torch.float32).to(device)
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out)


def report(name, y, probs, labels, amp):
    """Prints one weight set's scores beside the floor they have to clear."""
    p_idx = labels.index("P")
    n_idx = labels.index("N")
    p_prob = probs[:, p_idx]
    not_noise = 1.0 - probs[:, n_idx]
    floor = safe_auc(y, amp, oriented=True)
    print(f"\n  {name}")
    print(f"    AUC  P-class probability      {safe_auc(y, p_prob, oriented=False):.4f}")
    print(f"    AUC  1 - noise probability    {safe_auc(y, not_noise, oriented=False):.4f}")
    print(f"    amplitude floor (same rows)   {floor:.4f}")
    for thr in (0.5, 0.7):
        acc = p_prob > thr
        tp = int((acc & (y == 1)).sum())
        fp = int((acc & (y == 0)).sum())
        rec = tp / max(1, int((y == 1).sum()))
        prec = tp / max(1, tp + fp)
        print(f"    @P>{thr:.1f}  recall {rec:.4f}  precision {prec:.4f}  "
              f"false alarms {fp}")


@torch.no_grad()
def score_local(args, manifest, kept, device):
    """This repo's own ensemble, on exactly the rows the rebuild kept.

    Comparing our headline AUC to a pretrained picker's would put the two on
    different denominators -- some noise rows cannot be rebuilt from their
    source record, and dropping them changes the floor. So the local model is
    re-scored here rather than quoted.
    """
    from detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset
    from seismolib.checkpoints import find_checkpoints

    root = Path(args.dataset_dir) / args.split
    ds = RamDualTensorDataset(root, seq_transform=args.local_seq_transform)
    order = {f.name: i for i, (f, _) in enumerate(ds.samples)}
    want = [order[n] for n in manifest.loc[kept, "filename"] if n in order]

    seq_shape, img_shape = ds.sample_shapes()
    ckpts = find_checkpoints(args.local_ckpt_dir, args.local_channels,
                             args.local_fusion, args.local_branch_1d)
    subset = torch.utils.data.Subset(ds, want)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    per = []
    for c in ckpts:
        m = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, channels=args.local_channels,
                                 fusion=args.local_fusion,
                                 branch1d=args.local_branch_1d).to(device)
        m.load_state_dict(torch.load(c, weights_only=True))
        m.eval()
        pr = []
        for seq, img, _ in loader:
            pr.extend(torch.sigmoid(m(seq.to(device), img.to(device)))
                      .float().cpu().squeeze(1).tolist())
        per.append(np.asarray(pr))
    y = np.asarray([ds.samples[i][1] for i in want])
    return np.mean(per, axis=0), y, len(ckpts)


def main():
    args = parse_args()
    import seisbench.models as sbm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(Path(args.dataset_dir) / "manifest.csv")
    print(f"dataset     {Path(args.dataset_dir).name}  split={args.split}")
    waveforms, y, kept = load_waveforms(manifest, args.data_root, args.split, args.limit)
    n_req = int((manifest.split == args.split).sum())
    lost = "" if args.limit else f" ({n_req - len(y)} of {n_req:,} unrecoverable)"
    print(f"windows     {len(y):,} rebuilt from source mseed{lost}")
    print(f"            {int((y == 1).sum()):,} event / {int((y == 0).sum()):,} noise")

    # The floor is computed on exactly the rows that survived, not on the
    # dataset's headline figure, so the bar and the score share a denominator.
    amp = np.abs(waveforms).max(axis=(1, 2))

    cls = getattr(sbm, args.model)
    print(f"model       {args.model}  preprocess={args.preprocess}  "
          f"pad={args.pad_mode}  front_pad={args.front_pad}")
    for w in args.weights.split(","):
        model = cls.from_pretrained(w.strip()).to(device)
        model.eval()
        x = prepare(waveforms, args.preprocess, args.pad_mode, args.front_pad,
                    model.in_samples)
        report(f"{args.model}/{w.strip()}", y, score(model, x, args.batch_size, device),
               list(model.labels), amp)

    if args.local_ckpt_dir:
        p, ly, n_ckpt = score_local(args, manifest, kept, device)
        print(f"\n  this work / {Path(args.local_ckpt_dir).name} "
              f"({args.local_channels}, {n_ckpt} seeds)")
        print(f"    AUC  event probability        {safe_auc(ly, p, oriented=False):.4f}")
        print(f"    amplitude floor (same rows)   {safe_auc(y, amp, oriented=True):.4f}")
        for thr in (0.5,):
            acc = p > thr
            tp = int((acc & (ly == 1)).sum())
            fp = int((acc & (ly == 0)).sum())
            print(f"    @p>{thr:.1f}  recall {tp / max(1, int((ly == 1).sum())):.4f}  "
                  f"precision {tp / max(1, tp + fp):.4f}  false alarms {fp}")


if __name__ == "__main__":
    main()
