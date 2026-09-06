"""Is the 6 s detector reading the P wave, or is it partly reading S?

The 6 s window is [P - 2.0 s, P + 4.0 s] and generation only ever computed P
phases (`arrival_from_catalog.py`: PHASES = ["p", "P", "Pg", "Pn"]), so whether
S also landed inside was never checked. It does, in 28.8% of event windows --
and in 99.3% of them within 25 km, because S-P scales with distance.

That matters operationally rather than statistically: the conditional amplitude
floor is computed on the same windows, so the headline comparison is unaffected,
but a detection that only works once S has arrived carries no early-warning
value at that site.

Stratifying recall by S-present cannot settle it, because at fixed distance S-P
varies only through depth -- "S is present" and "the event is close" are nearly
the same statement in this corpus. So this intervenes instead: zero every sample
at and after the predicted S arrival and re-score with the existing weights.

Two controls make the result readable:

  * **duration-matched**  Zeroing a tail removes signal whether or not that
    signal is S. So S-absent windows are truncated by the same trailing-sample
    counts, drawn from the S-present distribution. If recall falls as far there,
    the effect is duration, not S.
  * **untouched S-absent** Left alone, as a check that the scoring path
    reproduces the unmasked number.

The mask is out of distribution for a model trained on unmasked windows, so a
drop is an upper bound on S-dependence and a recovered recall is the stronger
evidence. Same caveat as `zero_outside` in seisbench_stead_baseline.py.

Only the 1D arm is ablated (`--channels 1d`). The stored `img` spectrogram was
computed from the full window at generation time and cannot be re-masked
without recomputing its station-referenced dB baseline, so a two-branch run
would leave S visible in the channel we are trying to remove.

Usage:
    python3 src/detection/s_arrival_ablation.py \\
        --detector-dir  .../dataset_specdual_catalog_6s_matched_hard \\
        --magnitude-dir .../dataset_magreg_catalog_6s \\
        --catalog       .../catalogs/catalog_current.csv \\
        --ckpt-dir trained_model_branch1d_asinh --branch-1d cnn-lstm
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset
from seismolib.checkpoints import find_checkpoints

PRE_ARRIVAL_S = 2.0   # window starts this far before the predicted P
WINDOW_S = 6.0
P_PHASES = ["p", "P", "Pg", "Pn"]
S_PHASES = ["s", "S", "Sg", "Sn"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--detector-dir", required=True)
    p.add_argument("--magnitude-dir", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--branch-1d", default="cnn-lstm", choices=["lstm", "cnn", "cnn-lstm"])
    p.add_argument("--channels", default="1d", choices=["1d"],
                   help="1D only; see the module docstring for why img cannot be masked")
    p.add_argument("--fusion", default="linear")
    p.add_argument("--seq-transform", default="asinh", choices=["none", "asinh"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--margin-sec", type=float, default=0.0,
                   help="cut this long BEFORE predicted S, to absorb arrival error")
    p.add_argument("--seed", type=int, default=42, help="seed for the duration-matched draw")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def s_minus_p(distances_km, depths_km):
    """TauP S-P per (distance, depth), cached on the rounding generation used."""
    from obspy.taup import TauPyModel
    model = TauPyModel("iasp91")
    cache, out = {}, []
    for d, z in zip(distances_km, depths_km):
        key = (round(d / 111.195, 3), round(float(z), 0))
        if key not in cache:
            try:
                pa = model.get_travel_times(source_depth_in_km=max(0.0, key[1]),
                                            distance_in_degree=key[0], phase_list=P_PHASES)
                sa = model.get_travel_times(source_depth_in_km=max(0.0, key[1]),
                                            distance_in_degree=key[0], phase_list=S_PHASES)
                cache[key] = (min(a.time for a in sa) - min(a.time for a in pa)
                              if pa and sa else np.nan)
            except Exception:
                cache[key] = np.nan
        out.append(cache[key])
    return np.asarray(out, dtype=float)


@torch.no_grad()
def score(models, ds, idx, cut_samples, args, device):
    """Probability-averaged ensemble over `idx`, zeroing seq from `cut_samples`.

    `cut_samples[k]` is the first sample index to zero for `idx[k]`; a value at
    or beyond the window length leaves that window untouched.
    """
    probs = np.zeros(len(idx), dtype=np.float64)
    for bs in range(0, len(idx), args.batch_size):
        sl = slice(bs, bs + args.batch_size)
        seqs, imgs = [], []
        for k in range(sl.start, min(sl.stop, len(idx))):
            seq, img, _ = ds[idx[k]]
            c = int(cut_samples[k])
            if 0 <= c < seq.shape[0]:
                seq = seq.clone()
                seq[c:, :] = 0.0
            seqs.append(seq)
            imgs.append(img)
        sb = torch.stack(seqs).to(device)
        ib = torch.stack(imgs).to(device)
        acc = np.zeros(len(seqs), dtype=np.float64)
        for m in models:
            acc += torch.sigmoid(m(sb, ib)).float().cpu().squeeze(1).numpy()
        probs[sl] = acc / len(models)
    return probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    ds = RamDualTensorDataset(f"{args.detector_dir}/test", seq_transform=args.seq_transform)
    names = [Path(f).name for f, _ in ds.samples]
    labels = np.asarray([l for _, l in ds.samples])
    ev_idx = np.flatnonzero(labels == 1)
    print(f"[data] {len(ds):,} test windows, {len(ev_idx):,} events")

    seq_shape, img_shape = ds.sample_shapes()
    n_samp = seq_shape[0]
    fs = n_samp / WINDOW_S
    print(f"[geom] window {WINDOW_S:g}s @ {fs:g} Hz = {n_samp} samples, "
          f"P at sample {int(PRE_ARRIVAL_S * fs)}")

    mag = pd.read_csv(Path(args.magnitude_dir) / "manifest.csv")
    cat = pd.read_csv(args.catalog, encoding="utf-8-sig")
    mag["eid"] = mag.event_id.astype(str).str.extract(r"(\d+)")[0]
    cat["eid"] = cat.EventID.astype(str)
    mag = mag.merge(cat[["eid", "Depth"]], on="eid", how="left")

    meta = pd.DataFrame({"filename": [names[i] for i in ev_idx]}).merge(
        mag[["filename", "distance_km", "Depth", "magnitude", "log_snr"]],
        on="filename", how="left")
    miss = int(meta.distance_km.isna().sum() + meta.Depth.isna().sum())
    print(f"[join] {len(meta) - miss}/{len(meta)} events carry distance and depth")

    print("[taup] computing S-P ...", flush=True)
    meta["sp"] = s_minus_p(meta.distance_km.fillna(1e9).values, meta.Depth.fillna(0).values)

    post_s = WINDOW_S - PRE_ARRIVAL_S
    s_in = (meta.sp < post_s - args.margin_sec) & meta.sp.notna()
    # First sample to zero: predicted S, pulled `margin_sec` earlier.
    cut_at_s = np.full(len(meta), n_samp, dtype=int)
    cut_at_s[s_in.values] = np.round(
        (PRE_ARRIVAL_S + meta.sp.values[s_in.values] - args.margin_sec) * fs).astype(int)
    cut_at_s = np.clip(cut_at_s, 1, n_samp)
    trailing = n_samp - cut_at_s
    print(f"[mask] S inside for {int(s_in.sum()):,}/{len(meta):,} = {s_in.mean():.1%}; "
          f"median tail zeroed {int(np.median(trailing[s_in.values]))} samples "
          f"({np.median(trailing[s_in.values]) / fs:.2f} s)")

    # Duration-matched control: give S-absent windows the same tail lengths.
    cut_ctrl = np.full(len(meta), n_samp, dtype=int)
    absent = np.flatnonzero(~s_in.values)
    donor = trailing[s_in.values]
    if len(donor):
        cut_ctrl[absent] = n_samp - rng.choice(donor, size=len(absent), replace=True)
    cut_ctrl = np.clip(cut_ctrl, 1, n_samp)

    ckpts = find_checkpoints(args.ckpt_dir, args.channels, args.fusion, args.branch_1d)
    print(f"[ensemble] {len(ckpts)} checkpoints")
    models = []
    for c in ckpts:
        m = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                 fusion_dim=args.fusion_dim, channels=args.channels,
                                 fusion=args.fusion, branch1d=args.branch_1d).to(device)
        m.load_state_dict(torch.load(c, weights_only=True))
        m.eval()
        models.append(m)

    none_cut = np.full(len(meta), n_samp, dtype=int)
    print("\nscoring: baseline ...", flush=True)
    p_base = score(models, ds, ev_idx, none_cut, args, device)
    print("scoring: S masked ...", flush=True)
    p_smask = score(models, ds, ev_idx, cut_at_s, args, device)
    print("scoring: duration-matched control ...", flush=True)
    p_ctrl = score(models, ds, ev_idx, cut_ctrl, args, device)

    meta["p_base"], meta["p_smask"], meta["p_ctrl"] = p_base, p_smask, p_ctrl
    meta["s_inside"] = s_in.values
    if args.out_csv:
        meta.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")

    t = args.threshold
    A, B = s_in.values, ~s_in.values
    print(f"\n{'=' * 70}\nRECALL @ {t}  (event windows only; noise needs no S policy)\n{'=' * 70}")
    print(f"{'group':<34}{'n':>7}{'baseline':>11}{'masked':>10}{'change':>9}")
    rows = [
        ("S-present  -> zeroed at S", A, p_smask),
        ("S-absent   -> untouched (sanity)", B, p_smask),
        ("S-absent   -> same tail zeroed", B, p_ctrl),
        ("S-present  -> same tail (=above)", A, p_smask),
    ]
    for lbl, m, pm in rows[:3]:
        if m.sum() == 0:
            continue
        r0 = float((p_base[m] > t).mean())
        r1 = float((pm[m] > t).mean())
        print(f"{lbl:<34}{int(m.sum()):>7,}{r0:>11.4f}{r1:>10.4f}{r1 - r0:>+9.4f}")

    if A.sum() and B.sum():
        d_s = float((p_smask[A] > t).mean() - (p_base[A] > t).mean())
        d_c = float((p_ctrl[B] > t).mean() - (p_base[B] > t).mean())
        print(f"\n  S-attributable drop = {d_s:+.4f} - {d_c:+.4f} (duration) "
              f"= {d_s - d_c:+.4f}")
        print(f"  overall recall: baseline {float((p_base > t).mean()):.4f} -> "
              f"S masked {float((p_smask > t).mean()):.4f}")

    print("\nby distance band (S-present windows, masked at S):")
    for lo, hi in [(0, 25), (25, 50), (50, 100)]:
        m = A & (meta.distance_km.values >= lo) & (meta.distance_km.values < hi)
        if m.sum():
            print(f"  {lo:>3}-{hi:<3} km  n={int(m.sum()):>6,}  "
                  f"{float((p_base[m] > t).mean()):.4f} -> "
                  f"{float((p_smask[m] > t).mean()):.4f}  "
                  f"({float((p_smask[m] > t).mean() - (p_base[m] > t).mean()):+.4f})")


if __name__ == "__main__":
    main()
