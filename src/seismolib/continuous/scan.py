"""Windowing a whole record and scoring every window.

The expensive phase, and the one that should not be redone to change a guard
width: reading and decoding the archive is a fixed cost paid once for every
arm that wants to score it.
"""
import concurrent.futures
import glob
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

from detection.cnn_lstm_classify import DualChannelBinaryNet
from seismolib.checkpoints import find_checkpoints
from seismolib.continuous.chunks import (add_chunk_args, clean_block,
                                         component_segments, make_windows,
                                         pick_components, read_chunk,
                                         taper_vector)
from seismolib.continuous.spans import clip_spans, common_spans, merge_intervals

NAME = "scan"
HELP = "score every window in every chunk"


def add_args(q):
    add_chunk_args(q)
    q.add_argument("--baseline-json", required=True)
    q.add_argument("--arm", action="append", required=True, metavar="SPEC",
                   help="NAME:WINDOW_SECONDS:CKPT_DIR:BRANCH[:STEP_SECONDS], "
                        "repeatable. Each arm windows and scores the same "
                        "record independently; the archive is read once for "
                        "all of them, which is why running two costs well "
                        "under twice one. STEP defaults to WINDOW, giving "
                        "disjoint windows -- so an alarm count is also a count "
                        "of independent decisions.")
    q.add_argument("--channels", default="1d", choices=["all", "1d", "2d"])
    q.add_argument("--fusion", default="linear", choices=["linear", "gate"])
    q.add_argument("--hidden", type=int, default=48)
    q.add_argument("--fusion-dim", type=int, default=96)
    q.add_argument("--batch-size", type=int, default=1024)
    q.add_argument("--block-windows", type=int, default=20000,
                   help="windows preprocessed per vectorized block")
    q.add_argument("--workers", type=int, default=6,
                   help="threads for the filtering, which dominates the scan. "
                        "scipy's detrend and filtfilt drop the GIL, so threads "
                        "give real parallelism without pickling the blocks")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--limit-chunks", type=int, default=None,
                   help="stop after N chunks (for a timing probe)")
    q.add_argument("--near-csv", default=None,
                   help="restrict scoring to the neighbourhood of the epoch "
                        "times in this CSV's `p_epoch` column (a `timing` "
                        "output works directly). Catalogued events occupy well "
                        "under 1%% of a station-year, so a dense rescan of just "
                        "their guards costs minutes where a dense full-record "
                        "scan costs days -- which is how detection timing gets "
                        "resolved below the disjoint-window grid.")
    q.add_argument("--near-pre", type=float, default=30.0)
    q.add_argument("--near-post", type=float, default=90.0)


def load_models(ckpt_dir, branch_1d, args, device):
    """Loads every seed of one training arm as an evaluation ensemble."""
    ckpts = find_checkpoints(ckpt_dir, args.channels, args.fusion, branch_1d)
    models = []
    for c in ckpts:
        m = DualChannelBinaryNet(3, 3, hidden=args.hidden, fusion_dim=args.fusion_dim,
                                 channels=args.channels, fusion=args.fusion,
                                 branch1d=branch_1d).to(device)
        m.load_state_dict(torch.load(c, weights_only=True))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()


def score_block(models, seq, device):
    """Probability-averaged ensemble over a (n, win, 3) block."""
    x = torch.from_numpy(seq).float().to(device)
    x = torch.asinh(x)
    acc = None
    for m in models:
        p = torch.sigmoid(m(x, None)).squeeze(1)
        acc = p if acc is None else acc + p
    return (acc / len(models)).cpu().numpy().astype(np.float32)


class StaLtaArm:
    """Classical STA/LTA on the same stream, as the reference the models need.

    A deep detector's alarms-per-day means nothing on its own -- 10/day at 65%
    recall is good or embarrassing depending entirely on what the method it
    replaces achieves on the identical record. Almost no paper in this space
    reports that comparison on continuous data, and the ones that do compare
    against a batch-mode AR picker rather than a tuned operational one.

    The characteristic function is computed on the CONTINUOUS segment, not
    per window: a 10 s LTA does not fit inside a 6 s window, and computing it
    per window would handicap the baseline in a way that flatters the network.
    Each window then takes the maximum CF inside it, so the two are scored on
    exactly the same windows and the same report machinery applies.
    """

    def __init__(self, spec, args, device):
        parts = spec.split(":")
        self.name = parts[0]
        self.window_seconds = float(parts[1])
        self.ckpt_dir = parts[2]
        sta, lta = parts[3].split("-")
        self.sta, self.lta = float(sta), float(lta)
        self.branch_1d = parts[3]
        self.step_seconds = float(parts[4]) if len(parts) == 5 else self.window_seconds
        self.win = int(round(self.window_seconds * args.fs))
        self.step = int(round(self.step_seconds * args.fs))
        self.taper = taper_vector(self.win)
        self.models = None
        print(f"[arm {self.name}] STA/LTA {self.sta:g}s/{self.lta:g}s on the "
              f"vertical, {self.window_seconds:g}s windows every "
              f"{self.step_seconds:g}s")


class Arm:
    """One detector to score the record with: its window geometry and weights.

    Two arms answer different questions of the same data. The 6 s detector is
    the one TODO 2.3's 257-alarms-per-day extrapolation is derived from. The
    P-only detector is a **P-phase detector**, not an early-warning one: its
    3.4 s window is [P-2.0, P+1.4], and the 1.4 s tail excludes S only for
    events far enough away that S-P exceeds 1.4 s. Framing it as early warning
    would smuggle in a distance condition the window itself imposes, so its
    recall is reported against distance and read as phase detection.
    """

    def __init__(self, spec, args, device):
        parts = spec.split(":")
        if len(parts) not in (4, 5):
            sys.exit(f"--arm wants NAME:WINDOW:CKPT_DIR:BRANCH[:STEP], got {spec!r}")
        self.name = parts[0]
        self.window_seconds = float(parts[1])
        self.ckpt_dir = parts[2]
        self.branch_1d = parts[3]
        self.step_seconds = float(parts[4]) if len(parts) == 5 else self.window_seconds
        self.win = int(round(self.window_seconds * args.fs))
        self.step = int(round(self.step_seconds * args.fs))
        self.taper = taper_vector(self.win)
        self.models = load_models(self.ckpt_dir, self.branch_1d, args, device)
        print(f"[arm {self.name}] {self.window_seconds:g}s window every "
              f"{self.step_seconds:g}s ({self.win} samples), "
              f"{len(self.models)} seed(s) from {self.ckpt_dir}")


def sta_lta_scores(arm, spans, seg_lists, args, comps):
    """Max recursive STA/LTA inside each window, on the cleaned vertical.

    The vertical component only, which is what an operational STA/LTA trigger
    uses. The segment is cleaned whole with the same detrend/taper/bandpass the
    networks get, so the two see identical signal conditioning and differ only
    in the detector.
    """
    from obspy.signal.trigger import recursive_sta_lta

    times, probs = [], []
    nsta, nlta = int(arm.sta * args.fs), int(arm.lta * args.fs)
    cache = {}
    for t0, where, n_samp in spans:
        n_win = (n_samp - arm.win) // arm.step + 1
        if n_win < 1:
            continue
        si, off = where[0]                       # component 0 is the vertical
        if si not in cache:
            data = seg_lists[0][si][1]
            if len(data) < nlta * 2:
                cache[si] = None
            else:
                c = clean_block(data[None, :].copy(), args.fs, args.freqmin,
                                args.freqmax, taper_vector(len(data)))[0]
                cache[si] = recursive_sta_lta(c, nsta, nlta)
        cf = cache[si]
        if cf is None:
            continue
        idx = off + np.arange(n_win) * arm.step
        # max of the characteristic function inside each window
        win_max = np.array([cf[i:i + arm.win].max() for i in idx])
        times.append(t0 + (np.arange(n_win) * arm.step) / args.fs)
        probs.append(win_max.astype(np.float32))
    return times, probs


def run(args):
    """Windows every chunk, scores every window, writes (t, p) per chunk and arm.

    Each archive is read, merged and gap-split exactly once and every arm
    windows the result, because reading and decoding 876 MB of mseed is a fixed
    cost that should not be paid per detector.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = json.loads(pathlib.Path(args.baseline_json).read_text())
    arms = [(StaLtaArm if spec.split(":")[2] == "stalta" else Arm)(spec, args, device)
            for spec in args.arm]

    out_dir = pathlib.Path(args.out_dir)
    for a in arms:
        (out_dir / a.name).mkdir(parents=True, exist_ok=True)

    zips = sorted(glob.glob(args.zips))
    if args.limit_chunks:
        zips = zips[:args.limit_chunks]
    near = None
    if args.near_csv:
        col = pd.read_csv(args.near_csv)["p_epoch"].dropna().values
        near = merge_intervals([(x - args.near_pre, x + args.near_post) for x in col])
        print(f"[scan] restricted to {len(near):,} interval(s) around "
              f"{len(col):,} event(s) from {args.near_csv}")
    print(f"[scan] {len(zips)} chunk(s), {len(arms)} arm(s), device={device}, "
          f"{args.workers} filter thread(s)")
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)

    grand_n = 0
    grand_t = time.time()
    for zi, z in enumerate(zips, 1):
        stem = pathlib.Path(z).stem
        todo = [a for a in arms if not (out_dir / a.name / f"{stem}.npz").exists()]
        if not todo:
            print(f"[{zi}/{len(zips)}] {stem}: already scored, skipped", flush=True)
            continue
        t_chunk = time.time()
        # Announced BEFORE the read, because the read is a single opaque obspy
        # call that can take 17 minutes on a fragmented chunk (MANT_2025-02-19
        # decodes ~700 traces). Without this line a slow chunk is indistinguish-
        # able from a hang, which has already cost two false alarms of its own.
        print(f"[{zi}/{len(zips)}] {stem}: reading "
              f"({pathlib.Path(z).stat().st_size / 1e6:.0f} MB)...", flush=True)
        st = read_chunk(z)
        comps = pick_components(st)
        if comps is None:
            print(f"[{zi}/{len(zips)}] {stem}: incomplete components, skipped", flush=True)
            continue
        missing = [c for c in comps if c not in base]
        if missing:
            sys.exit(f"{stem}: no baseline for component(s) {missing}")

        seg_lists = [component_segments(st, c, args.fs) for c in comps]
        del st
        t_read = time.time() - t_chunk

        for arm in todo:
            t_arm = time.time()
            done_w, last_beat = 0, time.time()
            spans = clip_spans(common_spans(seg_lists, args.fs, arm.win),
                               near, args.fs, arm.win, arm.step)
            win, step, taper = arm.win, arm.step, arm.taper

            if isinstance(arm, StaLtaArm):
                times, probs = sta_lta_scores(arm, spans, seg_lists, args, comps)
                if times:
                    t_arr, p_arr = np.concatenate(times), np.concatenate(probs)
                    np.savez_compressed(out_dir / arm.name / f"{stem}.npz",
                                        t=t_arr, p=p_arr)
                    grand_n += len(t_arr)
                    dt = time.time() - t_arm
                    print(f"[{zi}/{len(zips)}] {stem} {arm.name:>6}: "
                          f"{len(t_arr):>8,} windows "
                          f"({len(t_arr) * arm.step_seconds / 86400:5.1f} d), "
                          f"median CF {np.median(p_arr):.2f}, {dt:.0f}s",
                          flush=True)
                continue

            # Batches are filled ACROSS span boundaries, not per span. A chunk
            # where the station drops out hundreds of times yields many tiny
            # spans, and dispatching a thread-pool job per span paid full
            # overhead on each: 3.4x slower on a synthetic 3,000-segment chunk
            # (480 vs 1,730 win/s). Window contents are unchanged -- only the
            # grouping into GPU batches is.
            #
            # This is NOT what made MANT_2025-02-19 slow, despite the guess that
            # prompted the change. That chunk spent 2215 s in the obspy read and
            # 85 s scoring, so the fragmentation cost there is in merge/split,
            # not here. Fixing the read is a separate problem.
            times, probs = [], []
            views_of, nwin_of = {}, {}
            for si_, (t0, where, n_samp) in enumerate(spans):
                n_win = (n_samp - win) // step + 1
                if n_win < 1:
                    continue
                views_of[si_] = [make_windows(seg_lists[k][sj][1], off,
                                              n_win, win, step)
                                 for k, (sj, off) in enumerate(where)]
                nwin_of[si_] = n_win
                times.append(t0 + (np.arange(n_win) * step) / args.fs)

            def flush(pending, total):
                """Cleans, standardizes and scores one cross-span batch."""
                blk = np.empty((total, win, 3), dtype=np.float32)
                tasks, dest = [], 0
                for si_, a, b in pending:
                    tasks.append((si_, a, b, dest))
                    dest += b - a

                def fill(task):
                    sj, a, b, d = task
                    for k in range(3):
                        c = clean_block(np.array(views_of[sj][k][a:b]), args.fs,
                                        args.freqmin, args.freqmax, taper)
                        mu, sigma = base[comps[k]]["mu"], base[comps[k]]["sigma"]
                        blk[d:d + (b - a), :, k] = ((c - mu) / max(sigma, 1e-12))

                list(pool.map(fill, tasks))
                for bl in range(0, total, args.batch_size):
                    bh = min(bl + args.batch_size, total)
                    probs.append(score_block(arm.models, blk[bl:bh], device))

            pending, total = [], 0
            for si_ in sorted(views_of):
                lo = 0
                while lo < nwin_of[si_]:
                    take = min(args.block_windows - total, nwin_of[si_] - lo)
                    pending.append((si_, lo, lo + take))
                    total += take
                    lo += take
                    if total >= args.block_windows:
                        flush(pending, total)
                        done_w += total
                        pending, total = [], 0
                        if time.time() - last_beat > 60:
                            print(f"    ... {stem} {arm.name}: {done_w:,} windows "
                                  f"scored, {time.time() - t_arm:.0f}s", flush=True)
                            last_beat = time.time()
            if total:
                flush(pending, total)
                done_w += total

            if not times:
                print(f"[{zi}/{len(zips)}] {stem} {arm.name}: no unbroken "
                      f"3-component span", flush=True)
                continue
            t_arr = np.concatenate(times)
            p_arr = np.concatenate(probs)
            np.savez_compressed(out_dir / arm.name / f"{stem}.npz", t=t_arr, p=p_arr)
            grand_n += len(t_arr)
            dt = time.time() - t_arm
            print(f"[{zi}/{len(zips)}] {stem} {arm.name:>6}: {len(t_arr):>8,} windows "
                  f"({len(t_arr) * arm.step_seconds / 86400:5.1f} d), "
                  f"{(p_arr > 0.5).mean() * 100:5.2f}% over 0.5, "
                  f"{dt:.0f}s ({len(t_arr) / dt:,.0f} win/s"
                  + (f", +{t_read:.0f}s read)" if arm is todo[0] else ")"), flush=True)

    total = time.time() - grand_t
    print(f"[scan] {grand_n:,} windows in {total / 60:.1f} min -> {out_dir}")
