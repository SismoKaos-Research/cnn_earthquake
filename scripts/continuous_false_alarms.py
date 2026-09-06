"""What does the 6 s detector cost in false alarms on continuous data?

Every detection number in this repo comes from a *curated* benchmark: balanced
classes, arrival-anchored positives, and negatives mined from a noise pool.
`docs/TODO.md` 2.3 asks the question that benchmark cannot answer -- put the
detector on an uninterrupted station record and count the alarms it raises when
nothing is happening.

The benchmark's own answer is a straight extrapolation and it is alarming: at
threshold 0.5 the 3-seed `1d/cnn-lstm` ensemble puts 141 of 7,906 test noise
windows on the wrong side, an FPR of 1.78%. A day holds 14,400 non-overlapping
6 s windows, so that rate is **257 false alarms per day** -- roughly one every
five minutes. Whether continuous noise actually behaves like mined noise is the
measurement.

**The catalogue bounds this from one side only.** An alarm that matches no
catalogued event is either a false positive or a real earthquake AFAD never
listed, and this script cannot tell them apart. Every "false alarm" figure it
prints is therefore an UPPER bound. That cuts the way you would hope -- a
detector worth deploying is one whose upper bound is already small.

Two detectors are scored, because they answer different questions of the same
record and the expensive part -- reading and filtering -- is shared:

    6s      the headline detector, window [P-2.0, P+4.0], 0.9896 AUC against a
            0.9049 amplitude floor. The 1.78% FPR above is its own, so this is
            the arm that actually tests the 257-per-day extrapolation.
    ponly   window [P-2.0, P+1.4], 0.8712 against a 0.6679 floor. Read this as
            **P-phase detection, not early warning**: the 1.4 s tail excludes S
            only where S-P exceeds 1.4 s, so calling it early warning would
            smuggle in a distance condition the window itself imposes. Its
            recall is reported against distance for that reason.

Three phases, because the second is the expensive one and should not be redone
to change a guard window:

    baseline  sample the archive, build the per-component (mu, sigma) the
              training windows were standardized against
    scan      window the whole record, score every window, write (t, p) per
              chunk and arm
    report    associate one arm's scores with the catalogue and tabulate

    python3 scripts/continuous_false_alarms.py baseline \\
        --zips 'afad_raw/MANT/*.zip' --out mant_baseline.json
    python3 scripts/continuous_false_alarms.py scan \\
        --zips 'afad_raw/MANT/*.zip' --baseline-json mant_baseline.json \\
        --arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm \\
        --arm ponly:3.4:trained_model_ponly_matched:cnn-lstm \\
        --out-dir scores_mant
    python3 scripts/continuous_false_alarms.py report \\
        --scores 'scores_mant/6s/*.npz' --station MANT \\
        --stations-csv catalogs/istasyon_katalog.csv \\
        --catalog catalogs/catalog_current.csv --out-prefix mant_fa_6s

The preprocessing here is not a reimplementation-by-eye: it is the same order of
operations `seismic_cli.core` applies when it builds a training window (detrend
twice, 5% Hann taper, 4th-order 1-45 Hz bandpass, resample to 100 Hz, then
standardize against the station baseline), applied to whole blocks of windows at
a time instead of one at a time. `verify` checks that claim against real dataset
tensors rather than asserting it.
"""
import argparse
import concurrent.futures
import glob
import json
import math
import pathlib
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from obspy import read, UTCDateTime
from scipy import signal
from sklearn.metrics import roc_auc_score


from detection.cnn_lstm_classify import DualChannelBinaryNet
from seismolib.arrivals import ArrivalTimes, P_PHASES, S_PHASES
from seismolib.catalog import haversine_km as haversine
from seismolib.checkpoints import find_checkpoints
from seismolib.continuous import (background_and_guards, clean_block, clip_spans,
                                  common_spans, component_segments, confirmed,
                                  coverage_spans, declarations, in_spans,
                                  intersect_spans, load_snr, make_windows,
                                  merge_intervals, pick_components,
                                  predicted_arrivals, read_chunk, taper_vector)

EARTH_KM = 6371.0

# Component roles, in the order the training encoder stacks them. Taking the
# first three channels alphabetically would grab ['1','2','E'] at a station with
# mixed sensor codes -- two horizontals and no vertical. See core._COMPONENT_ROLES.
COMPONENT_ROLES = (("Z",), ("N", "1"), ("E", "2"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--zips", required=True, help="glob of chunk archives")
        q.add_argument("--fs", type=float, default=100.0)
        q.add_argument("--freqmin", type=float, default=1.0)
        q.add_argument("--freqmax", type=float, default=45.0)

    b = sub.add_parser("baseline", help="build the station's long-term (mu, sigma)")
    common(b)
    b.add_argument("--sample-chunks", type=int, default=6,
                   help="chunks to scan, spread evenly across the archive")
    b.add_argument("--piece-seconds", type=float, default=3600.0,
                   help="length of the pieces each segment is cleaned in; see "
                        "cmd_baseline for why this is not a free parameter")
    b.add_argument("--out", required=True)

    s = sub.add_parser("scan", help="score every window in every chunk")
    common(s)
    s.add_argument("--baseline-json", required=True)
    s.add_argument("--arm", action="append", required=True, metavar="SPEC",
                   help="NAME:WINDOW_SECONDS:CKPT_DIR:BRANCH[:STEP_SECONDS], "
                        "repeatable. Each arm windows and scores the same "
                        "record independently; the archive is read once for "
                        "all of them, which is why running two costs well "
                        "under twice one. STEP defaults to WINDOW, giving "
                        "disjoint windows -- so an alarm count is also a count "
                        "of independent decisions.")
    s.add_argument("--channels", default="1d", choices=["all", "1d", "2d"])
    s.add_argument("--fusion", default="linear", choices=["linear", "gate"])
    s.add_argument("--hidden", type=int, default=48)
    s.add_argument("--fusion-dim", type=int, default=96)
    s.add_argument("--batch-size", type=int, default=1024)
    s.add_argument("--block-windows", type=int, default=20000,
                   help="windows preprocessed per vectorized block")
    s.add_argument("--workers", type=int, default=6,
                   help="threads for the filtering, which dominates the scan. "
                        "scipy's detrend and filtfilt drop the GIL, so threads "
                        "give real parallelism without pickling the blocks")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--limit-chunks", type=int, default=None,
                   help="stop after N chunks (for a timing probe)")
    s.add_argument("--near-csv", default=None,
                   help="restrict scoring to the neighbourhood of the epoch "
                        "times in this CSV's `p_epoch` column (a `timing` "
                        "output works directly). Catalogued events occupy well "
                        "under 1%% of a station-year, so a dense rescan of just "
                        "their guards costs minutes where a dense full-record "
                        "scan costs days -- which is how detection timing gets "
                        "resolved below the disjoint-window grid.")
    s.add_argument("--near-pre", type=float, default=30.0)
    s.add_argument("--near-post", type=float, default=90.0)

    r = sub.add_parser("report", help="associate with the catalogue and tabulate")
    r.add_argument("--scores", required=True, help="glob of scan .npz files")
    r.add_argument("--station", required=True)
    r.add_argument("--stations-csv", required=True)
    r.add_argument("--catalog", required=True)
    r.add_argument("--max-distance", type=float, default=500.0)
    r.add_argument("--guard-pre", type=float, default=10.0,
                   help="seconds before the predicted P a window may still be "
                        "explained by the event")
    r.add_argument("--guard-post", type=float, default=60.0,
                   help="seconds after it -- long enough to cover the coda a "
                        "regional event leaves in the record")
    r.add_argument("--window-seconds", type=float, required=True,
                   help="the arm's window length, needed so a guard test is an "
                        "overlap test and not a start-time test")
    r.add_argument("--snr-csv", default=None,
                   help="station_detection_range.py output. Without it, recall "
                        "is asked of every catalogued event including those the "
                        "station never recorded, which measures the catalogue's "
                        "reach rather than the detector's.")
    r.add_argument("--snr-min", type=float, default=3.0,
                   help="SNR a catalogued event must reach to count as a "
                        "positive in the confusion matrix. Below this the "
                        "station has no waveform to detect and the event says "
                        "nothing about the model.")
    r.add_argument("--signal-post", type=float, default=20.0,
                   help="seconds after P a window must overlap to be labelled "
                        "positive. Tighter than the guard, which is deliberately "
                        "generous about what an alarm may be excused by.")
    r.add_argument("--cluster-seconds", type=float, default=60.0,
                   help="alarms closer together than this are one declaration. "
                        "Without clustering a single noise burst spanning ten "
                        "windows counts as ten false positives.")
    r.add_argument("--out-prefix", required=True)

    c = sub.add_parser("coincidence",
                       help="require two stations to agree, and price what that costs")
    c.add_argument("--scores-a", required=True, help="glob of station A's .npz files")
    c.add_argument("--station-a", required=True)
    c.add_argument("--scores-b", required=True, help="glob of station B's .npz files")
    c.add_argument("--station-b", required=True)
    c.add_argument("--stations-csv", required=True)
    c.add_argument("--catalog", required=True)
    c.add_argument("--window-seconds", type=float, required=True,
                   help="the arm's window length; must be the same arm at both "
                        "stations or the two alarm streams are not comparable")
    c.add_argument("--coincidence-seconds", type=float, default=None,
                   help="how far apart two declarations may be and still count "
                        "as one. Defaults to the separation divided by Vp, which "
                        "is the largest P-arrival difference any event can "
                        "produce at this pair; smaller loses real events on the "
                        "line through both stations.")
    c.add_argument("--vp", type=float, default=6.0,
                   help="crustal Vp used for the default coincidence window")
    c.add_argument("--snr-csv-a", default=None,
                   help="station_detection_range.py output for A")
    c.add_argument("--snr-csv-b", default=None, help="the same for B")
    c.add_argument("--snr-min", type=float, default=3.0)
    c.add_argument("--max-distance", type=float, default=500.0)
    c.add_argument("--guard-pre", type=float, default=10.0)
    c.add_argument("--guard-post", type=float, default=60.0)
    c.add_argument("--signal-post", type=float, default=20.0)
    c.add_argument("--cluster-seconds", type=float, default=60.0)
    c.add_argument("--out-prefix", required=True)

    m = sub.add_parser("timing", help="per-event: when did it fire, relative to S")
    m.add_argument("--scores", required=True, help="glob of scan .npz files")
    m.add_argument("--station", required=True)
    m.add_argument("--stations-csv", required=True)
    m.add_argument("--catalog", required=True)
    m.add_argument("--max-distance", type=float, default=500.0)
    m.add_argument("--guard-pre", type=float, default=10.0)
    m.add_argument("--guard-post", type=float, default=60.0)
    m.add_argument("--window-seconds", type=float, required=True,
                   help="the arm's window length. A detection cannot be declared "
                        "before the whole window exists, so the alarm time is the "
                        "window's END -- this is what converts a start time into "
                        "one.")
    m.add_argument("--threshold", type=float, required=True,
                   help="take this from `report` -- the threshold that buys the "
                        "alarm budget you intend to run at. The benchmark's 0.5 "
                        "is not an operating point on continuous data.")
    m.add_argument("--snr-csv", default=None,
                   help="station_detection_range.py output; without it the "
                        "timing is diluted by events the station never recorded")
    m.add_argument("--snr-min", type=float, default=3.0)
    m.add_argument("--out", required=True)

    v = sub.add_parser("verify", help="check the preprocessing against real tensors")
    v.add_argument("--dataset-dir", required=True)
    v.add_argument("--ckpt-dir", required=True)
    v.add_argument("--branch-1d", default="cnn-lstm")
    v.add_argument("--channels", default="1d")
    v.add_argument("--fusion", default="linear")
    v.add_argument("--hidden", type=int, default=48)
    v.add_argument("--fusion-dim", type=int, default=96)
    v.add_argument("--limit", type=int, default=4000)
    v.add_argument("--expect-auc", default=None,
                   help="what this arm's training log reports, echoed for "
                        "comparison -- the subsample makes an exact match "
                        "neither expected nor meaningful")

    return p.parse_args()


# ---------------------------------------------------------------------------
# waveform handling
# ---------------------------------------------------------------------------



















# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def cmd_baseline(args):
    """Accumulates (mu, sigma) per component over whole cleaned traces.

    This mirrors `core.compute_station_noise_baselines`, which cleans each trace
    of each noise file whole and accumulates over all of them. Two differences,
    both stated rather than hidden:

    - That function is given noise-only files; a continuous record also contains
      the events. Earthquakes occupy a vanishing fraction of a station-year, so
      the effect on sigma is far below the precision that matters -- and it
      biases sigma UP, making the detector marginally more conservative.
    - It cleans whole traces, and the `noise_pre_3h` files it was pointed at are
      fragmented into pieces of seconds to minutes. Cleaning a 21-day trace whole
      is neither faithful to that nor affordable in memory, so segments are cut
      into `--piece-seconds` pieces first. This does not bias the comparison: the
      5% Hann taper always attenuates the same 10% *fraction* of any piece, so
      its effect on sigma is the same at any piece length.
    """
    zips = sorted(glob.glob(args.zips))
    if not zips:
        sys.exit(f"no archives matched {args.zips}")
    take = np.linspace(0, len(zips) - 1, min(args.sample_chunks, len(zips)))
    picked = [zips[int(round(i))] for i in take]

    print(f"[baseline] {len(picked)} of {len(zips)} chunks, whole-trace statistics")
    accum = {}
    for z in picked:
        t = time.time()
        st = read_chunk(z)
        comps = pick_components(st)
        if comps is None:
            print(f"  {pathlib.Path(z).stem}: incomplete components, skipped")
            continue
        piece = int(round(args.piece_seconds * args.fs))
        for comp in comps:
            for _, data in component_segments(st, comp, args.fs):
                for lo in range(0, len(data), piece):
                    part = data[lo:lo + piece]
                    if len(part) < args.fs * 10:
                        continue
                    c = clean_block(part[None, :].copy(), args.fs, args.freqmin,
                                    args.freqmax, taper_vector(len(part)))[0]
                    s, ss, n = accum.get(comp, (0.0, 0.0, 0))
                    accum[comp] = (s + float(c.sum()), ss + float((c ** 2).sum()),
                                   n + c.size)
        del st
        print(f"  {pathlib.Path(z).stem}: done in {time.time() - t:.0f}s", flush=True)

    out = {}
    for comp, (s, ss, n) in accum.items():
        mu = s / n
        sigma = math.sqrt(max(ss / n - mu ** 2, 0.0))
        out[comp] = {"mu": mu, "sigma": sigma, "n_samples": n}
        print(f"  {comp}: mu={mu:+.4g}  sigma={sigma:.6g}  ({n / args.fs / 3600:.1f} h)")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[baseline] wrote {args.out}")


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

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


def cmd_scan(args):
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


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------









def confusion(t, p, thr, cat, explained, args, win_s):
    """Confusion matrix at one threshold, at both the event and window level.

    Two units, because on continuous data neither alone is honest.

    **Event level** is what a detector is actually judged on, and it has no TN:
    there is no such thing as a "negative earthquake" to correctly not detect.
    Alarms are clustered first -- a noise burst spanning ten windows is one false
    declaration, not ten, and counting windows would inflate FP by whatever
    window length happened to be chosen.

    **Window level** has all four cells but TN is millions, so accuracy is
    meaningless there and only precision/recall are worth reading.

    Positives are events reaching `--snr-min`. Below that the station recorded no
    signal, so counting the event as a miss would charge the model for the
    catalogue's reach. Those events are excluded from BOTH classes rather than
    swept into the negatives, since a real arrival may well be present.
    """
    good = cat[(cat.snr >= args.snr_min) & cat.covered]

    # --- event level -------------------------------------------------------
    tp_ev = int((good.best_prob > thr).sum())
    fn_ev = int(len(good) - tp_ev)
    alarms = t[(p > thr) & ~explained]
    if len(alarms):
        breaks = np.flatnonzero(np.diff(alarms) > args.cluster_seconds)
        fp_ev = len(breaks) + 1
    else:
        fp_ev = 0

    # --- window level ------------------------------------------------------
    pos = np.zeros(len(t), dtype=bool)
    for c in good.p_epoch.values:
        i = np.searchsorted(t, c - win_s)
        j = np.searchsorted(t, c + args.signal_post, side="right")
        pos[i:j] = True
    neg = ~explained                      # outside every guard, any SNR
    tp_w = int(((p > thr) & pos).sum())
    fn_w = int(((p <= thr) & pos).sum())
    fp_w = int(((p > thr) & neg).sum())
    tn_w = int(((p <= thr) & neg).sum())
    return dict(tp_ev=tp_ev, fn_ev=fn_ev, fp_ev=fp_ev, n_ev=len(good),
                tp_w=tp_w, fn_w=fn_w, fp_w=fp_w, tn_w=tn_w)


def print_confusion(c, thr, days, label):
    """Prints one confusion block, with the metrics each unit can support."""
    pr_e = c["tp_ev"] / max(c["tp_ev"] + c["fp_ev"], 1)
    rc_e = c["tp_ev"] / max(c["tp_ev"] + c["fn_ev"], 1)
    f1_e = 2 * pr_e * rc_e / max(pr_e + rc_e, 1e-12)
    pr_w = c["tp_w"] / max(c["tp_w"] + c["fp_w"], 1)
    rc_w = c["tp_w"] / max(c["tp_w"] + c["fn_w"], 1)
    print(f"\n  CONFUSION MATRIX  --  {label}  (threshold {thr:.4f})")
    print(f"    EVENT level, n={c['n_ev']:,} events with signal"
          f"                    WINDOW level")
    print(f"                  alarm   no alarm                     "
          f"        alarm    no alarm")
    print(f"      event    {c['tp_ev']:>7,}   {c['fn_ev']:>8,}    <- TP / FN     "
          f"  event  {c['tp_w']:>7,}  {c['fn_w']:>10,}")
    print(f"      no event {c['fp_ev']:>7,}   {'n/a':>8}    <- FP / TN     "
          f"  none   {c['fp_w']:>7,}  {c['tn_w']:>10,}")
    print(f"      precision {pr_e:.4f}  recall {rc_e:.4f}  F1 {f1_e:.4f}"
          f"        precision {pr_w:.4f}  recall {rc_w:.4f}")
    print(f"      {c['fp_ev'] / max(days, 1e-9):.2f} false declarations/day. "
          f"TN is undefined at event level -- there is no negative earthquake, "
          f"and every FP may be an event AFAD never catalogued.")


def cmd_report(args):
    """The operating table: what a threshold costs per day, and what it buys.

    **The benchmark's 0.5 is not an operating point here and using it would be a
    mistake.** That threshold was fixed on balanced classes whose negatives were
    amplitude-mined, and it carries no meaning against continuous background: on
    MANT the 6 s detector scores a median of 0.83 on *noise*, so 0.5 flags 95% of
    a quiet station-day. Thresholds are therefore derived from the measured
    background distribution -- pick the alarm budget, read off the threshold --
    with the 0.5 row kept only to show how far off it is.

    **Recall is reported against events the station actually recorded.** Over the
    full 728-day MANT record, 27.0% of the catalogued events within 500 km with a
    measured SNR reach SNR 3, and the median is 1.39 -- the typical catalogued
    earthquake leaves no visible trace. Scoring a detector on events whose
    waveform does not exist measures the catalogue's reach, not the model's: the
    same arm scores AUC 0.675 against every event and 0.9403 against the ones
    with signal.

    (An earlier draft of this docstring quoted 11.5% and a median of 1.10. Those
    came from the first 195 days, when both the record and the SNR table were
    partial, and are not what the finished run says.)
    """
    files = sorted(glob.glob(args.scores))
    if not files:
        sys.exit(f"no score files matched {args.scores}")
    t = np.concatenate([np.load(f)["t"] for f in files])
    p = np.concatenate([np.load(f)["p"] for f in files])
    order = np.argsort(t)
    t, p = t[order], p[order]
    win_s = args.window_seconds

    step = float(np.median(np.diff(t[:100000]))) if len(t) > 1 else win_s
    days = len(t) * step / 86400.0
    print(f"{'=' * 78}\nCONTINUOUS OPERATING TABLE  --  {args.station}  "
          f"({win_s:g}s windows every {step:g}s)\n{'=' * 78}")
    print(f"  {len(t):,} windows, {days:.1f} days of record, "
          f"{pd.to_datetime(t.min(), unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(t.max(), unit='s'):%Y-%m-%d}")

    cat, (slat, slon) = predicted_arrivals(
        args.station, args.stations_csv, args.catalog, args.max_distance)
    cat = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)].copy()
    explained, idx = background_and_guards(t, p, cat, win_s, args.guard_pre, args.guard_post)
    bg = p[~explained]
    print(f"  {len(cat):,} catalogued events within {args.max_distance:g} km of "
          f"({slat:.4f}, {slon:.4f}); their guards cover "
          f"{100 * explained.mean():.2f}% of windows")

    print(f"\n  BACKGROUND score distribution ({len(bg):,} windows outside every guard)")
    print("    " + "  ".join(f"p{q}={np.percentile(bg, q):.4f}"
                             for q in (50, 90, 99, 99.9, 99.99)) +
          f"  max={bg.max():.4f}")
    if np.percentile(bg, 50) > 0.5:
        print("    ** the median NOISE window scores above 0.5: the benchmark "
              "threshold is meaningless here **")

    # Attach measured SNR so recall is asked only of events with a waveform.
    if args.snr_csv:
        cat = cat.merge(load_snr(args.snr_csv), left_on="EventID",
                        right_on="event_id", how="left")
    else:
        cat["snr"] = np.nan
    best = np.array([p[i:j].max() if j > i else np.nan for i, j in idx])
    cat["best_prob"] = best
    cat["covered"] = ~np.isnan(best)
    cov = cat[cat.covered]
    n_gap = int((~cat.covered).sum())
    print(f"  {n_gap:,} event(s) fall in a data gap and are excluded -- no "
          f"waveform, so not a miss")
    if args.snr_csv:
        print(f"  measured SNR available for {int(cov.snr.notna().sum()):,} of "
              f"{len(cov):,}; median {cov.snr.median():.2f}, "
              f"{100 * (cov.snr >= 3).mean():.1f}% reach SNR 3")

    # Thresholds chosen to buy a stated alarm budget, not inherited from the benchmark.
    print(f"\n  {'alarms/day':>11}{'threshold':>11}{'actual/day':>12}{'FPR':>10}"
          + "".join(f"{'R(SNR>=' + str(s) + ')':>13}" for s in (3, 5, 10)))
    rows = []
    for target in (100.0, 10.0, 1.0, 0.1):
        want = target * days
        if want >= len(bg):
            continue
        thr = float(np.quantile(bg, 1.0 - want / len(bg)))
        n_alarm = int((bg > thr).sum())
        rec = []
        for s in (3, 5, 10):
            g = cov[cov.snr >= s]
            rec.append((g.best_prob > thr).mean() if len(g) else np.nan)
        rows.append({"target_per_day": target, "threshold": thr,
                     "alarms_per_day": n_alarm / days, "fpr": n_alarm / len(bg),
                     **{f"recall_snr{s}": r for s, r in zip((3, 5, 10), rec)}})
        print(f"  {target:>11.4g}{thr:>11.4f}{n_alarm / days:>12.2f}"
              f"{n_alarm / len(bg):>10.6f}"
              + "".join(f"{r:>13.3f}" if r == r else f"{'-':>13}" for r in rec))

    n05 = int((bg > 0.5).sum())
    rec05 = [(cov[cov.snr >= s].best_prob > 0.5).mean() if len(cov[cov.snr >= s])
             else np.nan for s in (3, 5, 10)]
    print(f"  {'(0.5)':>11}{0.5:>11.4f}{n05 / days:>12.2f}{n05 / len(bg):>10.6f}"
          + "".join(f"{r:>13.3f}" if r == r else f"{'-':>13}" for r in rec05)
          + "   <- the benchmark threshold, for comparison only")
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_thresholds.csv", index=False)

    # Threshold-free: can a real guard be told from a random stretch of record?
    rng = np.random.default_rng(0)
    cand = rng.choice(t, size=min(8000, len(t)), replace=False)
    keep = np.ones(len(cand), bool)
    for c in cat.p_epoch.values:
        keep &= np.abs(cand - c) > (args.guard_pre + args.guard_post + 60)
    fake = []
    for c in cand[keep]:
        i, j = np.searchsorted(t, c - args.guard_pre - win_s), \
               np.searchsorted(t, c + args.guard_post, side="right")
        if j > i:
            fake.append(p[i:j].max())
    fake = np.asarray(fake)
    print(f"\n  event-level separation: max score in a real guard vs in "
          f"{len(fake):,} random ones")
    print(f"    {'SNR cut':>9}{'events':>8}{'AUC':>9}{'med real':>10}{'med random':>12}")
    for s in (0, 2, 3, 5, 10):
        g = cov[(cov.snr >= s) | (np.isnan(cov.snr) & (s == 0))].dropna(subset=["best_prob"])
        if len(g) < 8 or not len(fake):
            continue
        y = np.r_[np.ones(len(g)), np.zeros(len(fake))]
        auc = roc_auc_score(y, np.r_[g.best_prob.values, fake])
        print(f"    {s:>9}{len(g):>8,}{auc:>9.4f}{g.best_prob.median():>10.4f}"
              f"{np.median(fake):>12.4f}")

    # Diurnal cycle of the alarms. Cultural noise is strongly diurnal, so a
    # detector firing on anthropogenic transients shows a working-hours peak
    # that a detector firing on seismicity does not.
    thr10 = float(np.quantile(bg, 1.0 - min(10.0 * days, len(bg) - 1) / len(bg)))
    at = t[(p > thr10) & ~explained]
    if len(at) > 24:
        hod = (pd.to_datetime(at, unit="s").tz_localize("UTC")
               .tz_convert("Europe/Istanbul").hour)
        counts = np.bincount(np.asarray(hod), minlength=24)
        peak, trough = counts.max(), max(counts.min(), 1)
        print(f"\n  unexplained alarms by local hour at {thr10:.4f} "
              f"({len(at):,} alarms, peak/trough = {peak / trough:.2f}x)")
        for h in range(0, 24, 3):
            bar = "#" * int(38 * counts[h] / max(peak, 1))
            print(f"    {h:02d}:00 {counts[h]:>6,} {bar}")
        day = counts[6:20].sum() / 14
        night = (counts[:6].sum() + counts[20:].sum()) / 10
        print(f"    day (06-20) {day:.1f}/h vs night {night:.1f}/h "
              f"= {day / max(night, 1e-9):.2f}x"
              + ("   <- anthropogenic signature" if day > 1.5 * night else ""))

    # Recall by magnitude, on events the station actually recorded.
    mg = cov[cov.snr >= args.snr_min]
    if len(mg) > 20:
        print(f"\n  recall by magnitude at {thr10:.4f} (SNR>={args.snr_min:g} only, "
              f"n={len(mg):,})")
        print(f"    {'band':>12}{'events':>9}{'found':>8}{'recall':>9}{'med dist':>10}")
        for band, g in mg.groupby(pd.cut(mg.Magnitude, [0, 2, 2.5, 3, 3.5, 4, 10]),
                                  observed=True):
            if not len(g):
                continue
            hit = int((g.best_prob > thr10).sum())
            print(f"    {str(band):>12}{len(g):>9,}{hit:>8,}{hit / len(g):>9.3f}"
                  f"{g.dist.median():>10.0f}")

    # Confusion matrices at the budgets a deployment would actually pick.
    for target in (10.0, 1.0):
        want = target * days
        if want >= len(bg):
            continue
        thr = float(np.quantile(bg, 1.0 - want / len(bg)))
        c = confusion(t, p, thr, cat, explained, args, win_s)
        print_confusion(c, thr, days, f"{target:g} alarms/day budget")

    cov.to_csv(f"{args.out_prefix}_events.csv", index=False)
    print(f"\n  wrote {args.out_prefix}_thresholds.csv and "
          f"{args.out_prefix}_events.csv ({len(cov):,} rows)")


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



# ---------------------------------------------------------------------------
# Two stations
# ---------------------------------------------------------------------------

def load_scores(pattern):
    """Every scored window from one glob, sorted by time."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no score files matched {pattern}")
    t = np.concatenate([np.load(f)["t"] for f in files])
    p = np.concatenate([np.load(f)["p"] for f in files])
    order = np.argsort(t)
    return t[order], p[order]












def cmd_coincidence(args):
    """What requiring two stations to agree costs, and what it buys.

    Single-station continuous detection is dominated by false alarms: at the
    thresholds this detector needs to keep any recall, MANT alone declares tens
    of times a day. Requiring a second station to agree within the time an
    event's P wave could plausibly take to cross the pair is the standard
    network answer, and the reduction it delivers is usually quoted from an
    independence assumption.

    **That assumption is the thing worth measuring.** Two stations 130 km apart
    share weather, share the regional noise field, and share whatever diurnal
    cultural signal drives the day/night ratio already measured here. To the
    extent their false alarms are common-mode, the reduction is smaller than
    independence predicts -- and no amount of arithmetic can say by how much.
    So this reports the measured joint rate against the independent prediction,
    and their ratio.

    Two things it refuses to do:

    **It scores only the span both stations recorded.** Their coverage is
    intersected first. Counting an unconfirmed alarm as suppressed while the
    other station was simply off the air would read as a large gain and be
    nothing but missing data.

    **It asks recall only of events both stations actually recorded.** An event
    below SNR at either station cannot be confirmed by a network rule, and
    charging the rule for it measures the catalogue's reach, not the method.
    """
    ta_all, pa_all = load_scores(args.scores_a)
    tb_all, pb_all = load_scores(args.scores_b)
    win_s = args.window_seconds
    step = float(np.median(np.diff(ta_all[:100000])))

    st = pd.read_csv(args.stations_csv, encoding="utf-8-sig")
    st.columns = [c.strip() for c in st.columns]
    A = st[st.Code == args.station_a].iloc[0]
    B = st[st.Code == args.station_b].iloc[0]
    sep = float(haversine(float(A.Latitude), float(A.Longitude),
                          np.array([float(B.Latitude)]), np.array([float(B.Longitude)]))[0])
    w = args.coincidence_seconds
    if w is None:
        w = sep / args.vp
    print(f"{'=' * 78}\nTWO-STATION COINCIDENCE  --  {args.station_a} + "
          f"{args.station_b}  ({win_s:g}s windows)\n{'=' * 78}")
    print(f"  separation {sep:.0f} km -> coincidence window +/-{w:.1f} s "
          f"({'default: separation / Vp ' + format(args.vp, 'g') if args.coincidence_seconds is None else 'given'})",
          flush=True)

    # --- the span both stations recorded ----------------------------------
    spans = intersect_spans(coverage_spans(ta_all, step), coverage_spans(tb_all, step))
    joint_s = sum(hi - lo for lo, hi in spans)
    days = joint_s / 86400.0
    ka, kb = in_spans(ta_all, spans), in_spans(tb_all, spans)
    ta, pa, tb, pb = ta_all[ka], pa_all[ka], tb_all[kb], pb_all[kb]
    print(f"  {args.station_a}: {len(ta_all) * step / 86400:.1f} d scored, "
          f"{args.station_b}: {len(tb_all) * step / 86400:.1f} d scored, "
          f"both at once: {days:.1f} d in {len(spans)} span(s)")
    if days < 1:
        sys.exit("the two stations barely overlap; nothing to measure")

    # --- catalogue, at each station separately ----------------------------
    # Keyed "a"/"b", not by station name: passing the same station twice is the
    # obvious self-test, and a name-keyed dict silently collapses to one entry
    # for it -- the background of one station overwrites the other's and the
    # threshold table comes out empty.
    cats = {}
    for side, name, tt, snr_csv in (("a", args.station_a, ta, args.snr_csv_a),
                                    ("b", args.station_b, tb, args.snr_csv_b)):
        cat, _ = predicted_arrivals(
            name, args.stations_csv, args.catalog, args.max_distance)
        # `in_spans`, not a comprehension over `spans`: a gap-split archive has
        # tens of thousands of them (MANT's pnat scores have 43,215), and one
        # Python-level pass per event over all of them is hours rather than
        # seconds. p_epoch is sorted, which is what lets the searchsorted
        # version be used here.
        cat = cat[in_spans(cat.p_epoch.values, spans)].copy()
        if snr_csv:
            cat = cat.merge(load_snr(snr_csv), left_on="EventID",
                            right_on="event_id", how="left")
        else:
            cat["snr"] = np.nan
        cats[side] = cat.drop_duplicates(subset="EventID")
    both = cats["a"].merge(cats["b"][["EventID", "snr", "p_epoch"]],
                           on="EventID", suffixes=("_a", "_b"))
    good = both[(both.snr_a >= args.snr_min) & (both.snr_b >= args.snr_min)]
    print(f"  {len(both):,} catalogued event(s) in that span; "
          f"{len(good):,} reach SNR {args.snr_min:g} at BOTH stations")
    if len(good):
        dp = (good.p_epoch_b - good.p_epoch_a).abs()
        print(f"  their |P_A - P_B| spans {dp.min():.1f}..{dp.max():.1f} s "
              f"(median {dp.median():.1f}) -- the window must cover this")

    # --- background at each station ---------------------------------------
    # The guard mask is kept, not just the background scores. Declarations have
    # to be counted on UNEXPLAINED windows only: a catalogued earthquake is
    # detected at both stations by construction, so leaving real events in the
    # streams makes every one of them a guaranteed coincidence and the "excess"
    # then measures how many events the span contains rather than how much the
    # two stations' false alarms agree. On MANT+DEMI that is 11.6 catalogued
    # events per day at SNR>=3 against a measured 3.97 coincidences per day --
    # enough to account for all of them.
    bg, unexplained = {}, {}
    for side, name, tt, pp in (("a", args.station_a, ta, pa),
                               ("b", args.station_b, tb, pb)):
        explained, _ = background_and_guards(tt, pp, cats[side], win_s, args.guard_pre, args.guard_post)
        bg[side] = pp[~explained]
        unexplained[side] = ~explained

    # --- the table ---------------------------------------------------------
    print(f"\n  Each station is thresholded to the SAME alarm budget, not the same")
    print(f"  threshold: their backgrounds differ and a shared number would not")
    print(f"  mean the same thing at both.\n")
    print(f"  Alarm rates below count UNEXPLAINED declarations only -- windows")
    print(f"  overlapping a catalogued event's guard are removed from both")
    print(f"  streams first, since a real earthquake is seen at both stations by")
    print(f"  construction and would otherwise be counted as agreement.\n")
    print(f"  {'budget/day':>11}{'thr ' + args.station_a:>12}{'thr ' + args.station_b:>12}"
          f"{'A/day':>9}{'B/day':>9}{'2of2/day':>10}{'if indep':>10}{'excess':>8}"
          f"{'recall':>9}")
    rows = []
    for target in (100.0, 30.0, 10.0, 3.0, 1.0, 0.1):
        want = target * days
        if any(want >= len(bg[s]) for s in ("a", "b")):
            continue
        thr = {s: float(np.quantile(bg[s], 1.0 - want / len(bg[s])))
               for s in ("a", "b")}
        ua, ub = unexplained["a"], unexplained["b"]
        da_t, _ = declarations(ta[ua], pa[ua], thr["a"], args.cluster_seconds)
        db_t, _ = declarations(tb[ub], pb[ub], thr["b"], args.cluster_seconds)
        ok = confirmed(da_t, db_t, w)
        n_a, n_b, n_2 = len(da_t), len(db_t), int(ok.sum())
        # Independent Poisson streams of rate ra, rb coincide within +/-w at
        # rate ra * rb * 2w per unit time. This is the number the "1.78% ->
        # 0.03%" style estimate assumes; the measured one is next to it.
        ra, rb = n_a / joint_s, n_b / joint_s
        # What two INDEPENDENT streams of these rates would produce. The
        # measured quantity is "A declarations having at least one B within
        # +/-w", so the prediction must be for that and not for the number of
        # coincident pairs: a Poisson B stream puts 1 - exp(-rb*2w) of them in
        # the window, which is below rb*2w whenever B is busy. The two agree to
        # 0.25% at 10 alarms/day and diverge by 10% at 200, so the distinction
        # only matters at the loose end of this table -- which is exactly where
        # the reduction looks most impressive.
        indep = ra * (1.0 - np.exp(-rb * 2 * w)) * 86400
        rec = np.nan
        if len(good):
            fired_a = np.array([((pa[np.searchsorted(ta, c - win_s):
                                     np.searchsorted(ta, c + args.signal_post,
                                                     side="right")] > thr["a"]).any())
                                for c in good.p_epoch_a.values])
            fired_b = np.array([((pb[np.searchsorted(tb, c - win_s):
                                     np.searchsorted(tb, c + args.signal_post,
                                                     side="right")] > thr["b"]).any())
                                for c in good.p_epoch_b.values])
            rec = float((fired_a & fired_b).mean())
        excess = n_2 / days / indep if indep > 0 else np.nan
        rows.append({"budget_per_day": target,
                     "station_a": args.station_a, "station_b": args.station_b,
                     "thr_a": thr["a"], "thr_b": thr["b"],
                     "a_per_day": n_a / days, "b_per_day": n_b / days,
                     "both_per_day": n_2 / days, "independent_per_day": indep,
                     "excess_over_independent": excess, "recall_both": rec})
        print(f"  {target:>11.4g}{thr['a']:>12.4f}{thr['b']:>12.4f}"
              f"{n_a / days:>9.2f}{n_b / days:>9.2f}{n_2 / days:>10.3f}"
              f"{indep:>10.4f}{excess:>8.1f}x"
              + (f"{rec:>9.3f}" if rec == rec else f"{'-':>9}"))

    out = pathlib.Path(f"{args.out_prefix}_coincidence.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  wrote {out}")
    print(f"\n  `excess` is the measured two-station rate divided by what two")
    print(f"  independent alarm streams of the same rates would produce. 1.0x")
    print(f"  means the stations' false alarms are independent and the textbook")
    print(f"  reduction holds; above 1.0x they share a cause and the network")
    print(f"  rule buys less than the arithmetic promises.")

def cmd_verify(args):
    """Reproduces the benchmark score through this file's own scoring path.

    The scan path standardizes, asinh-compresses and batches windows itself
    rather than loading dataset tensors, so it can drift from the training
    pipeline silently. Two checks: the vectorized filter must equal the
    per-window one it replaces, and real dataset tensors pushed through
    `score_block` must recover the published AUC.
    """
    from sklearn.metrics import roc_auc_score
    from detection.cnn_lstm_classify import RamDualTensorDataset

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

def cmd_timing(args):
    """Per catalogued event: when the detector first fired, relative to P and S.

    **The alarm time is the window's END, not its start.** A detection cannot be
    declared before the whole window has been observed and scored, so a 6 s
    window starting at t announces at t+6. Using the start would credit the
    detector with information it did not yet have, and would make some events
    look detected before their P arrived.

    **Read the deltas against the window step, not below it.** Disjoint windows
    put the alarm time on a grid, so with a 6 s step a delta is only meaningful
    to +/-6 s -- which is coarser than S-P itself for anything inside ~50 km.
    Rescan the event guards densely (`scan --near-csv`, small `:STEP`) before
    reading a close event's number as a real lead time.
    """
    files = sorted(glob.glob(args.scores))
    if not files:
        sys.exit(f"no score files matched {args.scores}")
    t = np.concatenate([np.load(f)["t"] for f in files])
    p = np.concatenate([np.load(f)["p"] for f in files])
    order = np.argsort(t)
    t, p = t[order], p[order]

    cat, _ = predicted_arrivals(
        args.station, args.stations_csv, args.catalog, args.max_distance)
    ev = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)].copy()
    if args.snr_csv:
        # load_snr, not a raw read: a duplicated event_id expands `ev` on this
        # join, and here that does not raise -- the loops below simply score the
        # duplicated events twice, inflating the detection counts and the
        # before-S fractions. DEMI's table carries 269 such ids.
        ev = ev.merge(load_snr(args.snr_csv), left_on="EventID",
                      right_on="event_id", how="left")
        n_all = len(ev)
        ev = ev[ev.snr >= args.snr_min].copy()
        print(f"  {len(ev):,} of {n_all:,} events reach SNR {args.snr_min:g}; "
              f"the rest leave no trace in the record and are excluded")

    first, best = [], []
    for a, b in zip(ev.p_epoch - args.guard_pre - args.window_seconds,
                    ev.p_epoch + args.guard_post):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        if j <= i:
            first.append(np.nan)
            best.append(np.nan)
            continue
        best.append(float(p[i:j].max()))
        hit = np.flatnonzero(p[i:j] > args.threshold)
        first.append(t[i + hit[0]] + args.window_seconds if len(hit) else np.nan)

    ev["best_prob"] = best
    ev["alarm_epoch"] = first
    ev["dt_after_p"] = ev.alarm_epoch - ev.p_epoch
    ev["dt_vs_s"] = ev.alarm_epoch - ev.s_epoch
    ev["covered"] = ~np.isnan(best)
    ev["detected"] = ~np.isnan(first)

    cov = ev[ev.covered]
    det = ev[ev.detected]
    print(f"{'=' * 70}\nDETECTION TIMING  --  {args.station}  "
          f"(threshold {args.threshold}, {args.window_seconds:g}s window)\n{'=' * 70}")

    # A saturated model "detects" everything, so recall alone is not readable.
    # Quoting the background rate next to it makes that impossible to miss: at a
    # 95% background rate, a 98% recall is arithmetic, not detection.
    inside = np.zeros(len(t), dtype=bool)
    for a, b in zip(ev.p_epoch - args.guard_pre - args.window_seconds,
                    ev.p_epoch + args.guard_post):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        inside[i:j] = True
    bg_rate = float((p[~inside] > args.threshold).mean())
    print(f"  {len(ev):,} catalogued events in span, {len(cov):,} with data, "
          f"{len(det):,} detected ({len(det) / max(len(cov), 1):.1%})")
    print(f"  background alarm rate at this threshold: {bg_rate:.4%} of windows "
          f"outside every guard")
    if bg_rate > 0.05:
        print(f"  ** WARNING: at this background rate a guard of "
              f"{(args.guard_pre + args.guard_post) / args.window_seconds:.0f} "
              f"windows contains an alarm "
              f"{1 - (1 - bg_rate) ** ((args.guard_pre + args.guard_post) / args.window_seconds):.1%} "
              f"of the time BY CHANCE. The recall and timing below are not "
              f"measuring detection -- pick a threshold from `report` first. **")
    if len(det):
        before = int((det.dt_vs_s < 0).sum())
        print(f"  {before:,} of {len(det):,} ({before / len(det):.1%}) fired BEFORE "
              f"the predicted S arrival")
        print(f"  alarm after P: median {det.dt_after_p.median():.1f}s  "
              f"(quantized to the {args.window_seconds:g}s window grid)")
        print(f"\n  {'dist (km)':>12}{'events':>9}{'found':>8}{'recall':>9}"
              f"{'med S-P':>9}{'med dt vs S':>13}{'before S':>10}")
        for band, g in cov.groupby(pd.cut(cov.dist, [0, 25, 50, 100, 200, 500]),
                                   observed=True):
            d = g[g.detected]
            if not len(g):
                continue
            print(f"    {str(band):>10}{len(g):>9,}{len(d):>8,}"
                  f"{len(d) / len(g):>9.3f}{g.sp_seconds.median():>9.1f}"
                  + (f"{d.dt_vs_s.median():>13.1f}{(d.dt_vs_s < 0).mean():>10.2f}"
                     if len(d) else f"{'-':>13}{'-':>10}"))

    # p_epoch and s_epoch are carried so this file can drive `scan --near-csv`
    # for a dense rescan, which is how the deltas get resolved below the grid.
    cols = ["EventID", "t", "Magnitude", "dist", "Depth", "p_epoch", "s_epoch",
            "sp_seconds", "best_prob", "alarm_epoch", "dt_after_p", "dt_vs_s",
            "Location"]
    ev[ev.covered][cols].to_csv(args.out, index=False)
    print(f"\n  wrote {args.out} ({int(ev.covered.sum()):,} rows)")


def main():
    args = parse_args()
    {"baseline": cmd_baseline, "scan": cmd_scan, "report": cmd_report,
     "timing": cmd_timing, "verify": cmd_verify,
     "coincidence": cmd_coincidence}[args.cmd](args)


if __name__ == "__main__":
    main()
