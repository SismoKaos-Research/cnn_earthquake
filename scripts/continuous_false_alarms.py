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
from obspy.taup import TauPyModel
from scipy import signal
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from detection.cnn_lstm_classify import DualChannelBinaryNet
from seismolib.checkpoints import find_checkpoints

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
    r.add_argument("--out-prefix", required=True)

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
    m.add_argument("--threshold", type=float, default=0.5)
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

def read_chunk(zpath):
    """Reads one campaign archive into a Stream, merged and gap-split.

    `merge(fill_value=None)` leaves masked arrays where data is missing and
    `split()` turns those back into contiguous unmasked segments. Keeping the
    segmentation is the whole point: a window that would straddle a gap simply
    never gets built, rather than being built across interpolated samples.
    """
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as tmp:
        member = next(n for n in zf.namelist() if n.lower().endswith(".mseed"))
        zf.extract(member, tmp)
        st = read(str(pathlib.Path(tmp) / member))
    st.merge(method=1, fill_value=None)
    return st.split()


def pick_components(stream):
    """Returns the three channel codes to use, in Z/N/E role order."""
    have = {tr.stats.channel[-1].upper() for tr in stream}
    out = []
    for role in COMPONENT_ROLES:
        match = next((c for c in role if c in have), None)
        if match is None:
            return None
        out.append(match)
    return out


def component_segments(stream, comp, fs):
    """Contiguous (t0, data) segments for one component, at the nominal rate."""
    segs = []
    for tr in stream:
        if tr.stats.channel[-1].upper() != comp:
            continue
        if abs(tr.stats.sampling_rate - fs) > 1e-6:
            tr = tr.copy()
            tr.resample(fs)
        segs.append((tr.stats.starttime.timestamp, np.asarray(tr.data, dtype=np.float64)))
    segs.sort(key=lambda s: s[0])
    return segs


def merge_intervals(iv):
    """Sorts and coalesces overlapping (lo, hi) pairs."""
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def clip_spans(spans, near, fs, win, step):
    """Cuts spans down to the parts overlapping `near`, keeping the window grid.

    The grid matters. A window's start must stay on the same `t0 + k * step`
    lattice the unrestricted scan would have used, or a restricted rescan is not
    comparable with the full one -- so this advances the offset by whole steps
    rather than starting fresh at each interval's edge.
    """
    if near is None:
        return spans
    out = []
    for t0, where, n_samp in spans:
        span_end = t0 + n_samp / fs
        for a, b in near:
            lo, hi = max(t0, a), min(span_end, b)
            if hi - lo < win / fs:
                continue
            k0 = int(math.ceil((lo - t0) * fs / step))
            n_win = int((min(hi, span_end) - t0) * fs - k0 * step - win) // step + 1
            if n_win < 1:
                continue
            shifted = [(si, off + k0 * step) for si, off in where]
            out.append((t0 + k0 * step / fs, shifted, n_win * step + win - step))
    return out


def common_spans(seg_lists, fs, min_samples):
    """Intersects three components' coverage.

    Returns (t0, [(segment_index, sample_offset) per component], n_samples) for
    every interval where all three components have unbroken data. Windowing only
    inside these means no window ever spans a gap on any component -- and
    carrying the segment index is what lets the caller find the right array when
    a chunk has more than one.
    """
    bounds = [[(t0, t0 + len(d) / fs) for t0, d in segs] for segs in seg_lists]

    spans = []
    idx = [0, 0, 0]
    while all(idx[k] < len(bounds[k]) for k in range(3)):
        starts = [bounds[k][idx[k]][0] for k in range(3)]
        ends = [bounds[k][idx[k]][1] for k in range(3)]
        lo, hi = max(starts), min(ends)
        if hi - lo >= min_samples / fs:
            where = [(idx[k], int(round((lo - starts[k]) * fs))) for k in range(3)]
            spans.append((lo, where, int((hi - lo) * fs)))
        # advance whichever segment ends first; it cannot intersect anything later
        idx[int(np.argmin(ends))] += 1
    return spans


def make_windows(data, offset, n_windows, win, step):
    """A (n_windows, win) view over `data`, without copying."""
    sub = data[offset:offset + (n_windows - 1) * step + win]
    return np.lib.stride_tricks.as_strided(
        sub, shape=(n_windows, win),
        strides=(sub.strides[0] * step, sub.strides[0]), writeable=False)


def clean_block(x, fs, freqmin, freqmax, taper):
    """`core.clean_and_filter_1d`, applied to a (n, win) block at once.

    Identical operations in identical order; the only difference is that scipy
    is given an axis instead of being called once per window.
    """
    x = signal.detrend(x, type="linear", axis=-1)
    x = signal.detrend(x, type="constant", axis=-1)
    x = x * taper
    nyquist = fs / 2.0
    actual_freqmax = freqmax if nyquist > freqmax else nyquist - 1.0
    if actual_freqmax > freqmin:
        b, a = signal.butter(4, [freqmin, actual_freqmax], btype="bandpass", fs=fs)
        x = signal.filtfilt(b, a, x, axis=-1)
    return x


def taper_vector(n):
    """The 5% Hann cosine taper `clean_and_filter_1d` applies to each end."""
    t = np.ones(n)
    k = int(n * 0.05)
    if k > 0:
        w = signal.windows.hann(k * 2)
        t[:k] = w[:k]
        t[-k:] = w[k:]
    return t


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


def cmd_scan(args):
    """Windows every chunk, scores every window, writes (t, p) per chunk and arm.

    Each archive is read, merged and gap-split exactly once and every arm
    windows the result, because reading and decoding 876 MB of mseed is a fixed
    cost that should not be paid per detector.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = json.loads(pathlib.Path(args.baseline_json).read_text())
    arms = [Arm(spec, args, device) for spec in args.arm]

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
            spans = clip_spans(common_spans(seg_lists, args.fs, arm.win),
                               near, args.fs, arm.win, arm.step)
            win, step, taper = arm.win, arm.step, arm.taper

            times, probs = [], []
            for t0, where, n_samp in spans:
                n_win = (n_samp - win) // step + 1
                if n_win < 1:
                    continue
                views = [make_windows(seg_lists[k][si][1], off, n_win, win, step)
                         for k, (si, off) in enumerate(where)]
                for lo in range(0, n_win, args.block_windows):
                    hi = min(lo + args.block_windows, n_win)
                    blk = np.empty((hi - lo, win, 3), dtype=np.float32)

                    def fill(task, lo=lo, views=views, win=win, taper=taper, blk=blk):
                        k, a, b = task
                        comp = comps[k]
                        c = clean_block(np.array(views[k][a:b]), args.fs,
                                        args.freqmin, args.freqmax, taper)
                        mu, sigma = base[comp]["mu"], base[comp]["sigma"]
                        blk[a - lo:b - lo, :, k] = ((c - mu) / max(sigma, 1e-12))

                    rows = max(1, math.ceil((hi - lo) / max(args.workers // 3, 1)))
                    tasks = [(k, a, min(a + rows, hi))
                             for k in range(3) for a in range(lo, hi, rows)]
                    list(pool.map(fill, tasks))
                    for bl in range(0, len(blk), args.batch_size):
                        bh = min(bl + args.batch_size, len(blk))
                        probs.append(score_block(arm.models, blk[bl:bh], device))
                    times.append(t0 + (np.arange(lo, hi) * step) / args.fs)

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

def haversine(lat0, lon0, lat, lon):
    p1, p2 = np.radians(lat0), np.radians(lat)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predicted_arrivals(args):
    """Catalogued events near the station, with their predicted P arrival."""
    st_tab = pd.read_csv(args.stations_csv, encoding="utf-8-sig")
    st_tab.columns = [c.strip() for c in st_tab.columns]
    s = st_tab[st_tab.Code == args.station].iloc[0]
    slat, slon = float(s.Latitude), float(s.Longitude)

    cat = pd.read_csv(args.catalog, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat["dist"] = haversine(slat, slon, cat.Latitude.values, cat.Longitude.values)
    cat = cat[cat.dist <= args.max_distance].copy()

    model = TauPyModel(model="iasp91")
    cache = {}

    def travel(dist_km, depth_km, phases):
        # Travel time depends mostly on distance and depth; caching on a coarse
        # grid avoids one taup call per event without materially moving either
        # arrival. The S phase is wanted as well as P: the question "did the
        # detector fire before S got here" cannot be asked without it.
        key = (round(dist_km / 5.0), round(max(depth_km, 0.0) / 5.0), phases)
        if key not in cache:
            try:
                arr = model.get_travel_times(source_depth_in_km=key[1] * 5.0,
                                             distance_in_degree=key[0] * 5.0 / 111.195,
                                             phase_list=list(phases))
                cache[key] = arr[0].time if arr else None
            except Exception:
                cache[key] = None
        return cache[key]

    P = ("p", "P", "Pn", "Pg")
    S = ("s", "S", "Sn", "Sg")
    depth = [d if pd.notna(d) else 10.0 for d in cat.Depth.values]
    cat["tt_p"] = [travel(d, z, P) for d, z in zip(cat.dist.values, depth)]
    cat["tt_s"] = [travel(d, z, S) for d, z in zip(cat.dist.values, depth)]
    cat = cat.dropna(subset=["tt_p"])
    origin = cat.t.map(lambda x: UTCDateTime(x.to_pydatetime()).timestamp)
    cat["p_epoch"] = origin + cat.tt_p
    cat["s_epoch"] = origin + cat.tt_s
    cat["sp_seconds"] = cat.tt_s - cat.tt_p
    return cat.sort_values("p_epoch").reset_index(drop=True), (slat, slon)


def background_and_guards(t, p, cat, args, win_s):
    """Splits scored windows into event guards and background.

    A window is "explained" when it overlaps any catalogued event's guard. The
    `- win_s` on the lower edge is what makes that an overlap test rather than a
    start-time test: a window beginning before the guard still reaches into it.
    """
    lo = cat.p_epoch.values - args.guard_pre - win_s
    hi = cat.p_epoch.values + args.guard_post
    explained = np.zeros(len(t), dtype=bool)
    idx = []
    for a, b in zip(lo, hi):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        explained[i:j] = True
        idx.append((i, j))
    return explained, idx


def cmd_report(args):
    """The operating table: what a threshold costs per day, and what it buys.

    **The benchmark's 0.5 is not an operating point here and using it would be a
    mistake.** That threshold was fixed on balanced classes whose negatives were
    amplitude-mined, and it carries no meaning against continuous background: on
    MANT the 6 s detector scores a median of 0.83 on *noise*, so 0.5 flags 95% of
    a quiet station-day. Thresholds are therefore derived from the measured
    background distribution -- pick the alarm budget, read off the threshold --
    with the 0.5 row kept only to show how far off it is.

    **Recall is reported against events the station actually recorded.** Only
    11.5% of catalogued events within 500 km of MANT reach SNR 3, and the median
    is 1.10, i.e. no signal in the raw trace. Scoring a detector on events whose
    waveform does not exist measures the catalogue's reach, not the model's.
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

    cat, (slat, slon) = predicted_arrivals(args)
    cat = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)].copy()
    explained, idx = background_and_guards(t, p, cat, args, win_s)
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
        snr = pd.read_csv(args.snr_csv)[["event_id", "snr"]]
        cat = cat.merge(snr, left_on="EventID", right_on="event_id", how="left")
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

    cat, _ = predicted_arrivals(args)
    ev = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)].copy()

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
     "timing": cmd_timing, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
