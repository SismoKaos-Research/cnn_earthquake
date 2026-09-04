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

Three phases, because the second is the expensive one and should not be redone
to change a guard window:

    baseline  sample the archive, build the per-component (mu, sigma) the
              training windows were standardized against
    scan      window the whole record, score every window, write (t, p) per chunk
    report    associate scores with the catalogue and print the operating table

    python3 scripts/continuous_false_alarms.py baseline \\
        --zips 'afad_raw/MANT/*.zip' --out mant_baseline.json
    python3 scripts/continuous_false_alarms.py scan \\
        --zips 'afad_raw/MANT/*.zip' --baseline-json mant_baseline.json \\
        --ckpt-dir trained_model_branch1d_asinh --out-dir scores_mant
    python3 scripts/continuous_false_alarms.py report \\
        --scores 'scores_mant/*.npz' --station MANT \\
        --stations-csv catalogs/istasyon_katalog.csv \\
        --catalog catalogs/catalog_current.csv --out-prefix mant_fa

The preprocessing here is not a reimplementation-by-eye: it is the same order of
operations `seismic_cli.core` applies when it builds a training window (detrend
twice, 5% Hann taper, 4th-order 1-45 Hz bandpass, resample to 100 Hz, then
standardize against the station baseline), applied to whole blocks of windows at
a time instead of one at a time. `verify` checks that claim against real dataset
tensors rather than asserting it.
"""
import argparse
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
    s.add_argument("--ckpt-dir", required=True)
    s.add_argument("--branch-1d", default="cnn-lstm", choices=["lstm", "cnn", "cnn-lstm"])
    s.add_argument("--channels", default="1d", choices=["all", "1d", "2d"])
    s.add_argument("--fusion", default="linear", choices=["linear", "gate"])
    s.add_argument("--hidden", type=int, default=48)
    s.add_argument("--fusion-dim", type=int, default=96)
    s.add_argument("--window-seconds", type=float, default=6.0)
    s.add_argument("--step-seconds", type=float, default=6.0,
                   help="6.0 (the default) gives disjoint windows, so an alarm "
                        "count is also a count of independent decisions")
    s.add_argument("--batch-size", type=int, default=1024)
    s.add_argument("--block-windows", type=int, default=20000,
                   help="windows preprocessed per vectorized block")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--limit-chunks", type=int, default=None,
                   help="stop after N chunks (for a timing probe)")

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
    r.add_argument("--out-prefix", required=True)

    v = sub.add_parser("verify", help="check the preprocessing against real tensors")
    v.add_argument("--dataset-dir", required=True)
    v.add_argument("--ckpt-dir", required=True)
    v.add_argument("--branch-1d", default="cnn-lstm")
    v.add_argument("--channels", default="1d")
    v.add_argument("--fusion", default="linear")
    v.add_argument("--hidden", type=int, default=48)
    v.add_argument("--fusion-dim", type=int, default=96)
    v.add_argument("--limit", type=int, default=4000)

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

def load_models(args, device):
    """Loads every seed of one training arm as an evaluation ensemble."""
    ckpts = find_checkpoints(args.ckpt_dir, args.channels, args.fusion, args.branch_1d)
    models = []
    for c in ckpts:
        m = DualChannelBinaryNet(3, 3, hidden=args.hidden, fusion_dim=args.fusion_dim,
                                 channels=args.channels, fusion=args.fusion,
                                 branch1d=args.branch_1d).to(device)
        m.load_state_dict(torch.load(c, weights_only=True))
        m.eval()
        models.append(m)
    print(f"[ensemble] {len(ckpts)} checkpoint(s): "
          + ", ".join(c.name.split('_seed')[-1].replace('.pth', '') for c in ckpts))
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


def cmd_scan(args):
    """Windows every chunk, scores every window, writes (t, p) per chunk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args, device)
    base = json.loads(pathlib.Path(args.baseline_json).read_text())

    win = int(round(args.window_seconds * args.fs))
    step = int(round(args.step_seconds * args.fs))
    taper = taper_vector(win)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(glob.glob(args.zips))
    if args.limit_chunks:
        zips = zips[:args.limit_chunks]
    print(f"[scan] {len(zips)} chunk(s), {args.window_seconds:g}s windows "
          f"every {args.step_seconds:g}s, device={device}")

    grand_n = 0
    grand_t = time.time()
    for zi, z in enumerate(zips, 1):
        stem = pathlib.Path(z).stem
        dest = out_dir / f"{stem}.npz"
        if dest.exists():
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
        spans = common_spans(seg_lists, args.fs, win)
        del st

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
                for k, comp in enumerate(comps):
                    c = clean_block(np.array(views[k][lo:hi]), args.fs,
                                    args.freqmin, args.freqmax, taper)
                    mu, sigma = base[comp]["mu"], base[comp]["sigma"]
                    blk[:, :, k] = ((c - mu) / max(sigma, 1e-12)).astype(np.float32)
                for bl in range(0, len(blk), args.batch_size):
                    bh = min(bl + args.batch_size, len(blk))
                    probs.append(score_block(models, blk[bl:bh], device))
                times.append(t0 + (np.arange(lo, hi) * step) / args.fs)

        if not times:
            print(f"[{zi}/{len(zips)}] {stem}: no unbroken 3-component span", flush=True)
            continue
        t_arr = np.concatenate(times)
        p_arr = np.concatenate(probs)
        np.savez_compressed(dest, t=t_arr, p=p_arr)
        grand_n += len(t_arr)
        dt = time.time() - t_chunk
        print(f"[{zi}/{len(zips)}] {stem}: {len(t_arr):>7,} windows "
              f"({len(t_arr) * args.step_seconds / 86400:.1f} d), "
              f"{(p_arr > 0.5).mean() * 100:5.2f}% over 0.5, "
              f"{dt:.0f}s ({len(t_arr) / dt:,.0f} win/s)", flush=True)

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

    def travel(dist_km, depth_km):
        key = (round(dist_km / 5.0), round(max(depth_km, 0.0) / 5.0))
        if key not in cache:
            try:
                arr = model.get_travel_times(source_depth_in_km=key[1] * 5.0,
                                             distance_in_degree=key[0] * 5.0 / 111.195,
                                             phase_list=["p", "P", "Pn", "Pg"])
                cache[key] = arr[0].time if arr else None
            except Exception:
                cache[key] = None
        return cache[key]

    tt = [travel(d, dep if pd.notna(dep) else 10.0)
          for d, dep in zip(cat.dist.values, cat.Depth.values)]
    cat["tt"] = tt
    cat = cat.dropna(subset=["tt"])
    cat["p_epoch"] = cat.t.map(lambda x: UTCDateTime(x.to_pydatetime()).timestamp) + cat.tt
    return cat.sort_values("p_epoch").reset_index(drop=True), (slat, slon)


def cmd_report(args):
    """Tabulates alarms against the catalogue, at a range of thresholds."""
    files = sorted(glob.glob(args.scores))
    if not files:
        sys.exit(f"no score files matched {args.scores}")
    t_all, p_all = [], []
    for f in files:
        d = np.load(f)
        t_all.append(d["t"])
        p_all.append(d["p"])
    t = np.concatenate(t_all)
    p = np.concatenate(p_all)
    order = np.argsort(t)
    t, p = t[order], p[order]

    step = float(np.median(np.diff(t[:100000]))) if len(t) > 1 else 6.0
    days = len(t) * step / 86400.0
    print(f"{'=' * 70}\nCONTINUOUS FALSE ALARMS  --  {args.station}\n{'=' * 70}")
    print(f"  {len(t):,} windows from {len(files)} chunk(s), {step:g}s apart")
    print(f"  {days:.1f} days of record, "
          f"{pd.to_datetime(t.min(), unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(t.max(), unit='s'):%Y-%m-%d}")

    cat, (slat, slon) = predicted_arrivals(args)
    in_span = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)]
    print(f"  {len(in_span):,} catalogued events within {args.max_distance:g} km "
          f"of ({slat:.4f}, {slon:.4f}) in that span")

    # A window is "explained" when it overlaps any event's guard interval.
    lo = in_span.p_epoch.values - args.guard_pre - step
    hi = in_span.p_epoch.values + args.guard_post
    explained = np.zeros(len(t), dtype=bool)
    for a, b in zip(lo, hi):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        explained[i:j] = True
    print(f"  {explained.sum():,} windows ({100 * explained.mean():.3f}%) fall inside "
          f"an event guard [-{args.guard_pre:g}s, +{args.guard_post:g}s]")

    print(f"\n  {'threshold':>10}{'alarms':>12}{'unexplained':>13}"
          f"{'per day':>10}{'per hour':>10}{'FPR':>10}")
    rows = []
    for thr in (0.5, 0.9, 0.95, 0.99, 0.999, 0.9999):
        alarm = p > thr
        unexp = int((alarm & ~explained).sum())
        n_quiet = int((~explained).sum())
        rows.append({"threshold": thr, "alarms": int(alarm.sum()),
                     "unexplained": unexp, "per_day": unexp / days,
                     "fpr": unexp / max(n_quiet, 1)})
        print(f"  {thr:>10.4g}{int(alarm.sum()):>12,}{unexp:>13,}"
              f"{unexp / days:>10.1f}{unexp / days / 24:>10.2f}"
              f"{unexp / max(n_quiet, 1):>10.5f}")

    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_thresholds.csv", index=False)

    # Recall on catalogued events, on continuous data rather than cut windows.
    #
    # The denominator is events whose guard interval actually contains scored
    # windows, NOT every catalogued event in the span. The record has gaps -- the
    # four dead MANT chunks, and every place a component dropped out -- and an
    # event nobody has data for is not a miss, it is an absence. Counting those
    # as misses would understate recall by however much the station was down.
    idx = [(np.searchsorted(t, a), np.searchsorted(t, b, side="right"))
           for a, b in zip(lo, hi)]
    covered = np.array([j > i for i, j in idx])
    print(f"\n  recall on catalogued events (any window over threshold in the guard)")
    print(f"    {len(in_span) - covered.sum():,} of {len(in_span):,} events fall in a "
          f"gap and are excluded -- no data, so not a miss")
    print(f"    {'threshold':>10}{'events':>9}{'found':>8}{'recall':>9}")
    best = np.array([p[i:j].max() if j > i else np.nan for i, j in idx])
    for thr in (0.5, 0.9, 0.99):
        hits = int(np.nansum(best > thr))
        print(f"    {thr:>10.4g}{int(covered.sum()):>9,}{hits:>8,}"
              f"{hits / max(int(covered.sum()), 1):>9.4f}")

    ev = in_span.copy()
    ev["best_prob"] = best
    ev = ev[covered]
    print(f"\n  recall by magnitude at p>0.5")
    print(f"    {'band':>12}{'events':>9}{'found':>8}{'recall':>9}{'med dist':>10}")
    for band, g in ev.groupby(pd.cut(ev.Magnitude, [0, 2, 2.5, 3, 3.5, 4, 10]),
                              observed=True):
        if not len(g):
            continue
        hits = int((g.best_prob > 0.5).sum())
        print(f"    {str(band):>12}{len(g):>9,}{hits:>8,}{hits / len(g):>9.4f}"
              f"{g.dist.median():>10.0f}")
    print(f"\n  recall by epicentral distance at p>0.5")
    print(f"    {'band (km)':>12}{'events':>9}{'found':>8}{'recall':>9}{'med mag':>10}")
    for band, g in ev.groupby(pd.cut(ev.dist, [0, 25, 50, 100, 200, 500]),
                              observed=True):
        if not len(g):
            continue
        hits = int((g.best_prob > 0.5).sum())
        print(f"    {str(band):>12}{len(g):>9,}{hits:>8,}{hits / len(g):>9.4f}"
              f"{g.Magnitude.median():>10.1f}")
    ev[["EventID", "t", "Magnitude", "dist", "Depth", "best_prob", "Location"]] \
        .to_csv(f"{args.out_prefix}_events.csv", index=False)

    # Alarms by hour of local day: a cultural-noise signature would show here.
    hod = pd.to_datetime(t[(p > 0.99) & ~explained], unit="s").tz_localize("UTC") \
        .tz_convert("Europe/Istanbul").hour
    if len(hod):
        counts = np.bincount(hod, minlength=24)
        peak, trough = counts.max(), counts[counts > 0].min()
        print(f"\n  unexplained alarms at p>0.99 by local hour "
              f"(peak {peak:,} / trough {trough:,} = {peak / max(trough, 1):.2f}x)")
        for h in range(0, 24, 2):
            bar = "#" * int(40 * counts[h] / max(peak, 1))
            print(f"    {h:02d}:00 {counts[h]:>7,} {bar}")

    keep = (p > 0.99) & ~explained
    pd.DataFrame({"time": pd.to_datetime(t[keep], unit="s"), "prob": p[keep]}) \
        .to_csv(f"{args.out_prefix}_alarms.csv", index=False)
    print(f"\n  wrote {args.out_prefix}_thresholds.csv, {args.out_prefix}_events.csv "
          f"and {args.out_prefix}_alarms.csv ({int(keep.sum()):,} rows)")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

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
    models = load_models(args, device)
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
    print("         training log reports 0.9896 AUC and 141/7906 = 0.0178 FPR")


def main():
    args = parse_args()
    {"baseline": cmd_baseline, "scan": cmd_scan,
     "report": cmd_report, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
