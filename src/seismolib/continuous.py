"""Turning a continuous station archive into scored windows and alarms.

Extracted from `scripts/continuous_false_alarms.py`, which had grown to 1,522
lines and was the most scientifically important tool in the repo while sitting
in a directory that cannot be imported. `cut_event_windows.py` needed six of
these functions and reached them with `sys.path.insert(__file__.parent)` -- a
script importing a sibling script by path manipulation, which is what made the
two impossible to test together and easy to break apart.

Nothing here is new. The bodies are the ones that produced the published
continuous-detection figures, moved verbatim; only two signatures changed, from
taking an argparse namespace to taking their parameters, because three CLIs call
them and their flags do not have to agree.

Five things live here, in the order a scan uses them:

**Reading** an archive chunk and picking its three components.
**Spans** -- the interval algebra that keeps a window from ever straddling a
data gap, and that decides which stretch two stations both recorded.
**Preprocessing** one block the same way the training windows were made.
**Association** with the catalogue: predicted arrivals, measured SNR, and which
windows a catalogued event can excuse.
**Alarms** -- clustering scores into declarations and matching them across
stations.
"""
import glob
import math
import pathlib
import tempfile
import zipfile

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from scipy import signal

from seismolib.arrivals import P_PHASES, S_PHASES, ArrivalTimes
from seismolib.catalog import haversine_km as haversine

# Component roles, in the order the training encoder stacks them. Taking the
# first three channels alphabetically would grab ['1','2','E'] at a station with
# mixed sensor codes -- two horizontals and no vertical. See core._COMPONENT_ROLES.
COMPONENT_ROLES = (("Z",), ("N", "1"), ("E", "2"))


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


def predicted_arrivals(station, stations_csv, catalog, max_distance=500.0):
    """Catalogued events near the station, with their predicted P arrival.

    Takes its parameters rather than an argparse namespace: this is called from
    three CLIs whose flags do not have to agree, and a function that reaches
    into `args` cannot be tested without building one.

    Args:
        station: Station code, matched against the station table's `Code`.
        stations_csv: Station table with Code/Latitude/Longitude.
        catalog: Event catalogue CSV.
        max_distance: Events beyond this are dropped, in km.

    Returns:
        Tuple of (catalogue DataFrame with p_epoch/s_epoch/sp_seconds/dist,
        (station_lat, station_lon)).
    """
    st_tab = pd.read_csv(stations_csv, encoding="utf-8-sig")
    st_tab.columns = [c.strip() for c in st_tab.columns]
    s = st_tab[st_tab.Code == station].iloc[0]
    slat, slon = float(s.Latitude), float(s.Longitude)

    cat = pd.read_csv(catalog, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat["dist"] = haversine(slat, slon, cat.Latitude.values, cat.Longitude.values)
    cat = cat[cat.dist <= max_distance].copy()

    taup = ArrivalTimes(grid_km=5.0)

    def travel(dist_km, depth_km, phases):
        return taup.travel(dist_km, depth_km, phases)

    P, S = P_PHASES, S_PHASES
    depth = [d if pd.notna(d) else 10.0 for d in cat.Depth.values]
    cat["tt_p"] = [travel(d, z, P) for d, z in zip(cat.dist.values, depth)]
    cat["tt_s"] = [travel(d, z, S) for d, z in zip(cat.dist.values, depth)]
    cat = cat.dropna(subset=["tt_p"])
    origin = cat.t.map(lambda x: UTCDateTime(x.to_pydatetime()).timestamp)
    cat["p_epoch"] = origin + cat.tt_p
    cat["s_epoch"] = origin + cat.tt_s
    cat["sp_seconds"] = cat.tt_s - cat.tt_p
    return cat.sort_values("p_epoch").reset_index(drop=True), (slat, slon)


def load_snr(path):
    """The measured-SNR table, one row per event.

    `station_detection_range.py` can emit an event twice when it falls in two
    overlapping chunks, and a LEFT JOIN on a non-unique key silently expands the
    frame it is joined into. That is not hypothetical: DEMI's table has 269
    duplicated ids against MANT's and GCAM's zero, and the expansion desynced
    `best_prob` from the catalogue it was computed for -- which raised here, but
    would have quietly shifted every recall denominator if the lengths had
    happened to line up.

    The larger SNR is kept. A duplicate is the same event seen from two chunks,
    and the smaller reading is usually the one that fell near a chunk edge and
    was measured on a truncated window.
    """
    snr = pd.read_csv(path)[["event_id", "snr"]]
    return snr.sort_values("snr", ascending=False).drop_duplicates(subset="event_id")


def background_and_guards(t, p, cat, win_s, guard_pre=10.0, guard_post=60.0):
    """Splits scored windows into event guards and background.

    A window is "explained" when it overlaps any catalogued event's guard. The
    `- win_s` on the lower edge is what makes that an overlap test rather than a
    start-time test: a window beginning before the guard still reaches into it.
    """
    lo = cat.p_epoch.values - guard_pre - win_s
    hi = cat.p_epoch.values + guard_post
    explained = np.zeros(len(t), dtype=bool)
    idx = []
    for a, b in zip(lo, hi):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        explained[i:j] = True
        idx.append((i, j))
    return explained, idx


def coverage_spans(t, step, slack=2.5):
    """Intervals this station actually scored, from gaps in its window times.

    A station's record is not one continuous run: the archive is gap-split, and
    GCAM stops recording entirely in December 2024. Without this, every alarm at
    the other station during an outage would count as unconfirmed and be scored
    as a false alarm removed -- which reads as a spectacular coincidence gain
    and is only missing data.
    """
    if not len(t):
        return []
    breaks = np.flatnonzero(np.diff(t) > slack * step)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(t) - 1]])
    return [(float(t[i]), float(t[j] + step)) for i, j in zip(starts, ends)]


def intersect_spans(a, b):
    """The intervals covered by both stations."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def in_spans(t, spans):
    """Boolean mask: which of `t` fall inside any span."""
    keep = np.zeros(len(t), dtype=bool)
    for lo, hi in spans:
        keep[np.searchsorted(t, lo):np.searchsorted(t, hi, side="right")] = True
    return keep


def declarations(t, p, thr, cluster_seconds):
    """Alarms above `thr`, collapsed to one declaration per burst.

    Returns the time and score of each burst's peak. Clustering is the same
    rule `confusion` uses, and for the same reason: a noise burst spanning ten
    windows is one declaration, not ten.
    """
    hit = np.flatnonzero(p > thr)
    if not len(hit):
        return np.empty(0), np.empty(0)
    cuts = np.flatnonzero(np.diff(t[hit]) > cluster_seconds)
    groups = np.split(hit, cuts + 1)
    peak = [g[np.argmax(p[g])] for g in groups]
    return t[peak], p[peak]


def confirmed(ta, tb, window):
    """Mask over A's declarations: does B declare within +/- `window`?

    Both arrays are sorted, so this is two binary searches per declaration
    rather than a cross product -- the alarm lists run to tens of thousands at
    a loose threshold.
    """
    if not len(ta) or not len(tb):
        return np.zeros(len(ta), dtype=bool)
    lo = np.searchsorted(tb, ta - window, side="left")
    hi = np.searchsorted(tb, ta + window, side="right")
    return hi > lo

