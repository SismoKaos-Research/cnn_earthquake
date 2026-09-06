"""Interval algebra: which stretches of record a window may be cut from.

A window that straddles a data gap is samples from two different times glued
together, and it scores like a transient. Everything here exists so that never
happens, and so that two stations are only ever compared over the span they
both recorded.
"""
import math

import numpy as np


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
