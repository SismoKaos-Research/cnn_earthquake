"""Scores in, declarations out -- and matching them across stations.

A detector does not emit alarms, it emits a score per window. Turning that into
something countable is where two mistakes live: not clustering a burst into one
declaration inflates every rate built from it, and a confirmation search that
excludes its endpoints drops exactly the events arriving at the physical limit.
"""
import glob
import sys

import numpy as np


def load_scores(pattern):
    """Every scored window from one glob, sorted by time."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no score files matched {pattern}")
    t = np.concatenate([np.load(f)["t"] for f in files])
    p = np.concatenate([np.load(f)["p"] for f in files])
    order = np.argsort(t)
    return t[order], p[order]


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
