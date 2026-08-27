"""Hourly chaos-feature matrix and forecast labels, built once for every consumer.

The screen and the model must see the same rows, the same label and the same
floor, or their numbers cannot be compared -- and comparing them is the whole
point of running the screen first. So the preparation lives here rather than
being written twice.

**The label horizon is 6 HOURS, and that is a trap in this codebase.**
`seismolib.catalog.label_hours` does `np.timedelta64(int(horizon_days), "D")`,
so a 0.25-day horizon truncates to zero and every label comes back negative --
silently, with a plausible-looking all-zero array. `count_events_in_window`
takes fractional days correctly, so labels are built from that instead.

**Aggregation.** Extraction emits 72 windows per hour (200 s window, 50 s step).
`sequence_variance_check.py` measured that collapsing them to hourly means
*raises* within-sequence variation for a 24-step sequence (56.1% against 37.8%
native) -- the binding constraint is context length, not granularity, since 24
native steps span only 20 minutes. So hourly is the right granularity, and mean
is not a loss. std/min/max are carried anyway: they are free for a tree model
and describe the within-hour spread that the mean discards.

**`log1p_dsp` is a feature, not a leak.** The persistence floor ranks by days
since the previous qualifying event. Withholding it asks a model to beat a bar
built from a number it cannot see, which is how the ruled-out GRU experiment
produced its largest apparent gains (+0.069/+0.047).
"""

import numpy as np
import pandas as pd

from seismolib.catalog import count_events_in_window, days_since_prev_major, haversine_km

# BODT. The features are single-station, so the label's origin is too.
STATION_LAT, STATION_LON = 37.0622, 27.3103

# Chosen 2026-08-21 by label_sweep.py on statistical power: 232 events, 25%
# positive, persistence floor 0.543. Tighter radii are physically preferable
# and have too few events at 181 days to detect anything.
MIN_MAGNITUDE = 2.5
RADIUS_KM = 400.0
HORIZON_HOURS = 6.0

AGGS = ("mean", "std", "min", "max")
ID_COLUMNS = ("Pencere_ID", "Zaman_Dk", "hour_start", "index")


def load_chaos_hourly(parquet_path, aggs=AGGS):
    """Aggregates the 50 s feature stream to one row per hour.

    Returns:
        DataFrame indexed by hour_start, columns `{feature}_{agg}`.

    `Zaman_Dk` is absolute minutes since the Unix epoch, so hours align even
    across a multi-day outage -- which matters here, the archive has one.
    """
    df = pd.read_parquet(parquet_path)
    df = df.assign(hour_start=pd.to_datetime(df["Zaman_Dk"], unit="m").dt.floor("h"))
    cols = [c for c in df.columns if c not in ID_COLUMNS]
    g = df.groupby("hour_start")[cols]
    out = pd.concat({a: g.agg(a) for a in aggs}, axis=1)
    out.columns = [f"{c}_{a}" for a, c in out.columns]
    return out.sort_index()


def load_events(catalog_path, min_magnitude=MIN_MAGNITUDE, radius_km=RADIUS_KM):
    """Qualifying event times, sorted, as datetime64.

    Returns:
        np.ndarray of event times within `radius_km` of the station.
    """
    cat = pd.read_csv(catalog_path, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t", "Magnitude", "Latitude", "Longitude"])
    d = haversine_km(STATION_LAT, STATION_LON, cat.Latitude.values, cat.Longitude.values)
    sel = cat[(cat.Magnitude >= min_magnitude) & (d <= radius_km)]
    return np.sort(sel.t.values.astype("datetime64[s]"))


def build(parquet_path, catalog_path, horizon_hours=HORIZON_HOURS, aggs=AGGS):
    """Features, labels and the persistence predictor, on one shared hour index.

    Returns:
        Tuple of (features DataFrame, labels int array, days-since-previous
        array, hour index). Rows whose features are entirely absent are
        dropped; NaNs inside a kept row are left for the model to handle.
    """
    feats = load_chaos_hourly(parquet_path, aggs)
    events = load_events(catalog_path)
    idx = pd.DatetimeIndex(feats.index)

    # (t, t + horizon] -- an event at exactly t is already observable, so
    # counting it as future would let the label read its own input.
    horizon_days = horizon_hours / 24.0
    labels = (count_events_in_window(idx, events, horizon_days, forward=True) > 0).astype(int)
    dsp = days_since_prev_major(idx, events)
    return feats, labels, dsp, idx


def persistence_scores(dsp):
    """The trivial predictor the floor is built from: recency of the last event.

    Negated so that "more recent" ranks higher, which is the direction that
    makes physical sense. The floor is scored with `oriented=True` regardless,
    because a baseline's sign is free.
    """
    d = np.asarray(dsp, dtype=float)
    return -np.where(np.isnan(d), np.nanmax(d) if np.isfinite(d).any() else 0.0, d)
