"""Hourly chaos-feature matrix and forecast labels, built once for every consumer.

The screen and the model must see the same rows, the same label and the same
floor, or their numbers cannot be compared -- and comparing them is the whole
point of running the screen first. So the preparation lives here rather than
being written twice.

**The label horizon is 6 HOURS.** `label_hours` used to truncate the horizon to
whole days, so a 0.25-day horizon became zero and every label came back negative
-- silently, as a plausible all-zero array -- and this module counted events
itself to route around it. It no longer needs to: `label_hours` takes fractional
days, and it also opens the horizon at the END of the feature window, which
matters more here than the truncation did. Extraction emits 72 windows per hour
collapsed to an hourly mean, so the features at t cover [t, t+1h]; counting an
event inside that hour as future is one hour of a six-hour horizon.

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

from sismokaos.catalog import (count_events_in_window, days_since_prev_major,
                               haversine_km, label_hours)

# The features are single-station, so the label's origin is too. Default BODT,
# overridable so a second station can be scored against its OWN local label --
# which is what makes a cross-station comparison a replication rather than two
# views of one arbitrary point.
STATION_LAT, STATION_LON = 37.0622, 27.3103

# Chosen 2026-08-21 by label_sweep.py on statistical power: 232 events, 25%
# positive, persistence floor 0.543. Tighter radii are physically preferable
# and have too few events at 181 days to detect anything.
MIN_MAGNITUDE = 2.5
RADIUS_KM = 400.0
HORIZON_HOURS = 6.0

# Distance-graded alternative, from Sırdar et al. (AIMSA 2026) on ELBA.
# A flat threshold is physically inconsistent: an M2.5 at 20 km sits far above
# the station's noise floor while an M2.5 at 380 km sits under it, yet the flat
# label calls both positive. Grading the floor with distance approximates
# constant detectability, so the positive class is "an event this station could
# actually have seen" rather than "an event that happened somewhere nearby".
#
# (outer radius km, minimum magnitude), innermost first, contiguous.
MAGNITUDE_BANDS = ((100.0, 0.0), (300.0, 3.0), (500.0, 5.0), (1000.0, 6.0))

AGGS = ("mean", "std", "min", "max")
ID_COLUMNS = ("Pencere_ID", "Zaman_Dk", "hour_start", "index")

# Within-hour SHAPE, which mean/std/min/max discard entirely. A feature that
# climbs steadily through the hour, one that spikes at minute 40, and one that
# oscillates can share all four summary statistics.
#
# These are deliberately crude. They are the cheap test of the hypothesis a
# CNN encoder over the 50 s stream would embody: if within-hour shape carries
# association with the label, a slope term and a half-difference will show some
# of it, and a CNN is then worth its parameter budget. If they show nothing, a
# network learning a fancier function of the same 72 numbers is unlikely to.
SHAPE_AGGS = ("slope", "halfdiff", "argmax", "ac1")

# Cross-hour context: the recurrent half of a CNN+LSTM proposal. The shape
# statistics above stand in for what a convolutional encoder would read INSIDE
# an hour; these stand in for what a recurrent layer would read ACROSS hours.
# Lags plus their deltas, because a level and a change are different claims:
# "entropy is high" and "entropy has been rising for six hours" are not the
# same hypothesis and a model given only the current hour can express neither.
LAG_HOURS = (1, 3, 6, 12, 24)


def _shape_block(m):
    """Shape statistics for one hour's (n_windows, n_features) matrix.

    Returns:
        (4 * n_features,) array ordered slope, halfdiff, argmax, ac1.

    Missing samples are filled with their own column's within-hour mean before
    fitting, so one absent window tilts nothing; the feature stream is 0.04%
    NaN, so this touches almost nothing but keeps a whole hour from going NaN.
    """
    n = len(m)
    col_mean = np.nanmean(m, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    m = np.where(np.isfinite(m), m, col_mean)

    if n < 4:
        nan = np.full(m.shape[1], np.nan)
        return np.concatenate([nan, nan, nan, nan])

    # Least-squares slope in units of feature per hour: t is centred and scaled
    # to [-0.5, 0.5] so the coefficient does not depend on how many windows the
    # hour happened to contain.
    t = (np.arange(n) - (n - 1) / 2.0) / max(1, n - 1)
    slope = (t[:, None] * (m - m.mean(axis=0))).sum(axis=0) / (t ** 2).sum()

    half = m[n // 2:].mean(axis=0) - m[:n // 2].mean(axis=0)
    argmax = m.argmax(axis=0) / (n - 1)

    a = m[:-1] - m[:-1].mean(axis=0)
    b = m[1:] - m[1:].mean(axis=0)
    ac1 = (a * b).sum(axis=0) / np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0) + 1e-12)
    return np.concatenate([slope, half, argmax, ac1])


def load_chaos_hourly(parquet_path, aggs=AGGS, shape=False):
    """Aggregates the 50 s feature stream to one row per hour.

    Args:
        aggs: Summary statistics per feature per hour.
        shape: Also compute `SHAPE_AGGS` -- the within-hour trajectory
            statistics the summaries discard.

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
    out = out.sort_index()
    if not shape:
        return out

    hours, blocks = [], []
    for hour, grp in df.groupby("hour_start", sort=True):
        hours.append(hour)
        blocks.append(_shape_block(grp[cols].to_numpy(dtype=float)))
    names = [f"{c}_{a}" for a in SHAPE_AGGS for c in cols]
    shp = pd.DataFrame(blocks, index=pd.DatetimeIndex(hours, name="hour_start"),
                       columns=names)
    return out.join(shp, how="left")


def load_events(catalog_path, min_magnitude=MIN_MAGNITUDE, radius_km=RADIUS_KM,
                station=None, bands=None):
    """Qualifying event times, sorted, as datetime64.

    Args:
        station: (lat, lon), or a key in `sismokaos.catalog.STATION_COORDS`.
            Defaults to BODT.
        bands: Distance-graded magnitude floors, e.g. `MAGNITUDE_BANDS`. When
            given, `min_magnitude` and `radius_km` are ignored. Left None by
            default so every published number stays reproducible -- this
            changes the positive class AND the persistence floor together.

    Returns:
        np.ndarray of event times within `radius_km` of the station.
    """
    if station is None:
        lat, lon = STATION_LAT, STATION_LON
    elif isinstance(station, str):
        from sismokaos.catalog import STATION_COORDS
        lat, lon = STATION_COORDS[station]
    else:
        lat, lon = station
    cat = pd.read_csv(catalog_path, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t", "Magnitude", "Latitude", "Longitude"])
    d = haversine_km(lat, lon, cat.Latitude.values, cat.Longitude.values)
    if bands is None:
        keep = (cat.Magnitude.values >= min_magnitude) & (d <= radius_km)
    else:
        keep = band_selection(d, cat.Magnitude.values, bands)
    return np.sort(cat[keep].t.values.astype("datetime64[s]"))


def band_selection(distance_km, magnitude, bands=MAGNITUDE_BANDS):
    """Boolean mask for the distance-graded magnitude scheme.

    Bands are contiguous and given innermost first as (outer radius, minimum
    magnitude). An event beyond the outermost radius is excluded whatever its
    magnitude, which is what makes this a cap rather than an open tail.
    """
    distance_km = np.asarray(distance_km, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    keep = np.zeros(distance_km.shape, dtype=bool)
    inner = -np.inf
    for outer, m_min in bands:
        in_band = (distance_km > inner) & (distance_km <= outer)
        keep |= in_band & (magnitude >= m_min)
        inner = outer
    return keep


def add_lags(feats, lag_hours=LAG_HOURS, base=None):
    """Appends lagged levels and deltas for a subset of columns.

    Args:
        feats: Hourly feature frame, sorted by hour.
        base: Columns to lag. Lagging all ~1,000 would multiply the
            multiple-comparison burden sixfold for no reason; the caller passes
            the columns that carry association on their own.

    Returns:
        A new frame with `{col}_lag{h}` and `{col}_d{h}` added.

    Rows in the first `max(lag_hours)` hours get NaN lags, and gaps in the
    archive produce NaN rather than silently reaching across them, because the
    index is reindexed to a continuous hourly range first -- a shift() on a
    frame with missing hours would otherwise treat the row before a two-hour
    gap as if it were one hour earlier.
    """
    base = list(feats.columns if base is None else base)
    full = feats.reindex(pd.date_range(feats.index.min(), feats.index.max(), freq="h"))
    out = {}
    for h in lag_hours:
        sh = full[base].shift(h)
        for c in base:
            out[f"{c}_lag{h}"] = sh[c]
            out[f"{c}_d{h}"] = full[c] - sh[c]
    return full.join(pd.DataFrame(out, index=full.index)).reindex(feats.index)


def build(parquet_path, catalog_path, horizon_hours=HORIZON_HOURS, aggs=AGGS,
          shape=False, lags=False, lag_top=40, station=None, bands=None):
    """Features, labels and the persistence predictor, on one shared hour index.

    Returns:
        Tuple of (features DataFrame, labels int array, days-since-previous
        array, hour index). Rows whose features are entirely absent are
        dropped; NaNs inside a kept row are left for the model to handle.
    """
    feats = load_chaos_hourly(parquet_path, aggs, shape=shape)
    events = load_events(catalog_path, station=station, bands=bands)
    idx = pd.DatetimeIndex(feats.index)

    # (t + 1h, t + 1h + horizon]. Excluding only the exact instant t was not
    # enough: extraction emits 72 windows per HOUR and they are collapsed to an
    # hourly mean, so the features at t cover [t, t+1h] and an event inside that
    # hour is both visible to the model and counted as its future. At a 6-hour
    # horizon that is one hour in six -- 17% of the window, not a rounding
    # detail. label_hours applies the offset and now takes fractional days, so
    # the workaround this module was written around can go.
    horizon_days = horizon_hours / 24.0
    labels = label_hours(idx, events, horizon_days)
    dsp = days_since_prev_major(idx, events)

    if lags:
        # Rank columns by their own marginal association and lag only the top
        # ones. Selecting on the full series does leak into the fold structure,
        # so this is a screening convenience: it decides which columns to BUILD,
        # never which to trust, and the walk-forward evaluation is what judges
        # the result.
        from sismokaos.metrics import safe_auc
        score = {}
        for c in feats.columns:
            v = feats[c].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() > 100 and np.nanstd(v[ok]) > 0:
                score[c] = safe_auc(labels[ok], v[ok], oriented=True)
        top = sorted(score, key=score.get, reverse=True)[:lag_top]
        feats = add_lags(feats, base=top)
    return feats, labels, dsp, idx


def persistence_scores(dsp):
    """The trivial predictor the floor is built from: recency of the last event.

    Negated so that "more recent" ranks higher, which is the direction that
    makes physical sense. The floor is scored with `oriented=True` regardless,
    because a baseline's sign is free.
    """
    d = np.asarray(dsp, dtype=float)
    return -np.where(np.isnan(d), np.nanmax(d) if np.isfinite(d).any() else 0.0, d)
