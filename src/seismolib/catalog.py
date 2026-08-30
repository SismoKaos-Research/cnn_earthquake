"""Earthquake-catalog loading and hourly labelling.

Everything downstream of a catalog CSV and upstream of a model: event
selection by region and magnitude, per-hour labels over a forward horizon,
and the time-since/time-until features those labels are scored against."""

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

AEGEAN_BBOX = (36.0, 40.0, 25.0, 30.0)  # lat0, lat1, lon0, lon1


STATION_COORDS = {"BODT": (37.0622, 27.3103), "DAT": (36.7308, 27.5767)}


def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    r = 6371.0
    la1, lo1 = np.radians(lat0), np.radians(lon0)
    la2, lo2 = np.radians(lats), np.radians(lons)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def station_distance_mask(lats: np.ndarray, lons: np.ndarray, stations, max_dist_km: float):
    """Boolean mask for events within `max_dist_km` of the NEAREST named station."""
    if not max_dist_km or max_dist_km <= 0:
        return np.ones(len(lats), dtype=bool)
    best = None
    for s in stations:
        lat, lon = STATION_COORDS[s] if isinstance(s, str) else s
        d = haversine_km(lat, lon, lats, lons)
        best = d if best is None else np.minimum(best, d)
    return best <= max_dist_km


def parse_hour_start(pencere_id: str):
    """Parses a Sismokaos-featureExtract window ID into its containing hour.

    e.g. '2024_11_15_00_w01' -> datetime(2024,11,15,0,0,0). Ignores the small
    (<=150s) stitching offset from PREV_LEN carry-over -- negligible against
    a 30-day forecast horizon.

    Args:
        pencere_id: Window ID string, format 'YYYY_MM_DD_HH_wNN'.

    Returns:
        Tuple of (hour_start datetime, window index int), or (None, None) if
        `pencere_id` doesn't match the expected format.
    """
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_w(\d+)", pencere_id)
    if not m:
        return None, None
    y, mo, d, h, w = m.groups()
    return datetime(int(y), int(mo), int(d), int(h)), int(w)


def load_hourly_features(features_csv: str) -> pd.DataFrame:
    """Loads the combined features file and aggregates it to hourly means.

    Two producers write this file with *different* `Zaman_Dk` conventions, and
    picking the wrong one is silent rather than fatal:

    - the Rust engine writes absolute UTC minutes since the Unix epoch;
    - Sismokaos-featureExtract writes minutes **within the containing hour**
      (observed range 3.3--62.5), with the date living in `Pencere_ID`.

    Reading the second as the first maps every row into a single hour of
    1970-01-01, so 1.2M windows collapse to ~2 hourly vectors and every split
    comes out empty. `Pencere_ID` is therefore preferred whenever it parses,
    and `Zaman_Dk` is the fallback.
    """
    if str(features_csv).endswith(".npy"):
        df = pd.DataFrame.from_records(np.load(features_csv, allow_pickle=False))
    else:
        df = pd.read_csv(features_csv)

    hour_start = None
    if "Pencere_ID" in df.columns:
        # 'YYYY_MM_DD_HH_wNN' -- the first 13 characters are the hour. Sliced and
        # parsed with an explicit format rather than row-wise, which matters:
        # this file is ~1.2M rows and a per-row lambda takes minutes.
        parsed = pd.to_datetime(df["Pencere_ID"].astype(str).str[:13],
                                format="%Y_%m_%d_%H", errors="coerce")
        if parsed.notna().mean() > 0.5:
            hour_start = parsed

    if hour_start is None:
        hour_start = pd.to_datetime(df["Zaman_Dk"], unit="m").dt.floor("h")

    df = df.copy().assign(hour_start=hour_start)
    
    feature_cols = [c for c in df.columns if c not in ("Pencere_ID", "Zaman_Dk", "hour_start", "index")]
    hourly = df.groupby("hour_start")[feature_cols].mean().sort_index()
    return hourly


def load_aegean_events(catalog_path: str, min_magnitude: float = 4.5,
                       stations=None, max_dist_km: float = None) -> np.ndarray:
    """Loads catalog events within the Aegean bounding box at or above a magnitude."""
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                            aegean.Longitude.to_numpy(),
                                            stations, max_dist_km)]
    return np.sort(aegean.dt.to_numpy())


def load_aegean_events_with_magnitude(catalog_path: str, min_magnitude: float = 3.0,
                                      stations=None, max_dist_km: float = None):
    """Loads catalog events (times AND magnitudes) within the Aegean bounding box.

    Companion to `load_aegean_events`, which only returns times -- this is
    for magnitude-derived catalog features (mean magnitude, b-value, energy
    release, magnitude deficit) that need a lower completeness threshold
    than the M>=4.5 "major event" set used for labels/persistence, since
    b-value estimation needs more data points than the rare large events
    alone provide (16,724 M>=3.0 Aegean events vs. 261 M>=4.5 ones).

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include (completeness
            threshold for the returned "background" catalog).

    Returns:
        Tuple of (times, magnitudes) -- times is a sorted array of numpy
        datetime64 event times, magnitudes is the matching float64 array,
        same order.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()].sort_values("dt")
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                              aegean.Longitude.to_numpy(),
                                              stations, max_dist_km)]
    return aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64)


def load_aegean_events_with_location(catalog_path: str, min_magnitude: float = 3.0,
                                     stations=None, max_dist_km: float = None):
    """Loads catalog events (times, magnitudes, AND lat/lon) within the Aegean bbox.

    Companion to `load_aegean_events_with_magnitude`, adding coordinates for
    features that need event location -- nearest-neighbour distance
    (Zaliapin & Ben-Zion) and spatial Shannon entropy, both from Convertito
    et al. 2024 (Sci. Rep. 14:2964).

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include (completeness
            threshold for the returned "background" catalog).

    Returns:
        Tuple of (times, magnitudes, lats, lons), all sorted by time, same order.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    aegean = cat[(cat.Latitude.between(lat0, lat1)) & (cat.Longitude.between(lon0, lon1)) &
                (cat.Magnitude >= min_magnitude) & cat.dt.notna()].sort_values("dt")
    if stations and max_dist_km:
        aegean = aegean[station_distance_mask(aegean.Latitude.to_numpy(),
                                              aegean.Longitude.to_numpy(),
                                              stations, max_dist_km)]
    return (aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64),
           aegean.Latitude.to_numpy(dtype=np.float64), aegean.Longitude.to_numpy(dtype=np.float64))


def truncate_to_reliable_catalog_end(hour_index: pd.DatetimeIndex, raw: np.ndarray,
                                     major_times: np.ndarray, buffer_days: float = 0):
    """Drops hours past the point where the catalog can no longer reliably inform labels."""
    cutoff = major_times[-1] - np.timedelta64(int(buffer_days * 24), "h")
    n_keep = int((hour_index.to_numpy() <= cutoff).sum())
    if n_keep < len(hour_index):
        print(f"  [!] catalog's last event is {major_times[-1]} -- truncating archive from "
               f"{hour_index[-1]} to {hour_index[n_keep - 1]} ({len(hour_index) - n_keep} hours "
               f"dropped, buffer={buffer_days:.0f}d) to avoid right-censoring the forward-looking "
               f"label near the archive's end.")
    return hour_index[:n_keep], raw[:n_keep]


def count_events_in_window(hourly_index: pd.DatetimeIndex, times: np.ndarray,
                           window_days: float, forward: bool) -> np.ndarray:
    """Counts events in a trailing or leading window around each hour."""
    t = hourly_index.to_numpy()
    w = np.timedelta64(int(round(window_days * 24)), "h")
    if forward:
        return (np.searchsorted(times, t + w, side="right")
                - np.searchsorted(times, t, side="right")).astype(np.int64)
    return (np.searchsorted(times, t, side="right")
            - np.searchsorted(times, t - w, side="right")).astype(np.int64)


def label_hours_rate_change(hourly_index: pd.DatetimeIndex, rate_times: np.ndarray,
                            horizon_days: float, baseline_days: float = None):
    """Labels each hour with whether seismicity RATE will increase ("variant B").

    A different forecasting target from `label_hours`: instead of "does one
    rare M>=threshold event occur in the next horizon" (whose positive class,
    at M>=4.5, is driven by a handful of events per fold -- 4 in fold 1 --
    making the effective sample size far smaller than the hour count
    suggests), this asks "will the next window contain MORE events than the
    trailing window did".

    That is a rate/acceleration forecast, which is what ETAS-family models and
    CSEP evaluation actually target, and it uses a much lower magnitude
    threshold (typically M>=3.0), so the label is driven by ~10^3 events
    instead of ~10^1. It is also the quantity Convertito et al. 2024's
    beta-statistic measures -- but as the target itself rather than as a mask
    on a rare-event label (`label_hours_beta_precursor`), which is what made
    that earlier attempt fail.

    Note the natural baseline here is strongly ANTI-correlated: during an
    aftershock sequence a high trailing rate predicts a DECREASE (Omori
    decay). Score any model against `rate_persistence_auc`, not against 0.5.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        rate_times: Sorted array of event times defining the rate (e.g.
            M>=3.0 events -- a much lower threshold than the label-defining
            `major_times` used by `label_hours`).
        horizon_days: Length of the forward window being forecast.
        baseline_days: Length of the trailing comparison window. Defaults to
            `horizon_days` (a like-for-like comparison, so the label is a
            clean "up or down" with no window-length bias).

    Returns:
        Tuple of (labels, forward_counts, trailing_counts) -- labels is an
        int64 0/1 array (1 = rate increases), the counts are returned so
        callers can build the persistence floor and report diagnostics
        without recomputing them.
    """
    if baseline_days is None:
        baseline_days = horizon_days
    fwd = count_events_in_window(hourly_index, rate_times, horizon_days, forward=True)
    bwd = count_events_in_window(hourly_index, rate_times, baseline_days, forward=False)
    return (fwd > bwd).astype(np.int64), fwd, bwd


def label_hours(hourly_index: pd.DatetimeIndex, major_times: np.ndarray, horizon_days: float) -> np.ndarray:
    """Labels each hour with whether a qualifying event occurs within the horizon."""
    horizon = np.timedelta64(int(horizon_days), "D")
    t = hourly_index.to_numpy()
    labels = np.zeros(len(t), dtype=np.int64)
    for i, ti in enumerate(t):
        fut = major_times[(major_times > ti) & (major_times <= ti + horizon)]
        labels[i] = int(len(fut) > 0)
    return labels


def compute_beta_statistic(hourly_index: pd.DatetimeIndex, bg_times: np.ndarray,
                           recent_days: float = 7.0, baseline_days: float = 30.0) -> np.ndarray:
    """Backward-looking seismicity-rate-acceleration z-score (no leakage).

    The beta-statistic (Reasenberg & Simpson 1992; Matthews & Reasenberg) is
    the test Convertito et al. 2024 (Sci. Rep. 14:2964) use to identify
    precursor windows: compare the observed event count in a recent window
    against what a constant-rate Poisson process (calibrated on the
    immediately preceding baseline window) would predict.

        beta = (n_recent - E[n_recent]) / sqrt(E[n_recent])
        E[n_recent] = (n_baseline / baseline_days) * recent_days

    Large positive beta means the rate has significantly accelerated
    relative to the recent past; beta near/below 0 means no acceleration
    (or a slowdown). Uses the same lower-completeness-threshold background
    catalog as the other rate/regularity features (more data points than
    the rare M>=threshold events alone give).

    Args:
        hourly_index: Hour-start timestamps to score, one per sample.
        bg_times: Sorted array of lower-threshold "background" event times
            (see `load_aegean_events_with_magnitude`).
        recent_days: Length of the recent window being tested.
        baseline_days: Length of the preceding baseline window the recent
            rate is compared against.

    Returns:
        float64 array of beta values, same length as `hourly_index`; 0.0
        where the baseline window has no events (no rate to compare against).
    """
    t = hourly_index.to_numpy()
    recent_td = np.timedelta64(int(round(recent_days * 24)), "h")
    baseline_td = np.timedelta64(int(round(baseline_days * 24)), "h")
    beta = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        n_recent = np.sum((bg_times <= ti) & (bg_times > ti - recent_td))
        n_baseline = np.sum((bg_times <= ti - recent_td) & (bg_times > ti - recent_td - baseline_td))
        expected = (n_baseline / baseline_days) * recent_days
        beta[i] = (n_recent - expected) / np.sqrt(expected) if expected > 0 else 0.0
    return beta


def label_hours_beta_precursor(hourly_index: pd.DatetimeIndex, major_times: np.ndarray,
                               horizon_days: float, beta: np.ndarray,
                               beta_threshold: float = 1.645) -> np.ndarray:
    """Convertito et al. 2024-style precursor labeling.

    Narrows `label_hours`' "M>=threshold within horizon_days" positive
    class to hours that are ALSO showing a statistically significant
    seismicity-rate acceleration (see `compute_beta_statistic`) -- i.e.
    genuinely accelerating toward the event, not merely close to it in
    calendar time. Hours within the horizon of a qualifying event but
    without significant acceleration revert to label 0.

    `beta_threshold=1.645` is a one-sided z-critical value at alpha=0.05 --
    our own choice, standing in for the paper's per-sequence empirically
    tuned threshold (the paper doesn't give one universal number).

    Args:
        hourly_index: Hour-start timestamps to label, one per sample.
        major_times: Sorted array of qualifying event times.
        horizon_days: Forecast horizon in days (same semantics as `label_hours`).
        beta: Per-hour beta-statistic array from `compute_beta_statistic`.
        beta_threshold: Minimum beta to count as "significant acceleration".

    Returns:
        int64 array of 0/1 labels, same length as `hourly_index`.
    """
    base = label_hours(hourly_index, major_times, horizon_days)
    return (base & (beta > beta_threshold)).astype(np.int64)


def days_since_prev_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days elapsed since the previous qualifying event, per hour."""
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        prev = major_times[major_times < ti]
        if len(prev):
            out[i] = (ti - prev[-1]) / np.timedelta64(1, "D")
    return out


def days_until_next_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days until the next qualifying event, per hour.

    Symmetric counterpart to `days_since_prev_major`. Used by
    `cnn_proximity_classify.py` to build a "is this hour close to a
    qualifying event, looking either direction in time" label -- a regime
    classification (was there an event nearby) rather than a forecast (will
    one occur), so looking backward is fair game, not leakage: the label
    isn't claiming to predict anything unknown at the time.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        major_times: Sorted array of qualifying event times (see
            `load_aegean_events`).

    Returns:
        float64 array, same length as `hourly_index`; NaN where no future
        qualifying event exists (e.g. the last event in the catalog).
    """
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        nxt = major_times[major_times >= ti]
        if len(nxt):
            out[i] = (nxt[0] - ti) / np.timedelta64(1, "D")
    return out
