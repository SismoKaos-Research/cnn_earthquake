"""Tying scored windows to the catalogue.

Predicted arrivals at one station, the measured SNR that says whether the
station recorded an event at all, and the guard windows a catalogued event is
allowed to excuse.
"""
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from seismolib.arrivals import P_PHASES, S_PHASES, ArrivalTimes
from seismolib.catalog import haversine_km as haversine


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
