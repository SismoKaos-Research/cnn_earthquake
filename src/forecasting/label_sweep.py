"""Which (magnitude, radius, horizon) label is worth modelling at all?

Catalogue-only, no model and no features needed, so it runs in seconds and
belongs *before* any training run. For every cell it reports the three numbers
that decide it:

  events      distinct qualifying events in the span. This -- not the window
              count -- is the effective sample size: consecutive 50 s windows
              under a multi-hour horizon carry near-identical labels, so
              311,041 rows carry roughly `events` independent observations.
  pos_rate    fraction of windows labelled positive. Near 0 or 1 means the
              label is saturated and nothing can discriminate on it.
  floor       the *orientation-corrected* persistence AUC, max(a, 1-a). A
              persistence rule scoring below 0.5 ranks inversely and is just
              as exploitable, so the bar it sets is 1-a.

Chosen primary cell (2026-08-21): M>=2.5, 400 km, 6 h -- 232 events, 25%
positive, floor 0.543, detectable edge +/-0.064 at 95%. Picked by statistical
power, because at 181 days nothing with a tighter radius can detect anything:
the physically preferable M>=3.0 / 100 km cell has 19 events and would need
+0.229 to register. Declare secondaries in advance; 24 of 140 cells pass the
viability filter, so sweeping all of them and reporting the best would
manufacture a +0.10 out of noise.

Usage:
    python3 src/forecasting/label_sweep.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from seismolib.catalog import haversine_km

# Paths are arguments, not constants: this has to run on whichever machine the
# work currently lives on.
DEFAULT_CATALOG = ("~/Projects/Sismokaos/seismic_cli/catalogs/"
                   "catalog_current.csv")

MAGNITUDES = (2.5, 3.0, 3.5, 4.0, 4.5)
RADII_KM = (50, 100, 200, 400)
HORIZONS_H = (1, 6, 24, 72, 168, 336, 720)

# A cell is only worth modelling if it has events, is not saturated, and leaves
# headroom over the persistence floor.
MIN_EVENTS = 15
POS_RATE_RANGE = (0.05, 0.85)
MAX_FLOOR = 0.90


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", default=DEFAULT_CATALOG,
                    help="KOERI catalogue CSV (Date,Longitude,Latitude,...,Magnitude)")
    ap.add_argument("--lat", type=float, default=37.0622, help="station latitude (BODT)")
    ap.add_argument("--lon", type=float, default=27.3103, help="station longitude (BODT)")
    ap.add_argument("--start", default="2024-05-01", help="span start (inclusive)")
    ap.add_argument("--end", default="2024-10-28", help="span end (inclusive)")
    ap.add_argument("--step-sec", type=float, default=50.0,
                    help="seconds between window ends; must match the extraction config")
    ap.add_argument("--out-csv", default=None, help="optional path for the full grid")
    return ap.parse_args()


def auc(y, s):
    """Rank-based ROC-AUC, or NaN on a single-class `y`.

    Deliberately not `seismolib.metrics.safe_auc`: this is called 140 times
    over ~300k windows, and the rank formula avoids sklearn's per-call
    validation overhead. The two agree to floating-point equality, which
    `tests/test_label_sweep.py` asserts.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sweep_cell(event_times, window_times, horizon_seconds):
    """Scores one (magnitude, radius, horizon) cell.

    Args:
        event_times: Sorted int64 epoch seconds of qualifying events.
        window_times: Int64 epoch seconds of window ends, ascending.
        horizon_seconds: Forecast horizon.

    Returns:
        Dict of events / pos_rate / pers_auc / floor / headroom for the cell.

    The forward label is "any qualifying event in (t, t + horizon]" and the
    persistence predictor is its backward mirror, "was there one in the
    trailing horizon". The floor is `max(auc, 1 - auc)` because the sign of a
    baseline is free: an anti-correlated predictor is exactly as exploitable
    as a correlated one, so 0.46 is a floor of 0.54, not of 0.46.
    """
    et, wt, hz = event_times, window_times, horizon_seconds
    if len(et) == 0:
        # Tight (magnitude, radius) cells can be empty over a short span, and
        # an empty cell is a legitimate grid result -- it fails the viability
        # filter -- not a reason to abort the sweep.
        return dict(events=0, pos_rate=0.0, pers_auc=np.nan, floor=np.nan,
                    headroom=np.nan)
    nxt = np.searchsorted(et, wt, "right")
    fwd = (nxt < len(et)) & (et[np.minimum(nxt, len(et) - 1)] <= wt + hz)
    prv = np.searchsorted(et, wt, "right") - 1
    back = (prv >= 0) & (et[np.maximum(prv, 0)] >= wt - hz)
    n_ev = int(((et > wt[0]) & (et <= wt[-1] + hz)).sum())
    a = auc(fwd.astype(int), back.astype(float))
    floor = max(a, 1 - a) if a == a else np.nan
    return dict(events=n_ev, pos_rate=fwd.mean(), pers_auc=a, floor=floor,
                headroom=1 - floor if floor == floor else np.nan)


def sweep(cat, window_times, magnitudes=MAGNITUDES, radii_km=RADII_KM,
          horizons_h=HORIZONS_H):
    """Runs `sweep_cell` over the full grid.

    Args:
        cat: Catalogue frame with `t`, `Magnitude` and `dist_km` columns.
        window_times: Int64 epoch seconds of window ends, ascending.

    Returns:
        DataFrame, one row per cell.
    """
    rows = []
    for mmin in magnitudes:
        for rad in radii_km:
            sel = cat[(cat.Magnitude >= mmin) & (cat.dist_km <= rad)]
            et = np.sort(sel.t.values.astype("datetime64[s]").astype(np.int64))
            for hz_h in horizons_h:
                rows.append(dict(Mmin=mmin, radius=rad, horizon_h=hz_h,
                                 **sweep_cell(et, window_times, hz_h * 3600)))
    return pd.DataFrame(rows)


def worth_modelling(df):
    """The subset of cells that pass the viability filter."""
    return df[(df.events >= MIN_EVENTS)
              & (df.pos_rate.between(*POS_RATE_RANGE))
              & (df.floor < MAX_FLOOR)]


def load_catalog(path, lat, lon):
    """Reads the KOERI catalogue and adds `t` and `dist_km` columns."""
    cat = pd.read_csv(os.path.expanduser(path), encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t", "Magnitude", "Latitude", "Longitude"])
    cat["dist_km"] = haversine_km(lat, lon, cat.Latitude.values, cat.Longitude.values)
    return cat


def main():
    args = parse_args()
    t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
    cat = load_catalog(args.catalog, args.lat, args.lon)
    print(f"catalogue {len(cat):,} events   span {cat.t.min().date()} .. {cat.t.max().date()}")

    # Completeness: Mc ~ the mode of the magnitude histogram in-region.
    reg = cat[(cat.dist_km <= 200) & (cat.t >= t0) & (cat.t <= t1)]
    h, edges = np.histogram(reg.Magnitude, bins=np.arange(0, 7.05, 0.1))
    print(f"in 200 km / batch span: {len(reg):,} events, modal magnitude "
          f"(Mc estimate) ~ {edges[h.argmax()]:.1f}")

    wins = pd.date_range(t0, t1, freq=f"{int(args.step_sec)}s")
    wt = wins.values.astype("datetime64[s]").astype(np.int64)
    print(f"windows: {len(wt):,}\n")

    df = sweep(cat, wt)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}\n")

    ok = worth_modelling(df)
    print(f"=== cells worth modelling (>={MIN_EVENTS} events, "
          f"{POS_RATE_RANGE[0]:.0%}-{POS_RATE_RANGE[1]:.0%} positive, "
          f"floor < {MAX_FLOOR:.2f}) ===")
    print(ok.sort_values("headroom", ascending=False)
            .head(18).to_string(index=False,
            formatters={"pos_rate": "{:.3f}".format, "pers_auc": "{:.4f}".format,
                        "floor": "{:.4f}".format, "headroom": "{:.4f}".format}))
    print(f"\n{len(ok)}/{len(df)} cells pass.")


if __name__ == "__main__":
    main()
