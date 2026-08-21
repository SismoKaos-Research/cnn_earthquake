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
import numpy as np, pandas as pd

# Paths are arguments, not constants: this has to run on whichever machine the
# work currently lives on.
DEFAULT_CATALOG = ("~/Projects/Sismokaos/data_downloader/catalogs/"
                   "deprem_katalog_utc.csv")

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
args = ap.parse_args()

import os
C = os.path.expanduser(args.catalog)
BLAT, BLON = args.lat, args.lon
T0, T1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
STEP = args.step_sec

cat = pd.read_csv(C, encoding="utf-8-sig")
cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
cat = cat.dropna(subset=["t", "Magnitude", "Latitude", "Longitude"])

# great-circle distance from BODT
la1, lo1 = np.radians(BLAT), np.radians(BLON)
la2, lo2 = np.radians(cat.Latitude.values), np.radians(cat.Longitude.values)
cat["dist_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(
    np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))

print(f"catalogue {len(cat):,} events   span {cat.t.min().date()} .. {cat.t.max().date()}")

# Completeness: Mc ~ the mode of the magnitude histogram in-region.
reg = cat[(cat.dist_km <= 200) & (cat.t >= T0) & (cat.t <= T1)]
h, edges = np.histogram(reg.Magnitude, bins=np.arange(0, 7.05, 0.1))
print(f"in 200 km / batch span: {len(reg):,} events, modal magnitude "
      f"(Mc estimate) ~ {edges[h.argmax()]:.1f}")

wins = pd.date_range(T0, T1, freq=f"{int(STEP)}s")
wt = wins.values.astype("datetime64[s]").astype(np.int64)
print(f"windows: {len(wt):,}\n")

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

rows = []
for mmin in (2.5, 3.0, 3.5, 4.0, 4.5):
    for rad in (50, 100, 200, 400):
        sel = cat[(cat.Magnitude >= mmin) & (cat.dist_km <= rad)]
        et = np.sort(sel.t.values.astype("datetime64[s]").astype(np.int64))
        for hz_h in (1, 6, 24, 72, 168, 336, 720):
            hz = hz_h * 3600
            # forward label: any qualifying event in (t, t + horizon]
            nxt = np.searchsorted(et, wt, "right")
            fwd = (nxt < len(et)) & (et[np.minimum(nxt, len(et) - 1)] <= wt + hz)
            # persistence: was there one in the trailing horizon?
            prv = np.searchsorted(et, wt, "right") - 1
            back = (prv >= 0) & (et[np.maximum(prv, 0)] >= wt - hz)
            n_ev = int(((et > wt[0]) & (et <= wt[-1] + hz)).sum())
            a = auc(fwd.astype(int), back.astype(float))
            floor = max(a, 1 - a) if a == a else np.nan
            rows.append(dict(Mmin=mmin, radius=rad, horizon_h=hz_h, events=n_ev,
                             pos_rate=fwd.mean(), pers_auc=a, floor=floor,
                             headroom=1 - floor if floor == floor else np.nan))
df = pd.DataFrame(rows)
if args.out_csv:
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}\n")

# A cell is only worth modelling if it has events, isn't saturated, and leaves headroom.
ok = df[(df.events >= 15) & (df.pos_rate.between(0.05, 0.85)) & (df.floor < 0.90)]
print("=== cells worth modelling (>=15 events, 5-85% positive, floor < 0.90) ===")
print(ok.sort_values("headroom", ascending=False)
        .head(18).to_string(index=False,
        formatters={"pos_rate": "{:.3f}".format, "pers_auc": "{:.4f}".format,
                    "floor": "{:.4f}".format, "headroom": "{:.4f}".format}))
print(f"\n{len(ok)}/{len(df)} cells pass.")
