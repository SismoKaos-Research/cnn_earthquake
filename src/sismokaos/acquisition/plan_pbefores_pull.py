"""Plan a download campaign for magnitude estimation that beats the S wave.

The magnitude corpus caps at 56 km because it was requested that way. At 56 km
S-P is under 7 s, so no window longer than about 6 s can report before S reaches
the station -- the corpus cannot support a "magnitude before S" claim at any
window length, regardless of how the model is trained.

This plans the pull that can. Given a window length, the requirement is

    S - P  >  window_seconds - pre_seconds        (the post-arrival part)
    S - P  ~  distance / 8.18                     (Vp 6.0, Vs 3.46 km/s)

so a 10 s window with a 2 s pre-buffer needs epicentral distance beyond ~65 km,
and a 20 s window beyond ~147 km. The planner selects (station, event) pairs
satisfying that, and writes a ledger `afad_campaign.py` executes unchanged.

Three things it does that a naive "request every event" plan does not:

**Requests are grouped by station-day.** TDVMS bills one request per (station,
time range), not per event, so several events at one station on one day cost one
request. On a clustered catalogue this is a large saving and it is why the
request count is far below the event count.

**Stations are held disjoint from a reference manifest.** Pass the training
corpus's manifest and every station it used is excluded, so the result is a
station-disjoint test set by construction rather than by hoping the overlap is
small. Without this, 77% of one stage's test stations turned out to be the
other's training stations -- the same trap `--split-by detector` exists to fix.

**Magnitude is balanced by capping each band.** Left uncapped, any catalogue
selection is dominated by small events and reproduces the imbalance the pull is
meant to fix.

    sk plan-pull \\
        --catalog catalogs/catalog_current.csv \\
        --stations-csv catalogs/istasyon_katalog.csv \\
        --window-seconds 10 --min-magnitude 3.0 \\
        --exclude-manifest .../dataset_magreg_catalog_6s/manifest.csv \\
        --per-band 400 --out-ledger pbs10_ledger.jsonl
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from sismokaos.catalog import haversine_km as haversine

EARTH_KM = 6371.0
# S-P slowness for crustal Vp=6.0, Vs=3.46 km/s: (1/Vs - 1/Vp) s/km.
SP_PER_KM = 1.0 / 3.46 - 1.0 / 6.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--network", default="TU")
    p.add_argument("--window-seconds", type=float, default=10.0)
    p.add_argument("--pre-seconds", type=float, default=2.0)
    p.add_argument("--min-magnitude", type=float, default=3.0)
    p.add_argument("--max-distance", type=float, default=400.0,
                   help="beyond this the event is too faint to be worth a request")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2026-08-10")
    p.add_argument("--per-band", type=int, default=400,
                   help="cap per 0.5-magnitude band, so the plan is not swamped "
                        "by small events")
    p.add_argument("--max-stations-per-event", type=int, default=3)
    p.add_argument("--exclude-manifest", default=None,
                   help="manifest.csv whose station_key values must NOT appear")
    p.add_argument("--pad-minutes", type=float, default=5.0,
                   help="padding either side of the event when a day is not "
                        "already being requested")
    p.add_argument("--out-ledger", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()




def main():
    """Selects (station, event) pairs that can beat S, and writes the ledger."""
    args = parse_args()
    post = args.window_seconds - args.pre_seconds
    min_dist = post / SP_PER_KM
    print(f"[plan] {args.window_seconds:g}s window, {args.pre_seconds:g}s pre "
          f"-> needs S-P > {post:g}s -> distance > {min_dist:.0f} km")

    st = pd.read_csv(args.stations_csv, encoding="utf-8-sig")
    st.columns = [c.strip() for c in st.columns]
    st = st[st.Network == args.network]
    if args.exclude_manifest:
        man = pd.read_csv(args.exclude_manifest)
        used = {k.split(".")[-1] for k in man.station_key.astype(str).unique()}
        before = len(st)
        st = st[~st.Code.isin(used)]
        print(f"[plan] {before - len(st)} station(s) excluded as already used "
              f"in training; {len(st)} remain")

    cat = pd.read_csv(args.catalog, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat = cat[(cat.t >= args.start) & (cat.t < args.end)
              & (cat.Magnitude >= args.min_magnitude)]
    print(f"[plan] {len(cat):,} catalogued events in range, M>={args.min_magnitude:g}")

    # Cap per magnitude band before pairing, so the expensive per-station
    # distance work is only done for events that can survive the cap.
    cat["band"] = (cat.Magnitude * 2).astype(int) / 2.0
    keep = []
    for b, g in cat.groupby("band"):
        keep.append(g.sample(min(len(g), args.per_band), random_state=42))
    cat = pd.concat(keep).sort_values("t")
    print(f"[plan] {len(cat):,} after capping each 0.5-band at {args.per_band}")

    slat = st.Latitude.to_numpy(dtype=float)
    slon = st.Longitude.to_numpy(dtype=float)
    codes = st.Code.to_numpy()

    pairs = []
    for ev in cat.itertuples():
        d = haversine(ev.Latitude, ev.Longitude, slat, slon)
        ok = np.flatnonzero((d >= min_dist) & (d <= args.max_distance))
        if not len(ok):
            continue
        # nearest qualifying stations: closest to the distance floor means the
        # strongest signal that still clears the S-wave constraint
        for i in ok[np.argsort(d[ok])][:args.max_stations_per_event]:
            pairs.append((codes[i], ev.t, float(ev.Magnitude), float(d[i]),
                          int(ev.EventID)))
    print(f"[plan] {len(pairs):,} (station, event) pairs qualify")
    if not pairs:
        sys.exit("nothing qualifies -- loosen --max-distance or --per-band")

    pr = pd.DataFrame(pairs, columns=["station", "t", "mag", "dist", "event_id"])
    pr["day"] = pr.t.dt.floor("D")

    # One request per station-day: TDVMS charges per (station, time range), so
    # events sharing a day at a station are free after the first.
    grouped = pr.groupby(["station", "day"])
    rows = []
    for (station, day), g in grouped:
        rows.append({"station": str(station),
                     "start": day.strftime("%Y-%m-%dT00:00:00"),
                     "end": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00"),
                     "state": "pending",
                     "note": f"{len(g)} event(s), M {g.mag.min():.1f}-{g.mag.max():.1f}, "
                             f"{g.dist.min():.0f}-{g.dist.max():.0f} km"})
    rows.sort(key=lambda r: (r["station"], r["start"]))

    print(f"\n[plan] {len(rows):,} station-day requests "
          f"({len(pr):,} pairs / {len(rows):,} = {len(pr) / len(rows):.1f} events per request)")
    print(f"       at 6 parallel slots and ~2 min per single-day request: "
          f"~{len(rows) / 6 * 2 / 60:.1f} h of queue")
    print(f"       ~{len(rows) * 36 / 1024:.1f} GB at ~36 MB per station-day")
    print(f"\n  {'band':>8}{'pairs':>9}{'requests':>10}")
    pr["b"] = (pr.mag * 2).astype(int) / 2.0
    for b, g in pr.groupby("b"):
        print(f"  {b:>8.1f}{len(g):>9,}{g.groupby(['station', 'day']).ngroups:>10,}")
    print(f"\n  stations used: {pr.station.nunique()}")

    if args.dry_run:
        print("\n[plan] --dry-run: no ledger written")
        return
    with open(args.out_ledger, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n[plan] wrote {args.out_ledger}  "
          f"(run it with `afad_campaign.py --ledger {args.out_ledger} next --email ...`)")


if __name__ == "__main__":
    main()
