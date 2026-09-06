"""Plan and fetch a magnitude-regression corpus from KOERI's FDSN service.

The existing corpus has two defects that no amount of training fixes: 1.2% of
its windows are above M4, and nothing in it is further than 56 km, so it cannot
support a magnitude estimate that beats the S wave at any window length.

FDSN removes the cost argument for living with either. Measured against TDVMS:
a 60 s three-component window arrives in ~1 s under `get_waveforms_bulk` versus
~2 minutes for a whole station-day through the email queue, and the same plan
costs ~1.8 GB instead of ~206 GB. So this pulls generously and filters later
rather than deciding everything up front.

Three design choices follow from that, and each is the opposite of what the
TDVMS planner had to do:

**One 60 s window per (station, event), anchored at P-2 s.** Any window length
up to 58 s post-arrival can be cut from it afterwards, so window length becomes
a post-hoc variable rather than something fixed at download time. The TDVMS plan
had to commit to a length.

**Distance-stratified station selection.** For each event, near stations
(< `--split-km`) and far stations (>= it) are drawn separately, so the corpus
spans the attenuation range instead of collapsing onto whichever stations happen
to be closest. Far windows are the ones that can report before S: S-P is roughly
distance/8.18 s, so beating a 10 s window needs 65 km and a 20 s window 147 km.

**Station availability is checked per event.** A station installed in 2020
cannot record a 2015 event; the inventory's start/end dates are honoured rather
than assumed, which otherwise produces a plan full of requests that can only
return nothing.

Magnitude bands are still capped, because any uncapped catalogue selection is
dominated by small events and would reproduce the imbalance this exists to fix.

    sk fdsn plan \\
        --catalog catalogs/catalog_current.csv --per-band 800 \\
        --min-magnitude 3.0 --out requests.csv
    sk fdsn fetch \\
        --requests requests.csv --out-dir raw/fdsn_windows
"""
import argparse
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from sismokaos.arrivals import ArrivalTimes
from sismokaos.catalog import haversine_km as haversine

warnings.filterwarnings("ignore")

EARTH_KM = 6371.0
KOERI = "http://eida.koeri.boun.edu.tr"
# S-P slowness for crustal Vp=6.0, Vs=3.46 km/s, in s/km.
SP_PER_KM = 1.0 / 3.46 - 1.0 / 6.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("plan", help="choose (station, event) pairs and write a request list")
    q.add_argument("--catalog", required=True)
    q.add_argument("--network", default="KO")
    q.add_argument("--url", default=KOERI)
    q.add_argument("--start", default="2012-01-01")
    q.add_argument("--end", default="2026-08-01")
    q.add_argument("--min-magnitude", type=float, default=3.0)
    q.add_argument("--per-band", type=int, default=800,
                   help="cap per 0.5-magnitude band; bands with fewer are taken whole")
    q.add_argument("--split-km", type=float, default=65.0,
                   help="near/far boundary. 65 km is where S-P reaches 8 s, the "
                        "point a 10 s window can report before S.")
    q.add_argument("--near-per-event", type=int, default=3)
    q.add_argument("--far-per-event", type=int, default=3)
    q.add_argument("--rare-magnitude", type=float, default=4.5,
                   help="at or above this, take more stations per event. Above "
                        "~M4.5 the catalogue is exhausted -- no per-band cap "
                        "binds -- so extra stations are the ONLY way to get more "
                        "samples in the bands that matter most.")
    q.add_argument("--rare-per-event", type=int, default=8,
                   help="near AND far stations each, for rare events")
    q.add_argument("--max-distance", type=float, default=400.0)
    q.add_argument("--window-seconds", type=float, default=60.0)
    q.add_argument("--pre-seconds", type=float, default=2.0)
    q.add_argument("--exclude-manifest", default=None,
                   help="hold these stations out, for a station-disjoint pull")
    q.add_argument("--out", required=True)

    f = sub.add_parser("fetch", help="download the planned windows")
    f.add_argument("--requests", required=True)
    f.add_argument("--url", default=KOERI)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--miss-log", default="fdsn_misses.txt",
                   help="rows the archive returned nothing for. Skipped on the "
                        "next run so a resume does not re-walk them.")
    f.add_argument("--retry-misses", action="store_true",
                   help="attempt the rows in --miss-log again, for when they "
                        "failed because the archive was down rather than empty")
    f.add_argument("--batch", type=int, default=40,
                   help="windows per bulk call. Larger is faster but a single "
                        "bad row fails the whole call, so this trades speed "
                        "against retry granularity.")
    f.add_argument("--sleep", type=float, default=0.5,
                   help="pause between bulk calls; this is a shared academic "
                        "service, not a CDN")
    f.add_argument("--limit", type=int, default=None)
    return p.parse_args()




def load_inventory(url, network, start, end):
    """Station code, position and operating span for every HH* station."""
    inv = Client(url, timeout=180).get_stations(
        network=network, level="station", channel="HH*",
        starttime=UTCDateTime(start), endtime=UTCDateTime(end))
    rows = []
    for n in inv:
        for s in n:
            rows.append({"station": s.code, "lat": float(s.latitude),
                         "lon": float(s.longitude),
                         "start": s.start_date.timestamp if s.start_date else -1e18,
                         "end": s.end_date.timestamp if s.end_date else 1e18})
    return pd.DataFrame(rows).drop_duplicates("station").reset_index(drop=True)


def cmd_plan(args):
    """Selects distance-stratified (station, event) pairs and writes the list."""
    inv = load_inventory(args.url, args.network, args.start, args.end)
    print(f"[plan] {len(inv)} {args.network} stations with HH*")
    if args.exclude_manifest:
        man = pd.read_csv(args.exclude_manifest)
        used = {k.split(".")[-1] for k in man.station_key.astype(str)}
        before = len(inv)
        inv = inv[~inv.station.isin(used)].reset_index(drop=True)
        print(f"[plan] {before - len(inv)} held out as already used; {len(inv)} remain")
        if inv.empty:
            sys.exit("no stations left after exclusion -- this network is already "
                     "fully used; a station-disjoint pull needs a different network")

    cat = pd.read_csv(args.catalog, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat = cat[(cat.t >= args.start) & (cat.t < args.end)
              & (cat.Magnitude >= args.min_magnitude)]
    cat["band"] = (cat.Magnitude * 2).astype(int) / 2.0
    kept = [g.sample(min(len(g), args.per_band), random_state=42)
            for _, g in cat.groupby("band")]
    cat = pd.concat(kept).sort_values("t").reset_index(drop=True)
    print(f"[plan] {len(cat):,} events after capping each 0.5-band at {args.per_band}")

    # 10 km grid, the resolution this planner was written with; see
    # sismokaos.arrivals for why it is stated rather than defaulted.
    taup = ArrivalTimes(grid_km=10.0)

    def p_travel(dist_km, depth_km):
        return taup.travel(dist_km, depth_km)

    slat, slon = inv.lat.to_numpy(), inv.lon.to_numpy()
    s_start, s_end = inv.start.to_numpy(), inv.end.to_numpy()
    codes = inv.station.to_numpy()

    out = []
    for ev in cat.itertuples():
        epoch = UTCDateTime(ev.t.to_pydatetime()).timestamp
        live = (s_start <= epoch) & (s_end >= epoch)
        d = haversine(ev.Latitude, ev.Longitude, slat, slon)
        ok = live & (d <= args.max_distance)
        near = np.flatnonzero(ok & (d < args.split_km))
        far = np.flatnonzero(ok & (d >= args.split_km))
        rare = ev.Magnitude >= args.rare_magnitude
        n_near = args.rare_per_event if rare else args.near_per_event
        n_far = args.rare_per_event if rare else args.far_per_event
        pick = list(near[np.argsort(d[near])][:n_near]) \
             + list(far[np.argsort(d[far])][:n_far])
        depth = float(ev.Depth) if pd.notna(ev.Depth) else 10.0
        for i in pick:
            tt = p_travel(d[i], depth)
            if tt is None:
                continue
            p_epoch = epoch + tt
            out.append({"event_id": int(ev.EventID), "station": codes[i],
                        "network": args.network, "magnitude": float(ev.Magnitude),
                        "distance_km": float(d[i]), "depth_km": depth,
                        "p_epoch": p_epoch,
                        "start": p_epoch - args.pre_seconds,
                        "end": p_epoch - args.pre_seconds + args.window_seconds})
    req = pd.DataFrame(out)
    if req.empty:
        sys.exit("no (station, event) pairs qualify")

    print(f"\n[plan] {len(req):,} windows across {req.station.nunique()} stations "
          f"and {req.event_id.nunique():,} events")
    print(f"       ~{len(req) * 1.0 / 3600:.1f} h at ~1 s per window (bulk), "
          f"~{len(req) * args.window_seconds * 3 * 100 * 4 / 1e9:.2f} GB")
    print(f"\n  {'band':>7}{'events':>9}{'windows':>9}{'near':>8}{'far':>8}"
          f"{'med dist':>10}")
    for b, g in req.groupby((req.magnitude * 2).astype(int) / 2.0):
        print(f"  {b:>7.1f}{g.event_id.nunique():>9,}{len(g):>9,}"
              f"{int((g.distance_km < args.split_km).sum()):>8,}"
              f"{int((g.distance_km >= args.split_km).sum()):>8,}"
              f"{g.distance_km.median():>10.0f}")
    far = req[req.distance_km >= args.split_km]
    print(f"\n  windows able to report before S at 10 s: {len(far):,} "
          f"({100 * len(far) / len(req):.0f}%)")
    print(f"  ... at 20 s (>=147 km): "
          f"{int((req.distance_km >= 147).sum()):,}")

    req.to_csv(args.out, index=False)
    print(f"\n[plan] wrote {args.out}")


def cmd_fetch(args):
    """Downloads the planned windows in bulk, one mseed per (event, station)."""
    req = pd.read_csv(args.requests)
    if args.limit:
        req = req.head(args.limit)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, timeout=300)

    # Rows the archive has already said it has nothing for. Without this the
    # todo list is rebuilt in CSV ORDER on every restart, so a resumed fetch
    # re-attempts every previous miss before reaching any new work: after one
    # interrupted run that was ~8,500 rows and 2.6 hours of requests that
    # cannot succeed. A miss is usually permanent -- KO.GELI in 2013 carries
    # BH? at 50 Hz and no HH? at all, so this plan's channel set can never
    # match it -- but --retry-misses exists for the case where the archive was
    # simply down when they were tried.
    misses = set()
    miss_log = pathlib.Path(args.miss_log) if args.miss_log else None
    if miss_log and miss_log.exists() and not args.retry_misses:
        misses = {ln.strip() for ln in miss_log.read_text().splitlines() if ln.strip()}

    todo, skipped = [], 0
    for r in req.itertuples():
        # Station in the PATH, not the name. seismic-cli parses
        # `^(?:noise_)?event_(.+?)_raw$` non-greedily, so `event_153534_TASB_raw`
        # yields the event id "153534_TASB", matches no catalogue row, and the
        # dataset build ends with "No labelled windows" after the whole encode.
        # Flat files written before this are still honoured on resume, so an
        # existing pull is not re-downloaded.
        dest = out / str(r.station) / f"event_{r.event_id}_raw.mseed"
        legacy = out / f"event_{r.event_id}_{r.station}_raw.mseed"
        if dest.exists() or legacy.exists():
            continue
        key = f"{r.station}/{dest.name}"
        if key in misses:
            skipped += 1
            continue
        todo.append((r, dest, key))
    print(f"[fetch] {len(todo):,} to fetch, "
          f"{len(req) - len(todo) - skipped:,} already present, "
          f"{skipped:,} known-empty skipped"
          + (" (--retry-misses to try them again)" if skipped else ""))
    miss_fh = open(miss_log, "a") if miss_log else None

    got = miss = 0
    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        bulk = [(r.network, r.station, "*", "HH*",
                 UTCDateTime(r.start), UTCDateTime(r.end)) for r, _, _ in chunk]
        try:
            st = client.get_waveforms_bulk(bulk)
        except Exception as e:
            # A bulk call fails whole; fall back per row so one bad window does
            # not cost the other thirty-nine.
            st = None
            if "No data" not in str(e):
                print(f"  bulk failed ({type(e).__name__}), falling back", flush=True)
        for r, dest, key in chunk:
            # Trim to THIS row's window. A bulk response is one Stream for the
            # whole batch, so selecting by station alone also picks up the same
            # station's traces from other rows -- which silently wrote 112 s of
            # mixed windows into files that asked for 60 s.
            sel = None
            if st is not None:
                sel = st.select(station=r.station).slice(
                    UTCDateTime(r.start), UTCDateTime(r.end))
            if sel is None or not len(sel):
                try:
                    sel = client.get_waveforms(r.network, r.station, "*", "HH*",
                                               UTCDateTime(r.start), UTCDateTime(r.end))
                except Exception:
                    miss += 1
                    if miss_fh:
                        miss_fh.write(key + "\n"); miss_fh.flush()
                    continue
            sel = sel.copy()
            sel.merge(method=1, fill_value=None)
            sel = sel.split()
            want = int(round((r.end - r.start) * 100)) - 1
            keep = []
            for comp in ("Z", "N", "E"):
                cand = [x for x in sel if x.stats.channel[-1].upper() == comp
                        and x.stats.npts >= want * 0.99]
                if cand:
                    keep.append(cand[0])
            if len(keep) < 3:
                miss += 1
                if miss_fh:
                    miss_fh.write(key + "\n"); miss_fh.flush()
                continue
            from obspy import Stream
            dest.parent.mkdir(parents=True, exist_ok=True)
            Stream(keep).write(str(dest), format="MSEED")
            got += 1
        done = i + len(chunk)
        if done % (args.batch * 10) == 0 or done >= len(todo):
            el = time.time() - t0
            print(f"  {done:,}/{len(todo):,}  got {got:,} missing {miss:,}  "
                  f"{el / max(done, 1):.2f}s/window  eta "
                  f"{(len(todo) - done) * el / max(done, 1) / 60:.0f} min", flush=True)
        time.sleep(args.sleep)

    print(f"\n[fetch] {got:,} written, {miss:,} unavailable -> {out}")


def main():
    args = parse_args()
    {"plan": cmd_plan, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
