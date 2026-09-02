"""Measures how far out a station actually recovers catalogued events.

Limitation 4 of the TÜBİTAK report says the low-SNR distant regime is absent
from the training corpus: waveforms were only ever requested within ~56 km of an
epicentre, so no statement about attenuation with distance could be made from
it. Continuous data removes that ceiling -- every catalogued event in the window
has a waveform at the station whether or not anyone picked it.

Method: for each catalogued event, predict the P arrival at the station with
iasp91, measure RMS in a signal window against a pre-arrival noise window, and
call the event recovered when that ratio clears a threshold. This measures the
*data*, not a model -- no detector is involved.

    python3 scripts/station_detection_range.py \
        --zips 'afad_raw/MANT/*.zip' --station MANT --stations-csv istasyon_katalog.csv \
        --catalog catalog_current.csv --out mant_range.csv
"""
import argparse
import glob
import pathlib
import tempfile
import zipfile

import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel

EARTH_KM = 6371.0
NOISE_WIN = (-60.0, -10.0)   # seconds relative to predicted P
SIGNAL_WIN = (-1.0, 12.0)    # a little before, to tolerate model error


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zips", required=True, help="glob of chunk archives")
    p.add_argument("--station", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--max-distance", type=float, default=500.0)
    p.add_argument("--min-magnitude", type=float, default=0.0)
    p.add_argument("--freqmin", type=float, default=2.0)
    p.add_argument("--freqmax", type=float, default=20.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def haversine(lat0, lon0, lat, lon):
    p1, p2 = np.radians(lat0), np.radians(lat)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    args = parse_args()
    st_tab = pd.read_csv(args.stations_csv, encoding="utf-8-sig")
    st_tab.columns = [c.strip() for c in st_tab.columns]
    s = st_tab[st_tab.Code == args.station].iloc[0]
    slat, slon = float(s.Latitude), float(s.Longitude)

    cat = pd.read_csv(args.catalog, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat["dist"] = haversine(slat, slon, cat.Latitude.values, cat.Longitude.values)
    cat = cat[(cat.dist <= args.max_distance) & (cat.Magnitude >= args.min_magnitude)]

    model = TauPyModel(model="iasp91")
    # Travel time depends mostly on distance and depth; caching on a coarse grid
    # avoids ~7,000 taup calls without materially changing the arrival estimate.
    tt_cache = {}

    def p_travel(dist_km, depth_km):
        key = (round(dist_km / 5.0), round(max(depth_km, 0.0) / 5.0))
        if key not in tt_cache:
            deg = key[0] * 5.0 / 111.195
            try:
                arr = model.get_travel_times(source_depth_in_km=key[1] * 5.0,
                                             distance_in_degree=deg,
                                             phase_list=["p", "P", "Pn", "Pg"])
                tt_cache[key] = arr[0].time if arr else None
            except Exception:
                tt_cache[key] = None
        return tt_cache[key]

    rows = []
    for z in sorted(glob.glob(args.zips)):
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            f = next(p for p in pathlib.Path(tmp).rglob("*") if p.is_file())
            stream = read(str(f))
        stream = stream.select(channel="HHZ")
        if not len(stream):
            continue
        stream.merge(method=1, fill_value=None)
        stream.detrend("demean")
        stream.filter("bandpass", freqmin=args.freqmin, freqmax=args.freqmax,
                      corners=4, zerophase=True)
        t0 = min(tr.stats.starttime for tr in stream)
        t1 = max(tr.stats.endtime for tr in stream)
        sub = cat[(cat.t >= pd.Timestamp(t0.datetime)) & (cat.t <= pd.Timestamp(t1.datetime))]
        print(f"{pathlib.Path(z).stem}: {len(sub)} catalogued events in span", flush=True)

        for ev in sub.itertuples():
            tt = p_travel(ev.dist, float(ev.Depth) if pd.notna(ev.Depth) else 10.0)
            if tt is None:
                continue
            p_time = UTCDateTime(ev.t.to_pydatetime()) + tt
            try:
                noise = stream.slice(p_time + NOISE_WIN[0], p_time + NOISE_WIN[1])
                sig = stream.slice(p_time + SIGNAL_WIN[0], p_time + SIGNAL_WIN[1])
            except Exception:
                continue
            if not len(noise) or not len(sig):
                continue
            nd = np.ma.getdata(noise[0].data); nm = np.ma.getmaskarray(noise[0].data)
            sd = np.ma.getdata(sig[0].data);   sm = np.ma.getmaskarray(sig[0].data)
            # a window that is partly gap tells us nothing about the station
            if nm.any() or sm.any() or nd.size < 100 or sd.size < 100:
                continue
            n_rms = float(np.sqrt(np.mean(nd.astype(float) ** 2)))
            s_rms = float(np.sqrt(np.mean(sd.astype(float) ** 2)))
            if n_rms <= 0:
                continue
            rows.append({"event_id": ev.EventID, "time": ev.t, "mag": ev.Magnitude,
                         "dist_km": ev.dist, "depth": ev.Depth,
                         "snr": s_rms / n_rms, "location": ev.Location})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} events measured -> {args.out}")
    if len(df):
        print(f"median SNR {df.snr.median():.2f}   "
              f"fraction SNR>=3: {100*(df.snr>=3).mean():.1f}%")


if __name__ == "__main__":
    main()
