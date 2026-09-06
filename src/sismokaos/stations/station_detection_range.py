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

    sk station-range \
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

from sismokaos.arrivals import ArrivalTimes
from sismokaos.catalog import haversine_km as haversine

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

    taup = ArrivalTimes(grid_km=5.0)

    def p_travel(dist_km, depth_km):
        return taup.travel(dist_km, depth_km)

    rows = []
    sliced_away, last_slice_error = 0, None
    for z in sorted(glob.glob(args.zips)):
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            f = next(p for p in pathlib.Path(tmp).rglob("*") if p.is_file())
            stream = read(str(f))
        stream = stream.select(channel="HHZ")
        if not len(stream):
            continue
        # merge(fill_value=None) leaves masked arrays, which detrend and filter
        # refuse. split() turns those back into contiguous unmasked segments --
        # and keeping the segmentation is what makes the gap test below exact,
        # because a window spanning a gap then simply fails to return one piece.
        stream.merge(method=1, fill_value=None)
        stream = stream.split()
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
            except Exception as e:
                # Counted, not merely skipped. This table's row count is the
                # denominator of every "% of events reach SNR 3" figure in the
                # reports, so events silently vanishing here would shift those
                # percentages with nothing to show for it.
                sliced_away += 1
                last_slice_error = f"{type(e).__name__}: {e}"
                continue
            # exactly one segment each, at full length: anything else means the
            # window straddles a gap or an edge, and says nothing about the
            # station's reach
            if len(noise) != 1 or len(sig) != 1:
                continue
            sr = sig[0].stats.sampling_rate
            want_n = (NOISE_WIN[1] - NOISE_WIN[0]) * sr
            want_s = (SIGNAL_WIN[1] - SIGNAL_WIN[0]) * sr
            nd = np.asarray(noise[0].data, dtype=float)
            sd = np.asarray(sig[0].data, dtype=float)
            if nd.size < 0.98 * want_n or sd.size < 0.98 * want_s:
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
    if sliced_away:
        # This file's row count is the denominator of every "% of events reach
        # SNR 3" figure in the reports, so events dropped here move those
        # percentages. Saying how many, and why, is the difference between a
        # measurement and a number.
        print(f"[warn] {sliced_away:,} event(s) dropped by a slice error and are "
              f"NOT in the table; last was {last_slice_error}")
    if len(df):
        print(f"median SNR {df.snr.median():.2f}   "
              f"fraction SNR>=3: {100*(df.snr>=3).mean():.1f}%")


if __name__ == "__main__":
    main()
