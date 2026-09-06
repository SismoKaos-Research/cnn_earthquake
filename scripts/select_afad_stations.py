"""Greedy TU-station selection for the AFAD download campaign.

The ranking is *catalogue-sensitive*: run against `deprem_katalog_utc.csv` it
returns a different six stations than against AFAD's live catalogue, because
that file is missing 96% of the February 2025 Aegean swarm. Re-run this
whenever the catalogue changes and do not cache the answer.

    python3 scripts/select_afad_stations.py \
        --events ~/Projects/Sismokaos/seismic_cli/catalogs/catalog_current.csv \
        --stations ~/Projects/Sismokaos/seismic_cli/catalogs/istasyon_katalog.csv \
        --radius 100 --top 8

`--events` accepts either the local catalogue format (Date/Latitude/Longitude/
Magnitude) or a dump of AFAD's API (date/latitude/longitude/magnitude).
"""
import argparse

import numpy as np
import pandas as pd

EARTH_KM = 6371.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True, help="event catalogue CSV")
    p.add_argument("--stations", required=True, help="istasyon_katalog.csv")
    p.add_argument("--network", default="TU", help="network code to select within")
    p.add_argument("--radius", type=float, default=100.0, help="coverage radius, km")
    p.add_argument("--min-magnitude", type=float, default=2.5)
    p.add_argument("--start", default=None, help="ISO date, inclusive")
    p.add_argument("--end", default=None, help="ISO date, exclusive")
    p.add_argument("--top", type=int, default=8, help="stations to report")
    return p.parse_args()


def load_events(path, min_magnitude, start, end):
    """Normalises either catalogue layout to lat/lon/mag/t columns."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Latitude" in df.columns:
        out = pd.DataFrame({
            "lat": df.Latitude.astype(float),
            "lon": df.Longitude.astype(float),
            "mag": df.Magnitude.astype(float),
            "t": pd.to_datetime(df.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce"),
        })
    else:
        out = pd.DataFrame({
            "lat": df.latitude.astype(float),
            "lon": df.longitude.astype(float),
            "mag": df.magnitude.astype(float),
            "t": pd.to_datetime(df.date, errors="coerce"),
        })
    out = out.dropna(subset=["t"])
    out = out[out.mag >= min_magnitude]
    if start:
        out = out[out.t >= pd.Timestamp(start)]
    if end:
        out = out[out.t < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def coverage_matrix(stations, events, radius_km):
    """Boolean [station, event] — True where the event is within radius_km."""
    lat_s = np.radians(stations.lat.values)[:, None]
    lon_s = np.radians(stations.lon.values)[:, None]
    lat_e = np.radians(events.lat.values)[None, :]
    lon_e = np.radians(events.lon.values)[None, :]
    a = (np.sin((lat_e - lat_s) / 2) ** 2
         + np.cos(lat_s) * np.cos(lat_e) * np.sin((lon_e - lon_s) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a)) <= radius_km


def greedy(cov, stations, events, top):
    """Each pick maximises *newly* covered events, not total — otherwise
    neighbouring stations all score alike and the set is redundant."""
    covered = np.zeros(cov.shape[1], bool)
    picks = []
    for _ in range(top):
        gain = (cov & ~covered).sum(axis=1)
        i = int(np.argmax(gain))
        if gain[i] == 0:
            break
        covered |= cov[i]
        picks.append((stations.iloc[i], int(gain[i]), int(covered.sum())))
    return picks


def main():
    args = parse_args()
    events = load_events(args.events, args.min_magnitude, args.start, args.end)
    st = pd.read_csv(args.stations, encoding="utf-8-sig")
    st.columns = [c.strip() for c in st.columns]
    st = st[st.Network == args.network].copy()
    st["lat"] = st.Latitude.astype(float)
    st["lon"] = st.Longitude.astype(float)

    print(f"{len(st)} {args.network} stations, {len(events)} events "
          f"M>={args.min_magnitude}, radius {args.radius:.0f} km")
    cov = coverage_matrix(st, events, args.radius)
    print(f"\n{'#':>2s} {'code':7s} {'new':>6s} {'cum':>6s} {'cum%':>6s}  place")
    for k, (row, gain, cum) in enumerate(greedy(cov, st, events, args.top), 1):
        print(f"{k:2d} {row.Code:7s} {gain:6d} {cum:6d} {100*cum/len(events):5.1f}%  "
              f"{row.Province}")


if __name__ == "__main__":
    main()
