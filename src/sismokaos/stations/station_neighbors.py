"""Which stations sit within N km of one target station.

Every other distance in this repo is event-to-station. This is the
station-to-station one: the query you make to pick a coincidence partner, to
choose a second station for cross-station work, or to group a manifest's
stations by locality.

    sk station-neighbors --station MANT --radius 200 --min-km 20 \
        --stations catalogs/istasyon_katalog.csv \
        --station-coords ../data_downloader/catalogs/station_coords.csv

**Use `--min-km` when you are looking for a coincidence partner.** 189 pairs of
distinct station keys in these catalogues sit within 50 m of each other -- the
same physical site registered under two network codes, or two instruments in one
building. TU.MANT's nearest neighbour is TU.MANS, 8.7 m away; KO.SUTC and
TU.SUTC are 0 m apart. A pair like that is fully common-mode: it confirms
nothing, which is exactly the failure `continuous/coincidence.py` is written to
avoid. Nearest is very often not what you want.

**Both catalogues are read, and `station_coords.csv` wins where they overlap.**
Neither file is sufficient alone. `istasyon_katalog.csv` has 1,576 stations
(TK 832 / TU 494 / KO 184 / ...); `station_coords.csv` has 334 (KO 277 / 6G 49 /
IJ 8). They share 179 NET.STA keys and disagree on them -- KO.KIZT by 14.4 km,
KO.GMLD by 3.6 km -- and `manifest_distances.py` settles which is right:
`station_coords.csv` was checked against the KOERI FDSN station service and
agrees to 0.000 km on all 277 KO stations. So coords supplies the authoritative
position for the overlap, and istasyon supplies the ~1,400 TU/TK stations coords
does not carry at all.

**An ambiguous target is refused, not guessed at.** Station codes are not
unique. 15 bare codes resolve to two different places, and three collide even
with the network attached:

    TOKT   TU/KO     727.1 km   Balikesir vs Tokat
    KOCA   TU/TU     324.3 km   Denizli   vs Canakkale   <- one NET.STA key
    BALA   TU/GZ      22.8 km   Ankara    vs Ankara
    KULU   TU/KO      16.9 km   Konya     vs Konya

Picking the first row -- which is what a bare `df[df.Code == name].iloc[0]`
does -- answers a 727 km different question without saying so. This refuses and
lists the candidates instead, the way `manifest_distances.station_table()`
already refuses to guess at an ambiguous bare code.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from sismokaos.catalog import haversine_km

# Same key, same position to within ~1 m: one station listed twice, not two
# stations. Positions that differ by more than this are kept apart, because
# collapsing them is the guess this module exists to avoid.
SAME_SITE_DEGREES = 5

COLUMNS = ["network", "station", "key", "lat", "lon", "place", "src"]


def _normalise(df, network, station, lat, lon, place, src):
    """One catalogue's columns -> this module's."""
    out = pd.DataFrame({
        "network": df[network].astype(str).str.strip(),
        "station": df[station].astype(str).str.strip(),
        "lat": df[lat].astype(float),
        "lon": df[lon].astype(float),
        "place": (df[place].astype(str).str.strip() if place else ""),
        "src": src,
    })
    out["key"] = out.network + "." + out.station
    return out[COLUMNS]


def read_station_csv(path):
    """Reads either catalogue layout, deciding by header rather than by flag.

    The two files in this project do not share a single column name:
    `Network,Code,Longitude,Latitude,Height,Province,District` (utf-8-sig) for
    istasyon_katalog.csv, `network,station,latitude,longitude,elevation` for
    station_coords.csv. Callers should not have to know which they hold.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if {"Network", "Code", "Latitude", "Longitude"} <= set(df.columns):
        place = "Province" if "Province" in df.columns else None
        return _normalise(df, "Network", "Code", "Latitude", "Longitude", place, "istasyon")
    if {"network", "station", "latitude", "longitude"} <= set(df.columns):
        return _normalise(df, "network", "station", "latitude", "longitude", None, "coords")
    raise ValueError(
        f"{path}: unrecognised station table. Expected either "
        f"Network/Code/Latitude/Longitude or network/station/latitude/longitude, "
        f"got {list(df.columns)}")


def load_station_table(*paths):
    """Merges station tables, with EARLIER paths winning on a shared NET.STA.

    Callers pass the authority first. `main` passes `--station-coords` before
    `--stations` for the reason in the module docstring.

    Returns:
        DataFrame with one row per known station position. A key may appear
        more than once when a catalogue genuinely holds two positions for it;
        `resolve` is what refuses to pick between them.
    """
    tables = [read_station_csv(p) for p in paths if p]
    if not tables:
        raise ValueError("no station table given")

    out, seen = [], set()
    for t in tables:
        out.append(t[~t.key.isin(seen)])
        seen |= set(t.key)
    tab = pd.concat(out, ignore_index=True)

    # Losing a table's position does not mean losing its Province. coords has
    # no place column at all, so without this every KO station it overrides
    # would print blank where istasyon knew the answer.
    named = pd.concat([t[t.place.astype(str).str.len() > 0] for t in tables],
                      ignore_index=True).drop_duplicates(subset="key")
    fill = tab.key.map(named.set_index("key").place)
    tab["place"] = tab.place.where(tab.place.astype(str).str.len() > 0,
                                   fill).fillna("")

    # A row repeated verbatim is one station listed twice. Round before
    # de-duplicating so a position differing in the last decimal place does not
    # read as a second site -- but keep genuinely different positions apart, so
    # `resolve` can see the conflict and refuse.
    site = pd.DataFrame({"key": tab.key,
                         "lat": tab.lat.round(SAME_SITE_DEGREES),
                         "lon": tab.lon.round(SAME_SITE_DEGREES)})
    return tab[~site.duplicated()].reset_index(drop=True)


def resolve(tab, name, network=None, near=None):
    """Rows matching `name`, which may be 'NET.STA' or a bare 'STA'.

    Returns every candidate rather than choosing, so the caller can refuse. A
    bare code that two networks use is genuinely ambiguous and must not be
    resolved by row order.

    `near` is the escape hatch for the three keys -- TU.ERCT, TU.ERZM, TU.KOCA
    -- that collide even with the network attached, where naming the network
    cannot help. Given (lat, lon) it keeps the single nearest candidate, which
    is the one question that always separates two positions.
    """
    name = str(name).strip()
    hit = tab[tab.key == name] if "." in name else tab[tab.station == name]
    if network:
        hit = hit[hit.network == str(network).strip()]
    if near is not None and len(hit) > 1:
        d = haversine_km(float(near[0]), float(near[1]), hit.lat.values, hit.lon.values)
        hit = hit.iloc[[int(np.argmin(d))]]
    return hit


def neighbors(tab, lat, lon, radius_km, min_km=0.0, k=None, exclude_key=None):
    """Stations within [min_km, radius_km] of one position, nearest first.

    Both bounds are inclusive, matching `select_afad_stations.coverage_matrix`.
    `min_km` is the annulus that keeps a co-located twin out of the answer.
    """
    out = tab.copy()
    out["km"] = haversine_km(float(lat), float(lon), out.lat.values, out.lon.values)
    out = out[(out.km <= radius_km) & (out.km >= min_km)]
    if exclude_key is not None:
        out = out[out.key != exclude_key]
    out = out.sort_values(["km", "key"], kind="mergesort")
    # Truncate after sorting: `--top 5` means the five nearest, not five
    # arbitrary rows that then get ordered.
    return out.head(k).reset_index(drop=True) if k else out.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", required=True,
                   help="target, as NET.STA or a bare code. A bare code that "
                        "two networks use is refused, not guessed at.")
    p.add_argument("--radius", type=float, required=True, help="km, inclusive")
    p.add_argument("--stations", default=None, help="istasyon_katalog.csv")
    p.add_argument("--station-coords", default=None,
                   help="station_coords.csv. Authoritative where the two files "
                        "disagree; see this module's docstring.")
    p.add_argument("--min-km", type=float, default=0.0,
                   help="exclude anything closer than this. Use it when picking "
                        "a coincidence partner: the nearest station is often the "
                        "same site under another code, which confirms nothing.")
    p.add_argument("--top", type=int, default=None, help="keep only the N nearest")
    p.add_argument("--network", default=None,
                   help="restrict results to one network, and disambiguate the "
                        "target if a bare code needs it")
    p.add_argument("--near", default=None, metavar="LAT,LON",
                   help="break a tie the network cannot: keep whichever "
                        "candidate is closest to this point. Needed for "
                        "TU.ERCT, TU.ERZM and TU.KOCA, which are two stations "
                        "under one NET.STA key.")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def parse_near(s):
    """'38.5,28.5' -> (38.5, 28.5). Raises ValueError on anything else."""
    if s is None:
        return None
    parts = str(s).replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"--near wants LAT,LON, got {s!r}")
    return float(parts[0]), float(parts[1])


def main():
    args = parse_args()
    if not args.stations and not args.station_coords:
        print("station-neighbors: give --stations, --station-coords, or both",
              file=sys.stderr)
        return 2

    try:
        near = parse_near(args.near)
    except ValueError as e:
        print(f"station-neighbors: {e}", file=sys.stderr)
        return 2

    # Authority first: coords wins the 179 keys the two files share.
    tab = load_station_table(args.station_coords, args.stations)

    hit = resolve(tab, args.station, args.network, near)
    if len(hit) == 0:
        print(f"  unknown station {args.station!r} in "
              f"{tab.key.nunique():,} known stations", file=sys.stderr)
        return 2
    if len(hit) > 1:
        print(f"  {args.station!r} is ambiguous -- {len(hit)} stations carry "
              f"that code:", file=sys.stderr)
        for r in hit.itertuples():
            print(f"    {r.key:<12s} {r.lat:8.4f} {r.lon:9.4f}  {r.place}",
                  file=sys.stderr)
        # Only suggest what would actually work here. Telling someone to name
        # the network is useless when both candidates already share one, which
        # is the case for TU.ERCT, TU.ERZM and TU.KOCA.
        hint = ("give NET.STA, or --network" if hit.key.nunique() > 1
                else "these share one NET.STA key, so only --near LAT,LON "
                     "separates them")
        print(f"  disambiguate: {hint}", file=sys.stderr)
        return 2

    t = hit.iloc[0]
    res = neighbors(tab, t.lat, t.lon, args.radius, args.min_km, args.top,
                    exclude_key=t.key)
    if args.network:
        res = res[res.network == args.network.strip()].reset_index(drop=True)

    # A key appearing twice here is a station whose position is itself in
    # dispute; flag it rather than presenting one of the two as the answer.
    ndup = res.key.value_counts()

    print(f"{t.key} ({t.place or 'place unknown'})  {t.lat:.4f} {t.lon:.4f}")
    span = f"{args.min_km:g}-{args.radius:g}" if args.min_km else f"{args.radius:g}"
    print(f"{len(res)} station(s) within {span} km, of "
          f"{len(tab):,} rows / {tab.key.nunique():,} stations\n")
    if not len(res):
        print("  nothing in range")
        return 0

    print(f"  {'#':>3s} {'station':<12s} {'km':>8s}  {'src':<8s} place")
    for i, r in enumerate(res.itertuples(), 1):
        flag = f"  [AMBIGUOUS {ndup[r.key]}]" if ndup[r.key] > 1 else ""
        print(f"  {i:3d} {r.key:<12s} {r.km:8.2f}  {r.src:<8s} {r.place}{flag}")

    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
        print(f"\n  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
