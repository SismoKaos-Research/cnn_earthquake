"""Recompute `distance_km` in a dataset manifest, from station coordinates.

A magnitude manifest carries `distance_km` per row, and two things go wrong
with it. Both were found on `dataset_magreg_fdsn_10s` (2026-09-06) and both are
silent:

**Missing values are not random.** 1,331 of 13,150 rows had none -- and 42 of
161 stations had none for ANY row while 119 had one for every row. That is a
station-key mismatch, not unknown data. It matters twice over: the trainer
mean-imputes NaN aux after standardizing, so every row from those 42 stations
trains against one fixed wrong distance; and the
`ridge(log_snr, log_distance)` floor this project judges magnitude models
against cannot be fitted on rows that lack it.

**Present values can still be wrong.** 72 rows across 4 stations carried a
distance computed from a station position that is not the real one -- KO.KIZT
by 14.4 km.

**The guard is the point of this file.** It recomputes EVERY row, compares
against the values already there, and refuses to write unless it reproduces
them within `--tolerance`. A recomputation that disagrees broadly is a
recomputation that is wrong, and writing it would replace a partial column with
a confidently incorrect one.

    sk distances --manifest dataset_magreg_fdsn_10s/manifest.csv \
        --station-coords ../seismic_cli/catalogs/station_coords.csv \
        --catalog catalogs/catalog_current.csv
    sk distances ... --write        # after reading what it says it will do

`station_coords.csv` is the authority: checked 2026-09-06 against the KOERI
FDSN station service, it agrees to 0.000 km on all 277 KO stations.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sismokaos.catalog import haversine_km


def station_table(path):
    """NET.STA -> (lat, lon), plus the bare-code fallback for malformed keys."""
    sc = pd.read_csv(path)
    sc["key"] = (sc.network.astype(str).str.strip() + "."
                 + sc.station.astype(str).str.strip())
    full = sc.drop_duplicates(subset="key").set_index("key")
    # Two rows in the FDSN manifest carry an empty network ('.EDC'). A bare code
    # resolves those, but ONLY when it is unique -- a code that appears in two
    # networks is left unresolved rather than guessed at, which is how one
    # network's coordinates end up on another network's station.
    uniq = sc[~sc.station.astype(str).str.strip().duplicated(keep=False)]
    bare = uniq.set_index(uniq.station.astype(str).str.strip())
    return full, bare


def recompute(man, full, bare, cat):
    """Great-circle distance for every row, NaN where a lookup fails."""
    ev = cat.drop_duplicates(subset="EventID").set_index("EventID")
    k = man.station_key.astype(str).str.strip()
    slat = k.map(full.latitude).to_numpy(float)
    slon = k.map(full.longitude).to_numpy(float)
    need = ~np.isfinite(slat)
    if need.any():
        b = k.str.split(".").str[-1]
        slat = np.where(need, b.map(bare.latitude).to_numpy(float), slat)
        slon = np.where(need, b.map(bare.longitude).to_numpy(float), slon)
    elat = man.event_id.map(ev.Latitude).to_numpy(float)
    elon = man.event_id.map(ev.Longitude).to_numpy(float)
    ok = np.isfinite(slat) & np.isfinite(slon) & np.isfinite(elat) & np.isfinite(elon)
    out = np.full(len(man), np.nan)
    out[ok] = haversine_km(slat[ok], slon[ok], elat[ok], elon[ok])
    return out, ok


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--station-coords", required=True,
                   help="network,station,latitude,longitude,elevation")
    p.add_argument("--catalog", required=True,
                   help="event catalogue with EventID, Latitude, Longitude")
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="km. Refuse to write if recomputing a row that already "
                        "has a distance disagrees by more than this -- that "
                        "means the recomputation is wrong, not the column.")
    p.add_argument("--report-above", type=float, default=1.0,
                   help="km. Rows differing by more than this are listed as "
                        "corrections, with the stations they belong to.")
    p.add_argument("--write", action="store_true",
                   help="apply, after backing the manifest up alongside it")
    return p.parse_args()


def main():
    args = parse_args()
    man = pd.read_csv(args.manifest)
    cat = pd.read_csv(args.catalog, encoding="utf-8-sig")
    cat.columns = [c.strip() for c in cat.columns]
    full, bare = station_table(args.station_coords)

    new, ok = recompute(man, full, bare, cat)
    old = man.distance_km.to_numpy(float) if "distance_km" in man else np.full(len(man), np.nan)
    had = np.isfinite(old)

    print(f"  {len(man):,} rows, {man.station_key.nunique()} stations")
    print(f"  distance_km present before: {int(had.sum()):,} ({100*had.mean():.1f}%)")

    check = had & ok
    if not check.any():
        sys.exit("  nothing to validate against -- refusing to write blind")
    err = np.abs(new[check] - old[check])
    print(f"\n  VALIDATION against the {int(check.sum()):,} rows that already have one")
    print(f"    median {np.median(err):.6f} km   p99 {np.percentile(err, 99):.4f} km   "
          f"max {err.max():.4f} km")
    if err.max() > args.tolerance:
        who = sorted(man.loc[check, "station_key"][err > args.report_above].unique())
        print(f"    stations involved: {who}")
        sys.exit(f"  max disagreement {err.max():.2f} km exceeds --tolerance "
                 f"{args.tolerance:g} -- NOT writing. Recomputation looks wrong.")

    fill = ~had & ok
    corrected_idx = np.where(check)[0][err > args.report_above]
    print(f"\n  would fill      {int(fill.sum()):,} row(s) that had none")
    if len(corrected_idx):
        print(f"  would correct   {len(corrected_idx):,} row(s) by >"
              f"{args.report_above:g} km, stations "
              f"{sorted(man.iloc[corrected_idx].station_key.unique())}")
    still = ~ok
    if still.any():
        print(f"  UNRESOLVED      {int(still.sum()):,} row(s): "
              f"{sorted(man.loc[still, 'station_key'].unique())[:8]}")

    if not args.write:
        print("\n  dry run -- pass --write to apply")
        return 0

    backup = Path(args.manifest + ".pre-distance-repair")
    shutil.copy(args.manifest, backup)
    man["distance_km"] = np.where(ok, new, old)
    man.to_csv(args.manifest, index=False)
    print(f"\n  backed up to {backup}")
    print(f"  wrote {args.manifest}: distance_km "
          f"{100*man.distance_km.notna().mean():.2f}% present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
