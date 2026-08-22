"""Does any generated window actually contain the S arrival?

`arrival_from_catalog.py` only ever computes P phases, so a window's S content
is never checked at generation time. On the 6 s configuration S turned out to
be inside 28.8% of event windows corpus-wide and 99.3% of those within 25 km.
This re-runs TauP over the *generated* metadata -- the same (distance, depth)
pairs the windows were actually cut from -- and reports whether the cut holds.

Run it after every generation, before training on the output.

The guarantee this can give is relative to the velocity model, not absolute.
S-P is predicted from a catalogue hypocentre, so it inherits the catalogue's
location error (median RMS residual 0.42 s in this corpus; a 5 km distance
error moves S-P by ~0.6 s). The margin table reports how many recordings would
be contaminated if the prediction is off by a given amount, which is the honest
way to state the result: "P-only by construction under iasp91", not "zero S".

Usage:
    python3 src/detection/verify_ponly_windows.py \\
        --metadata .../window_post_3.4s_ponly/window_metadata.csv \\
        --window-seconds 3.4 --pre-arrival-seconds 2.0
"""

import argparse

import numpy as np
import pandas as pd
from obspy.taup import TauPyModel

P_PHASES = ["p", "P", "Pg", "Pn"]
S_PHASES = ["s", "S", "Sg", "Sn"]
DEG_KM = 111.195


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--metadata", required=True,
                   help="window_metadata.csv written by arrival_from_catalog.py")
    p.add_argument("--window-seconds", type=float, required=True)
    p.add_argument("--pre-arrival-seconds", type=float, required=True)
    p.add_argument("--taup-model", default="iasp91")
    p.add_argument("--out-csv", default=None, help="optional per-recording S-P table")
    return p.parse_args()


def main():
    args = parse_args()
    post = args.window_seconds - args.pre_arrival_seconds
    m = pd.read_csv(args.metadata)
    need = {"distance_km", "depth_km"}
    missing = need - set(m.columns)
    if missing:
        raise SystemExit(f"metadata lacks {missing}; expected columns from arrival_from_catalog.py")

    print(f"window {args.window_seconds:g} s = {args.pre_arrival_seconds:g} s pre-P "
          f"+ {post:g} s post-P")
    print(f"{len(m):,} station recordings, {m.event_id.nunique():,} events\n")

    # Cache on the same rounding generation used, so this costs seconds not hours.
    model = TauPyModel(args.taup_model)
    cache = {}

    def s_minus_p(dist_km, depth_km):
        key = (round(dist_km / DEG_KM, 3), round(float(depth_km), 0))
        if key not in cache:
            try:
                pa = model.get_travel_times(source_depth_in_km=max(0.0, key[1]),
                                            distance_in_degree=key[0], phase_list=P_PHASES)
                sa = model.get_travel_times(source_depth_in_km=max(0.0, key[1]),
                                            distance_in_degree=key[0], phase_list=S_PHASES)
                cache[key] = (min(a.time for a in sa) - min(a.time for a in pa)
                              if pa and sa else np.nan)
            except Exception:
                cache[key] = np.nan
        return cache[key]

    print(f"computing S-P with {args.taup_model} ...", flush=True)
    m["s_minus_p"] = [s_minus_p(d, z) for d, z in zip(m.distance_km, m.depth_km)]
    unresolved = int(m.s_minus_p.isna().sum())
    v = m.s_minus_p.dropna()
    print(f"  {len(cache):,} unique (degree, depth) cells; {unresolved} unresolved\n")

    intruding = int((v < post).sum())
    print(f"S-P seconds: min {v.min():.3f}  p1 {v.quantile(.01):.3f}  "
          f"median {v.median():.2f}  max {v.max():.2f}")
    print(f"\nwindows where S intrudes (S-P < {post:g} s): {intruding}  "
          f"({intruding / len(v):.4%})")
    print(f"margin at the closest arrival: {v.min() - post:+.3f} s")
    print("VERDICT:", "P-ONLY under this velocity model"
          if intruding == 0 else f"*** {intruding} CONTAMINATED ***")

    if intruding:
        cols = [c for c in ("event_id", "station_key", "distance_km", "depth_km",
                            "s_minus_p") if c in m.columns]
        print(m[m.s_minus_p < post][cols].head(10).to_string(index=False))

    # The margin is what matters, not just the count at zero error.
    print(f"\nif the S-P prediction is off by ... recordings that could contain S")
    for err in (0.0, 0.3, 0.5, 0.63):
        n = int((v < post + err).sum())
        print(f"  {err:.2f} s{'':>18}{n:>8,}  ({n / len(v):.2%})")

    print(f"\ndistance: min {m.distance_km.min():.1f} km  "
          f"median {m.distance_km.median():.1f}  max {m.distance_km.max():.1f}")
    if args.out_csv:
        m.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
