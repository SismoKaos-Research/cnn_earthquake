"""Per-station event loss from the province-filtered catalogue.

`deprem_katalog_utc.csv` only contains events that carry a Turkish province, so
everything offshore or across a border is absent. How much that costs is not
uniform -- it depends entirely on where a station sits. A station inland loses
almost nothing; one on the Aegean or Mediterranean coast loses most of what it
would actually have recorded.

    python3 scripts/station_catalog_loss.py --radius 100 --min-magnitude 2.5
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

EARTH_KM = 6371.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stations", required=True)
    p.add_argument("--broken", required=True, help="deprem_katalog_utc.csv")
    p.add_argument("--truth", required=True, help="the rebuilt AFAD catalogue")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--min-magnitude", type=float, default=2.5)
    p.add_argument("--network", default=None, help="restrict to one network code")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def load_catalog(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["t"] = pd.to_datetime(d.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return d.dropna(subset=["t"])


def main():
    args = parse_args()
    broken, truth = load_catalog(args.broken), load_catalog(args.truth)

    # Compare only across the window the broken file actually claims to cover,
    # so its 2010 start is not counted as "loss" here -- that is a separate,
    # already-documented defect.
    lo, hi = broken.t.min(), broken.t.max()
    truth = truth[(truth.t >= lo) & (truth.t <= hi)]
    truth = truth[truth.Magnitude >= args.min_magnitude]
    have = set(broken.EventID)
    truth = truth.assign(missing=~truth.EventID.isin(have))
    print(f"window {lo:%Y-%m-%d}..{hi:%Y-%m-%d}  M>={args.min_magnitude}  "
          f"radius {args.radius:.0f} km")
    print(f"reference events {len(truth)},  of which missing from the broken "
          f"catalogue {int(truth.missing.sum())} ({100*truth.missing.mean():.1f}%)\n")

    st = pd.read_csv(args.stations, encoding="utf-8-sig")
    st.columns = [c.strip() for c in st.columns]
    if args.network:
        st = st[st.Network == args.network]
    st = st.drop_duplicates("Code").reset_index(drop=True)

    ev_lat = np.radians(truth.Latitude.to_numpy(float))
    ev_lon = np.radians(truth.Longitude.to_numpy(float))
    miss = truth.missing.to_numpy()

    rows = []
    CHUNK = 40
    for i in range(0, len(st), CHUNK):
        blk = st.iloc[i:i + CHUNK]
        la = np.radians(blk.Latitude.to_numpy(float))[:, None]
        lo_ = np.radians(blk.Longitude.to_numpy(float))[:, None]
        a = (np.sin((ev_lat[None, :] - la) / 2) ** 2
             + np.cos(la) * np.cos(ev_lat[None, :])
             * np.sin((ev_lon[None, :] - lo_) / 2) ** 2)
        near = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= args.radius
        for j, (_, s) in enumerate(blk.iterrows()):
            n = int(near[j].sum())
            if n == 0:
                continue
            m = int((near[j] & miss).sum())
            rows.append({"network": s.Network, "code": s.Code,
                         "province": s.get("Province", ""),
                         "in_range": n, "missing": m, "pct": 100.0 * m / n})

    df = pd.DataFrame(rows).sort_values("pct", ascending=False)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    def show(sub, title):
        print(title)
        print(f"  {'code':8s} {'net':4s} {'province':16s} {'in range':>9s} "
              f"{'missing':>8s} {'loss':>7s}")
        for _, r in sub.iterrows():
            print(f"  {r.code:8s} {r.network:4s} {str(r.province)[:16]:16s} "
                  f"{r.in_range:9d} {r.missing:8d} {r.pct:6.1f}%")

    show(df.head(args.top), f"WORST {args.top} — stations that lose the most")
    print()
    show(df[df.in_range >= 200].tail(10).iloc[::-1], "BEST — stations that lose least (>=200 events in range)")

    print(f"\n{len(df)} stations with events in range")
    for lo_p, hi_p in ((50, 101), (20, 50), (5, 20), (0, 5)):
        n = ((df.pct >= lo_p) & (df.pct < hi_p)).sum()
        print(f"  losing {lo_p:3d}-{hi_p-1:3d}% : {n:4d} stations")


if __name__ == "__main__":
    main()
