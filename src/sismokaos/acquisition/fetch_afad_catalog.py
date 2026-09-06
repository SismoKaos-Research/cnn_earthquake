"""Rebuilds the event catalogue from AFAD's public API.

This is the fix for `data_large.csv`, which holds 51 of 1,256 February 2025
events in the Aegean -- almost none of the Santorini-Amorgos swarm -- despite
being current to 2026-08-12. The hole is spatial, not stale.

Nothing here is queued or throttled: this API is public, unauthenticated, and
unrelated to the TDVMS waveform portal. The whole 2000-present nationwide
catalogue comes back in **one request in under 30 seconds**, so the default is
a single request, with date paging only as a fallback for a flaky connection.

Writes the local catalogue schema (Date/Longitude/Latitude/Depth/Rms/Type/
Magnitude/Location/EventID) so it is a drop-in for `catalog_current.csv`.

    sk catalog --out catalog_afad_2026-08-30.csv

Default floor is M>=1.5, which covers every use in this repo with margin: the
forecasting work thresholds at M>=2.5 and the detection download list bottoms
out at M2.0. Lower it if a task needs to.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

API = "https://deprem.afad.gov.tr/apiv2/event/filter"

# Covers Turkey and the surrounding seismogenic region, including the Aegean
# events west of the coastline that `data_large.csv` is missing.
NATIONWIDE = (32.5, 44.5, 23.0, 47.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--start", default="2000-01-01", help="inclusive")
    p.add_argument("--end", default=None, help="exclusive; default today")
    p.add_argument("--min-magnitude", type=float, default=1.5)
    p.add_argument("--box", nargs=4, type=float, metavar=("MINLAT", "MAXLAT", "MINLON", "MAXLON"),
                   default=NATIONWIDE, help="bounding box; default covers Turkey and the Aegean")
    p.add_argument("--page-days", type=int, default=0,
                   help="0 = one request for the whole span (default). Set e.g. 90 to "
                        "page instead, for a connection that cannot hold a long response")
    p.add_argument("--retries", type=int, default=4)
    return p.parse_args()


def fetch_page(t0, t1, minmag, box, retries):
    minlat, maxlat, minlon, maxlon = box
    url = (f"{API}?start={t0:%Y-%m-%d}%2000:00:00&end={t1:%Y-%m-%d}%2000:00:00"
           f"&minmag={minmag}&minlat={minlat:.3f}&maxlat={maxlat:.3f}"
           f"&minlon={minlon:.3f}&maxlon={maxlon:.3f}")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [FAIL] {t0:%Y-%m-%d}: {type(e).__name__}", file=sys.stderr)
                return []
            time.sleep(2 ** attempt)
    return []


def to_local_schema(df):
    """AFAD's API field names -> the CSV layout every script here expects.

    `format="ISO8601"` is required, not cosmetic: pre-2010 records come back
    without fractional seconds while recent ones carry them, so a single
    inferred format fails partway through the history.
    """
    t = pd.to_datetime(df.date, format="ISO8601")
    return pd.DataFrame({
        "Date": t.dt.strftime("%d/%m/%Y %H:%M:%S"),
        "Longitude": df.longitude.astype(float),
        "Latitude": df.latitude.astype(float),
        "Depth": df.depth.astype(float),
        "Rms": df.rms.astype(float),
        "Type": df.type,
        "Magnitude": df.magnitude.astype(float),
        "Location": df.location,
        "EventID": df.eventID.astype(int),
    }).sort_values("EventID").reset_index(drop=True)


def main():
    args = parse_args()
    end = datetime.fromisoformat(args.end) if args.end else datetime.utcnow()
    start = datetime.fromisoformat(args.start)
    box = tuple(args.box)
    print(f"box lat {box[0]:.2f}..{box[1]:.2f}  lon {box[2]:.2f}..{box[3]:.2f}")
    print(f"{start:%Y-%m-%d} -> {end:%Y-%m-%d}, M>={args.min_magnitude}")

    rows, pages = [], 0
    if args.page_days <= 0:
        print("  single request...", flush=True)
        rows = fetch_page(start, end, args.min_magnitude, box, args.retries)
        pages = 1
        if not rows:
            print("  single request returned nothing; falling back to 90-day pages",
                  file=sys.stderr)
            args.page_days = 90
    if args.page_days > 0 and not rows:
        t = start
        while t < end:
            nxt = min(t + timedelta(days=args.page_days), end)
            got = fetch_page(t, nxt, args.min_magnitude, box, args.retries)
            rows.extend(got)
            pages += 1
            print(f"  {t:%Y-%m-%d} +{len(got):5d}  (total {len(rows)})", flush=True)
            t = nxt
    if not rows:
        sys.exit("no events returned")

    df = pd.DataFrame(rows).drop_duplicates("eventID")
    out = to_local_schema(df)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    import os
    print(f"\n{len(out)} events -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.2f} MB, {pages} requests)")


if __name__ == "__main__":
    main()
