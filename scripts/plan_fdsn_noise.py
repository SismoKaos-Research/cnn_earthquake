"""Plan noise windows for the stations the FDSN event pull actually returned.

`generate-regression-dataset` needs a per-station noise floor: the spectrogram
is normalised against it and `log_snr` is measured from it. The event pull has
no noise windows, and the alternative -- `--normalize per_window` -- normalises
each window by itself, which discards amplitude. Amplitude is the primary
magnitude predictor, so that is not an option here.

Noise is taken from the SAME station at `--offset` seconds before the event's
own P arrival, which is the convention `cut_event_windows.py` uses and for the
reason its docstring gives: the noise class then shares the station, the
instrument and the season with the signal class. Contamination by an
uncatalogued event is tolerable -- the baseline is a median over time frames
precisely so one event cannot drag a station's floor upward.

Only stations that actually returned event data are planned for, and each is
capped: a stable profile needs a handful of windows, not hundreds.
"""
import argparse
import pathlib

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", required=True, help="the event plan CSV")
    p.add_argument("--event-dir", required=True,
                   help="where the event pull landed; only stations that "
                        "actually returned data are planned for")
    p.add_argument("--per-station", type=int, default=20)
    p.add_argument("--offset", type=float, default=300.0,
                   help="seconds before P to take the noise window")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    req = pd.read_csv(args.requests)
    have = {f.name for f in pathlib.Path(args.event_dir).glob("*.mseed")}
    dest = ("event_" + req.event_id.astype(str) + "_"
            + req.station.astype(str) + "_raw.mseed")
    got = req[dest.isin(have)].copy()
    print(f"[noise] {len(got):,} event windows exist across "
          f"{got.station.nunique()} stations")

    rows = got.groupby("station", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.per_station), random_state=42))
    rows = rows.copy()
    span = rows.end - rows.start
    rows["start"] = rows.start - args.offset
    rows["end"] = rows.start + span
    # A distinct id so the noise file cannot collide with its own event window.
    rows["event_id"] = "n" + rows.event_id.astype(str)
    rows.to_csv(args.out, index=False)
    print(f"[noise] wrote {args.out}: {len(rows):,} window(s), "
          f"{rows.station.nunique()} station(s), "
          f"median {int(rows.groupby('station').size().median())} per station")


if __name__ == "__main__":
    main()
