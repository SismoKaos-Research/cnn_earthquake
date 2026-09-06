"""Recast STEAD as this project's own P-only corpus, so the generator can build it.

The reciprocal of `pretrained_picker_baseline.py`: that ran their model on our
data, this prepares their data for our model. Both halves are needed before
"our detector beats GPD" means anything -- a locally-trained model winning on
local data is the expected result, not evidence.

**Nothing here reimplements the preprocessing.** Windows are written out as
miniSEED in the exact layout `generate-spec-dual-dataset` already consumes, and
that command then builds the dataset with the same flags used for the Aegean
corpus. Filtering, per-(station, component) baseline normalisation, hard-negative
mining and the STFT geometry are therefore identical by construction rather than
by careful copying, which is the only way a cross-corpus number is worth
quoting.

**Component order is verified, not assumed.** STEAD stores (6000, 3) as E, N, Z
while this project's `_COMPONENT_ROLES` is Z, N, E. Getting that backwards would
feed the model horizontals where it expects the vertical and produce a plausible
low score that looks like a generalisation failure. Checked on 400 event traces:
column 2 carries the largest P onset jump on 66% of them (median post/pre
amplitude ratio 45.0, against 24.1 and 31.2 for columns 0 and 1), so column 2 is
the vertical, as documented. Channels are named from that.

**Only stations present in BOTH chunks are used.** The baseline normalisation
needs long-term noise per station, and STEAD's noise chunk covers 1,155 stations
against the event chunk's 203, overlapping on 95. Events at stations with no
noise record could not be normalised the way the model expects.

Usage:
    python3 src/sismokaos/detection/stead_to_ponly_mseed.py \\
        --stead-root ~/Projects/Sismokaos/stead_data_process/raw \\
        --out-root ~/Projects/Sismokaos/seismic_cli/raw/data
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

# STEAD column order. Verified empirically -- see the module docstring.
STEAD_COMPONENTS = ("E", "N", "Z")
WINDOW_SAMPLES = 340
PRE_P_SAMPLES = 200      # 2.0 s at 100 Hz, matching arrival_from_catalog.py
FS = 100.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stead-root", required=True,
                   help="Directory holding earthquake/chunk2.* and noise/chunk1.*")
    p.add_argument("--out-root", required=True,
                   help="`raw/data` under the data_downloader checkout.")
    p.add_argument("--event-chunk", default="earthquake/chunk2")
    p.add_argument("--noise-chunk", default="noise/chunk1")
    p.add_argument("--min-magnitude", type=float, default=2.0,
                   help="Matches the Aegean corpus's floor, so the comparison "
                        "is over a comparable event population.")
    p.add_argument("--max-distance-km", type=float, default=56.0)
    p.add_argument("--max-events", type=int, default=12000)
    p.add_argument("--max-noise", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def station_key(row):
    return f"{row.network_code}.{row.receiver_code}"


def to_stream(data, net, sta, start, n=None):
    """Builds a 3-component Stream, naming channels by STEAD's column order.

    `data` is (samples, 3) in E, N, Z order; the channel letter is what the
    generator's `select_components` reads to assign the Z/N/E roles, so the
    naming here is the entire mapping.
    """
    traces = []
    for i, comp in enumerate(STEAD_COMPONENTS):
        d = data[:n, i] if n else data[:, i]
        traces.append(Trace(np.ascontiguousarray(d, dtype=np.float32), header=dict(
            network=net, station=sta, location="", channel=f"HH{comp}",
            sampling_rate=FS, starttime=start)))
    return Stream(traces)


def select_events(csv_path, stations, args):
    """Event rows that are in range, at a usable station, with room for the cut."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["sta"] = df.apply(station_key, axis=1)
    keep = (df.sta.isin(stations)
            & (df.source_magnitude >= args.min_magnitude)
            & (df.source_distance_km <= args.max_distance_km)
            & (df.p_arrival_sample >= PRE_P_SAMPLES))
    df = df[keep].copy()
    # The window must fit inside the 60 s record on the far side too.
    df = df[df.p_arrival_sample + (WINDOW_SAMPLES - PRE_P_SAMPLES) <= 6000]
    if len(df) > args.max_events:
        df = df.sample(args.max_events, random_state=args.seed)
    return df


def main():
    args = parse_args()
    root = Path(args.stead_root).expanduser()
    out = Path(args.out_root).expanduser()
    ev_dir = out / "batched_waveforms" / "stead_ponly_3p4s"
    no_dir = out / "batched_noise_waveforms" / "stead_noise"
    ev_dir.mkdir(parents=True, exist_ok=True)
    no_dir.mkdir(parents=True, exist_ok=True)

    ev_csv = root / f"{args.event_chunk}.csv"
    no_csv = root / f"{args.noise_chunk}.csv"
    noise_meta = pd.read_csv(no_csv, low_memory=False)
    noise_meta["sta"] = noise_meta.apply(station_key, axis=1)
    ev_meta_all = pd.read_csv(ev_csv, low_memory=False)
    ev_meta_all["sta"] = ev_meta_all.apply(station_key, axis=1)
    stations = set(ev_meta_all.sta) & set(noise_meta.sta)
    print(f"stations  {len(stations)} present in both chunks "
          f"({ev_meta_all.sta.nunique()} event / {noise_meta.sta.nunique()} noise)")

    events = select_events(ev_csv, stations, args)
    print(f"events    {len(events):,} selected "
          f"(M>={args.min_magnitude}, <={args.max_distance_km:g} km)")
    print(f"          magnitude median {events.source_magnitude.median():.2f}  "
          f"distance median {events.source_distance_km.median():.1f} km")

    rows = []
    with h5py.File(root / f"{args.event_chunk}.hdf5", "r") as f:
        g = f["data"]
        for _, r in events.iterrows():
            if r.trace_name not in g:
                continue
            p = int(r.p_arrival_sample)
            w = g[r.trace_name][()][p - PRE_P_SAMPLES:p - PRE_P_SAMPLES + WINDOW_SAMPLES]
            if w.shape[0] != WINDOW_SAMPLES:
                continue
            start = UTCDateTime(str(r.trace_start_time))
            path = ev_dir / f"event_{r.trace_name}.mseed"
            to_stream(w, r.network_code, r.receiver_code, start).write(str(path), format="MSEED")
            rows.append(dict(event_id=r.trace_name, station_key=r.sta,
                             magnitude=r.source_magnitude, depth_km=r.source_depth_km,
                             distance_km=r.source_distance_km,
                             snr_db=r.snr_db, fs=FS, filename=path.name))
    pd.DataFrame(rows).to_csv(ev_dir / "window_metadata.csv", index=False)
    print(f"          {len(rows):,} event windows written")

    # Noise records go out whole: the generator slides its own windows over them
    # at --overlap 0.5 and builds the station baselines from the same files, so
    # pre-cutting here would remove exactly the freedom it needs.
    pool = noise_meta[noise_meta.sta.isin(stations)]
    if len(pool) > args.max_noise:
        pool = pool.sample(args.max_noise, random_state=args.seed)
    n_written = 0
    with h5py.File(root / f"{args.noise_chunk}.hdf5", "r") as f:
        g = f["data"]
        for _, r in pool.iterrows():
            if r.trace_name not in g:
                continue
            d = g[r.trace_name][()]
            start = UTCDateTime(str(r.trace_start_time))
            to_stream(d, r.network_code, r.receiver_code, start).write(
                str(no_dir / f"noise_{r.trace_name}.mseed"), format="MSEED")
            n_written += 1
    print(f"noise     {n_written:,} records written ({n_written * 60 / 3600:.1f} station-hours)")
    print(f"\nevent dir {ev_dir}\nnoise dir {no_dir}")


if __name__ == "__main__":
    main()
