"""Cut arrival-anchored event windows out of the continuous station archive.

The magnitude corpus was built by requesting waveforms per event within ~56 km
of an epicentre, which is why it is dominated by small local earthquakes: at
56 km you only catch a large one by luck. 1.2% of its windows are M>4, and its
distance axis is unusable because nothing in it is further than 56 km.

The continuous archive removes both limits at zero download cost. Every
catalogued event inside the record has a waveform whether or not anyone
requested it, out to whatever distance the station can hear -- 500 km for MANT
against 56 km for the event-window corpus.

Windows are written as `event_<id>_raw.mseed` under
`<out-dir>/<length>/<station>/{eq,noise}/`, the layout
`seismic-cli generate-regression-dataset` already consumes, so the spectrogram
encoding, the aux scalars and the split logic all come from the tested path
rather than being reimplemented here. This script only decides *which samples*
become a window, never how they are encoded.

**The station goes in the path, not the filename.** An earlier version wrote
`event_<id>_<station>_raw.mseed`, which reads as compatible and is not:
`parse_event_id` is `^(?:noise_)?event_(.+?)_raw$` with a non-greedy capture, so
that name yields the event id `627227_MANT`, matches no catalogue row, and every
window loses its magnitude label without an error anywhere.

**Noise windows come from the same record**, taken well before the P arrival of
the event they accompany, so the noise class shares the station, the instrument
and the season with the signal class. The detection work showed what happens
when it does not: negatives mined from a different amplitude regime leave a hole
the model extrapolates into.

    python3 scripts/cut_event_windows.py \\
        --zips 'afad_raw/MANT/*.zip' --station MANT \\
        --stations-csv catalogs/istasyon_katalog.csv \\
        --catalog catalogs/catalog_current.csv --snr-csv mant_range_full.csv \\
        --snr-min 3 --out-dir raw/continuous_windows
"""
import argparse
import glob
import pathlib
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from continuous_false_alarms import (component_segments, haversine,
                                     pick_components, predicted_arrivals,
                                     read_chunk)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zips", required=True)
    p.add_argument("--station", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--snr-csv", default=None,
                   help="station_detection_range.py output. Events below "
                        "--snr-min leave no usable trace and are skipped.")
    p.add_argument("--snr-min", type=float, default=3.0)
    p.add_argument("--max-distance", type=float, default=500.0)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--pre", type=float, default=2.0,
                   help="seconds before the predicted P (matches the corpus's "
                        "[P-2, P+4] geometry)")
    p.add_argument("--window-seconds", type=float, nargs="+", default=[6.0],
                   help="one or more window lengths. All are cut in the SAME "
                        "pass, because reading and decoding the archive costs "
                        "minutes per chunk while cutting costs milliseconds -- "
                        "a second pass would pay the whole read again for "
                        "nothing. Each length gets its own eq/noise subtree.")
    p.add_argument("--noise-offset", type=float, default=300.0,
                   help="seconds before P to take the paired noise window")
    p.add_argument("--anchor-csv", default=None,
                   help="CSV with event_id and an anchor time column, used "
                        "INSTEAD of the predicted P. `falsealarm timing` writes "
                        "one: its `alarm_epoch` is when the detector actually "
                        "fired. Every magnitude figure in this project so far "
                        "was measured on catalogue-anchored windows, i.e. with "
                        "the P arrival known -- which a deployed cascade does "
                        "not have. Cutting from the detector's own alarm times "
                        "is what prices that assumption.")
    p.add_argument("--anchor-column", default="alarm_epoch",
                   help="column in --anchor-csv holding the anchor epoch")
    p.add_argument("--anchor-lag", type=float, default=None,
                   help="seconds subtracted from the anchor before --pre is "
                        "applied. Set it to the detector's MEDIAN lag after P "
                        "so the median event lands exactly where training put "
                        "it; what then remains is the jitter, which is the "
                        "quantity being measured. Defaults to the median of "
                        "(anchor - predicted P) over the file itself.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit-chunks", type=int, default=None)
    return p.parse_args()


def cut(segs, comps, t0, n, fs):
    """Returns an (n, 3) array at epoch `t0`, or None if any component is short.

    A window is only returned when all three components cover it inside a
    single contiguous segment; anything spanning a gap is refused rather than
    padded, so no window is ever built across missing samples.
    """
    out = np.empty((n, 3), dtype=np.float64)
    for k in range(3):
        got = False
        for s0, data in segs[k]:
            i = int(round((t0 - s0) * fs))
            if 0 <= i and i + n <= len(data):
                out[:, k] = data[i:i + n]
                got = True
                break
        if not got:
            return None
    return out


def write_mseed(arr, comps, t0, station, path, fs):
    """Writes one (n, 3) window as a 3-trace mseed file."""
    from obspy import Stream, Trace
    trs = []
    for k, comp in enumerate(comps):
        tr = Trace(arr[:, k].astype(np.int32))
        tr.stats.network = "TU"
        tr.stats.station = station
        tr.stats.channel = f"HH{comp}"
        tr.stats.sampling_rate = fs
        tr.stats.starttime = UTCDateTime(t0)
        trs.append(tr)
    Stream(trs).write(str(path), format="MSEED")


def main():
    """Cuts event and paired-noise windows for every usable catalogued event."""
    args = parse_args()
    lengths = sorted(args.window_seconds)
    dirs = {}
    for w in lengths:
        tag = f"{w:g}s"
        # <length>/<station>/{eq,noise}: the station is what makes two cuts
        # distinguishable, and it cannot live in the filename without breaking
        # the consumer's event-id parse (see the module docstring).
        eq = pathlib.Path(args.out_dir) / tag / args.station / "eq"
        nz = pathlib.Path(args.out_dir) / tag / args.station / "noise"
        eq.mkdir(parents=True, exist_ok=True)
        nz.mkdir(parents=True, exist_ok=True)
        dirs[w] = (eq, nz)

    cat, (slat, slon) = predicted_arrivals(args)
    if args.snr_csv:
        snr = pd.read_csv(args.snr_csv)[["event_id", "snr"]]
        cat = cat.merge(snr, left_on="EventID", right_on="event_id", how="left")
        before = len(cat)
        cat = cat[cat.snr >= args.snr_min]
        print(f"[events] {len(cat):,} of {before:,} reach SNR {args.snr_min:g}")

    # The anchor. `cut_epoch` is what every window is cut relative to: normally
    # the predicted P, or the detector's own alarm when --anchor-csv is given.
    cat["cut_epoch"] = cat.p_epoch
    if args.anchor_csv:
        a = pd.read_csv(args.anchor_csv)
        # The id column is spelled both ways in this project: `falsealarm
        # timing` writes `EventID` (it comes straight off the catalogue frame)
        # and `station_detection_range` writes `event_id`. Accept either rather
        # than making the caller know which tool produced the file.
        idcol = next((c for c in ("EventID", "event_id") if c in a.columns), None)
        if idcol is None:
            sys.exit(f"{args.anchor_csv} has neither EventID nor event_id; "
                     f"columns are {list(a.columns)}")
        if args.anchor_column not in a.columns:
            sys.exit(f"{args.anchor_csv} has no column {args.anchor_column!r}; "
                     f"columns are {list(a.columns)}")
        a = a[[idcol, args.anchor_column]].dropna()
        a = a.sort_values(args.anchor_column).drop_duplicates(subset=idcol)
        before = len(cat)
        cat = cat.merge(a.rename(columns={idcol: "_anchor_id"}),
                        left_on="EventID", right_on="_anchor_id", how="inner")
        lag = args.anchor_lag
        if lag is None:
            lag = float((cat[args.anchor_column] - cat.p_epoch).median())
        cat["cut_epoch"] = cat[args.anchor_column] - lag
        resid = (cat.cut_epoch - cat.p_epoch)
        print(f"[anchor] {len(cat):,} of {before:,} event(s) have a "
              f"{args.anchor_column}; lag {lag:.2f}s removed")
        print(f"[anchor] residual offset from the true P after the lag: "
              f"median {resid.median():+.2f}s, "
              f"p10 {resid.quantile(0.1):+.2f}s, p90 {resid.quantile(0.9):+.2f}s")
        print(f"[anchor] that spread IS the quantity being measured -- the "
              f"median event lands where training put it and the rest do not")
    cat = cat.sort_values("p_epoch").reset_index(drop=True)

    zips = sorted(glob.glob(args.zips))
    if args.limit_chunks:
        zips = zips[:args.limit_chunks]
    print(f"[cut] {len(zips)} chunk(s), lengths " + ", ".join(
        f"{w:g}s [P-{args.pre:g}, P+{w - args.pre:g}]" for w in lengths))

    kept = skipped = 0
    per_band = {}
    for zi, z in enumerate(zips, 1):
        stem = pathlib.Path(z).stem
        st = read_chunk(z)
        comps = pick_components(st)
        if comps is None:
            print(f"[{zi}/{len(zips)}] {stem}: incomplete components, skipped", flush=True)
            continue
        segs = [component_segments(st, c, args.fs) for c in comps]
        del st
        lo = min(s0 for seg in segs for s0, _ in seg)
        hi = max(s0 + len(d) / args.fs for seg in segs for s0, d in seg)
        sub = cat[(cat.p_epoch >= lo + args.noise_offset) & (cat.p_epoch <= hi - 60)]

        got = 0
        for ev in sub.itertuples():
            # `event_<id>_raw`, exactly. seismic-cli's parse_event_id is
            # `^(?:noise_)?event_(.+?)_raw$` and its capture is non-greedy, so
            # an extra `_MANT` infix parses as the event id "627227_MANT",
            # which matches no catalogue EventID -- every window would lose its
            # magnitude label, silently, and the dataset would come out empty.
            # The station belongs in the path (below), where it distinguishes
            # two stations' cuts without breaking the consumer.
            tag = f"event_{int(ev.EventID)}"
            # An event is kept only if EVERY requested length is clean, so the
            # length series covers identical events and a difference between
            # lengths cannot come from a difference in which events survived.
            cuts = {}
            for w in lengths:
                nw = int(round(w * args.fs))
                a = cut(segs, comps, ev.cut_epoch - args.pre, nw, args.fs)
                b_ = cut(segs, comps, ev.p_epoch - args.noise_offset, nw, args.fs)
                if a is None or b_ is None:
                    cuts = None
                    break
                cuts[w] = (a, b_)
            if cuts is None:
                skipped += 1
                continue
            for w, (a, b_) in cuts.items():
                eq, nzd = dirs[w]
                write_mseed(a, comps, ev.cut_epoch - args.pre, args.station,
                            eq / f"{tag}_raw.mseed", args.fs)
                write_mseed(b_, comps, ev.p_epoch - args.noise_offset, args.station,
                            nzd / f"noise_{tag}_raw.mseed", args.fs)
            band = int(min(ev.Magnitude, 6.0) * 2) / 2.0
            per_band[band] = per_band.get(band, 0) + 1
            kept += 1
            got += 1
        print(f"[{zi}/{len(zips)}] {stem}: {got} window(s) "
              f"({len(sub)} events in span)", flush=True)

    print(f"\n[cut] wrote {kept:,} event(s) x {len(lengths)} length(s), "
          f"skipped {skipped:,} (gap or edge)")
    for w in lengths:
        print(f"  -> {dirs[w][0].parent}")
    if per_band:
        print("\n  magnitude distribution of what was cut")
        for b in sorted(per_band):
            print(f"    M {b:>4.1f}+ : {per_band[b]:>6,}")


if __name__ == "__main__":
    main()
