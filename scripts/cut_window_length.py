"""Re-cut the 60 s catalogue-anchored event windows to a shorter length.

Window length is the one lever the magnitude corpus has never varied under a
controlled comparison. `window_post_60s` holds every event in
`window_post_6s_catalog` at the SAME anchor time, so cutting it to N seconds
gives a series that differs from the 6 s corpus in length and nothing else --
same events, same stations, same catalogue anchoring, same instrument.

Geometry follows the 6 s corpus: the anchor is P-2.0 s, so an N-second cut is
[P-2, P+(N-2)].

Only samples are selected here; the spectrogram encoding, the aux scalars and
the splits all come from `seismic-cli generate-regression-dataset` afterwards,
so nothing about the comparison depends on this script reproducing an encoder.

**Fragmented files are merged before cutting.** Some 60 s records arrive as
several short traces per component; `merge` then a length check means a window
is written only when the component is genuinely continuous across it, rather
than silently short or gap-filled.

    python3 scripts/cut_window_length.py \\
        --src .../window_post_60s --seconds 10 \\
        --out .../window_post_10s_catalog
"""
import argparse
import pathlib
import sys

from obspy import Stream, read

# Components the regression encoder expects, in Z/N/E role order.
ROLES = (("Z",), ("N", "1"), ("E", "2"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="directory of 60 s event mseed files")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="stop after N files (smoke test)")
    return p.parse_args()


def main():
    """Cuts every source record to `--seconds` from its own start time."""
    args = parse_args()
    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.mseed"))
    if args.limit:
        files = files[:args.limit]
    print(f"[cut] {len(files):,} file(s) -> {args.seconds:g}s  ->  {out}")

    kept = short = nocomp = 0
    for i, f in enumerate(files, 1):
        try:
            st = read(str(f))
        except Exception:
            short += 1
            continue
        # Merge first: a component split across several traces is continuous
        # data, but slicing it unmerged would take only the first fragment.
        st.merge(method=1, fill_value=None)
        st = st.split()

        have = {tr.stats.channel[-1].upper() for tr in st}
        picked = [next((c for c in role if c in have), None) for role in ROLES]
        if any(c is None for c in picked):
            nocomp += 1
            continue

        t0 = min(tr.stats.starttime for tr in st)
        keep = []
        for comp in picked:
            trs = [tr for tr in st if tr.stats.channel[-1].upper() == comp]
            # Pick the trace that actually COVERS the window, rather than
            # requiring the component to be a single trace. 44% of these files
            # carry stray overlapping fragments alongside a complete 60 s
            # trace; demanding exactly one threw away nearly half the corpus
            # for a reason that has nothing to do with data quality.
            chosen = None
            for tr in trs:
                fs = tr.stats.sampling_rate
                want = int(round(args.seconds * fs))
                off = int(round((t0 - tr.stats.starttime) * fs))
                if off >= 0 and off + want <= tr.stats.npts:
                    chosen = tr.copy()
                    chosen.data = chosen.data[off:off + want]
                    chosen.stats.starttime = t0
                    break
            if chosen is None:
                keep = []
                break
            keep.append(chosen)
        if not keep:
            short += 1
            continue

        Stream(keep).write(str(out / f.name), format="MSEED")
        kept += 1
        if i % 5000 == 0:
            print(f"  ...{i:,}/{len(files):,}  kept {kept:,}", flush=True)

    print(f"[cut] kept {kept:,}  short/gapped {short:,}  missing component {nocomp:,}")
    if kept == 0:
        sys.exit("nothing written")


if __name__ == "__main__":
    main()
