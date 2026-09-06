"""The station's long-term (mu, sigma), which every scored window is standardized
against.
"""
import glob
import json
import math
import pathlib
import sys
import time

import numpy as np

from sismokaos.continuous.chunks import (add_chunk_args, clean_block,
                                         component_segments, pick_components,
                                         read_chunk, taper_vector)

NAME = "baseline"
HELP = "build the station's long-term (mu, sigma)"


def add_args(q):
    add_chunk_args(q)
    q.add_argument("--sample-chunks", type=int, default=6,
                   help="chunks to scan, spread evenly across the archive")
    q.add_argument("--piece-seconds", type=float, default=3600.0,
                   help="length of the pieces each segment is cleaned in; see "
                        "`sismokaos.continuous.baseline` for why this is not a free "
                        "parameter")
    q.add_argument("--out", required=True)


def run(args):
    """Accumulates (mu, sigma) per component over whole cleaned traces.

    This mirrors `core.compute_station_noise_baselines`, which cleans each trace
    of each noise file whole and accumulates over all of them. Two differences,
    both stated rather than hidden:

    - That function is given noise-only files; a continuous record also contains
      the events. Earthquakes occupy a vanishing fraction of a station-year, so
      the effect on sigma is far below the precision that matters -- and it
      biases sigma UP, making the detector marginally more conservative.
    - It cleans whole traces, and the `noise_pre_3h` files it was pointed at are
      fragmented into pieces of seconds to minutes. Cleaning a 21-day trace whole
      is neither faithful to that nor affordable in memory, so segments are cut
      into `--piece-seconds` pieces first. This does not bias the comparison: the
      5% Hann taper always attenuates the same 10% *fraction* of any piece, so
      its effect on sigma is the same at any piece length.
    """
    zips = sorted(glob.glob(args.zips))
    if not zips:
        sys.exit(f"no archives matched {args.zips}")
    take = np.linspace(0, len(zips) - 1, min(args.sample_chunks, len(zips)))
    picked = [zips[int(round(i))] for i in take]

    print(f"[baseline] {len(picked)} of {len(zips)} chunks, whole-trace statistics")
    accum = {}
    for z in picked:
        t = time.time()
        st = read_chunk(z)
        comps = pick_components(st)
        if comps is None:
            print(f"  {pathlib.Path(z).stem}: incomplete components, skipped")
            continue
        piece = int(round(args.piece_seconds * args.fs))
        for comp in comps:
            for _, data in component_segments(st, comp, args.fs):
                for lo in range(0, len(data), piece):
                    part = data[lo:lo + piece]
                    if len(part) < args.fs * 10:
                        continue
                    c = clean_block(part[None, :].copy(), args.fs, args.freqmin,
                                    args.freqmax, taper_vector(len(part)))[0]
                    s, ss, n = accum.get(comp, (0.0, 0.0, 0))
                    accum[comp] = (s + float(c.sum()), ss + float((c ** 2).sum()),
                                   n + c.size)
        del st
        print(f"  {pathlib.Path(z).stem}: done in {time.time() - t:.0f}s", flush=True)

    out = {}
    for comp, (s, ss, n) in accum.items():
        mu = s / n
        sigma = math.sqrt(max(ss / n - mu ** 2, 0.0))
        out[comp] = {"mu": mu, "sigma": sigma, "n_samples": n}
        print(f"  {comp}: mu={mu:+.4g}  sigma={sigma:.6g}  ({n / args.fs / 3600:.1f} h)")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[baseline] wrote {args.out}")
