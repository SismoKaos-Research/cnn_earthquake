"""Gap and continuity audit across downloaded TDVMS chunks.

`paste` only checks that an archive is non-empty. That catches an expired link
or a too-long window, but not a chunk that arrived short, misaligned, or with
one component missing -- all of which look like a valid zip. This reads every
archive and reports actual coverage.

    python3 scripts/afad_audit.py --dir afad_raw/MANT
"""
import argparse
import pathlib
import tempfile
import zipfile
from datetime import datetime

from obspy import read


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="directory of chunk .zip files")
    return p.parse_args()


def audit(zpath):
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            f = next(p for p in pathlib.Path(tmp).rglob("*") if p.is_file())
            st = read(str(f))
    cov, span = {}, {}
    for tr in st:
        ch = tr.stats.channel
        cov[ch] = cov.get(ch, 0.0) + (tr.stats.endtime - tr.stats.starttime)
        s, e = tr.stats.starttime.datetime, tr.stats.endtime.datetime
        lo, hi = span.get(ch, (s, e))
        span[ch] = (min(lo, s), max(hi, e))
    any_ch = next(iter(span))
    total = (span[any_ch][1] - span[any_ch][0]).total_seconds()
    return names[0], len(st), cov, span, total


def main():
    args = parse_args()
    files = sorted(pathlib.Path(args.dir).glob("*.zip"))
    if not files:
        raise SystemExit(f"no .zip files in {args.dir}")
    print(f"{'chunk':<12s} {'traces':>7s} {'chans':>5s} {'gapE':>7s} {'gapN':>7s} {'gapZ':>7s}")
    worst, gaps_all = None, []
    for z in files:
        member, ntr, cov, span, total = audit(z)
        row = []
        for ch in ("HHE", "HHN", "HHZ"):
            if ch in cov:
                g = 100 * (1 - cov[ch] / total)
                row.append(f"{g:6.3f}%")
                gaps_all.append(g)
            else:
                row.append("MISSING")
        flag = "" if len(cov) == 3 else "   <- not 3 components"
        print(f"{z.stem[-10:]:<12s} {ntr:7d} {len(cov):5d} " + " ".join(row) + flag)
        m = max((100 * (1 - cov[c] / total)) for c in cov)
        if worst is None or m > worst[1]:
            worst = (z.stem, m)
    print(f"\n{len(files)} chunk(s)   mean gap {sum(gaps_all)/len(gaps_all):.3f}%   "
          f"worst {worst[0][-10:]} at {worst[1]:.3f}%")
    print("for comparison, the KOERI archive measures 2.316% (BODT) / 2.428% (DAT)")


if __name__ == "__main__":
    main()
