"""For which events does detection AND magnitude complete before S arrives?

This is the early-warning question the project has been careful not to claim,
and it is answerable from a `sk falsealarm timing` CSV with no new training and
no rescanning.

**The arithmetic.** `alarm_epoch` is the detection window's END -- a detector
cannot declare before the whole window exists, so the alarm is at the end, not
the start. The magnitude stage then needs its own window filled. Anchored on the
alarm with `--anchor-lag L` and `--pre P`, a `W`-second magnitude window starts
at `alarm_epoch - L - P` and is complete at:

    mag_ready = alarm_epoch - L - P + W

so the full cascade beats the S arrival when

    dt_vs_s + (W - P - L) < 0            where dt_vs_s = alarm_epoch - s_epoch

Detection alone beating S (`dt_vs_s < 0`) is a strictly weaker claim, and the
gap between the two columns below is the price of estimating magnitude.

**Why a break-even distance exists.** S-P grows with distance while the cascade's
own latency does not, so beyond some distance the cascade always wins and inside
it always loses. That distance is a property of the window geometry, not of the
model, and this prints the measured one rather than assuming a Vp.

    python -m experiments.analyses.cascade_lead_time \
        --timing mant_alarm_times.csv --mag-window 6 10 20
"""
import argparse

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timing", required=True,
                   help="a `sk falsealarm timing` CSV: needs dt_vs_s, sp_seconds, "
                        "dist, Magnitude, alarm_epoch")
    p.add_argument("--mag-window", type=float, nargs="+", default=[6.0, 10.0, 20.0],
                   help="magnitude window length(s) to price, seconds")
    p.add_argument("--pre", type=float, default=2.0,
                   help="seconds of pre-anchor lead in the magnitude window "
                        "(cut_event_windows --pre)")
    p.add_argument("--anchor-lag", type=float, default=0.0,
                   help="cut_event_windows --anchor-lag, subtracted from the anchor")
    p.add_argument("--snr-min", type=float, default=None,
                   help="if the CSV carries snr, keep only events reaching this")
    p.add_argument("--bands", type=float, nargs="+",
                   default=[0, 25, 50, 100, 150, 200, 300, 500],
                   help="distance band edges, km")
    p.add_argument("--out", default=None, help="write the per-event table here")
    return p.parse_args()


def main():
    args = parse_args()
    ev = pd.read_csv(args.timing)
    need = {"dt_vs_s", "sp_seconds", "dist"}
    missing = need - set(ev.columns)
    if missing:
        raise SystemExit(f"{args.timing} lacks {sorted(missing)} -- is it a "
                         f"`sk falsealarm timing` output?")
    if args.snr_min is not None and "snr" in ev.columns:
        ev = ev[ev.snr >= args.snr_min]

    det = ev[ev.dt_vs_s.notna()].copy()          # detected events only
    print(f"{'=' * 74}")
    print(f"CASCADE LEAD TIME  --  {args.timing}")
    print(f"{'=' * 74}")
    print(f"  {len(ev):,} events in the file, {len(det):,} detected "
          f"({len(det) / max(len(ev), 1):.1%})")
    print(f"  detection alone beats S for {int((det.dt_vs_s < 0).sum()):,} "
          f"({(det.dt_vs_s < 0).mean():.1%}) of them\n")

    # S-P against distance, measured rather than assumed -- this is what turns
    # a latency in seconds into a break-even distance in km.
    ok = det.sp_seconds.notna() & det.dist.notna() & (det.dist > 0)
    slope = float(np.polyfit(det.loc[ok, "dist"], det.loc[ok, "sp_seconds"], 1)[0])
    vps = 1.0 / slope if slope > 0 else float("nan")
    print(f"  measured S-P vs distance: {slope * 100:.2f} s per 100 km "
          f"(equivalent {vps:.2f} km/s), n={int(ok.sum()):,}\n")

    print(f"  {'mag window':>11}{'latency added':>15}{'cascade beats S':>18}"
          f"{'break-even':>13}")
    rows = []
    for W in args.mag_window:
        add = W - args.pre - args.anchor_lag       # seconds after the alarm
        beats = det.dt_vs_s + add < 0
        # break-even distance: the smallest distance at which more than half the
        # detected events in a band still beat S. Measured, not derived.
        be = np.nan
        edges = args.bands
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = det[(det.dist >= lo) & (det.dist < hi)]
            if len(g) >= 20 and (g.dt_vs_s + add < 0).mean() > 0.5:
                be = lo
                break
        print(f"  {W:>9.0f} s{add:>13.1f} s{beats.mean():>17.1%}"
              + (f"{be:>11.0f} km" if be == be else f"{'none':>13}"))
        rows.append((W, add, beats.mean(), be))

    print(f"\n  by distance band (fraction of DETECTED events where the whole "
          f"cascade beats S)")
    hdr = "".join(f"{'W=' + format(W, '.0f') + 's':>10}" for W in args.mag_window)
    print(f"    {'band (km)':>12}{'events':>9}{'det<S':>8}{hdr}")
    for lo, hi in zip(args.bands[:-1], args.bands[1:]):
        g = det[(det.dist >= lo) & (det.dist < hi)]
        if not len(g):
            continue
        cells = "".join(
            f"{(g.dt_vs_s + (W - args.pre - args.anchor_lag) < 0).mean():>10.1%}"
            for W in args.mag_window)
        print(f"    {f'{lo:.0f}-{hi:.0f}':>12}{len(g):>9,}"
              f"{(g.dt_vs_s < 0).mean():>8.1%}{cells}")

    if args.out:
        for W in args.mag_window:
            det[f"beats_s_w{W:.0f}"] = det.dt_vs_s + (W - args.pre - args.anchor_lag) < 0
        det.to_csv(args.out, index=False)
        print(f"\n  wrote {args.out} ({len(det):,} rows)")

    print(f"\n  Read this as P-phase timing, not a deployment claim: the alarm "
          f"time is\n  quantized to the detection window's step, so a delta is "
          f"only meaningful to\n  within one step. Rescan the event guards "
          f"densely (`scan --near-csv`) before\n  reading a close band's number "
          f"as a real lead time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
