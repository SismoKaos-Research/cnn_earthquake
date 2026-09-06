"""Per catalogued event: when the detector first fired, relative to P and S."""
import numpy as np
import pandas as pd

from seismolib.continuous.alarms import load_scores
from seismolib.continuous.association import load_snr, predicted_arrivals

NAME = "timing"
HELP = "per-event: when did it fire, relative to S"


def add_args(q):
    q.add_argument("--scores", required=True, help="glob of scan .npz files")
    q.add_argument("--station", required=True)
    q.add_argument("--stations-csv", required=True)
    q.add_argument("--catalog", required=True)
    q.add_argument("--max-distance", type=float, default=500.0)
    q.add_argument("--guard-pre", type=float, default=10.0)
    q.add_argument("--guard-post", type=float, default=60.0)
    q.add_argument("--window-seconds", type=float, required=True,
                   help="the arm's window length. A detection cannot be declared "
                        "before the whole window exists, so the alarm time is the "
                        "window's END -- this is what converts a start time into "
                        "one.")
    q.add_argument("--threshold", type=float, required=True,
                   help="take this from `report` -- the threshold that buys the "
                        "alarm budget you intend to run at. The benchmark's 0.5 "
                        "is not an operating point on continuous data.")
    q.add_argument("--snr-csv", default=None,
                   help="station_detection_range.py output; without it the "
                        "timing is diluted by events the station never recorded")
    q.add_argument("--snr-min", type=float, default=3.0)
    q.add_argument("--out", required=True)


def run(args):
    """Per catalogued event: when the detector first fired, relative to P and S.

    **The alarm time is the window's END, not its start.** A detection cannot be
    declared before the whole window has been observed and scored, so a 6 s
    window starting at t announces at t+6. Using the start would credit the
    detector with information it did not yet have, and would make some events
    look detected before their P arrived.

    **Read the deltas against the window step, not below it.** Disjoint windows
    put the alarm time on a grid, so with a 6 s step a delta is only meaningful
    to +/-6 s -- which is coarser than S-P itself for anything inside ~50 km.
    Rescan the event guards densely (`scan --near-csv`, small `:STEP`) before
    reading a close event's number as a real lead time.
    """
    t, p = load_scores(args.scores)

    cat, _ = predicted_arrivals(
        args.station, args.stations_csv, args.catalog, args.max_distance)
    ev = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)].copy()
    if args.snr_csv:
        # load_snr, not a raw read: a duplicated event_id expands `ev` on this
        # join, and here that does not raise -- the loops below simply score the
        # duplicated events twice, inflating the detection counts and the
        # before-S fractions. DEMI's table carries 269 such ids.
        ev = ev.merge(load_snr(args.snr_csv), left_on="EventID",
                      right_on="event_id", how="left")
        n_all = len(ev)
        ev = ev[ev.snr >= args.snr_min].copy()
        print(f"  {len(ev):,} of {n_all:,} events reach SNR {args.snr_min:g}; "
              f"the rest leave no trace in the record and are excluded")

    first, best = [], []
    for a, b in zip(ev.p_epoch - args.guard_pre - args.window_seconds,
                    ev.p_epoch + args.guard_post):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        if j <= i:
            first.append(np.nan)
            best.append(np.nan)
            continue
        best.append(float(p[i:j].max()))
        hit = np.flatnonzero(p[i:j] > args.threshold)
        first.append(t[i + hit[0]] + args.window_seconds if len(hit) else np.nan)

    ev["best_prob"] = best
    ev["alarm_epoch"] = first
    ev["dt_after_p"] = ev.alarm_epoch - ev.p_epoch
    ev["dt_vs_s"] = ev.alarm_epoch - ev.s_epoch
    ev["covered"] = ~np.isnan(best)
    ev["detected"] = ~np.isnan(first)

    cov = ev[ev.covered]
    det = ev[ev.detected]
    print(f"{'=' * 70}\nDETECTION TIMING  --  {args.station}  "
          f"(threshold {args.threshold}, {args.window_seconds:g}s window)\n{'=' * 70}")

    # A saturated model "detects" everything, so recall alone is not readable.
    # Quoting the background rate next to it makes that impossible to miss: at a
    # 95% background rate, a 98% recall is arithmetic, not detection.
    inside = np.zeros(len(t), dtype=bool)
    for a, b in zip(ev.p_epoch - args.guard_pre - args.window_seconds,
                    ev.p_epoch + args.guard_post):
        i, j = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        inside[i:j] = True
    bg_rate = float((p[~inside] > args.threshold).mean())
    print(f"  {len(ev):,} catalogued events in span, {len(cov):,} with data, "
          f"{len(det):,} detected ({len(det) / max(len(cov), 1):.1%})")
    print(f"  background alarm rate at this threshold: {bg_rate:.4%} of windows "
          f"outside every guard")
    if bg_rate > 0.05:
        print(f"  ** WARNING: at this background rate a guard of "
              f"{(args.guard_pre + args.guard_post) / args.window_seconds:.0f} "
              f"windows contains an alarm "
              f"{1 - (1 - bg_rate) ** ((args.guard_pre + args.guard_post) / args.window_seconds):.1%} "
              f"of the time BY CHANCE. The recall and timing below are not "
              f"measuring detection -- pick a threshold from `report` first. **")
    if len(det):
        before = int((det.dt_vs_s < 0).sum())
        print(f"  {before:,} of {len(det):,} ({before / len(det):.1%}) fired BEFORE "
              f"the predicted S arrival")
        print(f"  alarm after P: median {det.dt_after_p.median():.1f}s  "
              f"(quantized to the {args.window_seconds:g}s window grid)")
        print(f"\n  {'dist (km)':>12}{'events':>9}{'found':>8}{'recall':>9}"
              f"{'med S-P':>9}{'med dt vs S':>13}{'before S':>10}")
        for band, g in cov.groupby(pd.cut(cov.dist, [0, 25, 50, 100, 200, 500]),
                                   observed=True):
            d = g[g.detected]
            if not len(g):
                continue
            print(f"    {str(band):>10}{len(g):>9,}{len(d):>8,}"
                  f"{len(d) / len(g):>9.3f}{g.sp_seconds.median():>9.1f}"
                  + (f"{d.dt_vs_s.median():>13.1f}{(d.dt_vs_s < 0).mean():>10.2f}"
                     if len(d) else f"{'-':>13}{'-':>10}"))

    # p_epoch and s_epoch are carried so this file can drive `scan --near-csv`
    # for a dense rescan, which is how the deltas get resolved below the grid.
    cols = ["EventID", "t", "Magnitude", "dist", "Depth", "p_epoch", "s_epoch",
            "sp_seconds", "best_prob", "alarm_epoch", "dt_after_p", "dt_vs_s",
            "Location"]
    ev[ev.covered][cols].to_csv(args.out, index=False)
    print(f"\n  wrote {args.out} ({int(ev.covered.sum()):,} rows)")
