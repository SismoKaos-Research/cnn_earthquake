"""The operating table: what a threshold costs per day, and what it buys."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sismokaos.continuous.alarms import load_scores
from sismokaos.continuous.association import (background_and_guards, load_snr,
                                              predicted_arrivals)

NAME = "report"
HELP = "associate with the catalogue and tabulate"


def add_args(q):
    q.add_argument("--scores", required=True, help="glob of scan .npz files")
    q.add_argument("--station", required=True)
    q.add_argument("--stations-csv", required=True)
    q.add_argument("--catalog", required=True)
    q.add_argument("--max-distance", type=float, default=500.0)
    q.add_argument("--guard-pre", type=float, default=10.0,
                   help="seconds before the predicted P a window may still be "
                        "explained by the event")
    q.add_argument("--guard-post", type=float, default=60.0,
                   help="seconds after it -- long enough to cover the coda a "
                        "regional event leaves in the record")
    q.add_argument("--window-seconds", type=float, required=True,
                   help="the arm's window length, needed so a guard test is an "
                        "overlap test and not a start-time test")
    q.add_argument("--snr-csv", default=None,
                   help="station_detection_range.py output. Without it, recall "
                        "is asked of every catalogued event including those the "
                        "station never recorded, which measures the catalogue's "
                        "reach rather than the detector's.")
    q.add_argument("--snr-min", type=float, default=3.0,
                   help="SNR a catalogued event must reach to count as a "
                        "positive in the confusion matrix. Below this the "
                        "station has no waveform to detect and the event says "
                        "nothing about the model.")
    q.add_argument("--signal-post", type=float, default=20.0,
                   help="seconds after P a window must overlap to be labelled "
                        "positive. Tighter than the guard, which is deliberately "
                        "generous about what an alarm may be excused by.")
    q.add_argument("--cluster-seconds", type=float, default=60.0,
                   help="alarms closer together than this are one declaration. "
                        "Without clustering a single noise burst spanning ten "
                        "windows counts as ten false positives.")
    q.add_argument("--out-prefix", required=True)


def confusion(t, p, thr, cat, explained, args, win_s):
    """Confusion matrix at one threshold, at both the event and window level.

    Two units, because on continuous data neither alone is honest.

    **Event level** is what a detector is actually judged on, and it has no TN:
    there is no such thing as a "negative earthquake" to correctly not detect.
    Alarms are clustered first -- a noise burst spanning ten windows is one false
    declaration, not ten, and counting windows would inflate FP by whatever
    window length happened to be chosen.

    **Window level** has all four cells but TN is millions, so accuracy is
    meaningless there and only precision/recall are worth reading.

    Positives are events reaching `--snr-min`. Below that the station recorded no
    signal, so counting the event as a miss would charge the model for the
    catalogue's reach. Those events are excluded from BOTH classes rather than
    swept into the negatives, since a real arrival may well be present.
    """
    good = cat[(cat.snr >= args.snr_min) & cat.covered]

    # --- event level -------------------------------------------------------
    tp_ev = int((good.best_prob > thr).sum())
    fn_ev = int(len(good) - tp_ev)
    alarms = t[(p > thr) & ~explained]
    if len(alarms):
        breaks = np.flatnonzero(np.diff(alarms) > args.cluster_seconds)
        fp_ev = len(breaks) + 1
    else:
        fp_ev = 0

    # --- window level ------------------------------------------------------
    pos = np.zeros(len(t), dtype=bool)
    for c in good.p_epoch.values:
        i = np.searchsorted(t, c - win_s)
        j = np.searchsorted(t, c + args.signal_post, side="right")
        pos[i:j] = True
    neg = ~explained                      # outside every guard, any SNR
    tp_w = int(((p > thr) & pos).sum())
    fn_w = int(((p <= thr) & pos).sum())
    fp_w = int(((p > thr) & neg).sum())
    tn_w = int(((p <= thr) & neg).sum())
    return dict(tp_ev=tp_ev, fn_ev=fn_ev, fp_ev=fp_ev, n_ev=len(good),
                tp_w=tp_w, fn_w=fn_w, fp_w=fp_w, tn_w=tn_w)


def print_confusion(c, thr, days, label):
    """Prints one confusion block, with the metrics each unit can support."""
    pr_e = c["tp_ev"] / max(c["tp_ev"] + c["fp_ev"], 1)
    rc_e = c["tp_ev"] / max(c["tp_ev"] + c["fn_ev"], 1)
    f1_e = 2 * pr_e * rc_e / max(pr_e + rc_e, 1e-12)
    pr_w = c["tp_w"] / max(c["tp_w"] + c["fp_w"], 1)
    rc_w = c["tp_w"] / max(c["tp_w"] + c["fn_w"], 1)
    print(f"\n  CONFUSION MATRIX  --  {label}  (threshold {thr:.4f})")
    print(f"    EVENT level, n={c['n_ev']:,} events with signal"
          f"                    WINDOW level")
    print(f"                  alarm   no alarm                     "
          f"        alarm    no alarm")
    print(f"      event    {c['tp_ev']:>7,}   {c['fn_ev']:>8,}    <- TP / FN     "
          f"  event  {c['tp_w']:>7,}  {c['fn_w']:>10,}")
    print(f"      no event {c['fp_ev']:>7,}   {'n/a':>8}    <- FP / TN     "
          f"  none   {c['fp_w']:>7,}  {c['tn_w']:>10,}")
    print(f"      precision {pr_e:.4f}  recall {rc_e:.4f}  F1 {f1_e:.4f}"
          f"        precision {pr_w:.4f}  recall {rc_w:.4f}")
    print(f"      {c['fp_ev'] / max(days, 1e-9):.2f} false declarations/day. "
          f"TN is undefined at event level -- there is no negative earthquake, "
          f"and every FP may be an event AFAD never catalogued.")


def run(args):
    """The operating table: what a threshold costs per day, and what it buys.

    **The benchmark's 0.5 is not an operating point here and using it would be a
    mistake.** That threshold was fixed on balanced classes whose negatives were
    amplitude-mined, and it carries no meaning against continuous background: on
    MANT the 6 s detector scores a median of 0.83 on *noise*, so 0.5 flags 95% of
    a quiet station-day. Thresholds are therefore derived from the measured
    background distribution -- pick the alarm budget, read off the threshold --
    with the 0.5 row kept only to show how far off it is.

    **Recall is reported against events the station actually recorded.** Over the
    full 728-day MANT record, 27.0% of the catalogued events within 500 km with a
    measured SNR reach SNR 3, and the median is 1.39 -- the typical catalogued
    earthquake leaves no visible trace. Scoring a detector on events whose
    waveform does not exist measures the catalogue's reach, not the model's: the
    same arm scores AUC 0.675 against every event and 0.9403 against the ones
    with signal.

    (An earlier draft of this docstring quoted 11.5% and a median of 1.10. Those
    came from the first 195 days, when both the record and the SNR table were
    partial, and are not what the finished run says.)
    """
    t, p = load_scores(args.scores)
    win_s = args.window_seconds

    step = float(np.median(np.diff(t[:100000]))) if len(t) > 1 else win_s
    days = len(t) * step / 86400.0
    print(f"{'=' * 78}\nCONTINUOUS OPERATING TABLE  --  {args.station}  "
          f"({win_s:g}s windows every {step:g}s)\n{'=' * 78}")
    print(f"  {len(t):,} windows, {days:.1f} days of record, "
          f"{pd.to_datetime(t.min(), unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(t.max(), unit='s'):%Y-%m-%d}")

    cat, (slat, slon) = predicted_arrivals(
        args.station, args.stations_csv, args.catalog, args.max_distance)
    cat = cat[(cat.p_epoch >= t.min() - 300) & (cat.p_epoch <= t.max() + 300)].copy()
    explained, idx = background_and_guards(t, p, cat, win_s, args.guard_pre, args.guard_post)
    bg = p[~explained]
    print(f"  {len(cat):,} catalogued events within {args.max_distance:g} km of "
          f"({slat:.4f}, {slon:.4f}); their guards cover "
          f"{100 * explained.mean():.2f}% of windows")

    print(f"\n  BACKGROUND score distribution ({len(bg):,} windows outside every guard)")
    print("    " + "  ".join(f"p{q}={np.percentile(bg, q):.4f}"
                             for q in (50, 90, 99, 99.9, 99.99)) +
          f"  max={bg.max():.4f}")
    if np.percentile(bg, 50) > 0.5:
        print("    ** the median NOISE window scores above 0.5: the benchmark "
              "threshold is meaningless here **")

    # Attach measured SNR so recall is asked only of events with a waveform.
    if args.snr_csv:
        cat = cat.merge(load_snr(args.snr_csv), left_on="EventID",
                        right_on="event_id", how="left")
    else:
        cat["snr"] = np.nan
    best = np.array([p[i:j].max() if j > i else np.nan for i, j in idx])
    cat["best_prob"] = best
    cat["covered"] = ~np.isnan(best)
    cov = cat[cat.covered]
    n_gap = int((~cat.covered).sum())
    print(f"  {n_gap:,} event(s) fall in a data gap and are excluded -- no "
          f"waveform, so not a miss")
    if args.snr_csv:
        print(f"  measured SNR available for {int(cov.snr.notna().sum()):,} of "
              f"{len(cov):,}; median {cov.snr.median():.2f}, "
              f"{100 * (cov.snr >= 3).mean():.1f}% reach SNR 3")

    # Thresholds chosen to buy a stated alarm budget, not inherited from the benchmark.
    print(f"\n  {'alarms/day':>11}{'threshold':>11}{'actual/day':>12}{'FPR':>10}"
          + "".join(f"{'R(SNR>=' + str(s) + ')':>13}" for s in (3, 5, 10)))
    rows = []
    for target in (100.0, 10.0, 1.0, 0.1):
        want = target * days
        if want >= len(bg):
            continue
        thr = float(np.quantile(bg, 1.0 - want / len(bg)))
        n_alarm = int((bg > thr).sum())
        rec = []
        for s in (3, 5, 10):
            g = cov[cov.snr >= s]
            rec.append((g.best_prob > thr).mean() if len(g) else np.nan)
        rows.append({"target_per_day": target, "threshold": thr,
                     "alarms_per_day": n_alarm / days, "fpr": n_alarm / len(bg),
                     **{f"recall_snr{s}": r for s, r in zip((3, 5, 10), rec)}})
        print(f"  {target:>11.4g}{thr:>11.4f}{n_alarm / days:>12.2f}"
              f"{n_alarm / len(bg):>10.6f}"
              + "".join(f"{r:>13.3f}" if r == r else f"{'-':>13}" for r in rec))

    n05 = int((bg > 0.5).sum())
    rec05 = [(cov[cov.snr >= s].best_prob > 0.5).mean() if len(cov[cov.snr >= s])
             else np.nan for s in (3, 5, 10)]
    print(f"  {'(0.5)':>11}{0.5:>11.4f}{n05 / days:>12.2f}{n05 / len(bg):>10.6f}"
          + "".join(f"{r:>13.3f}" if r == r else f"{'-':>13}" for r in rec05)
          + "   <- the benchmark threshold, for comparison only")
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_thresholds.csv", index=False)

    # Threshold-free: can a real guard be told from a random stretch of record?
    rng = np.random.default_rng(0)
    cand = rng.choice(t, size=min(8000, len(t)), replace=False)
    keep = np.ones(len(cand), bool)
    for c in cat.p_epoch.values:
        keep &= np.abs(cand - c) > (args.guard_pre + args.guard_post + 60)
    fake = []
    for c in cand[keep]:
        i, j = np.searchsorted(t, c - args.guard_pre - win_s), \
               np.searchsorted(t, c + args.guard_post, side="right")
        if j > i:
            fake.append(p[i:j].max())
    fake = np.asarray(fake)
    print(f"\n  event-level separation: max score in a real guard vs in "
          f"{len(fake):,} random ones")
    print(f"    {'SNR cut':>9}{'events':>8}{'AUC':>9}{'med real':>10}{'med random':>12}")
    for s in (0, 2, 3, 5, 10):
        g = cov[(cov.snr >= s) | (np.isnan(cov.snr) & (s == 0))].dropna(subset=["best_prob"])
        if len(g) < 8 or not len(fake):
            continue
        y = np.r_[np.ones(len(g)), np.zeros(len(fake))]
        auc = roc_auc_score(y, np.r_[g.best_prob.values, fake])
        print(f"    {s:>9}{len(g):>8,}{auc:>9.4f}{g.best_prob.median():>10.4f}"
              f"{np.median(fake):>12.4f}")

    # Diurnal cycle of the alarms. Cultural noise is strongly diurnal, so a
    # detector firing on anthropogenic transients shows a working-hours peak
    # that a detector firing on seismicity does not.
    thr10 = float(np.quantile(bg, 1.0 - min(10.0 * days, len(bg) - 1) / len(bg)))
    at = t[(p > thr10) & ~explained]
    if len(at) > 24:
        hod = (pd.to_datetime(at, unit="s").tz_localize("UTC")
               .tz_convert("Europe/Istanbul").hour)
        counts = np.bincount(np.asarray(hod), minlength=24)
        peak, trough = counts.max(), max(counts.min(), 1)
        print(f"\n  unexplained alarms by local hour at {thr10:.4f} "
              f"({len(at):,} alarms, peak/trough = {peak / trough:.2f}x)")
        for h in range(0, 24, 3):
            bar = "#" * int(38 * counts[h] / max(peak, 1))
            print(f"    {h:02d}:00 {counts[h]:>6,} {bar}")
        day = counts[6:20].sum() / 14
        night = (counts[:6].sum() + counts[20:].sum()) / 10
        print(f"    day (06-20) {day:.1f}/h vs night {night:.1f}/h "
              f"= {day / max(night, 1e-9):.2f}x"
              + ("   <- anthropogenic signature" if day > 1.5 * night else ""))

    # Recall by magnitude, on events the station actually recorded.
    mg = cov[cov.snr >= args.snr_min]
    if len(mg) > 20:
        print(f"\n  recall by magnitude at {thr10:.4f} (SNR>={args.snr_min:g} only, "
              f"n={len(mg):,})")
        print(f"    {'band':>12}{'events':>9}{'found':>8}{'recall':>9}{'med dist':>10}")
        for band, g in mg.groupby(pd.cut(mg.Magnitude, [0, 2, 2.5, 3, 3.5, 4, 10]),
                                  observed=True):
            if not len(g):
                continue
            hit = int((g.best_prob > thr10).sum())
            print(f"    {str(band):>12}{len(g):>9,}{hit:>8,}{hit / len(g):>9.3f}"
                  f"{g.dist.median():>10.0f}")

    # Confusion matrices at the budgets a deployment would actually pick.
    for target in (10.0, 1.0):
        want = target * days
        if want >= len(bg):
            continue
        thr = float(np.quantile(bg, 1.0 - want / len(bg)))
        c = confusion(t, p, thr, cat, explained, args, win_s)
        print_confusion(c, thr, days, f"{target:g} alarms/day budget")

    cov.to_csv(f"{args.out_prefix}_events.csv", index=False)
    print(f"\n  wrote {args.out_prefix}_thresholds.csv and "
          f"{args.out_prefix}_events.csv ({len(cov):,} rows)")
