"""Requiring two stations to agree, and pricing what that costs."""
import pathlib
import sys

import numpy as np
import pandas as pd

from seismolib.catalog import haversine_km as haversine
from seismolib.continuous.alarms import confirmed, declarations, load_scores
from seismolib.continuous.association import (background_and_guards, load_snr,
                                              predicted_arrivals)
from seismolib.continuous.spans import coverage_spans, in_spans, intersect_spans

NAME = "coincidence"
HELP = "require two stations to agree, and price what that costs"


def add_args(q):
    q.add_argument("--scores-a", required=True, help="glob of station A's .npz files")
    q.add_argument("--station-a", required=True)
    q.add_argument("--scores-b", required=True, help="glob of station B's .npz files")
    q.add_argument("--station-b", required=True)
    q.add_argument("--stations-csv", required=True)
    q.add_argument("--catalog", required=True)
    q.add_argument("--window-seconds", type=float, required=True,
                   help="the arm's window length; must be the same arm at both "
                        "stations or the two alarm streams are not comparable")
    q.add_argument("--coincidence-seconds", type=float, default=None,
                   help="how far apart two declarations may be and still count "
                        "as one. Defaults to the separation divided by Vp, which "
                        "is the largest P-arrival difference any event can "
                        "produce at this pair; smaller loses real events on the "
                        "line through both stations.")
    q.add_argument("--vp", type=float, default=6.0,
                   help="crustal Vp used for the default coincidence window")
    q.add_argument("--snr-csv-a", default=None,
                   help="station_detection_range.py output for A")
    q.add_argument("--snr-csv-b", default=None, help="the same for B")
    q.add_argument("--snr-min", type=float, default=3.0)
    q.add_argument("--max-distance", type=float, default=500.0)
    q.add_argument("--guard-pre", type=float, default=10.0)
    q.add_argument("--guard-post", type=float, default=60.0)
    q.add_argument("--signal-post", type=float, default=20.0)
    q.add_argument("--cluster-seconds", type=float, default=60.0)
    q.add_argument("--out-prefix", required=True)


def run(args):
    """What requiring two stations to agree costs, and what it buys.

    Single-station continuous detection is dominated by false alarms: at the
    thresholds this detector needs to keep any recall, MANT alone declares tens
    of times a day. Requiring a second station to agree within the time an
    event's P wave could plausibly take to cross the pair is the standard
    network answer, and the reduction it delivers is usually quoted from an
    independence assumption.

    **That assumption is the thing worth measuring.** Two stations 130 km apart
    share weather, share the regional noise field, and share whatever diurnal
    cultural signal drives the day/night ratio already measured here. To the
    extent their false alarms are common-mode, the reduction is smaller than
    independence predicts -- and no amount of arithmetic can say by how much.
    So this reports the measured joint rate against the independent prediction,
    and their ratio.

    Two things it refuses to do:

    **It scores only the span both stations recorded.** Their coverage is
    intersected first. Counting an unconfirmed alarm as suppressed while the
    other station was simply off the air would read as a large gain and be
    nothing but missing data.

    **It asks recall only of events both stations actually recorded.** An event
    below SNR at either station cannot be confirmed by a network rule, and
    charging the rule for it measures the catalogue's reach, not the method.
    """
    ta_all, pa_all = load_scores(args.scores_a)
    tb_all, pb_all = load_scores(args.scores_b)
    win_s = args.window_seconds
    step = float(np.median(np.diff(ta_all[:100000])))

    st = pd.read_csv(args.stations_csv, encoding="utf-8-sig")
    st.columns = [c.strip() for c in st.columns]
    A = st[st.Code == args.station_a].iloc[0]
    B = st[st.Code == args.station_b].iloc[0]
    sep = float(haversine(float(A.Latitude), float(A.Longitude),
                          np.array([float(B.Latitude)]), np.array([float(B.Longitude)]))[0])
    w = args.coincidence_seconds
    if w is None:
        w = sep / args.vp
    print(f"{'=' * 78}\nTWO-STATION COINCIDENCE  --  {args.station_a} + "
          f"{args.station_b}  ({win_s:g}s windows)\n{'=' * 78}")
    print(f"  separation {sep:.0f} km -> coincidence window +/-{w:.1f} s "
          f"({'default: separation / Vp ' + format(args.vp, 'g') if args.coincidence_seconds is None else 'given'})",
          flush=True)

    # --- the span both stations recorded ----------------------------------
    spans = intersect_spans(coverage_spans(ta_all, step), coverage_spans(tb_all, step))
    joint_s = sum(hi - lo for lo, hi in spans)
    days = joint_s / 86400.0
    ka, kb = in_spans(ta_all, spans), in_spans(tb_all, spans)
    ta, pa, tb, pb = ta_all[ka], pa_all[ka], tb_all[kb], pb_all[kb]
    print(f"  {args.station_a}: {len(ta_all) * step / 86400:.1f} d scored, "
          f"{args.station_b}: {len(tb_all) * step / 86400:.1f} d scored, "
          f"both at once: {days:.1f} d in {len(spans)} span(s)")
    if days < 1:
        sys.exit("the two stations barely overlap; nothing to measure")

    # --- catalogue, at each station separately ----------------------------
    # Keyed "a"/"b", not by station name: passing the same station twice is the
    # obvious self-test, and a name-keyed dict silently collapses to one entry
    # for it -- the background of one station overwrites the other's and the
    # threshold table comes out empty.
    cats = {}
    for side, name, tt, snr_csv in (("a", args.station_a, ta, args.snr_csv_a),
                                    ("b", args.station_b, tb, args.snr_csv_b)):
        cat, _ = predicted_arrivals(
            name, args.stations_csv, args.catalog, args.max_distance)
        # `in_spans`, not a comprehension over `spans`: a gap-split archive has
        # tens of thousands of them (MANT's pnat scores have 43,215), and one
        # Python-level pass per event over all of them is hours rather than
        # seconds. p_epoch is sorted, which is what lets the searchsorted
        # version be used here.
        cat = cat[in_spans(cat.p_epoch.values, spans)].copy()
        if snr_csv:
            cat = cat.merge(load_snr(snr_csv), left_on="EventID",
                            right_on="event_id", how="left")
        else:
            cat["snr"] = np.nan
        cats[side] = cat.drop_duplicates(subset="EventID")
    both = cats["a"].merge(cats["b"][["EventID", "snr", "p_epoch"]],
                           on="EventID", suffixes=("_a", "_b"))
    good = both[(both.snr_a >= args.snr_min) & (both.snr_b >= args.snr_min)]
    print(f"  {len(both):,} catalogued event(s) in that span; "
          f"{len(good):,} reach SNR {args.snr_min:g} at BOTH stations")
    if len(good):
        dp = (good.p_epoch_b - good.p_epoch_a).abs()
        print(f"  their |P_A - P_B| spans {dp.min():.1f}..{dp.max():.1f} s "
              f"(median {dp.median():.1f}) -- the window must cover this")

    # --- background at each station ---------------------------------------
    # The guard mask is kept, not just the background scores. Declarations have
    # to be counted on UNEXPLAINED windows only: a catalogued earthquake is
    # detected at both stations by construction, so leaving real events in the
    # streams makes every one of them a guaranteed coincidence and the "excess"
    # then measures how many events the span contains rather than how much the
    # two stations' false alarms agree. On MANT+DEMI that is 11.6 catalogued
    # events per day at SNR>=3 against a measured 3.97 coincidences per day --
    # enough to account for all of them.
    bg, unexplained = {}, {}
    for side, name, tt, pp in (("a", args.station_a, ta, pa),
                               ("b", args.station_b, tb, pb)):
        explained, _ = background_and_guards(tt, pp, cats[side], win_s, args.guard_pre, args.guard_post)
        bg[side] = pp[~explained]
        unexplained[side] = ~explained

    # --- the table ---------------------------------------------------------
    print(f"\n  Each station is thresholded to the SAME alarm budget, not the same")
    print(f"  threshold: their backgrounds differ and a shared number would not")
    print(f"  mean the same thing at both.\n")
    print(f"  Alarm rates below count UNEXPLAINED declarations only -- windows")
    print(f"  overlapping a catalogued event's guard are removed from both")
    print(f"  streams first, since a real earthquake is seen at both stations by")
    print(f"  construction and would otherwise be counted as agreement.\n")
    print(f"  {'budget/day':>11}{'thr ' + args.station_a:>12}{'thr ' + args.station_b:>12}"
          f"{'A/day':>9}{'B/day':>9}{'2of2/day':>10}{'if indep':>10}{'excess':>8}"
          f"{'recall':>9}")
    rows = []
    for target in (100.0, 30.0, 10.0, 3.0, 1.0, 0.1):
        want = target * days
        if any(want >= len(bg[s]) for s in ("a", "b")):
            continue
        thr = {s: float(np.quantile(bg[s], 1.0 - want / len(bg[s])))
               for s in ("a", "b")}
        ua, ub = unexplained["a"], unexplained["b"]
        da_t, _ = declarations(ta[ua], pa[ua], thr["a"], args.cluster_seconds)
        db_t, _ = declarations(tb[ub], pb[ub], thr["b"], args.cluster_seconds)
        ok = confirmed(da_t, db_t, w)
        n_a, n_b, n_2 = len(da_t), len(db_t), int(ok.sum())
        # Independent Poisson streams of rate ra, rb coincide within +/-w at
        # rate ra * rb * 2w per unit time. This is the number the "1.78% ->
        # 0.03%" style estimate assumes; the measured one is next to it.
        ra, rb = n_a / joint_s, n_b / joint_s
        # What two INDEPENDENT streams of these rates would produce. The
        # measured quantity is "A declarations having at least one B within
        # +/-w", so the prediction must be for that and not for the number of
        # coincident pairs: a Poisson B stream puts 1 - exp(-rb*2w) of them in
        # the window, which is below rb*2w whenever B is busy. The two agree to
        # 0.25% at 10 alarms/day and diverge by 10% at 200, so the distinction
        # only matters at the loose end of this table -- which is exactly where
        # the reduction looks most impressive.
        indep = ra * (1.0 - np.exp(-rb * 2 * w)) * 86400
        rec = np.nan
        if len(good):
            fired_a = np.array([((pa[np.searchsorted(ta, c - win_s):
                                     np.searchsorted(ta, c + args.signal_post,
                                                     side="right")] > thr["a"]).any())
                                for c in good.p_epoch_a.values])
            fired_b = np.array([((pb[np.searchsorted(tb, c - win_s):
                                     np.searchsorted(tb, c + args.signal_post,
                                                     side="right")] > thr["b"]).any())
                                for c in good.p_epoch_b.values])
            rec = float((fired_a & fired_b).mean())
        excess = n_2 / days / indep if indep > 0 else np.nan
        rows.append({"budget_per_day": target,
                     "station_a": args.station_a, "station_b": args.station_b,
                     "thr_a": thr["a"], "thr_b": thr["b"],
                     "a_per_day": n_a / days, "b_per_day": n_b / days,
                     "both_per_day": n_2 / days, "independent_per_day": indep,
                     "excess_over_independent": excess, "recall_both": rec})
        print(f"  {target:>11.4g}{thr['a']:>12.4f}{thr['b']:>12.4f}"
              f"{n_a / days:>9.2f}{n_b / days:>9.2f}{n_2 / days:>10.3f}"
              f"{indep:>10.4f}{excess:>8.1f}x"
              + (f"{rec:>9.3f}" if rec == rec else f"{'-':>9}"))

    out = pathlib.Path(f"{args.out_prefix}_coincidence.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  wrote {out}")
    print(f"\n  `excess` is the measured two-station rate divided by what two")
    print(f"  independent alarm streams of the same rates would produce. 1.0x")
    print(f"  means the stations' false alarms are independent and the textbook")
    print(f"  reduction holds; above 1.0x they share a cause and the network")
    print(f"  rule buys less than the arithmetic promises.")
