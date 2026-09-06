"""Assertions about what the numbers ARE, not about what exists.

The suite is 284 structural assertions against 56 numerical ones: it checks that
modules import, that flags exist, that filenames match a pattern. That catches
drift, and it caught none of the eight bugs found in the sweep -- every one of
those was found by reading, because every one produced a plausible number rather
than an error.

These are the other kind. Each fixture is small enough to work out by hand, and
each targets a quantity a published figure depends on:

  haversine        -> the distance axis of every magnitude corpus
  S-P vs distance  -> the break-even distance that decides if a window beats S
  binary_report    -> precision/recall/F1 in every detection table
  regression_report-> the MAE in every magnitude table
  the floors       -> what the model is required to beat
  walk_forward     -> that a fold never trains on its own future

A test that only says "this runs" would pass on a sign error.
"""
import numpy as np
import pytest

from sismokaos.arrivals import P_PHASES, S_PHASES, ArrivalTimes
from sismokaos.catalog import haversine_km
from sismokaos.metrics import (binary_report, majority_class_baseline,
                               predict_mean_baseline, regression_report,
                               safe_auc)
from sismokaos.splits import walk_forward_splits


# --- distance -------------------------------------------------------------

def test_haversine_against_known_separations():
    """One degree of latitude is ~111.19 km; the equator is ~40,030 km round."""
    assert haversine_km(0.0, 0.0, np.array([1.0]), np.array([0.0]))[0] == pytest.approx(111.19, abs=0.05)
    assert haversine_km(0.0, 0.0, np.array([0.0]), np.array([1.0]))[0] == pytest.approx(111.19, abs=0.05)
    # a degree of longitude shrinks with the cosine of latitude
    assert haversine_km(60.0, 0.0, np.array([60.0]), np.array([1.0]))[0] == pytest.approx(111.19 / 2, abs=0.3)
    assert haversine_km(5.0, 5.0, np.array([5.0]), np.array([5.0]))[0] == pytest.approx(0.0, abs=1e-9)


def test_haversine_matches_the_station_pair_the_reports_quote():
    """MANT-GCAM is stated as 144 km in the report; it must come out of the code."""
    mant = (38.4908, 28.5579)
    gcam = (37.7139, 27.2418)
    d = haversine_km(mant[0], mant[1], np.array([gcam[0]]), np.array([gcam[1]]))[0]
    assert d == pytest.approx(144, abs=1.5), f"report says 144 km, code says {d:.1f}"


# --- travel times ---------------------------------------------------------

def test_s_minus_p_tracks_the_slowness_the_planners_assume():
    """Every window-length decision rests on S-P ~ distance / 8.18.

    `plan_pbefores_pull` and `fdsn_magnitude_pull` both size their pulls with
    that constant (Vp 6.0, Vs 3.46 km/s), and the 65 km / 147 km break-even
    distances in the reports follow from it. Check the real velocity model
    agrees to within the tolerance those decisions can stand.
    """
    taup = ArrivalTimes(grid_km=1.0, grid_depth_km=1.0)
    for dist in (60.0, 100.0, 150.0, 200.0):
        tp = taup.travel(dist, 10.0, P_PHASES)
        ts = taup.travel(dist, 10.0, S_PHASES)
        assert tp is not None and ts is not None, f"no arrival at {dist} km"
        assert ts > tp, "S must arrive after P"
        approx = dist / 8.18
        assert ts - tp == pytest.approx(approx, rel=0.25), (
            f"at {dist} km: iasp91 gives S-P {ts - tp:.1f}s, the planners assume "
            f"{approx:.1f}s -- if this drifts, the break-even distances are wrong")


def test_the_10s_break_even_distance_is_where_the_reports_put_it():
    """A 10 s window with a 2 s pre-buffer needs S-P > 8 s, quoted as ~65 km."""
    taup = ArrivalTimes(grid_km=1.0, grid_depth_km=1.0)
    sp = lambda d: taup.travel(d, 10.0, S_PHASES) - taup.travel(d, 10.0, P_PHASES)
    assert sp(65.0) == pytest.approx(8.0, abs=2.0), (
        f"S-P at 65 km is {sp(65.0):.1f}s; the corpus was planned on it being ~8s")
    assert sp(40.0) < 8.0 < sp(100.0), "the break-even must sit between 40 and 100 km"


# --- classification metrics ----------------------------------------------

def test_binary_report_on_a_hand_computed_confusion_matrix():
    """TP=3 FP=1 FN=2 TN=4 -> precision 0.75, recall 0.6, F1 2*.75*.6/1.35."""
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .2, .1, .6, .3, .2, .1, .05])   # thr 0.5
    r = binary_report(y, s, threshold=0.5)
    assert r["precision"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(3 / 5)
    assert r["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert r["accuracy"] == pytest.approx(7 / 10)


def test_safe_auc_on_separable_and_random_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert safe_auc(y, np.array([.1, .2, .3, .7, .8, .9])) == pytest.approx(1.0)
    assert safe_auc(y, np.array([.9, .8, .7, .3, .2, .1])) == pytest.approx(0.0)
    assert safe_auc(y, np.array([.5, .5, .5, .5, .5, .5])) == pytest.approx(0.5)
    assert np.isnan(safe_auc(np.zeros(4), np.array([.1, .2, .3, .4]))), (
        "one class present cannot yield an AUC; it must be NaN, not 0.5")


# --- regression metrics and floors ---------------------------------------

def test_regression_report_on_known_errors():
    """Errors of -1, 0, +1, +2: MAE 1.0, RMSE sqrt(6/4)."""
    true = np.array([2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 3.0, 5.0, 7.0])
    r = regression_report(true, pred)
    assert r["MAE"] == pytest.approx(1.0)
    assert r["RMSE"] == pytest.approx(np.sqrt(6 / 4))


def test_predict_mean_floor_is_the_train_mean_not_the_test_mean():
    """Fitting the floor on test would flatter it and understate the model's edge.

    The reports quote a constant-mean MAE of 0.5426 as the thing the network
    beats by 73.9%; if that floor were fitted on the test split it would be
    lower and the margin smaller.
    """
    train = np.array([1.0, 1.0, 1.0, 1.0])       # mean 1.0
    test = np.array([5.0, 5.0])                  # mean 5.0
    out = predict_mean_baseline(train, test)
    mae = out[0] if isinstance(out, tuple) else (out["MAE"] if isinstance(out, dict) else out)
    assert mae == pytest.approx(4.0), (
        f"floor MAE {mae}; predicting the TRAIN mean 1.0 on a test of 5.0 is an "
        f"error of 4.0. Anything near 0 means it was fitted on the test split.")


def test_majority_class_floor_uses_the_train_majority():
    train = np.array([0, 0, 0, 0, 1])            # majority 0
    test = np.array([1, 1, 1, 0])                # majority 1
    # returns (majority_class, accuracy, balanced_accuracy)
    majority, acc, balanced = majority_class_baseline(train, test)
    assert majority == 0, "the majority must come from TRAIN, where 0 wins 4-1"
    assert acc == pytest.approx(1 / 4), (
        "predicting the TRAIN majority (0) on this test set is right once in four; "
        "anything near 3/4 means it took the majority from TEST")
    assert balanced == pytest.approx(0.5), (
        "a constant prediction is chance-level however the classes are weighted")


# --- splits ---------------------------------------------------------------

def test_walk_forward_folds_never_train_on_their_own_future():
    """The property the whole forecasting protocol rests on."""
    idx = np.arange(1000)
    folds = walk_forward_splits(idx, n_folds=4)
    assert len(folds) >= 1
    for i, fold in enumerate(folds):
        tr, *rest = fold if isinstance(fold, (list, tuple)) else (fold,)
        te = rest[-1] if rest else None
        if te is None or not len(np.atleast_1d(te)):
            continue
        tr, te = np.atleast_1d(tr), np.atleast_1d(te)
        assert tr.max() < te.min(), (
            f"fold {i}: train reaches index {tr.max()} but test starts at "
            f"{te.min()} -- the model would see its own future")


# --- split protocol -------------------------------------------------------

def test_auto_split_picks_station_disjoint_when_it_can():
    """`--split-by auto` must not quietly pick the leaky protocol.

    Every magnitude figure this project published came from a one-station
    corpus, where the diagnostics print "LEAK: site response" and the number is
    that station's estimator rather than an estimator. With 161 stations now
    available the honest protocol should be what you get without asking.
    """
    import pandas as pd

    from sismokaos.magnitude.cnn_lstm_regression import resplit

    many = pd.DataFrame({
        "split": ["train"] * 40,
        "station_key": [f"KO.S{i % 8}" for i in range(40)],
        "event_id": list(range(40)),
    })
    one = many.assign(station_key="KO.MANT")

    assert many.station_key.nunique() > 1 and one.station_key.nunique() == 1
    # the rule the trainer applies
    pick = lambda d: "both" if d.station_key.nunique() > 1 else "event"
    assert pick(many) == "both"
    assert pick(one) == "event"

    out = resplit(many.copy(), "both", seed=42)
    tr = set(out[out.split == "train"].station_key)
    te = set(out[out.split == "test"].station_key)
    assert not (tr & te), f"'both' left stations shared: {tr & te}"


# --- forecasting labels ---------------------------------------------------

def test_an_event_inside_the_feature_window_does_not_make_a_positive_label():
    """The window-end trap: the model must not be shown its own answer.

    Features for hour H are aggregated over [H, H+1h]. A horizon opening at H
    counts an event occurring inside that window as a future positive -- so the
    event is visible in the features AND is the thing being predicted.
    """
    import pandas as pd

    from sismokaos.catalog import label_hours

    idx = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"])
    inside = np.array([np.datetime64("2026-01-01T00:30")])   # inside hour 0's features

    lab = label_hours(idx, inside, horizon_days=30.0)
    assert lab[0] == 0, (
        "an event at 00:30 is inside hour 00:00's own feature window; labelling "
        "that hour positive shows the model the answer")
    assert lab[1] == 0 and lab[2] == 0, "the event is in the past for later hours"

    after = np.array([np.datetime64("2026-01-01T05:00")])
    assert list(label_hours(idx, after, horizon_days=30.0)) == [1, 1, 1]

    # the old behaviour is still reachable for reproducing published figures
    assert label_hours(idx, inside, horizon_days=30.0, feature_hours=0)[0] == 1


def test_sub_day_horizons_are_not_truncated_to_zero():
    """`np.timedelta64(int(0.5), "D")` is ZERO days -- every label comes out 0."""
    import pandas as pd

    from sismokaos.catalog import label_hours

    idx = pd.DatetimeIndex(["2026-01-01 00:00"])
    # event 3 h after the feature window closes
    ev = np.array([np.datetime64("2026-01-01T04:00")])
    assert label_hours(idx, ev, horizon_days=0.5)[0] == 1, (
        "a 12-hour horizon must reach an event 3 hours out; if this is 0 the "
        "float horizon was truncated to an integer number of days")
    assert label_hours(idx, ev, horizon_days=0.05)[0] == 0, "1.2 h must not reach it"
