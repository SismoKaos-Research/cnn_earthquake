"""The label grid that chose M>=2.5 / 400 km / 6 h.

Cell selection decided what the forecasting work is even trying to predict, so
the two numbers behind it -- the forward positive rate and the persistence floor
-- have to be right. The floor especially: it is `max(auc, 1 - auc)` because a
baseline's sign is free, and under Omori decay the trailing-event predictor is
usually anti-correlated. Dropping that correction turns a floor of 0.57 into a
reported 0.43 and manufactures a gain out of nothing.
"""

import numpy as np
import pandas as pd
import pytest

from sismokaos.forecasting.label_sweep import auc, sweep, sweep_cell, worth_modelling
from sismokaos.metrics import safe_auc

HOUR = 3600


# Epoch seconds, because `sweep` reads event times off a datetime column and
# converts them the same way.
T2024 = int(pd.Timestamp("2024-01-01").value // 10**9)


def windows(n, step=HOUR, t0=0):
    return np.arange(t0, t0 + n * step, step, dtype=np.int64)


def test_fast_auc_matches_the_shared_implementation():
    """`auc` is a hand-rolled rank formula for speed; it must not be a second
    definition of the statistic."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        y = rng.integers(0, 2, 60)
        s = rng.normal(size=60)
        if len(np.unique(y)) < 2:
            continue
        assert auc(y, s) == pytest.approx(safe_auc(y, s))


def test_fast_auc_handles_ties_like_sklearn():
    y = np.array([0, 0, 1, 1])
    s = np.array([1.0, 1.0, 1.0, 1.0])
    assert auc(y, s) == pytest.approx(safe_auc(y, s))


def test_fast_auc_is_nan_on_a_single_class():
    assert np.isnan(auc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])))


def test_forward_label_excludes_an_event_on_the_window_end():
    """The label is 'any event in (t, t + horizon]'. An event at exactly t is
    already in the window the model is reading."""
    wt = np.array([0, HOUR], dtype=np.int64)
    on_edge = sweep_cell(np.array([0], dtype=np.int64), wt, HOUR)
    assert on_edge["pos_rate"] == 0.0

    just_after = sweep_cell(np.array([1], dtype=np.int64), wt, HOUR)
    assert just_after["pos_rate"] == pytest.approx(0.5)   # labels window 0 only


def test_forward_label_includes_the_far_edge():
    wt = np.array([0], dtype=np.int64)
    assert sweep_cell(np.array([HOUR], dtype=np.int64), wt, HOUR)["pos_rate"] == 1.0
    assert sweep_cell(np.array([HOUR + 1], dtype=np.int64), wt, HOUR)["pos_rate"] == 0.0


def test_floor_is_orientation_corrected():
    """An anti-correlated persistence predictor sets the flipped bar. Without
    the correction the floor is reported below chance, which no baseline is."""
    rng = np.random.default_rng(3)
    wt = windows(400)
    et = np.sort(rng.integers(0, 400 * HOUR, 40).astype(np.int64))
    cell = sweep_cell(et, wt, 6 * HOUR)
    assert cell["floor"] >= 0.5
    assert cell["floor"] == pytest.approx(max(cell["pers_auc"], 1 - cell["pers_auc"]))


def test_headroom_is_what_is_left_above_the_floor():
    cell = sweep_cell(np.array([5 * HOUR], dtype=np.int64), windows(50), 6 * HOUR)
    assert cell["headroom"] == pytest.approx(1 - cell["floor"])


def test_a_cell_with_no_events_is_degenerate_not_an_exception():
    cell = sweep_cell(np.array([], dtype=np.int64), windows(10), HOUR)
    assert cell["events"] == 0 and cell["pos_rate"] == 0.0
    assert np.isnan(cell["pers_auc"]) and np.isnan(cell["floor"])


def test_event_count_covers_the_span_plus_one_horizon():
    """Events after the last window still create labels, so they count."""
    wt = windows(10)                       # 0 .. 9h
    late = np.array([12 * HOUR], dtype=np.int64)
    assert sweep_cell(late, wt, 6 * HOUR)["events"] == 1
    assert sweep_cell(late, wt, 1 * HOUR)["events"] == 0


def test_sweep_covers_the_whole_grid():
    cat = pd.DataFrame({
        "t": pd.to_datetime(["2024-01-01 05:00", "2024-01-02 05:00"]),
        "Magnitude": [2.6, 4.6],
        "dist_km": [80.0, 300.0],
    })
    df = sweep(cat, windows(72, t0=T2024), magnitudes=(2.5, 4.5),
               radii_km=(100, 400), horizons_h=(1, 6))
    assert len(df) == 2 * 2 * 2
    assert set(df.columns) >= {"Mmin", "radius", "horizon_h", "events", "pos_rate",
                               "pers_auc", "floor", "headroom"}


def test_sweep_magnitude_and_radius_filters_are_inclusive_lower_bounds():
    cat = pd.DataFrame({
        "t": pd.to_datetime(["2024-01-01 05:00"]),
        "Magnitude": [2.5],
        "dist_km": [100.0],
    })
    df = sweep(cat, windows(72, t0=T2024), magnitudes=(2.5,), radii_km=(100,),
               horizons_h=(24,))
    assert df.events.iloc[0] == 1


def test_worth_modelling_rejects_the_three_ways_a_cell_is_useless():
    df = pd.DataFrame([
        dict(events=100, pos_rate=0.30, floor=0.60),   # keep
        dict(events=3,   pos_rate=0.30, floor=0.60),   # too few events
        dict(events=100, pos_rate=0.99, floor=0.60),   # saturated
        dict(events=100, pos_rate=0.30, floor=0.95),   # no headroom
    ])
    assert worth_modelling(df).index.tolist() == [0]
