"""Catalogue geometry and labelling windows.

The interesting content here is boundary semantics. An event landing exactly on
an hour boundary must count as past, not future -- a forward label that includes
`t` is a label the model can read off its own input, which is leakage rather
than forecasting. Those asymmetries are invisible in normal use and silent when
broken, so they are pinned here.
"""

import numpy as np
import pandas as pd
import pytest

from sismokaos.catalog import (count_events_in_window, days_since_prev_major,
                               days_until_next_major, haversine_km, label_hours,
                               parse_hour_start, station_distance_mask)

DEG_KM = 111.195  # one degree of latitude on a 6371 km sphere


def hours(start, n):
    return pd.date_range(start, periods=n, freq="h")


def times(*stamps):
    return np.array([np.datetime64(s) for s in stamps])


def test_haversine_zero_distance():
    assert haversine_km(37.0, 27.0, np.array([37.0]), np.array([27.0]))[0] == pytest.approx(0.0)


def test_haversine_one_degree_of_latitude():
    d = haversine_km(0.0, 0.0, np.array([1.0]), np.array([0.0]))[0]
    assert d == pytest.approx(DEG_KM, rel=1e-3)


def test_haversine_is_symmetric():
    a = haversine_km(37.06, 27.31, np.array([39.0]), np.array([29.0]))[0]
    b = haversine_km(39.0, 29.0, np.array([37.06]), np.array([27.31]))[0]
    assert a == pytest.approx(b)


def test_haversine_longitude_shrinks_with_latitude():
    """A degree of longitude is ~cos(lat) of a degree at the equator."""
    eq = haversine_km(0.0, 0.0, np.array([0.0]), np.array([1.0]))[0]
    hi = haversine_km(60.0, 0.0, np.array([60.0]), np.array([1.0]))[0]
    assert hi == pytest.approx(eq * 0.5, rel=1e-2)


def test_station_mask_takes_the_nearest_station():
    lats, lons = np.array([37.0622]), np.array([27.3103])   # exactly on BODT
    assert station_distance_mask(lats, lons, ["DAT", "BODT"], 10.0).tolist() == [True]
    assert station_distance_mask(lats, lons, ["DAT"], 10.0).tolist() == [False]


def test_station_mask_disabled_by_a_nonpositive_radius():
    lats, lons = np.array([0.0, 45.0]), np.array([0.0, 90.0])
    assert station_distance_mask(lats, lons, ["BODT"], 0).all()
    assert station_distance_mask(lats, lons, ["BODT"], None).all()


def test_station_mask_accepts_explicit_coordinates():
    lats, lons = np.array([10.0]), np.array([10.0])
    assert station_distance_mask(lats, lons, [(10.0, 10.0)], 1.0).tolist() == [True]


@pytest.mark.parametrize("wid,expect_hour,expect_idx", [
    ("2024_11_15_00_w01", pd.Timestamp("2024-11-15 00:00"), 1),
    ("2024_05_01_23_w72", pd.Timestamp("2024-05-01 23:00"), 72),
])
def test_parse_hour_start(wid, expect_hour, expect_idx):
    h, i = parse_hour_start(wid)
    assert h == expect_hour and i == expect_idx


@pytest.mark.parametrize("bad", ["", "not-an-id", "24_11_15_00_w01", "2024-11-15-00-w01"])
def test_parse_hour_start_rejects_malformed_ids(bad):
    assert parse_hour_start(bad) == (None, None)


def test_forward_window_excludes_an_event_on_the_boundary():
    """(t, t+w]. An event at exactly t is already observable, so counting it as
    future would let a forecast label read its own input."""
    h = hours("2024-01-01", 1)
    assert count_events_in_window(h, times("2024-01-01T00:00"), 1.0, forward=True)[0] == 0
    assert count_events_in_window(h, times("2024-01-01T00:01"), 1.0, forward=True)[0] == 1


def test_forward_window_includes_the_far_edge():
    h = hours("2024-01-01", 1)
    assert count_events_in_window(h, times("2024-01-02T00:00"), 1.0, forward=True)[0] == 1
    assert count_events_in_window(h, times("2024-01-02T00:01"), 1.0, forward=True)[0] == 0


def test_trailing_window_includes_an_event_on_the_boundary():
    """(t-w, t] -- the mirror of the forward rule, so an event at t is counted
    exactly once across the two windows, never zero or twice."""
    h = hours("2024-01-01", 1)
    assert count_events_in_window(h, times("2024-01-01T00:00"), 1.0, forward=False)[0] == 1


def test_every_event_is_counted_exactly_once_across_the_two_windows():
    h = hours("2024-01-05", 6)
    ev = times("2024-01-05T00:00", "2024-01-05T03:30", "2024-01-05T05:00")
    fwd = count_events_in_window(h, ev, 1.0, forward=True)
    back = count_events_in_window(h, ev, 1.0, forward=False)
    assert (fwd + back == 3).all()


def test_label_hours_is_the_forward_count_shifted_past_the_feature_window():
    """They no longer agree, and that is the point.

    `count_events_in_window(forward=True)` counts (t, t+w]. `label_hours` counts
    (t+1h, t+1h+w], because the features at t cover [t, t+1h] and an event in
    there is visible to the model. The two are the same quantity offset by one
    hour, which is what this now pins -- an equality test would have quietly
    reverted the fix.
    """
    h = hours("2024-01-01", 48)
    ev = times("2024-01-01T12:00", "2024-01-02T06:00")
    shifted = h + pd.Timedelta(hours=1)
    assert (label_hours(h, ev, 1)
            == (count_events_in_window(shifted, ev, 1.0, forward=True) > 0)).all()
    assert label_hours(h, times("2024-01-01T00:30"), 1)[0] == 0, (
        "an event inside hour 0's own feature window must not label it positive")


def test_days_since_prev_is_strictly_before_and_until_is_inclusive():
    """The two are deliberately asymmetric at t: an event at exactly t is
    'now', so distance-to-next is 0 while distance-since-previous is unknown."""
    h = hours("2024-01-02", 1)
    ev = times("2024-01-02T00:00")
    assert np.isnan(days_since_prev_major(h, ev)[0])
    assert days_until_next_major(h, ev)[0] == pytest.approx(0.0)


def test_days_since_prev_measures_in_days():
    h = hours("2024-01-03", 1)
    assert days_since_prev_major(h, times("2024-01-01T00:00"))[0] == pytest.approx(2.0)


def test_days_until_next_is_nan_past_the_last_event():
    h = hours("2024-01-05", 1)
    assert np.isnan(days_until_next_major(h, times("2024-01-01T00:00"))[0])


def test_days_since_prev_uses_the_most_recent_event():
    h = hours("2024-01-10", 1)
    ev = times("2024-01-01T00:00", "2024-01-09T00:00")
    assert days_since_prev_major(h, ev)[0] == pytest.approx(1.0)
