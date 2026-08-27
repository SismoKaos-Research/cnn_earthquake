"""Chaos-feature labelling and aggregation.

One bug here would invalidate every downstream number silently: the chosen cell
has a 6 HOUR horizon, and `seismolib.catalog.label_hours` truncates fractional
days to zero. Calling it with 0.25 returns an all-negative label array with no
error, no warning, and a perfectly plausible shape. `chaos_dataset` uses
`count_events_in_window` instead, and this is what keeps it that way.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting.chaos_dataset import (HORIZON_HOURS, MIN_MAGNITUDE, RADIUS_KM,
                                       persistence_scores)
from seismolib.catalog import count_events_in_window, label_hours


def hours(start, n):
    return pd.date_range(start, periods=n, freq="h")


def times(*stamps):
    return np.array([np.datetime64(s) for s in stamps])


def test_label_hours_would_silently_zero_a_six_hour_horizon():
    """The trap this module exists to route around. If label_hours is ever
    fixed to take fractional days, this test fails and the workaround can go."""
    h = hours("2024-01-01", 24)
    ev = times("2024-01-01T03:00")
    assert label_hours(h, ev, HORIZON_HOURS / 24.0).sum() == 0


def test_count_events_in_window_handles_the_same_horizon_correctly():
    h = hours("2024-01-01", 24)
    ev = times("2024-01-01T03:00")
    got = count_events_in_window(h, ev, HORIZON_HOURS / 24.0, forward=True) > 0
    # Hours 21:00(prev day)..02:00 would see it; within this index, 00,01,02.
    assert got[:3].all() and not got[3:].any()


def test_horizon_is_six_hours_expressed_in_days():
    assert HORIZON_HOURS == 6.0
    assert HORIZON_HOURS / 24.0 == pytest.approx(0.25)


def test_chosen_cell_matches_the_label_sweep():
    """M>=2.5 / 400 km / 6 h, picked on statistical power. Pinned so a silent
    edit cannot decouple the model from the cell that was actually selected."""
    assert (MIN_MAGNITUDE, RADIUS_KM, HORIZON_HOURS) == (2.5, 400.0, 6.0)


def test_persistence_ranks_recent_events_higher():
    """Negated days-since, so a smaller gap scores larger. The floor is scored
    oriented anyway, but the sign should still mean what it says."""
    s = persistence_scores(np.array([0.5, 10.0]))
    assert s[0] > s[1]


def test_persistence_puts_never_seen_rows_at_the_bottom():
    """NaN means no prior qualifying event on record, which is the least
    'recent' a row can be -- not a missing value to impute with the mean."""
    s = persistence_scores(np.array([1.0, np.nan, 5.0]))
    assert s[1] == min(s)
    assert np.isfinite(s).all()


def test_persistence_survives_an_all_nan_column():
    s = persistence_scores(np.array([np.nan, np.nan]))
    assert np.isfinite(s).all()


def test_labels_exclude_an_event_on_the_hour_boundary():
    """(t, t+h]. An event at exactly t is already observable, so labelling it
    as future would let the forecast read its own input."""
    h = hours("2024-01-01", 2)
    on_edge = count_events_in_window(h, times("2024-01-01T00:00"), 0.25, forward=True)
    assert on_edge[0] == 0
