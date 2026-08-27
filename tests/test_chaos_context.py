"""Trailing-context aggregation for the configuration sweep.

The sweep's whole claim is that context length is the parameter every
architecture proposal is really about, so the trailing window has to be exactly
what it says: closed on the right, never reaching across the archive's outage,
and reporting a trend that means what a recurrent layer would be credited with
noticing. A window that leaked one hour forward would manufacture a forecast.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting.chaos_config_sweep import context_features


def hourly(n=200, start="2024-05-01"):
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"a": np.arange(n, dtype=float),
                         "b": np.ones(n)}, index=idx)


def test_context_of_one_hour_is_the_current_hour_only():
    """The already-tested configuration, and the sweep's own control row."""
    h = hourly(10)
    out = context_features(h, 1)
    assert list(out.columns) == ["a_ctxmean", "b_ctxmean"]
    assert out["a_ctxmean"].tolist() == h["a"].tolist()


def test_longer_context_adds_spread_and_trend():
    out = context_features(hourly(50), 24)
    for suffix in ("_ctxmean", "_ctxstd", "_ctxslope"):
        assert f"a{suffix}" in out.columns


def test_window_is_closed_on_the_right_and_never_looks_forward():
    """The failure that would fabricate a forecast: a row must summarise history
    up to and including itself, never the hour after."""
    h = hourly(60)
    out = context_features(h, 6)
    # Rolling mean of a 0..n ramp over 6 closed-right hours ending at t is
    # the mean of t-5..t.
    t = 30
    assert out["a_ctxmean"].iloc[t] == pytest.approx(np.mean(np.arange(t - 5, t + 1)))


def test_slope_recovers_a_known_hourly_trend():
    """A ramp rising 1 per hour has trailing slope 1, whatever the window."""
    out = context_features(hourly(80), 24)
    assert out["a_ctxslope"].iloc[60] == pytest.approx(1.0, rel=1e-6)


def test_slope_is_zero_on_a_flat_series():
    out = context_features(hourly(80), 24)
    assert out["b_ctxslope"].iloc[60] == pytest.approx(0.0, abs=1e-9)


def test_std_is_zero_on_a_flat_series():
    out = context_features(hourly(80), 24)
    assert out["b_ctxstd"].iloc[60] == pytest.approx(0.0, abs=1e-9)


def test_a_gap_does_not_get_bridged_by_the_window():
    """The archive has a 1.9 h outage. A time-based window must span wall-clock
    hours, not row positions, or a window after a gap silently reaches further
    back than it claims."""
    h = hourly(60)
    h.loc[h.index[20:28]] = np.nan          # eight missing hours
    out = context_features(h, 6)
    # Six wall-clock hours after the gap starts, every contributing row is NaN.
    assert np.isnan(out["a_ctxmean"].iloc[24])


def test_early_rows_are_nan_until_the_window_can_fill():
    out = context_features(hourly(80), 24)
    assert np.isnan(out["a_ctxstd"].iloc[0])
    assert np.isfinite(out["a_ctxstd"].iloc[40])


def test_column_count_scales_with_the_statistics_kept():
    h = hourly(60)
    assert context_features(h, 1).shape[1] == h.shape[1]
    assert context_features(h, 24).shape[1] == h.shape[1] * 3


def test_index_is_preserved_so_labels_stay_aligned():
    h = hourly(60)
    assert context_features(h, 24).index.equals(h.index)
    assert context_features(h, 1).index.equals(h.index)
