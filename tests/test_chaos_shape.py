"""Within-hour shape statistics.

These are the cheap stand-in for a CNN encoder over the 50 s stream: if the
trajectory inside an hour carries association with the label, a slope and a
half-difference should show some of it. That argument only holds if the
statistics actually measure what they claim, so each is checked against a signal
whose answer is known by construction.

The one that would fail quietly is `slope`. It is scaled by window count on
purpose -- hours do not all contain 72 windows, and an unscaled least-squares
coefficient would make a short hour look flatter than a full one for reasons
that have nothing to do with seismology.
"""

import numpy as np
import pytest

from sismokaos.forecasting.chaos_dataset import SHAPE_AGGS, _shape_block

N = 72          # windows per hour at 200 s / 50 s step
F = 3           # features


def block(sig):
    """Runs one (n, F) matrix through and splits the four statistics out."""
    out = _shape_block(np.asarray(sig, dtype=float))
    return dict(zip(SHAPE_AGGS, np.split(out, len(SHAPE_AGGS))))


def test_slope_recovers_a_known_ramp():
    """t is centred and scaled to [-0.5, 0.5], so a ramp spanning `d` over the
    hour has slope `d` regardless of how many windows the hour held."""
    for d in (1.0, -3.0):
        ramp = np.linspace(0.0, d, N)[:, None] * np.ones(F)
        assert block(ramp)["slope"] == pytest.approx(np.full(F, d), rel=1e-6)


def test_slope_is_invariant_to_window_count():
    """The whole point of the scaling: a short hour must not read as flatter."""
    a = block(np.linspace(0, 2, 72)[:, None] * np.ones(F))["slope"]
    b = block(np.linspace(0, 2, 40)[:, None] * np.ones(F))["slope"]
    assert a == pytest.approx(b, rel=1e-6)


def test_slope_is_zero_on_a_flat_hour():
    assert block(np.ones((N, F)))["slope"] == pytest.approx(np.zeros(F), abs=1e-9)


def test_halfdiff_recovers_a_step():
    sig = np.zeros((N, F))
    sig[N // 2:] = 5.0
    assert block(sig)["halfdiff"] == pytest.approx(np.full(F, 5.0))


def test_halfdiff_is_zero_on_a_symmetric_spike():
    """A spike in the middle moves no mass between halves -- which is exactly
    why halfdiff alone is not enough and argmax is carried too."""
    sig = np.zeros((N, F))
    sig[N // 2] = 10.0
    assert abs(block(sig)["halfdiff"]).max() < 10.0 / (N // 2) + 1e-9


def test_argmax_locates_the_peak_as_a_fraction_of_the_hour():
    for pos in (0, N // 2, N - 1):
        sig = np.zeros((N, F))
        sig[pos] = 1.0
        assert block(sig)["argmax"] == pytest.approx(np.full(F, pos / (N - 1)))


def test_ac1_is_high_for_a_smooth_trajectory_and_low_for_noise():
    smooth = np.sin(np.linspace(0, 2 * np.pi, N))[:, None] * np.ones(F)
    noise = np.random.default_rng(0).normal(size=(N, F))
    assert block(smooth)["ac1"].min() > 0.9
    assert abs(block(noise)["ac1"]).max() < 0.5


def test_ac1_is_negative_for_an_alternating_series():
    alt = (np.arange(N) % 2 * 2.0 - 1.0)[:, None] * np.ones(F)
    assert block(alt)["ac1"].max() < -0.9


def test_features_do_not_bleed_into_each_other():
    sig = np.zeros((N, F))
    sig[:, 0] = np.linspace(0, 4, N)
    got = block(sig)["slope"]
    assert got[0] == pytest.approx(4.0, rel=1e-6)
    assert got[1:] == pytest.approx(np.zeros(F - 1), abs=1e-9)


def test_missing_samples_are_filled_rather_than_poisoning_the_hour():
    """0.04% of the stream is NaN. One absent window must not NaN a whole hour."""
    sig = np.linspace(0, 2, N)[:, None] * np.ones(F)
    sig[5, 1] = np.nan
    got = block(sig)
    assert np.isfinite(got["slope"]).all()
    assert got["slope"][0] == pytest.approx(2.0, rel=1e-6)


def test_a_too_short_hour_returns_nan_rather_than_a_fabricated_shape():
    got = block(np.ones((3, F)))
    assert all(np.isnan(v).all() for v in got.values())


def test_output_length_matches_the_declared_statistics():
    assert len(_shape_block(np.ones((N, F)))) == len(SHAPE_AGGS) * F


# --- cross-hour lags -------------------------------------------------------

import pandas as pd

from sismokaos.forecasting.chaos_dataset import LAG_HOURS, add_lags


def frame(n=48, start="2024-05-01"):
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"a": np.arange(n, dtype=float),
                         "b": np.arange(n, dtype=float) * -2.0}, index=idx)


def test_lag_and_delta_columns_exist_for_every_horizon():
    out = add_lags(frame(), base=["a"])
    for h in LAG_HOURS:
        assert f"a_lag{h}" in out and f"a_d{h}" in out
    assert "b_lag1" not in out          # only the requested base columns


def test_lag_looks_backward_not_forward():
    """A lag that reached forward would hand the model its own future."""
    out = add_lags(frame(), base=["a"])
    assert out["a_lag1"].iloc[10] == 9.0
    assert out["a_lag24"].iloc[30] == 6.0


def test_delta_is_current_minus_lagged():
    out = add_lags(frame(), base=["a"])
    assert out["a_d3"].iloc[10] == pytest.approx(3.0)


def test_first_rows_are_nan_rather_than_wrapped():
    out = add_lags(frame(), base=["a"])
    assert out["a_lag24"].iloc[:24].isna().all()
    assert out["a_lag24"].iloc[24:].notna().all()


def test_a_gap_in_the_archive_does_not_get_bridged():
    """The archive has an outage. shift() on a frame with missing hours would
    treat the row before a gap as one hour earlier than it is."""
    f = frame(10)
    f = f.drop(f.index[4:7])            # three missing hours
    out = add_lags(f, base=["a"])
    assert out["a_lag1"].loc[f.index[4]] != f["a"].iloc[3]
    assert np.isnan(out["a_lag1"].loc[f.index[4]])


def test_row_index_is_preserved():
    f = frame(30)
    out = add_lags(f, base=["a"])
    assert out.index.equals(f.index)
