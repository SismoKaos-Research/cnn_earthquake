"""The waveform path of `seismolib.continuous`, which nothing else exercised.

`test_coincidence.py` covers the span algebra and the alarm matching -- the
parts that read scored `.npz` files. Everything upstream of a score, from the
archive to a standardized block, had no test at all, and that is where the
Phase-1 move broke four functions without turning the suite red.

These are cheap checks on synthetic input. They are not a substitute for the
`verify` subcommand, which pushes real dataset tensors through the real
checkpoints; they exist so that "the scan path still runs" is answered in half
a second rather than by a two-day rescan.
"""
import numpy as np
import pytest

from seismolib import continuous as C


class _Trace:
    """The two attributes `pick_components` reads off an obspy trace."""

    def __init__(self, channel):
        self.stats = type("Stats", (), {"channel": channel})()


def test_components_come_back_in_encoder_order():
    """Z first, then N, then E -- the order the training encoder stacks them."""
    assert C.pick_components([_Trace("HHE"), _Trace("HHN"), _Trace("HHZ")]) == ["Z", "N", "E"]


def test_numeric_horizontals_map_to_their_roles():
    """A station coding its horizontals 1/2 still yields Z, then N, then E."""
    assert C.pick_components([_Trace("HH1"), _Trace("HH2"), _Trace("HHZ")]) == ["Z", "1", "2"]


def test_two_horizontals_and_no_vertical_is_rejected():
    """The trap COMPONENT_ROLES exists for.

    Taking the first three channel codes alphabetically at a station with mixed
    sensor codes gives ['1', '2', 'E'] -- two horizontals and no vertical, which
    the models would happily score as if it were Z/N/E.
    """
    assert C.pick_components([_Trace("HH1"), _Trace("HH2"), _Trace("HHE")]) is None


def test_the_taper_reaches_zero_at_both_ends_and_one_in_the_middle():
    t = C.taper_vector(600)
    assert t.shape == (600,)
    assert t[0] == pytest.approx(0.0) and t[-1] == pytest.approx(0.0)
    assert t[300] == pytest.approx(1.0)
    assert int(600 * 0.05) == 30 and t[30] > 0.99


def test_clean_block_filters_a_block_without_changing_its_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 600)) * 1e3
    out = C.clean_block(x.copy(), 100.0, 1.0, 45.0, C.taper_vector(600))
    assert out.shape == (4, 600)
    assert np.isfinite(out).all()
    # A 1-45 Hz bandpass must remove the DC the detrend did not.
    assert abs(out.mean()) < abs(x.mean()) or abs(out.mean()) < 1e-6


def test_clip_spans_keeps_the_window_grid():
    """A restricted rescan must land on the same lattice as the full scan.

    Window starts are `t0 + k * step`; if a clipped span started fresh at the
    interval edge, its scores could not be compared with the unrestricted run's.
    """
    span = (1000.0, [(0, 0)], 100_000)          # 1000 s at 100 Hz from t=1000
    out = C.clip_spans([span], [(1234.0, 1500.0)], 100.0, 600, 600)
    assert len(out) == 1
    t0, where, n_samp = out[0]
    assert (t0 - 1000.0) % 6.0 == pytest.approx(0.0), t0
    assert t0 >= 1234.0 - 6.0 and where[0][1] % 600 == 0


def test_clip_spans_without_a_restriction_is_the_identity():
    spans = [(0.0, [(0, 0)], 100_000)]
    assert C.clip_spans(spans, None, 100.0, 600, 600) is spans
