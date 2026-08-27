"""Rebuilding our windows for someone else's model.

The comparison this script exists for is only worth anything if the waveform
handed to the pretrained picker is the same waveform our model was scored on,
positioned where that picker expects it. Two things decide that, and both are
silent when wrong: the sample arithmetic that locates a window inside its
source record, and which end the padding goes on.

Getting the padding backwards moves P from sample 200 to sample 260 while
producing an array of exactly the right shape, a valid softmax, and a plausible
AUC. Nothing raises. So it is pinned here.
"""

import numpy as np
import pytest

from detection.pretrained_picker_baseline import (NOISE_STEP, P_SAMPLE,
                                                  WINDOW_SAMPLES, prepare,
                                                  window_start)

IN_SAMPLES = 400


def wave(n=1, samples=WINDOW_SAMPLES):
    """A 10 Hz tone per component, at three distinct amplitudes.

    In the passband on purpose: a ramp or a step is exactly what the 2 Hz
    highpass removes, so testing padding placement on one would compare
    numerical noise against numerical noise and pass no matter which end the
    padding went on.
    """
    t = np.arange(samples, dtype=np.float64) / 100.0
    return np.stack([np.stack([(c + 1) * np.sin(2 * np.pi * 10 * t)
                               for c in range(3)]) for _ in range(n)])


def test_event_windows_start_at_the_record_start():
    """One window per station, cut by arrival_from_catalog.py as win000."""
    assert window_start("event_209452_raw_KO.RKY_win000.pt", is_event=True) == 0


def test_noise_window_index_is_a_50_percent_overlap_stride():
    assert window_start("noise_event_209836_raw_6G.KMRM_win000.pt", False) == 0
    assert window_start("noise_event_209836_raw_6G.KMRM_win001.pt", False) == NOISE_STEP
    assert window_start("noise_event_209836_raw_6G.KMRM_win027.pt", False) == 27 * NOISE_STEP


def test_noise_stride_is_half_the_window():
    """--overlap 0.5 on a 3.4 s window. If either changes, the sample range
    recovered from the filename silently points at the wrong data."""
    assert NOISE_STEP * 2 == WINDOW_SAMPLES


def test_prepare_returns_the_models_input_length():
    x = prepare(wave(2), "gpd", "edge", 0, IN_SAMPLES)
    assert x.shape == (2, 3, IN_SAMPLES)


def test_padding_goes_on_the_end_so_p_stays_on_the_prediction_sample():
    """P sits at sample 200 of 340, and GPD predicts at sample 200 of 400.
    That only holds if every padded sample is added after the window."""
    x = prepare(wave(), "gpd", "edge", 0, IN_SAMPLES)[0]
    for c in range(3):
        assert np.allclose(x[c, WINDOW_SAMPLES:], x[c, WINDOW_SAMPLES - 1])


def test_front_pad_shifts_the_arrival_off_the_prediction_sample():
    """The sensitivity knob has to actually move the window, or a null result
    from it would be meaningless."""
    k = 40
    x = prepare(wave(), "gpd", "edge", k, IN_SAMPLES)[0]
    for c in range(3):
        assert np.allclose(x[c, :k], x[c, k])
        assert np.allclose(x[c, k + WINDOW_SAMPLES:], x[c, k + WINDOW_SAMPLES - 1])


def test_p_sample_matches_gpd_prediction_sample():
    """2.0 s pre-arrival at 100 Hz. Stated as a constant because the whole
    alignment argument rests on it."""
    assert P_SAMPLE == 200
    assert P_SAMPLE < WINDOW_SAMPLES


def test_zero_padding_is_flat_and_reflect_is_not():
    z = prepare(wave(), "gpd", "zero", 0, IN_SAMPLES)[0]
    r = prepare(wave(), "gpd", "reflect", 0, IN_SAMPLES)[0]
    assert np.allclose(np.diff(z[0, WINDOW_SAMPLES:]), 0.0)
    assert not np.allclose(np.diff(r[0, WINDOW_SAMPLES:]), 0.0)


def test_every_trace_is_demeaned():
    """GPD's own annotate_batch_pre. Applied after padding, so the padded
    samples count toward the mean -- which is why it is checked on the output."""
    x = prepare(wave(4), "gpd", "edge", 0, IN_SAMPLES)
    assert np.allclose(x.mean(-1), 0.0, atol=1e-8)


def test_components_keep_their_order_and_relative_scale():
    """ZNE order is load-bearing -- the generator's _COMPONENT_ROLES and GPD's
    component_order agree on it, and nothing downstream would notice a swap.
    Relative amplitude has to survive too: GPD max-normalises the block
    globally, so between-component ratios are information it reads."""
    x = prepare(wave(), "gpd", "edge", 0, IN_SAMPLES)[0]
    spread = np.array([x[c].std() for c in range(3)])
    assert (np.diff(spread) > 0).all()
    assert spread[1] / spread[0] == pytest.approx(2.0, rel=1e-3)
    assert spread[2] / spread[0] == pytest.approx(3.0, rel=1e-3)


def test_front_pad_larger_than_the_shortfall_is_refused():
    with pytest.raises(SystemExit):
        prepare(wave(), "gpd", "edge", 200, IN_SAMPLES)


@pytest.mark.parametrize("mode", ["gpd", "pipeline"])
def test_both_preprocessing_paths_produce_finite_output(mode):
    x = prepare(wave(2), mode, "edge", 0, IN_SAMPLES)
    assert np.isfinite(x).all()
