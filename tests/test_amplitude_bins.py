"""Within-amplitude binning -- the guard on the morphology claim.

The claim "the detector reads waveform shape, not loudness" rests on
discrimination surviving inside a bin where amplitude barely varies. Whether a
bin qualifies is decided by its WIDTH, and equal-count bins on a heavy-tailed
corpus are wildly unequal in width: the top decile here spans ~530x. Reading a
near-chance AUC in a 508x-wide bin as evidence about SNR is a mistake I have
actually made; the width tag is what stops it, so it is tested.
"""

import numpy as np
import pytest

from detection.within_amplitude_auc import NARROW_RATIO, amplitude_bins


def alternating(n):
    """Labels that keep every bin two-class, so nothing is dropped for that."""
    return np.arange(n) % 2


def test_equal_count_bins_are_not_equal_width():
    """The whole reason width has to be reported alongside the AUC."""
    amp = np.concatenate([np.linspace(1.0, 2.0, 90), np.linspace(2.0, 1000.0, 10)])
    bins = amplitude_bins(amp, alternating(100), n_bins=10, min_bin=1)
    ratios = [b["ratio"] for b in bins]
    assert max(ratios) > 100 * min(ratios)


def test_narrow_flag_follows_the_ratio_threshold():
    amp = np.concatenate([np.full(50, 1.0), np.full(50, 1000.0)])
    bins = amplitude_bins(amp, alternating(100), n_bins=2, min_bin=1)
    for b in bins:
        assert b["narrow"] == (b["ratio"] <= NARROW_RATIO)


def test_a_wide_bin_is_not_evidence_however_good_its_auc():
    """A 500x bin leaves amplitude free to vary by two and a half orders of
    magnitude, so a high AUC inside it is not a statement about shape."""
    amp = np.linspace(1.0, 500.0, 100)
    bins = amplitude_bins(amp, alternating(100), n_bins=2, min_bin=1)
    assert not bins[0]["narrow"]
    assert bins[0]["ratio"] > NARROW_RATIO


def test_a_constant_amplitude_bin_is_maximally_narrow():
    amp = np.full(100, 7.0)
    bins = amplitude_bins(amp, alternating(100), n_bins=2, min_bin=1)
    assert all(b["ratio"] == pytest.approx(1.0) and b["narrow"] for b in bins)


def test_bins_partition_the_samples():
    amp = np.exp(np.linspace(0, 6, 200))
    bins = amplitude_bins(amp, alternating(200), n_bins=10, min_bin=1)
    counts = np.zeros(200, dtype=int)
    for b in bins:
        counts += b["mask"].astype(int)
    assert (counts == 1).all()          # every sample in exactly one bin


def test_the_largest_amplitude_lands_in_the_top_bin():
    """The last bin's upper edge is inclusive; without that the maximum
    silently drops out of the table."""
    amp = np.exp(np.linspace(0, 6, 200))
    bins = amplitude_bins(amp, alternating(200), n_bins=10, min_bin=1)
    assert bins[-1]["mask"][amp.argmax()]


def test_bins_are_ascending_and_contiguous():
    amp = np.exp(np.linspace(0, 6, 200))
    bins = amplitude_bins(amp, alternating(200), n_bins=10, min_bin=1)
    for a, b in zip(bins, bins[1:]):
        assert a["hi"] == pytest.approx(b["lo"])
        assert a["lo"] <= a["hi"]


def test_undersized_bins_are_dropped():
    amp = np.exp(np.linspace(0, 6, 100))
    assert amplitude_bins(amp, alternating(100), n_bins=10, min_bin=11) == []
    assert len(amplitude_bins(amp, alternating(100), n_bins=10, min_bin=10)) == 10


def test_single_class_bins_are_dropped_because_auc_is_undefined():
    amp = np.linspace(1.0, 2.0, 100)
    y = np.zeros(100, dtype=int)
    y[50:] = 1                              # top half all positive
    bins = amplitude_bins(amp, y, n_bins=2, min_bin=1)
    assert bins == []


def test_class_balance_does_not_disqualify_a_bin():
    """AUC is invariant to class balance -- that is why it is the statistic
    here and accuracy is not. A 95%-positive bin is still usable."""
    amp = np.linspace(1.0, 2.0, 100)
    y = np.ones(100, dtype=int)
    y[::20] = 0                             # 5% negative, spread across bins
    bins = amplitude_bins(amp, y, n_bins=2, min_bin=1)
    assert len(bins) == 2


def test_zero_amplitude_gives_an_infinite_ratio_not_a_crash():
    amp = np.concatenate([np.zeros(50), np.linspace(1.0, 2.0, 50)])
    bins = amplitude_bins(amp, alternating(100), n_bins=2, min_bin=1)
    assert np.isinf(bins[0]["ratio"]) and not bins[0]["narrow"]
