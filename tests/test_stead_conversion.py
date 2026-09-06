"""Recasting STEAD into this project's window geometry.

Two constants and one mapping decide whether the cross-corpus number means
anything, and all three fail quietly. The window arithmetic can be off by a
sample and still produce a valid tensor. The component naming can be reversed
and still produce a three-channel record -- the model would then read
horizontals where it expects the vertical and score badly, which looks exactly
like a generalisation failure and is not one.
"""

import numpy as np
import pytest
from obspy import UTCDateTime

from sismokaos.detection.stead_to_ponly_mseed import (FS, PRE_P_SAMPLES, STEAD_COMPONENTS,
                                            WINDOW_SAMPLES, to_stream)


def test_window_geometry_matches_the_aegean_corpus():
    """2.0 s before P, 3.4 s long, at 100 Hz. If these drift apart from
    arrival_from_catalog.py, the two corpora stop being comparable."""
    assert WINDOW_SAMPLES == 340
    assert PRE_P_SAMPLES == 200
    assert PRE_P_SAMPLES / FS == pytest.approx(2.0)
    assert WINDOW_SAMPLES / FS == pytest.approx(3.4)


def test_post_p_is_the_one_point_four_seconds_that_make_it_p_only():
    assert (WINDOW_SAMPLES - PRE_P_SAMPLES) / FS == pytest.approx(1.4)


def test_stead_column_order_is_east_north_vertical():
    """Verified empirically on 400 event traces: column 2 carries the largest
    P onset jump. Pinned because reversing it is silent."""
    assert STEAD_COMPONENTS == ("E", "N", "Z")
    assert STEAD_COMPONENTS[2] == "Z"


def test_channel_names_follow_the_column_they_carry():
    """The generator's select_components reads the channel LETTER to assign
    Z/N/E roles, so this naming is the entire column-to-role mapping."""
    data = np.stack([np.full(WINDOW_SAMPLES, float(i)) for i in range(3)], axis=1)
    st = to_stream(data, "TA", "109C", UTCDateTime("2006-01-01"))
    got = {tr.stats.channel[-1]: float(tr.data[0]) for tr in st}
    assert got == {"E": 0.0, "N": 1.0, "Z": 2.0}


def test_stream_carries_the_station_identity_the_generator_groups_by():
    st = to_stream(np.zeros((WINDOW_SAMPLES, 3)), "TA", "109C", UTCDateTime("2006-01-01"))
    for tr in st:
        assert tr.stats.network == "TA" and tr.stats.station == "109C"
        assert tr.stats.sampling_rate == FS
        assert tr.id.rsplit(".", 1)[0] == "TA.109C."


def test_stream_has_exactly_three_components():
    st = to_stream(np.zeros((WINDOW_SAMPLES, 3)), "TA", "109C", UTCDateTime("2006-01-01"))
    assert len(st) == 3
    assert sorted(tr.stats.channel for tr in st) == ["HHE", "HHN", "HHZ"]


def test_truncation_keeps_the_window_length():
    st = to_stream(np.zeros((6000, 3)), "TA", "109C", UTCDateTime("2006-01-01"),
                   n=WINDOW_SAMPLES)
    assert all(len(tr.data) == WINDOW_SAMPLES for tr in st)


def test_full_record_is_written_whole_when_no_length_is_given():
    """Noise records go out at full length so the generator can slide its own
    windows and build station baselines from the same files."""
    st = to_stream(np.zeros((6000, 3)), "TA", "109C", UTCDateTime("2006-01-01"))
    assert all(len(tr.data) == 6000 for tr in st)


def test_written_data_is_float32_and_contiguous():
    """miniSEED will not round-trip a non-contiguous float64 view cleanly."""
    data = np.asfortranarray(np.random.default_rng(0).normal(size=(WINDOW_SAMPLES, 3)))
    st = to_stream(data, "TA", "109C", UTCDateTime("2006-01-01"))
    for tr in st:
        assert tr.data.dtype == np.float32
        assert tr.data.flags["C_CONTIGUOUS"]


def test_columns_do_not_get_mixed_between_components():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(WINDOW_SAMPLES, 3))
    st = to_stream(data, "TA", "109C", UTCDateTime("2006-01-01"))
    for i, comp in enumerate(STEAD_COMPONENTS):
        tr = [t for t in st if t.stats.channel.endswith(comp)][0]
        assert np.allclose(tr.data, data[:, i].astype(np.float32))
