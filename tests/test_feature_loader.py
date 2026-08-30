"""Time decoding for the hand-feature archives.

Two producers disagree on what `Zaman_Dk` means, and reading the wrong one is
silent: every row lands in a single hour of 1970, 1.2M windows collapse to ~2
hourly vectors, and the forecasters then report "not enough data" while exiting
zero. That failure shipped -- these tests exist so it cannot ship twice.
"""
import numpy as np
import pandas as pd
import pytest

from seismolib.catalog import load_hourly_features


def _npy(tmp_path, rows):
    """Writes a structured .npy shaped like Sismokaos-featureExtract's output."""
    dt = np.dtype([("Pencere_ID", "<U50"), ("Zaman_Dk", "<f8"), ("E_RMS", "<f8")])
    a = np.array(rows, dtype=dt)
    p = tmp_path / "feat.npy"
    np.save(p, a)
    return str(p)


def test_pencere_id_wins_over_within_hour_zaman_dk(tmp_path):
    """The featureExtract convention: Zaman_Dk is minutes *within* the hour."""
    path = _npy(tmp_path, [
        ("2024_05_01_00_w01", 3.33, 1.0),
        ("2024_05_01_00_w02", 62.5, 3.0),
        ("2024_05_01_01_w01", 3.33, 5.0),
    ])
    h = load_hourly_features(path)
    assert list(h.index) == [pd.Timestamp("2024-05-01 00:00"),
                             pd.Timestamp("2024-05-01 01:00")]
    # the two windows inside hour 0 are averaged, not dropped
    assert h.loc[pd.Timestamp("2024-05-01 00:00"), "E_RMS"] == pytest.approx(2.0)


def test_does_not_collapse_into_1970(tmp_path):
    """The actual regression: distinct hours must stay distinct."""
    rows = [(f"2024_05_{d:02d}_{hh:02d}_w01", 30.0, float(d))
            for d in range(1, 6) for hh in range(24)]
    h = load_hourly_features(_npy(tmp_path, rows))
    assert len(h) == 120, "hours collapsed -- Zaman_Dk was read as epoch minutes"
    assert h.index.min().year == 2024


def test_falls_back_to_epoch_minutes_without_pencere_id(tmp_path):
    """The Rust engine's convention: absolute minutes since the Unix epoch."""
    dt = np.dtype([("Zaman_Dk", "<f8"), ("E_RMS", "<f8")])
    # computed, not hardcoded -- a wrong magic constant here would make the
    # test pass against an equally wrong loader
    minute = pd.Timestamp("2025-01-01").value / 1e9 / 60
    a = np.array([(minute, 1.0), (minute + 30, 2.0), (minute + 60, 4.0)], dtype=dt)
    p = tmp_path / "engine.npy"
    np.save(p, a)
    h = load_hourly_features(str(p))
    assert h.index.min() == pd.Timestamp("2025-01-01 00:00")
    assert len(h) == 2


def test_unparseable_pencere_id_falls_back_rather_than_erroring(tmp_path):
    minute = pd.Timestamp("2025-01-01").value / 1e9 / 60
    path = _npy(tmp_path, [("garbage", minute, 1.0),
                           ("also_garbage", minute + 60, 2.0)])
    h = load_hourly_features(path)
    assert h.index.min().year == 2025
