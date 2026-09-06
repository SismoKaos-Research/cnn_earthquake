"""The guard that decides whether a recomputed distance column may be written.

`sk distances` replaces a whole column. That is safe only while the
recomputation reproduces the values already there -- so the interesting tests
are the refusals, not the successes.
"""
import numpy as np
import pandas as pd
import pytest

from sismokaos.stations.manifest_distances import recompute, station_table


@pytest.fixture
def coords(tmp_path):
    p = tmp_path / "station_coords.csv"
    pd.DataFrame([
        ("KO", "AAA", 39.0, 27.0, 100),
        ("KO", "BBB", 40.0, 28.0, 100),
        ("TU", "AAA", 37.0, 30.0, 100),   # same bare code, different network
        ("KO", "SOLO", 41.0, 29.0, 100),
    ], columns=["network", "station", "latitude", "longitude", "elevation"]).to_csv(p, index=False)
    return p


@pytest.fixture
def cat():
    return pd.DataFrame({"EventID": [1, 2], "Latitude": [39.0, 40.5],
                         "Longitude": [27.0, 28.5]})


def test_a_station_at_the_epicentre_is_zero_km(coords, cat):
    full, bare = station_table(coords)
    man = pd.DataFrame({"station_key": ["KO.AAA"], "event_id": [1]})
    d, ok = recompute(man, full, bare, cat)
    assert ok.all() and d[0] == pytest.approx(0.0, abs=1e-6)


def test_the_network_decides_which_station_is_meant(coords, cat):
    """KO.AAA and TU.AAA are different places; the key must carry the network."""
    full, bare = station_table(coords)
    man = pd.DataFrame({"station_key": ["KO.AAA", "TU.AAA"], "event_id": [1, 1]})
    d, ok = recompute(man, full, bare, cat)
    assert ok.all()
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert d[1] > 200          # TU.AAA is far from event 1


def test_an_empty_network_resolves_only_when_the_bare_code_is_unique(coords, cat):
    """`.SOLO` is unambiguous. `.AAA` is not, and must NOT be guessed."""
    full, bare = station_table(coords)
    man = pd.DataFrame({"station_key": [".SOLO", ".AAA"], "event_id": [1, 1]})
    d, ok = recompute(man, full, bare, cat)
    assert ok[0], "a unique bare code should resolve"
    assert not ok[1], "an ambiguous bare code must stay unresolved, not pick one"


def test_an_unknown_station_or_event_yields_nan_rather_than_a_wrong_number(coords, cat):
    full, bare = station_table(coords)
    man = pd.DataFrame({"station_key": ["KO.ZZZ", "KO.AAA"], "event_id": [1, 999]})
    d, ok = recompute(man, full, bare, cat)
    assert not ok.any() and np.isnan(d).all()
