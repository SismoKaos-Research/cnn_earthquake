"""Per-station labelling for the replication check.

The comparison is only a replication if each station is scored against its OWN
local label. Silently falling back to BODT's coordinates for both would compare
two waveforms against one label and call agreement a result -- which is the
failure mode this whole cross-station exercise exists to rule out.
"""

import numpy as np
import pytest

from forecasting.chaos_dataset import RADIUS_KM, load_events
from seismolib.catalog import STATION_COORDS, haversine_km

CATALOG = ("/home/hogib/Projects/Sismokaos/data_downloader/catalogs/"
           "catalog_current.csv")


@pytest.fixture(scope="module")
def counts():
    return {k: len(load_events(CATALOG, station=k)) for k in ("BODT", "DAT")}


def test_each_station_gets_its_own_event_set(counts):
    """If these were identical, the station parameter would not be reaching
    the distance filter."""
    assert counts["BODT"] != counts["DAT"]


def test_default_station_is_bodt(counts):
    assert len(load_events(CATALOG)) == counts["BODT"]


def test_explicit_coordinates_match_the_named_station(counts):
    assert len(load_events(CATALOG, station=STATION_COORDS["DAT"])) == counts["DAT"]


def test_unknown_station_name_raises_rather_than_defaulting():
    """Falling back to BODT on a typo would produce a plausible number for the
    wrong station -- the exact silent failure this suite is for."""
    with pytest.raises(KeyError):
        load_events(CATALOG, station="NOSUCH")


def test_radius_is_applied_from_the_named_station():
    """A tight radius around DAT must not admit events that only BODT sees."""
    near = load_events(CATALOG, radius_km=25.0, station="DAT")
    far = load_events(CATALOG, radius_km=RADIUS_KM, station="DAT")
    assert 0 < len(near) < len(far)


def test_the_two_stations_are_far_enough_apart_to_differ():
    b, d = STATION_COORDS["BODT"], STATION_COORDS["DAT"]
    sep = haversine_km(b[0], b[1], np.array([d[0]]), np.array([d[1]]))[0]
    assert 40 < sep < 50            # 43.8 km; pinned so a coordinate typo shows


def test_magnitude_threshold_still_applies_per_station():
    assert len(load_events(CATALOG, min_magnitude=4.5, station="DAT")) < \
           len(load_events(CATALOG, min_magnitude=2.5, station="DAT"))


def test_event_times_come_back_sorted():
    ev = load_events(CATALOG, station="DAT")
    assert (np.diff(ev) >= np.timedelta64(0, "s")).all()
