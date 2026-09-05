"""Per-station labelling for the replication check.

The comparison is only a replication if each station is scored against its OWN
local label. Silently falling back to BODT's coordinates for both would compare
two waveforms against one label and call agreement a result -- which is the
failure mode this whole cross-station exercise exists to rule out.
"""

import os
import pathlib

import numpy as np
import pytest

from forecasting.chaos_dataset import RADIUS_KM, load_events
from seismolib.catalog import STATION_COORDS, haversine_km

# The catalogue lives in a sibling checkout whose name is not stable: it has
# been `data_downloader` and is now `seismic_cli`, under a parent that is
# `Sismokaos` here and `sismokaos` on the box. The previous version enumerated
# four absolute paths, which is a list that can only ever be right about renames
# someone thought of -- the `seismic_cli` rename turned eight passing tests into
# eight skips whose only explanation was "not found".
#
# So search rather than enumerate: this repo, then every sibling of it. That
# survives a rename, and it fails only when the file is genuinely absent.
_LEAF = pathlib.Path("catalogs") / "catalog_current.csv"


def _search_roots():
    """This repo, then its siblings -- the two places the catalogue has lived."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    yield repo
    try:
        yield from sorted(d for d in repo.parent.iterdir() if d.is_dir())
    except OSError:
        pass


def _find_catalog():
    env = os.environ.get("SEISMO_CATALOG")
    if env:
        return pathlib.Path(env)
    return next((r / _LEAF for r in _search_roots() if (r / _LEAF).exists()), None)


_FOUND = _find_catalog()
CATALOG = str(_FOUND) if _FOUND else None

pytestmark = pytest.mark.skipif(
    CATALOG is None,
    # Name what was searched. A skip that says only "not found" is what made the
    # rename take a while to spot.
    reason=(f"no {_LEAF} under "
            f"{pathlib.Path(__file__).resolve().parents[2]}/*; "
            f"set SEISMO_CATALOG to point at one"),
)


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
