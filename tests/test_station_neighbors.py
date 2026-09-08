"""The ways a station-radius query returns a confident wrong answer.

Asking "what is within N km of MANT" looks like it cannot fail. Every way it
does fail is silent, and each one is pinned here:

**Picking one of two stations that share a code.** 15 bare codes in these
catalogues name two different places, and three collide even with the network
attached. TOKT is Balikesir or Tokat, 727 km apart; TU.KOCA is Denizli or
Canakkale, 324 km apart under one NET.STA key. `df[df.Code == name].iloc[0]`
answers a different question without saying so, so an ambiguous target must be
refused rather than resolved by row order.

**Merging two catalogues that disagree.** The same 179 stations appear in both
files at different positions -- KO.KIZT by 14.4 km. Whichever file loses must
lose *everywhere*, or a station's position depends on which query found it.

**Returning the co-located twin.** 189 pairs of distinct keys sit within 50 m:
the same site under two network codes. For coincidence work that is the one
answer that is useless, so `min_km` must exclude on distance, not on name.

The fixtures mirror those real shapes at a size that can be checked by hand.
"""
import numpy as np
import pandas as pd
import pytest

from sismokaos.catalog import haversine_km
from sismokaos.stations.station_neighbors import (load_station_table, neighbors,
                                                  parse_near, read_station_csv,
                                                  resolve)


def write_istasyon(path, rows):
    """The istasyon_katalog.csv layout: Network/Code/Province, utf-8-sig."""
    pd.DataFrame(rows, columns=["Network", "Code", "Longitude", "Latitude",
                                "Height", "Province", "District"]
                 ).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_coords(path, rows):
    """The station_coords.csv layout: lowercase, no province at all."""
    pd.DataFrame(rows, columns=["network", "station", "latitude", "longitude",
                                "elevation"]).to_csv(path, index=False)
    return path


@pytest.fixture
def istasyon(tmp_path):
    return write_istasyon(tmp_path / "istasyon.csv", [
        ("TU", "MANT", 28.5579, 38.4908, 100, "Manisa", "x"),
        ("TU", "MANS", 28.5580, 38.4908, 100, "Manisa", "x"),   # 9 m away
        ("TU", "FAR", 29.5579, 38.4908, 100, "Usak", "x"),      # ~87 km east
        ("KO", "KIZT", 31.7163, 38.8808, 100, "Konya", "x"),    # the wrong lon
        ("TU", "TOKT", 28.0311, 39.7617, 100, "Balikesir", "x"),
        ("KO", "TOKT", 36.5443, 40.3173, 100, "Tokat", "x"),    # 727 km away
        ("TU", "KOCA", 29.2088, 37.8407, 100, "Denizli", "x"),
        ("TU", "KOCA", 26.1383, 39.5022, 100, "Canakkale", "x"),  # ONE key
        ("KO", "DUP", 27.0000, 39.0000, 100, "Mugla", "x"),
        ("KO", "DUP", 27.0000, 39.0000, 100, "Mugla", "x"),     # verbatim twin
    ])


@pytest.fixture
def coords(tmp_path):
    return write_coords(tmp_path / "coords.csv", [
        ("KO", "KIZT", 38.8808, 31.8830, 100.0),   # the FDSN-checked position
        ("KO", "ONLY", 38.4908, 28.5579, 100.0),   # exists nowhere else
    ])


@pytest.fixture
def tab(istasyon, coords):
    """Authority first, exactly as `main` builds it."""
    return load_station_table(coords, istasyon)


# --- reading the two layouts ---------------------------------------------

def test_each_layout_is_recognised_by_its_header(istasyon, coords):
    """Neither file names a single column the same way; the caller should not
    have to say which one it is holding."""
    a, b = read_station_csv(istasyon), read_station_csv(coords)
    assert a.src.eq("istasyon").all() and b.src.eq("coords").all()
    assert set(a.columns) == set(b.columns)
    assert a.loc[a.key == "TU.MANT", "lat"].iloc[0] == pytest.approx(38.4908)
    assert b.loc[b.key == "KO.KIZT", "lon"].iloc[0] == pytest.approx(31.8830)


def test_an_unrecognised_table_raises_rather_than_returning_nothing(tmp_path):
    p = tmp_path / "junk.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="unrecognised station table"):
        read_station_csv(p)


# --- the merge ------------------------------------------------------------

def test_the_authority_file_wins_the_position_it_shares(tab):
    """KO.KIZT is in both files 14.4 km apart. station_coords.csv is the one
    checked against FDSN, so its longitude must be the one that survives."""
    row = tab[tab.key == "KO.KIZT"]
    assert len(row) == 1, "the loser's row must not survive alongside the winner"
    assert row.lon.iloc[0] == pytest.approx(31.8830)
    assert row.src.iloc[0] == "coords"


def test_merging_does_not_cost_the_stations_only_the_other_file_has(tab):
    """The authority covers 334 of 1,576 stations. If overriding it dropped the
    rest, the merge would trade 14.4 km of accuracy for 1,200 missing stations."""
    assert "TU.FAR" in set(tab.key)
    assert "KO.ONLY" in set(tab.key)


def test_the_winner_keeps_the_province_only_the_loser_carried(tab):
    """station_coords.csv has no province column, so a KO station it overrides
    would print blank where istasyon knew the answer."""
    assert tab.loc[tab.key == "KO.KIZT", "place"].iloc[0] == "Konya"


def test_a_row_repeated_verbatim_is_one_station_not_two(tab):
    assert len(tab[tab.key == "KO.DUP"]) == 1


def test_two_real_positions_under_one_key_are_both_kept(tab):
    """The opposite of the case above, and the reason de-duplication cannot key
    on the code alone: collapsing these picks Denizli or Canakkale silently."""
    assert len(tab[tab.key == "TU.KOCA"]) == 2


# --- resolving a name -----------------------------------------------------

def test_a_unique_bare_code_resolves(tab):
    assert len(resolve(tab, "MANT")) == 1


def test_a_bare_code_two_networks_use_is_not_resolved_by_row_order(tab):
    """TU.TOKT and KO.TOKT are 727 km apart. Returning either is a wrong answer
    delivered as a right one."""
    assert len(resolve(tab, "TOKT")) == 2


def test_the_network_decides_which_station_is_meant(tab):
    assert resolve(tab, "KO.TOKT").place.iloc[0] == "Tokat"
    assert resolve(tab, "TU.TOKT").place.iloc[0] == "Balikesir"


def test_the_network_flag_disambiguates_a_bare_code(tab):
    assert resolve(tab, "TOKT", network="KO").place.iloc[0] == "Tokat"


def test_a_full_key_can_still_be_ambiguous(tab):
    """TU.KOCA is two stations under one key, so naming the network cannot
    help -- which is why `main` must not suggest it here."""
    assert len(resolve(tab, "TU.KOCA")) == 2
    assert len(resolve(tab, "KOCA", network="TU")) == 2


def test_near_breaks_the_tie_the_network_cannot(tab):
    """The escape hatch for the three keys that collide with the network on."""
    assert resolve(tab, "TU.KOCA", near=(37.8, 29.2)).place.iloc[0] == "Denizli"
    assert resolve(tab, "TU.KOCA", near=(39.5, 26.1)).place.iloc[0] == "Canakkale"


def test_near_does_not_invent_a_station_that_is_not_there(tab):
    assert len(resolve(tab, "NOPE", near=(39.0, 27.0))) == 0


def test_parse_near_rejects_what_it_cannot_read():
    assert parse_near("38.5,28.5") == (38.5, 28.5)
    assert parse_near(" 38.5 , 28.5 ") == (38.5, 28.5)
    assert parse_near(None) is None
    for bad in ("garbage", "1,2,3", "1"):
        with pytest.raises(ValueError):
            parse_near(bad)


# --- the query ------------------------------------------------------------

def _at(tab, key):
    r = tab[tab.key == key].iloc[0]
    return float(r.lat), float(r.lon)


def test_a_station_is_not_its_own_neighbour(tab):
    lat, lon = _at(tab, "TU.MANT")
    out = neighbors(tab, lat, lon, 500, exclude_key="TU.MANT")
    assert "TU.MANT" not in set(out.key)


def test_min_km_excludes_the_co_located_twin(tab):
    """TU.MANS is 9 m from TU.MANT: the same site, so it confirms nothing. It
    has to go by distance -- nothing in its name says it is a twin."""
    lat, lon = _at(tab, "TU.MANT")
    near = neighbors(tab, lat, lon, 500, exclude_key="TU.MANT")
    assert "TU.MANS" in set(near.key)
    far = neighbors(tab, lat, lon, 500, min_km=1.0, exclude_key="TU.MANT")
    assert "TU.MANS" not in set(far.key)
    assert "TU.FAR" in set(far.key), "the floor must not empty the answer"


def test_the_radius_includes_a_station_exactly_on_it(tab):
    """`<=`, matching select_afad_stations.coverage_matrix. An exclusive bound
    drops the station at exactly the limit and nothing reports the loss."""
    lat, lon = _at(tab, "TU.MANT")
    flat, flon = _at(tab, "TU.FAR")
    d = float(haversine_km(lat, lon, np.array([flat]), np.array([flon]))[0])
    assert "TU.FAR" in set(neighbors(tab, lat, lon, d, exclude_key="TU.MANT").key)
    assert "TU.FAR" not in set(
        neighbors(tab, lat, lon, d * 0.999, exclude_key="TU.MANT").key)


def test_the_min_bound_is_inclusive_too(tab):
    lat, lon = _at(tab, "TU.MANT")
    flat, flon = _at(tab, "TU.FAR")
    d = float(haversine_km(lat, lon, np.array([flat]), np.array([flon]))[0])
    assert "TU.FAR" in set(
        neighbors(tab, lat, lon, d + 1, min_km=d, exclude_key="TU.MANT").key)


def test_results_come_back_nearest_first(tab):
    lat, lon = _at(tab, "TU.MANT")
    out = neighbors(tab, lat, lon, 500, exclude_key="TU.MANT")
    assert list(out.km) == sorted(out.km)


def test_top_keeps_the_nearest_n_not_the_first_n_found(tab):
    """Truncating before the sort returns whichever rows the catalogue happened
    to list first, which is not a radius query at all."""
    lat, lon = _at(tab, "TU.MANT")
    full = neighbors(tab, lat, lon, 500, exclude_key="TU.MANT")
    top2 = neighbors(tab, lat, lon, 500, k=2, exclude_key="TU.MANT")
    assert list(top2.key) == list(full.key[:2])
    assert len(top2) == 2


def test_an_empty_answer_is_an_empty_frame_not_a_crash(tab):
    lat, lon = _at(tab, "TU.MANT")
    out = neighbors(tab, lat, lon, 0.001, min_km=0.0005, exclude_key="TU.MANT")
    assert len(out) == 0
    assert "km" in out.columns, "an empty result still has to have the columns"


def test_the_distance_is_the_repo_s_haversine_not_a_second_copy(tab):
    """Three copies of this formula already exist. If this module grew a
    fourth, a fix to one would leave the others wrong."""
    lat, lon = _at(tab, "TU.MANT")
    out = neighbors(tab, lat, lon, 500, exclude_key="TU.MANT")
    expect = haversine_km(lat, lon, out.lat.values, out.lon.values)
    assert np.allclose(out.km.values, expect)
