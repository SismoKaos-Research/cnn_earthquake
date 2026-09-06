"""The pieces of two-station coincidence where a bug reads as a great result.

Requiring a second station to confirm an alarm is supposed to remove false
alarms. Every mistake available here removes them too -- and removes them
faster:

**Not intersecting coverage.** GCAM stops recording in December 2024. Every
MANT alarm after that would go unconfirmed and be counted as suppressed, which
looks like a near-total false-alarm reduction and is only missing data.

**Not clustering.** A noise burst spanning ten windows is one declaration; ten
would inflate both the single-station rate it is compared against and the
independence prediction built from it.

**An off-by-one in the confirmation search.** A window that excludes its
endpoints drops real events arriving at exactly the physical limit -- an event
on the line through both stations -- and the loss shows up as improved
precision.

So each is pinned separately, on inputs small enough to check by hand.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuous_false_alarms.py"


def _module():
    spec = importlib.util.spec_from_file_location("_cfa", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfa = _module()


# --- coverage -------------------------------------------------------------

def test_coverage_spans_splits_on_a_real_gap():
    """Two runs of windows an hour apart are two spans, not one."""
    t = np.concatenate([np.arange(0, 100, 1.0), np.arange(3600, 3700, 1.0)])
    spans = cfa.coverage_spans(t, step=1.0)
    assert len(spans) == 2
    assert spans[0] == (0.0, 100.0)
    assert spans[1] == (3600.0, 3700.0)


def test_coverage_spans_tolerates_jitter_within_the_step():
    """A window step is not exact; a 1.5x hiccup is not an outage."""
    t = np.array([0.0, 1.0, 2.5, 3.5, 4.5])
    assert len(cfa.coverage_spans(t, step=1.0)) == 1


def test_coverage_spans_of_nothing():
    assert cfa.coverage_spans(np.empty(0), step=1.0) == []


def test_intersect_spans_keeps_only_the_shared_time():
    a = [(0.0, 10.0), (20.0, 40.0)]
    b = [(5.0, 25.0), (30.0, 50.0)]
    assert cfa.intersect_spans(a, b) == [(5.0, 10.0), (20.0, 25.0), (30.0, 40.0)]


def test_intersect_spans_of_disjoint_records_is_empty():
    """The GCAM-stops-recording case: no shared time, so nothing to score."""
    assert cfa.intersect_spans([(0.0, 10.0)], [(100.0, 200.0)]) == []


def test_touching_spans_do_not_count_as_overlap():
    assert cfa.intersect_spans([(0.0, 10.0)], [(10.0, 20.0)]) == []


def test_in_spans_masks_exactly_the_covered_windows():
    t = np.arange(0.0, 20.0, 1.0)
    m = cfa.in_spans(t, [(3.0, 6.0), (15.0, 17.0)])
    assert list(t[m]) == [3.0, 4.0, 5.0, 6.0, 15.0, 16.0, 17.0]


# --- declarations ---------------------------------------------------------

def test_one_burst_is_one_declaration_at_its_peak():
    t = np.arange(0.0, 10.0, 1.0)
    p = np.array([0.1, 0.1, 0.9, 0.95, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1])
    dt, dp = cfa.declarations(t, p, thr=0.5, cluster_seconds=60.0)
    assert len(dt) == 1, "a three-window burst counted as more than one alarm"
    assert dt[0] == 3.0 and dp[0] == 0.95


def test_bursts_further_apart_than_the_cluster_are_separate():
    t = np.array([0.0, 1.0, 500.0, 501.0])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    dt, _ = cfa.declarations(t, p, thr=0.5, cluster_seconds=60.0)
    assert list(dt) == [0.0, 500.0]


def test_nothing_above_threshold_declares_nothing():
    dt, dp = cfa.declarations(np.arange(5.0), np.zeros(5), 0.5, 60.0)
    assert len(dt) == len(dp) == 0


# --- confirmation ---------------------------------------------------------

def test_confirmation_window_includes_its_endpoints():
    """An event on the line through both stations arrives exactly `w` apart."""
    a = np.array([100.0])
    assert cfa.confirmed(a, np.array([100.0 + 20.0]), window=20.0)[0]
    assert cfa.confirmed(a, np.array([100.0 - 20.0]), window=20.0)[0]
    assert not cfa.confirmed(a, np.array([100.0 + 20.1]), window=20.0)[0]


def test_confirmation_is_per_declaration():
    a = np.array([0.0, 100.0, 200.0])
    b = np.array([5.0, 195.0])
    assert list(cfa.confirmed(a, b, window=10.0)) == [True, False, True]


def test_confirmation_against_a_silent_station_confirms_nothing():
    a = np.array([0.0, 100.0])
    assert not cfa.confirmed(a, np.empty(0), window=10.0).any()
    assert len(cfa.confirmed(np.empty(0), a, window=10.0)) == 0


def test_the_independence_prediction_matches_a_simulation():
    """The number the whole result is compared against must be the right one.

    The measured quantity is "A declarations having at least one B within
    +/-w", not the number of coincident pairs. For a Poisson B stream that is
    ra * (1 - exp(-rb*2w)), which sits below the pair rate ra*rb*2w whenever B
    is busy. Simulated here at a rate where the two differ by 9%, so the test
    would fail if the tool used the pair formula.
    """
    rng = np.random.default_rng(0)
    span, ra, rb, w = 2_000_000.0, 5e-3, 5e-3, 20.0
    a = np.sort(rng.uniform(0, span, int(ra * span)))
    b = np.sort(rng.uniform(0, span, int(rb * span)))
    measured = cfa.confirmed(a, b, w).sum() / span

    correct = ra * (1.0 - np.exp(-rb * 2 * w))
    pairs = ra * rb * 2 * w
    assert measured == pytest.approx(correct, rel=0.06), (
        f"measured {measured:.4e} vs prediction {correct:.4e}")
    assert pairs / correct > 1.08, (
        "this simulation is meant to separate the two formulas; at these rates "
        "it no longer does, so the test has stopped checking anything")


def test_the_two_formulas_agree_in_the_regime_that_matters():
    """At a realistic alarm budget the distinction is negligible, as claimed."""
    w, rb = 21.7, 10.0 / 86400.0          # 130 km apart, 10 alarms/day
    correct = 1.0 - np.exp(-rb * 2 * w)
    pairs = rb * 2 * w
    assert abs(pairs - correct) / correct < 0.005


def test_in_spans_handles_a_gap_split_archive_quickly():
    """The span list is not small: MANT's 3.4 s scores split into 43,215.

    A per-event Python pass over all of them is hours, which is what the first
    version of the catalogue filter did. This pins the vectorised path by size
    rather than by clock, so it fails on a machine of any speed if the
    quadratic version comes back.
    """
    n_spans = 40_000
    spans = [(float(i * 100), float(i * 100 + 50)) for i in range(n_spans)]
    t = np.arange(0.0, n_spans * 100.0, 250.0)
    m = cfa.in_spans(t, spans)
    # every t is either inside its span's first 50 s or in the gap after it
    expected = ((t % 100.0) <= 50.0)
    assert np.array_equal(m, expected)


def test_load_snr_collapses_duplicate_events_keeping_the_best(tmp_path):
    """A LEFT JOIN on a non-unique key silently expands the frame it joins into.

    DEMI's SNR table carries 269 duplicated event ids where MANT's and GCAM's
    carry none, so `cat.merge(snr, how="left")` grew the catalogue by 269 rows
    after the per-event guards had already been computed against it. That raised
    a length mismatch here; had the lengths happened to line up it would instead
    have shifted every recall denominator without a word.
    """
    import pandas as pd
    f = tmp_path / "range.csv"
    pd.DataFrame({"event_id": [1, 2, 2, 3], "snr": [5.0, 1.2, 9.9, 4.0],
                  "other": ["a", "b", "c", "d"]}).to_csv(f, index=False)
    out = cfa.load_snr(str(f))
    assert len(out) == 3, "duplicate event_id survived"
    assert out.set_index("event_id").snr.to_dict() == {1: 5.0, 2: 9.9, 3: 4.0}, (
        "the larger reading should win -- the smaller is usually a chunk-edge "
        "truncation of the same event")


def test_merging_a_duplicated_table_would_have_grown_the_catalogue(tmp_path):
    """Pins the mechanism, so the reason for load_snr is not lost to a tidy-up."""
    import pandas as pd
    cat = pd.DataFrame({"EventID": [1, 2, 3]})
    raw = pd.DataFrame({"event_id": [1, 2, 2, 3], "snr": [5.0, 1.2, 9.9, 4.0]})
    assert len(cat.merge(raw, left_on="EventID", right_on="event_id",
                         how="left")) == 4, "the duplicate no longer expands"

    f = tmp_path / "r.csv"
    raw.to_csv(f, index=False)
    assert len(cat.merge(cfa.load_snr(str(f)), left_on="EventID",
                         right_on="event_id", how="left")) == 3
