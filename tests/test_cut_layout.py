"""The cut layout must be what `seismic-cli` can actually read back.

`cut_event_windows.py` writes into a layout another repo consumes, and that
contract is a filename regex living in a repo this test cannot import. The
first version wrote `event_<id>_<station>_raw.mseed`, which reads as compatible
and is not: seismic-cli's `parse_event_id` is `^(?:noise_)?event_(.+?)_raw$`
with a NON-GREEDY capture, so that name yields the event id `627227_MANT`. That
matches no catalogue EventID, so every window loses its magnitude label -- with
no error, no warning, and an empty dataset at the end of a long encode.

The regex is copied here rather than imported, and the test says so: if the
consumer changes it, this test keeps passing while the pipeline breaks. What it
does buy is that a change on THIS side -- putting the station back in the name,
or renaming the suffix -- fails immediately.
"""
import re

import pytest

# Copied verbatim from seismic_cli/regression.py (EVENT_ID_RE). Not importable
# from here: it lives in the Sismokaos repo, which is not a dependency.
CONSUMER_RE = re.compile(r"^(?:noise_)?event_(.+?)_raw$")


def consumer_parse(stem):
    m = CONSUMER_RE.match(stem)
    return m.group(1) if m else None


@pytest.mark.parametrize("event_id", ["627227", "721442", "1"])
def test_cut_names_parse_back_to_the_event_id(event_id):
    """What the cutter writes must round-trip through the consumer's regex."""
    assert consumer_parse(f"event_{event_id}_raw") == event_id
    assert consumer_parse(f"noise_event_{event_id}_raw") == event_id


def test_a_station_in_the_filename_is_what_broke_it():
    """Pinning the actual failure, so the reason is not lost to a tidy-up."""
    assert consumer_parse("event_627227_MANT_raw") == "627227_MANT", (
        "the non-greedy capture is the whole problem: this parses cleanly and "
        "returns an id no catalogue contains")


def test_the_cutter_builds_the_name_the_consumer_expects():
    """Reads the tag out of the source, so a change to it fails here."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "cut_event_windows.py").read_text()
    assert 'tag = f"event_{int(ev.EventID)}"' in src, (
        "cut_event_windows.py no longer builds `event_<id>`; if the station "
        "went back into the filename, seismic-cli cannot resolve the magnitude")
    assert 'args.out_dir) / tag / args.station / "eq"' in src, (
        "the station must stay in the path -- it is what keeps two stations' "
        "cuts apart now that it is out of the filename")


# ---------------------------------------------------------------------------
# Alarm anchoring
# ---------------------------------------------------------------------------

def test_the_cutter_anchors_on_cut_epoch_not_p_epoch():
    """Windows must be cut relative to `cut_epoch`, which --anchor-csv sets.

    Every magnitude figure in this project was measured on catalogue-anchored
    windows -- the P arrival known exactly. A deployed cascade has an alarm
    time instead, declared at the window's END and lagging P by a few seconds
    with spread. Cutting from the detector's own alarm times is what prices
    that assumption, and it only works if the cut and the write agree on the
    anchor: an earlier version had one on `cut_epoch` and the other still on
    `p_epoch`, which writes correct samples under a wrong start time.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "cut_event_windows.py").read_text()
    assert "cut(segs, comps, ev.cut_epoch - args.pre" in src
    assert "write_mseed(a, comps, ev.cut_epoch - args.pre" in src, (
        "the cut and the written start time must use the same anchor")
    assert "ev.p_epoch - args.noise_offset" in src, (
        "the paired noise window stays on the true P -- it is a reference "
        "window, not something a detector found")
    assert 'cat["cut_epoch"] = cat.p_epoch' in src, (
        "cut_epoch must default to the predicted P, so runs without "
        "--anchor-csv are unchanged")
