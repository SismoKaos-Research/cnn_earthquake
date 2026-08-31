"""Concurrency on the TDVMS campaign ledger.

Two email addresses run in parallel, so `next` and a minutes-long `paste` overlap
routinely. The original code loaded the ledger, held that copy across the whole
download, then wrote it back wholesale -- discarding anything recorded meanwhile.
That erased a live submission, put its window back to pending, and handed the
same window to the other address: a duplicate that wastes a queue slot and, in
one case, 825 MB of re-downloaded data.
"""
import importlib.util
import json
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "afad_campaign", pathlib.Path(__file__).resolve().parents[1] / "scripts/afad_campaign.py")
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)


@pytest.fixture
def led(tmp_path):
    p = tmp_path / "ledger.jsonl"
    rows = [{"station": "MANT", "start": f"2024-0{i}-01T00:00:00", "end": "x",
             "state": "pending", "email": None, "url": None, "bytes": None,
             "note": None} for i in (1, 2, 3)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_two_addresses_never_claim_the_same_window(led):
    """The duplicate that actually happened, twice, on 2026-08-31."""
    a, _ = m.claim_next_pending(led, "a@x")
    b, _ = m.claim_next_pending(led, "b@x")
    assert a["start"] != b["start"], "both addresses claimed the same window"
    assert a["start"].startswith("2024-01")
    assert b["start"].startswith("2024-02")


def test_one_address_cannot_hold_two_claims(led):
    """The TDVMS queue slot is per address, so a second claim must be refused."""
    first, _ = m.claim_next_pending(led, "a@x")
    nothing, busy = m.claim_next_pending(led, "a@x")
    assert nothing is None
    assert busy["start"] == first["start"]


def test_a_stale_writer_cannot_clobber_a_concurrent_claim(led):
    """`paste` holds its copy for minutes; writing it back wholesale is the bug."""
    m.load_ledger(led)                      # the stale copy a long paste would hold
    m.update_chunk(led, "MANT", "2024-03-01T00:00:00", state="submitted", email="c@x")
    m.update_chunk(led, "MANT", "2024-01-01T00:00:00", state="fetched")
    final = {r["start"][:7]: (r["state"], r["email"]) for r in m.load_ledger(led)}
    assert final["2024-03"] == ("submitted", "c@x"), "concurrent claim was clobbered"
    assert final["2024-01"][0] == "fetched"


def test_update_chunk_rejects_an_unknown_window(led):
    with pytest.raises(KeyError):
        m.update_chunk(led, "MANT", "1999-01-01T00:00:00", state="fetched")
