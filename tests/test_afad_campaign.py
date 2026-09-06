"""A failed submission must not leave the ledger holding a claim.

`next` claims a pending chunk, writes the claim, then looks the station's
device code up at TDVMS. That lookup raises SystemExit for a station TDVMS does
not list -- and the claim was already on disk. The row stayed `claimed` with an
address attached and nothing in flight, so no refill ever fired and the poller
sat waiting for a link nobody had requested.

That is not hypothetical either: BAKC and IRLI stalled two of the station
probe's four queue slots this way, and from the outside it was indistinguishable
from mail that never arrived.
"""
import json
from pathlib import Path

import pytest

# An ordinary import now. These were loaded from a file path because the
# tools lived in `scripts/`, which is not importable; that is the whole
# reason the directory is gone.
from sismokaos.acquisition import afad_campaign as camp


class Args:
    def __init__(self, ledger, email="you+a1@x.com"):
        # A Path, as argparse builds it: ledger_lock takes .parent.
        self.ledger = Path(ledger)
        self.email = email
        self.timeout = 30


def ledger_with_one_pending(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({
        "station": "BAKC", "start": "2025-06-01T00:00:00",
        "end": "2025-06-02T00:00:00", "state": "pending", "email": None,
        "url": None, "bytes": None, "note": None}) + "\n")
    return p


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_unlistable_station_releases_its_claim(tmp_path, monkeypatch):
    led = ledger_with_one_pending(tmp_path)
    monkeypatch.setattr(camp, "_device_code", lambda code: (_ for _ in ()).throw(
        SystemExit(f"{code}: not in the TDVMS station list")))
    monkeypatch.setattr(camp, "requests", None)   # must never be reached

    assert camp.cmd_next(Args(led)) == 1
    r, = rows(led)
    assert r["state"] == "failed", (
        "a station TDVMS does not list left the row claimed, holding a queue "
        "slot against a request that was never made")
    assert r["email"] is None, "the address stayed attached to a dead row"
    assert "not in the TDVMS station list" in (r["note"] or "")


def test_a_listable_station_still_submits(tmp_path, monkeypatch):
    led = ledger_with_one_pending(tmp_path)
    monkeypatch.setattr(camp, "_device_code", lambda code: "H")

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"Result": camp.RESULT_OK} if hasattr(camp, "RESULT_OK") else {"Result": 109}

    monkeypatch.setattr(camp.requests, "post", lambda *a, **k: Resp())
    camp.cmd_next(Args(led))
    r, = rows(led)
    assert r["state"] == "submitted"
    assert r["email"] == "you+a1@x.com"
