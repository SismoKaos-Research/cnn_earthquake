"""A poller must not consume mail addressed to another ledger.

Several campaigns run against one mailbox at a time -- plus-addressing is what
keeps their TDVMS request slots apart -- and every poller runs the same
`(UNSEEN FROM "tdvms@afad.gov.tr")` search, so every poller sees every message.
On 2026-09-05 that cost two requests: the station-probe poller downloaded the
links for `+dep3` and `+dep5`, could not match them to its own ledger, and
consumed the messages, recording both as permanent failures in its own log.
`depth.jsonl` went on listing those two windows as `submitted`, waiting for
links that no longer existed.

The fix is to leave a foreign message unread rather than flag it, so the ledger
that owns the address still finds it. These tests pin both halves: the address
set is read from the ledger, and `handle` returns None -- the value `poll`
already treats as "do not mark seen" -- without running the campaign.
"""
import email.message
import json

import pytest

# An ordinary import now. These were loaded from a file path because the
# tools lived in `scripts/`, which is not importable; that is the whole
# reason the directory is gone.
from sismokaos.acquisition import afad_imap as _imap


imap = _imap


def write_ledger(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


class FakeIMAP:
    """Just enough of imaplib to hand `handle` one message."""

    def __init__(self, raw):
        self.raw = raw
        self.stored = []

    def uid(self, command, uid, *rest):
        if command == "fetch":
            return "OK", [(b"1 (BODY[] {0}", self.raw)]
        self.stored.append((uid, rest))
        return "OK", [b""]


class Args:
    def __init__(self, ledger, **kw):
        self.ledger = ledger
        self.claim_unknown = False
        self.dry_run = True
        self.out_dir = "/tmp/nowhere"
        self.fail_log = "/tmp/nowhere.log"
        self.pump = False
        self.__dict__.update(kw)


def message(to, link="https://tdvms.afad.gov.tr/files/TDVMS_1_abc.zip"):
    m = email.message.EmailMessage()
    m["From"] = "tdvms@afad.gov.tr"
    m["To"] = to
    m["Subject"] = "TDVMS Verisi"
    m.set_content(f"Buyurun: {link}")
    return m.as_bytes()


def test_ledger_addresses_reads_every_submitted_slot(tmp_path):
    led = write_ledger(tmp_path / "l.jsonl", [
        {"station": "MANT", "state": "submitted", "email": "you+dep1@gmail.com"},
        {"station": "MANT", "state": "fetched", "email": "You+DEP2@Gmail.com"},
        {"station": "MANT", "state": "pending", "email": None},
    ])
    assert imap.ledger_addresses(led) == {"you+dep1@gmail.com", "you+dep2@gmail.com"}


def test_ledger_addresses_survives_a_partial_line(tmp_path):
    """The ledger is appended to live; a torn last line must not stop the poll."""
    p = tmp_path / "l.jsonl"
    p.write_text('{"email": "you+a1@x.com"}\n\n{"email": "you+a2@x.co')
    assert imap.ledger_addresses(str(p)) == {"you+a1@x.com"}


def test_missing_ledger_owns_nothing(tmp_path):
    assert imap.ledger_addresses(str(tmp_path / "absent.jsonl")) == set()


def test_foreign_link_is_left_unread_and_not_pasted(tmp_path, monkeypatch):
    """The exact 2026-09-05 failure: depth's link arriving at the scr poller."""
    led = write_ledger(tmp_path / "scr.jsonl", [
        {"station": "PASA", "state": "submitted", "email": "you+s1@gmail.com"},
    ])
    ran = []
    monkeypatch.setattr(imap, "run", lambda cmd, dry: ran.append(cmd) or 0)
    imap._foreign.clear()

    m = FakeIMAP(message("you+dep3@gmail.com"))
    assert imap.handle(m, b"7", Args(led)) is None, (
        "a foreign message must return None -- poll() only leaves a message "
        "unread when handle returns None")
    assert ran == [], "the campaign was run on another ledger's link"
    assert m.stored == [], "the message was flagged and is now invisible to its owner"


def test_own_link_is_still_acted_on(tmp_path, monkeypatch):
    led = write_ledger(tmp_path / "scr.jsonl", [
        {"station": "PASA", "state": "submitted", "email": "you+s1@gmail.com"},
    ])
    ran = []
    monkeypatch.setattr(imap, "run", lambda cmd, dry: ran.append(cmd) or 0)
    imap._foreign.clear()

    result = imap.handle(FakeIMAP(message("you+s1@gmail.com")), b"8",
                         Args(led, dry_run=False))
    assert result is True
    assert any("paste" in c for c in ran[0]), ran


def test_claim_unknown_restores_the_old_behaviour(tmp_path, monkeypatch):
    """A single poller that owns the whole mailbox can still opt back in."""
    led = write_ledger(tmp_path / "scr.jsonl", [
        {"station": "PASA", "state": "submitted", "email": "you+s1@gmail.com"},
    ])
    ran = []
    monkeypatch.setattr(imap, "run", lambda cmd, dry: ran.append(cmd) or 0)
    imap._foreign.clear()

    imap.handle(FakeIMAP(message("you+dep3@gmail.com")), b"9",
                Args(led, claim_unknown=True, dry_run=False))
    assert ran, "--claim-unknown should have pasted the link"


def test_a_pending_only_ledger_claims_nothing(tmp_path, monkeypatch):
    """Having submitted from no address, it can own no link."""
    led = write_ledger(tmp_path / "fresh.jsonl", [
        {"station": "PASA", "state": "pending", "email": None},
    ])
    monkeypatch.setattr(imap, "run", lambda cmd, dry: pytest.fail("ran the campaign"))
    imap._foreign.clear()
    assert imap.handle(FakeIMAP(message("you+s1@gmail.com")), b"10", Args(led)) is None
