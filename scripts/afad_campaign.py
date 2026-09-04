"""TDVMS download campaign: a restartable ledger plus one-request-at-a-time submission.

TDVMS emails a download link rather than returning data, and processes one
request at a time **per email address**. A campaign is therefore hundreds of
human-in-the-loop round trips, and the only way that survives interruption is a
ledger that records exactly which (station, window) chunks are done.

Deliberately NOT automated end-to-end: requests are submitted one at a time and
links are pasted back in. That keeps the same human-in-the-loop shape as
`sismokaos/download/afad.py`, which documents why.

    python3 scripts/afad_campaign.py plan   --station MANT --chunk-days 21
    python3 scripts/afad_campaign.py next   --email you@example.com
    python3 scripts/afad_campaign.py paste  --url https://tdvms.afad.gov.tr/files/....zip
    python3 scripts/afad_campaign.py status

`plan` is idempotent; re-running it never duplicates or re-requests a chunk.
"""
import argparse
import contextlib
import fcntl
import json
import os
import pathlib
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone

import requests

STATIONS_URL = "https://tdvms.afad.gov.tr/api/Data/GetStations"
REQUEST_URL = "https://tdvmservis.afad.gov.tr/GetData"
LEDGER = pathlib.Path("afad_campaign_ledger.jsonl")

# TDVMS result codes, from the portal's own client.
RESULT_OK, RESULT_QUEUED, RESULT_ERROR, RESULT_BUSY = 0, 109, 110, 111


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_ledger(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def save_ledger(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@contextlib.contextmanager
def ledger_lock(path):
    """Serialises read-modify-write on the ledger.

    `paste` holds its in-memory copy for the several minutes a download takes,
    so writing that whole copy back silently discards anything `next` recorded
    meanwhile. That really happened: a submission to one address was erased, its
    window went back to pending, and the other address was then handed the same
    window -- a duplicate that costs a queue slot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def update_chunk(path, station, start, **changes):
    """Applies changes to one chunk against the CURRENT on-disk ledger.

    Never writes back a copy loaded earlier -- that is the race above.
    """
    with ledger_lock(path):
        rows = load_ledger(path)
        for r in rows:
            if r["station"] == station and r["start"] == start:
                r.update(changes)
                save_ledger(path, rows)
                return r
    raise KeyError(f"{station} {start} not in ledger")


def claim_next_pending(path, email):
    """Atomically takes the oldest pending chunk and marks it submitted, so two
    concurrent `next` calls cannot claim the same window."""
    with ledger_lock(path):
        rows = load_ledger(path)
        # "claimed" counts as in-flight too: a submission that is mid-POST holds
        # the address's queue slot just as much as a confirmed one. Checking only
        # for "submitted" lets a second call claim another window while the first
        # is still talking to TDVMS.
        held = [r for r in rows
                if r["state"] in ("claimed", "submitted") and r["email"] == email]
        if held:
            return None, held[0]
        todo = next((r for r in rows if r["state"] == "pending"), None)
        if todo is None:
            return None, None
        todo["state"], todo["email"] = "claimed", email
        save_ledger(path, rows)
        return todo, None


def cmd_plan(args):
    """Enumerates chunks. Never re-adds one already in the ledger."""
    with ledger_lock(args.ledger):
        return _plan_locked(args)


def _plan_locked(args):
    rows = load_ledger(args.ledger)
    have = {(r["station"], r["start"]) for r in rows}
    t, added = datetime.fromisoformat(args.start), 0
    end = datetime.fromisoformat(args.end)
    while t < end:
        nxt = min(t + timedelta(days=args.chunk_days), end)
        key = (args.station, t.isoformat())
        if key not in have:
            rows.append({"station": args.station, "start": t.isoformat(),
                         "end": nxt.isoformat(), "state": "pending",
                         "email": None, "url": None, "bytes": None, "note": None})
            added += 1
        t = nxt
    save_ledger(args.ledger, rows)
    pend = sum(1 for r in rows if r["state"] == "pending")
    print(f"planned {added} new chunk(s) for {args.station} at {args.chunk_days} d")
    print(f"ledger now holds {len(rows)} chunk(s), {pend} pending")


def _device_code(station_code):
    r = requests.post(STATIONS_URL, json={"netcodes": ["TU"], "deviceCode": "",
                                          "component": ""}, timeout=30)
    r.raise_for_status()
    for s in r.json():
        if s["code"] == station_code:
            for flag, code in (("deviceH", "H"), ("deviceL", "L"), ("deviceN", "N")):
                if s.get(flag):
                    return code
            raise SystemExit(f"{station_code}: no H/L/N device")
    raise SystemExit(f"{station_code}: not in the TDVMS station list")


def cmd_next(args):
    """Submits the oldest pending chunk. One at a time -- the queue slot is per email."""
    todo, busy = claim_next_pending(args.ledger, args.email)
    if todo is None and busy is not None:
        print(f"[BUSY] {args.email} already has a chunk in flight: "
              f"{busy['station']} {busy['start'][:10]}..{busy['end'][:10]}")
        print("       Paste its link first:  afad_campaign.py paste --url ...")
        return 1
    if todo is None:
        print("nothing pending — campaign complete")
        return 0

    dev = _device_code(todo["station"])
    payload = {
        "start_time": datetime.fromisoformat(todo["start"]).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromisoformat(todo["end"]).strftime("%Y-%m-%d %H:%M:%S"),
        "data_type": "mseed", "instrument": False,
        "networks": ["TU"], "stations": [todo["station"]], "location": [None],
        "device_codes": [dev], "components": [["Z", "N", "E"]],
        "e_mail": args.email,
    }
    print(f"submitting  TU.{todo['station']} ({dev})  "
          f"{todo['start'][:10]} -> {todo['end'][:10]}  -> {args.email}")
    try:
        resp = requests.post(REQUEST_URL, json=payload, timeout=args.timeout)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # AMBIGUOUS, not failed: TDVMS often accepts and then goes quiet. Keep the
        # claim -- releasing it would hand the same window to the other address.
        update_chunk(args.ledger, todo["station"], todo["start"], state="submitted",
                     submitted_at=_now(),
                     note=f"unconfirmed ({type(e).__name__}) — link may still arrive")
        print(f"[AMBIGUOUS] {type(e).__name__} — no answer from TDVMS.")
        print("            Kept as submitted; these are usually accepted anyway.")
        print("            If no email arrives, reset it with `reset --start <ISO>`.")
        return 0
    if resp.status_code != 200:
        update_chunk(args.ledger, todo["station"], todo["start"], state="pending", email=None)
        print(f"[HTTP {resp.status_code}] {resp.text[:200]}")
        return 1
    result = resp.json().get("Result")
    if result == RESULT_BUSY:
        update_chunk(args.ledger, todo["station"], todo["start"], state="pending", email=None)
        print("[111] the previous request for this address is still processing")
        return 1
    if result == RESULT_ERROR:
        update_chunk(args.ledger, todo["station"], todo["start"], state="pending", email=None)
        print(f"[110] TDVMS returned a general error: {resp.json()}")
        return 1
    update_chunk(args.ledger, todo["station"], todo["start"], state="submitted",
                 submitted_at=_now())
    print(f"[{result}] accepted — the link will arrive by email at {args.email}")
    return 0


def _window_from_name(name):
    """TDVMS names members TU_<STA>_<DDMMYYYY>_<HHMMSS>_<DDMMYYYY>_<HHMMSS>_<CH>.mseed.

    Matching on this rather than on ledger order matters: with two addresses in
    flight the links arrive interleaved, and "the oldest submitted chunk" is then
    simply the wrong answer -- it would file one window's data under another
    window's name.
    """
    m = re.match(r"TU_([A-Z0-9]+)_(\d{8})_(\d{6})_(\d{8})_(\d{6})_",
                 pathlib.Path(name).name)
    if not m:
        return None, None, None
    sta, d1, t1, d2, _ = m.groups()
    return sta, datetime.strptime(d1 + t1, "%d%m%Y%H%M%S"), \
        datetime.strptime(d2 + "000000", "%d%m%Y%H%M%S")


def _discard(zpath, from_file):
    """Removes the failed download -- unless it is the operator's own file.

    `--from-file` exists to re-ingest an archive already on disk. Deleting that
    on a failed parse destroys the very thing the flag was there to salvage.
    """
    if from_file:
        print(f"       kept {zpath} (yours, supplied with --from-file)")
        return
    zpath.unlink(missing_ok=True)


def cmd_paste(args):
    """Downloads a link, then matches it to its ledger chunk by the window
    encoded in the archive -- not by ledger order, which is wrong when two
    addresses have requests in flight simultaneously."""
    rows = load_ledger(args.ledger)
    # A link already banked usually means an older email resurfacing while
    # working through an inbox. Say so instead of spending 800 MB re-fetching
    # bytes we already verified.
    dup = next((r for r in rows if r.get("url") == args.url
                and r["state"] == "fetched"), None)
    if dup and not args.force:
        print(f"[SKIP] already fetched as {dup['station']} "
              f"{dup['start'][:10]}..{dup['end'][:10]} ({(dup['bytes'] or 0)/1e6:.1f} MB)")
        # Exit 2, not 0: nothing went wrong, but no archive landed either. A
        # caller driving the queue has to tell those apart -- treating a
        # duplicate as a fresh fetch made the poller refill a slot that was
        # never freed, and TDVMS answered [BUSY].
        return 2
    if not any(r["state"] == "submitted" for r in rows):
        print("no chunk is awaiting a link")
        return 1
    # Staged at the out-dir root, NOT under a station directory: which station
    # this archive belongs to is only known after the window is matched below.
    # Taking it from rows[0] filed every station's data under whichever station
    # happened to be first in the ledger, so a second station in the same ledger
    # wrote GCAM archives into afad_raw/MANT/.
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Unique per invocation: two links can be in flight at once (one per email
    # address), and a shared scratch name would have them overwrite each other.
    tgt = None
    if args.from_file:
        # Recovery path: the transfer succeeded but verification crashed, so the
        # bytes are already on disk. Re-downloading 800 MB to re-run a regex is
        # pure waste.
        zpath = pathlib.Path(args.from_file)
        if not zpath.exists():
            print(f"[FAIL] {zpath} does not exist")
            return 1
        print(f"ingesting existing {zpath} ({zpath.stat().st_size/1e6:.1f} MB)")
    else:
        zpath = out / f"incoming.{os.getpid()}.zip.part"
        print(f"fetching -> {zpath}")
        with requests.get(args.url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            # An expired or invalid link returns AFAD's homepage as text/html with
            # HTTP 200 -- not a 404. Checking the type first avoids streaming a
            # few KB of HTML into a .zip, and would avoid streaming 800 MB of
            # anything else that is not an archive.
            ctype = resp.headers.get("Content-Type", "")
            if "html" in ctype.lower():
                print(f"[FAIL] link returned {ctype} ({resp.headers.get('Content-Length','?')} B), "
                      "not an archive — it has expired or the request errored.")
                print("       Reset the window and request it again:")
                print("         afad_campaign.py reset --start <YYYY-MM-DD>")
                return 1
            with open(zpath, "wb") as f:
                for chunk in resp.iter_content(1 << 16):
                    f.write(chunk)
    size = zpath.stat().st_size

    # A 181-day request once returned a 4 KB empty zip. Verify, do not assume.
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
    except zipfile.BadZipFile:
        print(f"[FAIL] {size} bytes and not a valid zip — nothing recorded")
        _discard(zpath, args.from_file)
        return 1
    if not names:
        # 22 bytes is a bare end-of-central-directory: TDVMS built an archive and
        # put nothing in it. Seen on 21-day windows, which are known good, so the
        # old "window is too long" guess was wrong and sent operators chasing it.
        print(f"[FAIL] zip is empty ({size} B) — TDVMS produced no archive content. "
              "Reset the window and request it again.")
        _discard(zpath, args.from_file)
        return 1

    # "No data" does NOT arrive as an empty zip. TDVMS returns a well-formed
    # archive holding one 82-byte notice --
    #   "Seçilen istasyon içerisinde geçerli zaman aralığında veri bulunamamıştır."
    # -- which passes the emptiness check above and was banked as a successful
    # fetch for three windows before anyone looked at the byte counts. Require
    # actual waveform members, not merely members.
    if not any(n.lower().endswith(".mseed") for n in names):
        with zipfile.ZipFile(zpath) as zf:
            notice = zf.read(names[0])[:200].decode("utf-8", errors="replace").strip()
        sta, start, _ = _window_from_name(names[0])
        tgt = next((r for r in rows if r["station"] == sta
                    and r["start"][:10] == start.strftime("%Y-%m-%d")), None) \
            if start is not None else None
        print(f"[NO DATA] TDVMS holds no waveform for this window:")
        print(f"          {notice}")
        if tgt is not None:
            update_chunk(args.ledger, tgt["station"], tgt["start"],
                         state="nodata", fetched_at=_now(), url=args.url,
                         bytes=size, note=notice)
            print(f"          recorded as nodata: {tgt['start'][:10]}..{tgt['end'][:10]}")
        else:
            print("          could not match it to a ledger chunk — nothing recorded")
        _discard(zpath, args.from_file)
        return 1

    sta, start, end = _window_from_name(names[0])
    if start is not None:
        tgt = next((r for r in rows if r["station"] == sta
                    and r["start"][:10] == start.strftime("%Y-%m-%d")), None)
    if tgt is None:
        print(f"[WARN] could not match '{names[0]}' to a ledger chunk; "
              f"falling back to the oldest submitted one")
        tgt = next(r for r in rows if r["state"] == "submitted")
    else:
        print(f"       matched to {tgt['station']} {tgt['start'][:10]}..{tgt['end'][:10]}"
              f" ({tgt['email'] or 'unknown address'})")

    dest = out / tgt["station"]
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / f"{tgt['station']}_{tgt['start'][:10]}.zip"
    # Never replace an existing archive silently. A mislabelled earlier download
    # once left the wrong window under this name, and the corrected download then
    # overwrote 825 MB of good data that had to be re-fetched. If something is
    # already here, keep both and let the operator decide.
    if final.exists() and not args.force:
        keep = dest / f"{tgt['station']}_{tgt['start'][:10]}.dup{os.getpid()}.zip"
        zpath.replace(keep)
        print(f"[WARN] {final.name} already exists — saved this one as {keep.name}")
        print("       Inspect both, delete the wrong one, then rename. "
              "Re-run with --force to overwrite instead.")
        return 1
    zpath.replace(final)
    update_chunk(args.ledger, tgt["station"], tgt["start"], state="fetched",
                 fetched_at=_now(), url=args.url, bytes=size,
                 note=f"{len(names)} file(s)")
    print(f"[OK] {size/1e6:.1f} MB, {len(names)} file(s) — {final}")
    print(f"     {sum(1 for r in rows if r['state']=='pending')} chunk(s) still pending")
    return 0


def cmd_reset(args):
    """Puts a chunk back to pending — for an unconfirmed submission whose email
    never arrived."""
    with ledger_lock(args.ledger):
        return _reset_locked(args)


def _reset_locked(args):
    rows = load_ledger(args.ledger)
    hit = [r for r in rows if r["start"].startswith(args.start)]
    if not hit:
        print(f"no chunk starting {args.start}")
        return 1
    for r in hit:
        print(f"  {r['station']} {r['start'][:10]}: {r['state']} -> pending")
        r["state"], r["email"], r["note"] = "pending", None, None
    save_ledger(args.ledger, rows)
    return 0


def cmd_status(args):
    rows = load_ledger(args.ledger)
    if not rows:
        print("ledger is empty — run `plan` first")
        return 0
    by = {}
    for r in rows:
        by.setdefault(r["state"], []).append(r)
    print(f"{len(rows)} chunk(s) in {args.ledger}")
    for st in ("pending", "claimed", "submitted", "fetched", "nodata", "failed"):
        if st in by:
            mb = sum(r["bytes"] or 0 for r in by[st]) / 1e6
            print(f"  {st:10s} {len(by[st]):4d}" + (f"   {mb:.1f} MB" if mb else ""))
    for r in by.get("submitted", []):
        print(f"  awaiting link: {r['station']} {r['start'][:10]}..{r['end'][:10]} ({r['email']})")
    for r in by.get("claimed", []):
        print(f"  STUCK in claimed: {r['station']} {r['start'][:10]} ({r['email']}) — "
              f"a submission died mid-POST; `reset --start {r['start'][:10]}` to requeue")
    for r in by.get("nodata", []):
        print(f"  no waveform at source: {r['station']} {r['start'][:10]}..{r['end'][:10]}"
              " — re-requesting cannot recover it")
    for r in by.get("failed", []):
        print(f"  FAILED: {r['station']} {r['start'][:10]} — {r['note']}")
    _print_turnaround(rows)
    return 0


def _print_turnaround(rows):
    """Turnaround was guessed at for the first eleven chunks because nothing
    recorded when a request went out. Rows written before the stamps exist are
    simply skipped rather than estimated."""
    laps = []
    for r in rows:
        if r.get("submitted_at") and r.get("fetched_at"):
            laps.append((datetime.fromisoformat(r["fetched_at"])
                         - datetime.fromisoformat(r["submitted_at"])).total_seconds() / 60)
    if not laps:
        return
    laps.sort()
    mid = laps[len(laps) // 2]
    print(f"  turnaround  n={len(laps)}  median {mid:.0f} min  "
          f"range {laps[0]:.0f}-{laps[-1]:.0f} min")
    if len(laps) < len(rows):
        print(f"              ({len(rows) - len(laps)} chunk(s) predate the stamps)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", type=pathlib.Path, default=LEDGER)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan"); pl.set_defaults(fn=cmd_plan)
    pl.add_argument("--station", required=True)
    pl.add_argument("--start", default="2024-05-01")
    pl.add_argument("--end", default="2026-08-10")
    pl.add_argument("--chunk-days", type=int, default=21)

    nx = sub.add_parser("next"); nx.set_defaults(fn=cmd_next)
    nx.add_argument("--email", required=True)
    nx.add_argument("--timeout", type=int, default=180,
                    help="seconds to wait for TDVMS to answer a submission")
    nx.add_argument("--force", action="store_true",
                    help="submit even if this address already has one in flight")

    pa = sub.add_parser("paste"); pa.set_defaults(fn=cmd_paste)
    pa.add_argument("--url", required=True)
    pa.add_argument("--from-file", default=None,
                    help="ingest an already-downloaded archive instead of fetching")
    pa.add_argument("--out-dir", default="afad_raw")
    pa.add_argument("--force", action="store_true",
                    help="overwrite an existing archive for this window")

    rs = sub.add_parser("reset"); rs.set_defaults(fn=cmd_reset)
    rs.add_argument("--start", required=True, help="ISO start of the chunk, e.g. 2024-05-01")

    st = sub.add_parser("status"); st.set_defaults(fn=cmd_status)

    args = p.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
