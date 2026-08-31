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
import json
import pathlib
import re
import sys
import zipfile
from datetime import datetime, timedelta

import requests

STATIONS_URL = "https://tdvms.afad.gov.tr/api/Data/GetStations"
REQUEST_URL = "https://tdvmservis.afad.gov.tr/GetData"
LEDGER = pathlib.Path("afad_campaign_ledger.jsonl")

# TDVMS result codes, from the portal's own client.
RESULT_OK, RESULT_QUEUED, RESULT_ERROR, RESULT_BUSY = 0, 109, 110, 111


def load_ledger(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def save_ledger(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def cmd_plan(args):
    """Enumerates chunks. Never re-adds one already in the ledger."""
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
    rows = load_ledger(args.ledger)
    inflight = [r for r in rows if r["state"] == "submitted" and r["email"] == args.email]
    if inflight and not args.force:
        r = inflight[0]
        print(f"[BUSY] {args.email} already has a chunk in flight: "
              f"{r['station']} {r['start'][:10]}..{r['end'][:10]}")
        print("       Paste its link first:  afad_campaign.py paste --url ...")
        return 1
    todo = next((r for r in rows if r["state"] == "pending"), None)
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
    # 30 s is not enough: TDVMS does real queueing work before answering, and a
    # client-side timeout here is ambiguous -- the request may well have been
    # accepted. Retrying after one is safe, because a live request makes the next
    # submission return 111 (busy) rather than silently duplicating.
    try:
        resp = requests.post(REQUEST_URL, json=payload, timeout=args.timeout)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # AMBIGUOUS, not failed: TDVMS frequently accepts the request and then
        # times out or drops the connection instead of answering. Leaving the
        # chunk `pending` is the wrong call -- the next `next` then re-grabs the
        # same window and burns the other address's queue slot on a duplicate,
        # which is exactly what happened on 2026-08-31. Record it as submitted
        # and flag the uncertainty instead.
        todo["state"], todo["email"] = "submitted", args.email
        todo["note"] = f"unconfirmed ({type(e).__name__}) — link may still arrive"
        save_ledger(args.ledger, rows)
        print(f"[AMBIGUOUS] {type(e).__name__} — no answer from TDVMS.")
        print("            Recorded as submitted: these are usually accepted anyway,")
        print("            and a duplicate wastes a slot. If no email arrives, reset")
        print("            it with `reset --start <ISO>`.")
        return 0
    if resp.status_code != 200:
        print(f"[HTTP {resp.status_code}] {resp.text[:200]}")
        return 1
    result = resp.json().get("Result")
    if result == RESULT_BUSY:
        print("[111] the previous request for this address is still processing")
        return 1
    if result == RESULT_ERROR:
        print(f"[110] TDVMS returned a general error: {resp.json()}")
        return 1
    todo["state"], todo["email"] = "submitted", args.email
    save_ledger(args.ledger, rows)
    print(f"[{result}] accepted — the link will arrive by email at {args.email}")
    return 0


def _window_from_name(name):
    """TDVMS names members TU_<STA>_<DDMMYYYY>_<HHMMSS>_<DDMMYYYY>_<HHMMSS>_<CH>.mseed.

    Matching on this rather than on ledger order matters: with two addresses in
    flight the links arrive interleaved, and 'the oldest submitted chunk' is then
    simply the wrong answer -- it would file a window's data under a different
    window's name.
    """
    m = re.match(r"TU_([A-Z0-9]+)_(\d{8})_(\d{6})_(\d{8})_(\d{6})_", pathlib.Path(name).name)
    if not m:
        return None, None, None
    sta, d1, t1, d2, _ = m.groups()
    start = datetime.strptime(d1 + t1, "%d%m%Y%H%M%S")
    end = datetime.strptime(d2 + "000000", "%d%m%Y%H%M%S")
    return sta, start, end


def cmd_paste(args):
    """Downloads a link, then matches it to its ledger chunk by the window
    encoded in the archive -- not by ledger order, which is wrong when two
    addresses have requests in flight simultaneously."""
    rows = load_ledger(args.ledger)
    if not any(r["state"] == "submitted" for r in rows):
        print("no chunk is awaiting a link")
        return 1
    out = pathlib.Path(args.out_dir) / rows[0]["station"]
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "incoming.zip.part"
    tgt = None
    print(f"fetching -> {zpath}")
    with requests.get(args.url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
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
        zpath.unlink(missing_ok=True)
        return 1
    if not names:
        print(f"[FAIL] zip is empty ({size} B) — the window is probably too long")
        zpath.unlink(missing_ok=True)
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

    final = out / f"{tgt['station']}_{tgt['start'][:10]}.zip"
    zpath.replace(final)
    tgt.update(state="fetched", url=args.url, bytes=size, note=f"{len(names)} file(s)")
    save_ledger(args.ledger, rows)
    print(f"[OK] {size/1e6:.1f} MB, {len(names)} file(s) — {final}")
    print(f"     {sum(1 for r in rows if r['state']=='pending')} chunk(s) still pending")
    return 0


def cmd_reset(args):
    """Puts a chunk back to pending — for an unconfirmed submission whose email
    never arrived."""
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
    for st in ("pending", "submitted", "fetched", "failed"):
        if st in by:
            mb = sum(r["bytes"] or 0 for r in by[st]) / 1e6
            print(f"  {st:10s} {len(by[st]):4d}" + (f"   {mb:.1f} MB" if mb else ""))
    for r in by.get("submitted", []):
        print(f"  awaiting link: {r['station']} {r['start'][:10]}..{r['end'][:10]} ({r['email']})")
    for r in by.get("failed", []):
        print(f"  FAILED: {r['station']} {r['start'][:10]} — {r['note']}")
    return 0


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
    pa.add_argument("--out-dir", default="afad_raw")

    rs = sub.add_parser("reset"); rs.set_defaults(fn=cmd_reset)
    rs.add_argument("--start", required=True, help="ISO start of the chunk, e.g. 2024-05-01")

    st = sub.add_parser("status"); st.set_defaults(fn=cmd_status)

    args = p.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
