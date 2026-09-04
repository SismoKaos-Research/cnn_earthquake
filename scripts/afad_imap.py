"""Watches a mailbox for TDVMS download links and feeds them to the campaign.

The queue itself cannot be made faster -- TDVMS takes what it takes -- but there
is no reason for a human to sit in front of it. This closes the loop: poll the
inbox, pull the link out of each new mail, hand it to `afad_campaign.py paste`,
and refill the queue slot that just freed.

Plus-addressing is what makes the refill precise. TDVMS keys its one-request-at-
a-time limit on the literal address string, so `you+a1@x` and `you+a2@x` are
separate slots while the mail all lands in one inbox. The address a link was
delivered to therefore names exactly which slot is now free, and only that one
is refilled -- refilling blindly would draw a `111` from every busy address.

    export AFAD_IMAP_HOST=imap.gmail.com
    export AFAD_IMAP_USER=you@gmail.com
    export AFAD_IMAP_PASS='<app password, not the account password>'

    python3 scripts/afad_imap.py --interval 60 --pump
    python3 scripts/afad_imap.py --once --dry-run    # see what it would do

Credentials come from the environment only; nothing is written to disk but the
failure log.
"""
import argparse
import email
import email.header
import email.utils
import imaplib
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

CAMPAIGN = pathlib.Path(__file__).resolve().parent / "afad_campaign.py"
LINK_RE = re.compile(r'https?://tdvms\.afad\.gov\.tr/[^\s"\'<>\\]+\.zip')
DEFAULT_FAIL_LOG = "afad_imap_failures.log"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", default="afad_campaign_ledger.jsonl")
    p.add_argument("--out-dir", default="afad_raw")
    p.add_argument("--folder", default=os.environ.get("AFAD_IMAP_FOLDER", "INBOX"))
    p.add_argument("--fail-log", default=DEFAULT_FAIL_LOG,
                   help="where dead links are recorded; give each station its own "
                        "when several pollers run side by side")
    p.add_argument("--interval", type=int, default=60,
                   help="seconds between polls; ignored with --once")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--pump", action="store_true",
                   help="after a successful paste, submit the next chunk to the "
                        "address that link was delivered to")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would happen; touches neither mail nor ledger")
    return p.parse_args()


def failed_urls(path):
    """URLs already known dead, so a permanent failure is not re-downloaded.

    Read fresh each pass rather than cached: the log is the durable record, and
    a run that restarts must not start retrying links it already gave up on.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out.add(parts[2])
    return out


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def connect(args):
    host = os.environ.get("AFAD_IMAP_HOST")
    user = os.environ.get("AFAD_IMAP_USER")
    pw = os.environ.get("AFAD_IMAP_PASS")
    missing = [n for n, v in (("AFAD_IMAP_HOST", host), ("AFAD_IMAP_USER", user),
                              ("AFAD_IMAP_PASS", pw)) if not v]
    if missing:
        raise SystemExit(f"missing environment: {', '.join(missing)}")
    m = imaplib.IMAP4_SSL(host)
    m.login(user, pw)
    m.select(args.folder)
    return m


def recipient_of(msg):
    """Which plus-address this link was sent to -- i.e. which slot just freed.

    `To` is what TDVMS addressed; Delivered-To/X-Original-To are what the server
    actually routed, and survive some forwarding setups that rewrite `To`.
    """
    for header in ("To", "Delivered-To", "X-Original-To"):
        for _, addr in email.utils.getaddresses(msg.get_all(header, [])):
            if "@" in addr:
                return addr.strip().lower()
    return None


def links_in(msg):
    """Every TDVMS link in the message, deduped, order preserved.

    Walks all parts: the mail arrives as HTML often enough that scanning only
    text/plain silently sees nothing at all.
    """
    found = []
    for part in msg.walk():
        if part.get_content_maintype() != "text":
            continue
        try:
            body = part.get_payload(decode=True)
        except Exception:
            continue
        if not body:
            continue
        text = body.decode(part.get_content_charset() or "utf-8", errors="replace")
        for url in LINK_RE.findall(text):
            if url not in found:
                found.append(url)
    return found


def run(cmd, dry):
    if dry:
        log(f"  DRY  would run: {' '.join(cmd[1:])}")
        return 0
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.strip():
            log(f"  | {line}")
    return r.returncode


def handle(m, uid, args):
    """Returns True when the message is fully dealt with and can be marked seen."""
    # PEEK: a plain fetch sets \Seen as a side effect, which would consume the
    # message even if the paste below fails and the link still needs acting on.
    typ, data = m.uid("fetch", uid, "(BODY.PEEK[])")
    if typ != "OK" or not data or not data[0]:
        log(f"uid {uid.decode()}: could not fetch")
        return False
    msg = email.message_from_bytes(data[0][1])
    subj = str(email.header.make_header(email.header.decode_header(msg.get("Subject", ""))))
    urls = links_in(msg)
    if not urls:
        return True  # not a TDVMS mail; nothing to do with it, stop re-reading it
    to = recipient_of(msg)
    log(f"uid {uid.decode()}: {len(urls)} link(s), to {to or 'unknown'} — {subj[:60]}")

    pasted = failures = 0
    dead = failed_urls(args.fail_log)
    for url in urls:
        if url in dead:
            log(f"  already failed before, not re-fetching: {url.rsplit('/', 1)[-1]}")
            failures += 1
            continue
        rc = run([str(CAMPAIGN), "--ledger", args.ledger, "paste",
                  "--url", url, "--out-dir", args.out_dir], args.dry_run)
        if rc == 0:
            pasted += 1
        elif rc == 2:
            # Already banked -- an older mail resurfacing. Handled, but no slot
            # was freed, so it must not trigger a refill.
            log("  link was already banked; nothing new landed")
        else:
            failures += 1
            # The bytes are gone but the URL must not be: an expired link has to
            # be reset and re-requested, and that needs the window it belonged to.
            if not args.dry_run:
                with pathlib.Path(args.fail_log).open("a") as fh:
                    fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
                             f"\t{to}\t{url}\n")
                log(f"  recorded in {args.fail_log} — `reset --start <ISO>` to requeue")
    # Only an archive that actually landed frees a queue slot. Refilling after a
    # skipped dead link asks an address that still has a chunk in flight for
    # another one, and TDVMS answers [BUSY] -- noise that reads like a fault.
    if pasted and args.pump and to:
        run([str(CAMPAIGN), "--ledger", args.ledger, "next", "--email", to], args.dry_run)
    return failures == 0


def poll(m, args):
    typ, data = m.uid("search", None, "(UNSEEN)")
    if typ != "OK":
        log("search failed")
        return
    uids = data[0].split()
    if not uids:
        return
    log(f"{len(uids)} unread message(s)")
    for uid in uids:
        ok = handle(m, uid, args)
        if args.dry_run:
            continue
        # Mark it read either way. Leaving a failure unread meant a dead link was
        # re-fetched every single tick, forever -- three of them ran 21 times in
        # seven minutes. The URL is preserved in the failure log, which is what
        # makes the message safe to consume; \Flagged leaves it visible in the
        # mailbox so it is not simply lost.
        flags = "(\\Seen)" if ok else "(\\Seen \\Flagged)"
        m.uid("store", uid, "+FLAGS", flags)


def main():
    args = parse_args()
    if args.dry_run:
        log("dry run — no mail flagged, no ledger writes")
    while True:
        try:
            m = connect(args)
            try:
                poll(m, args)
            finally:
                try:
                    m.logout()
                except Exception:
                    pass
        except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError) as e:
            # Long-lived IMAP connections get dropped by the server routinely.
            # Reconnecting next tick is normal operation, not an error worth exiting on.
            log(f"connection problem ({type(e).__name__}: {e}) — retrying next tick")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
