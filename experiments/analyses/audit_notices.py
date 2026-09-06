"""Cross-checks each auto-retired chunk against the notice that retired it.

A "veri bulunmamaktadir" email carries no window -- only the address it was
sent to -- so the poller matched it to whatever chunk that address held. That
is only valid if the notice arrived AFTER the chunk was submitted. This finds
the ones where it did not.
"""
import datetime
import email
import email.utils
import imaplib
import json
import os

m = imaplib.IMAP4_SSL(os.environ["AFAD_IMAP_HOST"])
m.login(os.environ["AFAD_IMAP_USER"], os.environ["AFAD_IMAP_PASS"])
m.select("INBOX", readonly=True)
typ, data = m.uid("search", None, '(FROM "tdvms@afad.gov.tr")')

notices = []
for uid in (data[0].split() if data[0] else []):
    typ, dd = m.uid("fetch", uid, "(BODY.PEEK[])")
    if not dd or not dd[0]:
        continue
    msg = email.message_from_bytes(dd[0][1])
    body = ""
    for part in msg.walk():
        if part.get_content_maintype() == "text":
            b = part.get_payload(decode=True)
            if b:
                body += b.decode(part.get_content_charset() or "utf-8", errors="replace")
    if "bulunmamaktad" in body and "tdvms.afad.gov.tr/files/" not in body:
        to = email.utils.parseaddr(msg.get("To"))[1].lower()
        dt = email.utils.parsedate_to_datetime(msg.get("Date")).astimezone(datetime.timezone.utc)
        notices.append((dt, to))
m.logout()
notices.sort()

print(f"{len(notices)} no-data notice email(s) in the mailbox:")
for dt, to in notices:
    print(f"   {dt:%H:%M:%S}Z  {to}")

rows = [json.loads(l) for l in open("gcam_ledger.jsonl") if l.strip()]
retired = [r for r in rows if r["state"] == "nodata" and "email, no link" in str(r.get("note"))]

print(f"\n{len(retired)} chunk(s) retired by an email notice:\n")
print(f"  {'window':<12}{'submitted':>11}{'earliest notice after it':>26}   verdict")
bad = []
for r in sorted(retired, key=lambda r: r["start"]):
    sub = datetime.datetime.fromisoformat(r["submitted_at"])
    later = [dt for dt, to in notices if to == r["email"] and dt > sub]
    if later:
        print(f"  {r['start'][:10]:<12}{sub:%H:%M:%S}Z{min(later):>21:%H:%M:%S}Z   ok")
    else:
        print(f"  {r['start'][:10]:<12}{sub:%H:%M:%S}Z{'none':>22}   STALE -> reset")
        bad.append(r["start"][:10])

print("\nwindows to reset:", " ".join(bad) if bad else "(none)")
