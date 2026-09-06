"""Does one TDVMS request return more than one station?

The whole cost model turns on this. A request occupies one queue slot for
~31 minutes regardless of size, so if the `stations` list is honoured rather
than ignored, N stations cost one slot instead of N and every plan involving
more than a couple of stations divides by N.

The payload's parallel-array shape (stations / location / device_codes /
components) suggests it is supported, but nothing has ever tested it.

Deliberately a 1-day window: if this returns two stations' data the answer is
yes, and a short window keeps the failure cheap if the API rejects the shape.
CGC and MTOP are chosen because they are in no ledger, so the reply cannot be
mis-filed into a running campaign -- paste refuses an unmatched archive.
"""
import json
import sys
from datetime import datetime

import requests

REQUEST_URL = "https://tdvmservis.afad.gov.tr/GetData"
RESULT = {0: "OK", 109: "accepted/queued", 110: "general error", 111: "address busy"}

EMAIL = sys.argv[1] if len(sys.argv) > 1 else None
if not EMAIL:
    sys.exit("usage: multistation_probe.py <email>")

STATIONS = ["CGC"]
START, END = "2024-06-01 00:00:00", "2024-06-02 00:00:00"

variants = [
    ("networks once", {
        "networks": ["TU"],
        "stations": STATIONS,
        "location": [None] * len(STATIONS),
        "device_codes": ["H"] * len(STATIONS),
        "components": [["Z", "N", "E"]] * len(STATIONS),
    }),
    ("networks per station", {
        "networks": ["TU"] * len(STATIONS),
        "stations": STATIONS,
        "location": [None] * len(STATIONS),
        "device_codes": ["H"] * len(STATIONS),
        "components": [["Z", "N", "E"]] * len(STATIONS),
    }),
]

for name, extra in variants:
    payload = {"start_time": START, "end_time": END,
               "data_type": "mseed", "instrument": False,
               "e_mail": EMAIL, **extra}
    print(f"--- {name} ---")
    print("   ", json.dumps({k: v for k, v in payload.items() if k != "e_mail"}))
    try:
        r = requests.post(REQUEST_URL, json=payload, timeout=180)
    except Exception as e:
        print(f"    {type(e).__name__}: {e}")
        print("    ambiguous — TDVMS often accepts then goes quiet; watch the mailbox")
        break
    print(f"    HTTP {r.status_code}")
    body = r.text[:300]
    try:
        res = r.json().get("Result")
        print(f"    Result {res} = {RESULT.get(res, 'unknown')}")
        if res in (0, 109):
            print("    ACCEPTED — wait for the mail and inspect the archive members")
            break
        if res == 111:
            print("    this address already has a request in flight; use another")
            break
    except Exception:
        print(f"    non-JSON body: {body}")
