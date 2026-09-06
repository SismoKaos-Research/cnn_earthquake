"""Does ONE TDVMS request return more than one station?

The whole cost model turns on this. A request occupies one queue slot for ~31
minutes regardless of size, so if the `stations` list is honoured rather than
ignored, N stations cost one slot instead of N and every plan involving more
than a couple of stations divides by N.

The payload's parallel-array shape (stations / location / device_codes /
components) suggests it is supported.

**ANSWERED 2026-09-06: no.** A request for `[MANT, DEMI]` on a day both
stations demonstrably recorded came back with one member, MANT, and a mail
saying the data for "the station(s) you requested" was ready. N stations cost N
slots. See `docs/experiment_tdvms_multistation_2026-09-06.md`; this script is
kept because the reversed-order case is still open and because a service can
change.

**It had not been tested when this was rewritten (2026-09-06), despite two
scripts existing that said they did.** `multistation_probe.py` requested
`["GCAM"]` and `singlestation_control.py` requested `["CGC"]` -- both single
-station, differing only in which station and which day. They were the same
experiment twice, and the question they name went unanswered. The two are one
parameterised script now, so probe and control cannot drift apart again.

    # the probe: two stations in one request
    multistation_probe.py --email you+ms1@example.com --stations CGC MTOP
    # the control: one station, same shape
    multistation_probe.py --email you+ms2@example.com --stations CGC

Deliberately a 1-day window: a short window keeps the failure cheap if the API
rejects the shape. CGC and MTOP are chosen because they are in no ledger, so the
reply cannot be mis-filed into a running campaign -- `paste` refuses an
unmatched archive.

**An accepted request is not the answer.** TDVMS may accept the payload and
still return one station's data. The evidence is what the ARCHIVE holds, so the
reply has to be fetched and its members listed; this prints the command for that
rather than implying the Result code settles it.
"""
import argparse
import json
import sys
from datetime import datetime

import requests

REQUEST_URL = "https://tdvmservis.afad.gov.tr/GetData"
STATIONS_URL = "https://tdvms.afad.gov.tr/api/Data/GetStations"
RESULT = {0: "OK", 109: "accepted/queued", 110: "general error", 111: "address busy"}


def servable(codes, network):
    """Device code per station, or exits naming the ones TDVMS will not serve.

    Checked BEFORE the request is sent. BAKC and IRLI sat `claimed` for an
    afternoon holding two queue slots because an unlistable station was only
    discovered after the ledger had been written.
    """
    r = requests.post(STATIONS_URL, json={"netcodes": [network], "deviceCode": "",
                                          "component": ""}, timeout=30)
    r.raise_for_status()
    have = {s["code"]: s for s in r.json()}
    out, missing = [], []
    for c in codes:
        s = have.get(c)
        dev = next((code for flag, code in (("deviceH", "H"), ("deviceL", "L"),
                                            ("deviceN", "N")) if s and s.get(flag)), None)
        (out.append(dev) if dev else missing.append(c))
    if missing:
        sys.exit(f"not in the TDVMS {network} station list: {missing}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--email", required=True, help="a plus-address in NO ledger")
    p.add_argument("--stations", nargs="+", required=True,
                   help="two or more for the probe, one for the control")
    p.add_argument("--network", default="TU")
    p.add_argument("--start", default="2024-06-01 00:00:00")
    p.add_argument("--end", default="2024-06-02 00:00:00")
    p.add_argument("--dry-run", action="store_true",
                   help="check the stations and print the payload; send nothing")
    return p.parse_args()


def main():
    """Sends one request for N stations and reports what TDVMS said."""
    args = parse_args()
    n = len(args.stations)
    devs = servable(args.stations, args.network)
    print(f"  {n} station(s): " + ", ".join(f"{c}({d})" for c, d in zip(args.stations, devs)))
    print(f"  {'PROBE -- more than one station in one request' if n > 1 else 'CONTROL -- one station'}")

    payload = {
        "start_time": args.start, "end_time": args.end,
        "data_type": "mseed", "instrument": False,
        "networks": [args.network], "stations": list(args.stations),
        "location": [None] * n, "device_codes": devs,
        "components": [["Z", "N", "E"]] * n,
        "e_mail": args.email,
    }
    if args.dry_run:
        print("  --dry-run, payload:\n" + json.dumps(payload, indent=2))
        return 0

    print(f"  sending to {args.email} at {datetime.now():%H:%M:%S} ...")
    r = requests.post(REQUEST_URL, json=payload, timeout=180)
    print(f"  HTTP {r.status_code}")
    try:
        res = r.json().get("Result")
        print(f"  Result {res} = {RESULT.get(res, 'unknown')}")
    except ValueError:
        print(f"  non-JSON body: {r.text[:300]}")
        return 1

    if res in (0, 109):
        print("\n  ACCEPTED -- and that is NOT the answer. TDVMS may accept the")
        print("  payload and still return one station. Fetch the reply and list")
        print("  the archive members; the count of distinct stations in it is")
        print("  the result:")
        print(f"    sk poll --ledger /dev/null --out-dir /tmp/msprobe \\")
        print(f"        --search '(UNSEEN FROM \"tdvms@afad.gov.tr\")' --once --claim-unknown")
        print(f"    unzip -l /tmp/msprobe/*.zip | grep -oE '[A-Z]{{3,5}}\\.[A-Z]+' | sort -u")
        return 0
    if res == 111:
        print("  this address already has a request in flight; use another")
    return 1


if __name__ == "__main__":
    sys.exit(main())
