# AFAD TDVMS download campaign — plan

Written 2026-08-30, after the catalogue audit that revised the previous
station shortlist.

---

## 0. What changed since the last shortlist

Last session's greedy pick (`ZEDA, KHAL, AKUM, BEYE, IZZE, STEP`) was computed
over `deprem_katalog_utc.csv`, which is missing 96% of the February 2025 Aegean
swarm. Re-running over AFAD's live catalogue (9,301 nationwide events M>=2.5,
2024-05-01 → 2026-08-10) shows the defect matters **only at the wider radius**:

**R = 100 km — ranking is unaffected by the catalogue defect.** Both catalogues
return the same six stations with the same absolute event counts:

| # | station | new events | cum. (complete cat.) |
|---|---|---|---|
| 1 | **MANT** (Kula, Manisa) | 2,582 | 27.8% |
| 2 | MTOP (Malatya) | 940 | 37.9% |
| 3 | KRT (Osmaniye) | 588 | 44.2% |
| 4 | BAND (Bandırma) | 340 | 47.8% |
| 5 | KARB (Karaburun, İzmir) | 319 | 51.3% |
| 6 | TASL (Marmaris, Muğla) | 295 | 54.4% |

The counts are *identical* between catalogues because the missing swarm sits
offshore, beyond 100 km of every TU station — so it cannot influence a 100 km
selection either way.

**R = 200 km — ranking does flip.** Here the swarm is in range and the pick
changes at rank 1:

| # | complete catalogue | | incomplete (`deprem_katalog_utc`) | |
|---|---|---|---|---|
| 1 | **GCAM** (Kuşadası, Aydın) | 3,957 | STEP (Balıkesir) | 3,379 |
| 2 | CGC (K.maraş) | 1,727 | CGC | 1,724 |
| 3 | BALY (Balya, Balıkesir) | 856 | TASL (Muğla) | 653 |
| 4 | YAZI (Datça, Muğla) | 590 | MUSM (Muş) | 498 |
| 5 | MUSM (Muş) | 500 | KEPZ (Antalya) | 200 |

GCAM does not appear in the incomplete top five at all; STEP — one of last
session's six — does not appear in the complete one. **STEP was selected on the
strength of events the catalogue was missing.**

All candidates carry a broadband `H` device in TDVMS (387 of 390 TU stations
do), so the `device_codes` payload field resolves to `"H"` for each.

Re-run the selection when the replacement catalogue lands:
`scripts/select_afad_stations.py`, which reads either catalogue layout.

---

## 1. The binding constraint is disk, not the queue

Measured, not estimated: one 100 Hz 3-component station-day of miniSEED is
**37.1 MB** (TU.ANDN, 2025-06-10, via ORFEUS).

| stations | full span (831 d) | fits in 166 GB free? |
|---|---|---|
| 1 | 30.8 GB | yes |
| 4 | 123.3 GB | yes, 43 GB spare |
| 6 | 184.9 GB | **no** |
| 8 | 246.6 GB | no |

Existing footprint already on disk: 55 GB raw (BODT+DAT) + 26 GB `sismokaos-cli`.

**Recommended: extract and discard.** The 34 GB BODT archive yields an 819 MB
feature parquet — a 40× reduction. Pipeline each chunk as
`download → verify → extract features → delete raw`, and 6 stations costs
~5 GB of features instead of 185 GB of waveform.

Trade-off to accept deliberately: discarding raw means a change to window
geometry or sample rate requires re-downloading through the queue. Mitigation —
**keep raw for one station** (GCAM, the Aegean anchor) as the re-extraction
reference, discard for the rest. That is 31 GB, which fits comfortably.

---

## 2. Chunk size — the one open variable

Verified last session: 1 h / 1 d / 3 d / 7 d all return complete data at
0.000% gaps. 30 d errors. 181 d returns an empty 4 KB zip.

Two bracket probes were queued and their results were never recorded:

- **21-day** → `oguz.bolat@std.yildiz.edu.tr`
- **14-day** → `hoguzbolat@gmail.com`

Check whether those two emails arrived and whether their zips are complete.
Request count for the 831-day span turns on the answer:

| chunk | requests/station | 4 stations | 6 stations |
|---|---|---|---|
| 7 d (verified) | 119 | 476 | 714 |
| 14 d (unverified) | 60 | 240 | 360 |
| 21 d (unverified) | 40 | 160 | 240 |

7 d is the safe fallback and costs ~3× the requests of 21 d.

---

## 3. Queue mechanics

- TDVMS processes **one request at a time per email address** — the slot is
  per-email, not per-IP. Two addresses = two concurrent slots.
- Result codes: `0` success, `109` accepted/queued, `110` general error,
  `111` wait for previous request.
- `111` is the normal signal that a slot is still busy; it is the poll
  condition, not a failure.
- Results arrive as an emailed `https://tdvms.afad.gov.tr/files/*.zip` link.
  Links have been directly fetchable with `curl`/`wget`.

Per-request turnaround has not been measured. **Measure it on the first
half-dozen** before committing to a schedule — total wall-clock is
`requests / 2 slots × turnaround`, and that number is the whole plan.

---

## 4. What to build

`sismokaos/download/afad.py` today is one station, one request, manual confirm,
manual link paste. It is right for a one-off and wrong for 240–714 requests.
Needed:

1. **A request ledger** — SQLite or a JSONL file, one row per
   `(station, start, end)` with state `pending → submitted → link → fetched →
   extracted → verified`. Restartable; never re-requests a completed chunk.
2. **A slot manager** — submits the next pending chunk when a `111` clears,
   one worker per email address. Pacing between requests is a policy call, so
   make it a config value rather than a hard-coded constant.
3. **Link ingestion** — either IMAP polling of the two mailboxes, or a
   `paste-link` subcommand that attaches a URL to the oldest submitted chunk.
   IMAP is the difference between attended and unattended operation.
4. **On-arrival verification** — checksum, unzip, assert the returned span
   matches the requested span, assert gap rate below threshold, and *only then*
   mark the chunk done and delete the raw.
5. **Feature extraction hook** — run `sismokaos-cli` per chunk so storage stays
   flat instead of growing to 185 GB.

Item 4 matters most: last session a 181-day request returned a **4 KB empty
zip** that a naive runner would have recorded as success.

---

## 5. Sequencing

The campaign is not the critical path. In order:

1. Refresh the catalogue; re-run the forecasting work on corrected labels.
   The two-station null was computed on labels missing a third of the events
   and 96% of the defining swarm — it is not currently a result either way.
2. Re-run the station selection on the replacement catalogue.
3. Read the 14 d / 21 d bracket emails; fix the chunk size.
4. Build the ledger + slot manager; pilot on **GCAM** (R=200 anchor) or **MANT** (R=100 anchor), full span, raw
   retained. One station is ~31 GB and 40–119 requests — enough to measure
   turnaround honestly and shake out the runner.
5. Decide on stations 2–6 from what the GCAM pilot costs in wall-clock.
