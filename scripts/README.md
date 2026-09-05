# Tools

Reusable tooling, as opposed to `experiments/`, which is a record of things run
once. Everything here is expected to be run again against new data.

| tool | what it does |
|---|---|
| `afad_campaign.py` | TDVMS download ledger: plan, submit, paste, status |
| `afad_imap.py` | polls the mailbox, fetches links, refills freed queue slots |
| `afad_pump.sh` | wrapper for the poller |
| `fdsn_magnitude_pull.py` | plan and fetch event windows from KOERI FDSN |
| `plan_pbefores_pull.py` | plan a TDVMS pull that can report before S |
| `fetch_afad_catalog.py` | rebuild the event catalogue from AFAD's API |
| `select_afad_stations.py` | rank stations by catalogue coverage |
| `station_detection_range.py` | measure per-event SNR at a station |
| `station_catalog_loss.py` | what a station's catalogue misses |
| `cut_event_windows.py` | cut arrival-anchored windows from continuous record |
| `cut_window_length.py` | re-cut existing windows to another length |
| `continuous_false_alarms.py` | false-alarm rate on continuous data (baseline/scan/report/timing/verify) |
| `magnitude_error_profile.py` | where a magnitude regressor's error lives |
| `md2docx.py` | Markdown to .docx (no pandoc on this box) |
| `make_report_figures.py` | report figures |

**Source of truth for which waveform service to use:** FDSN (KOERI) for the KO
network, TDVMS only for TU, which no FDSN node serves. See the data-sources note
in the project memory.
