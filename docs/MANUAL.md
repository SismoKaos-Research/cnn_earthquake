# Operations manual

How to perform each operation in this repo. Every command below was read off the
live `--help`; where a flag is in brackets it is optional and the tool's own
default applies.

`sk` is the only entry point. Every tool is also a module, so anything here runs
standalone as `python -m sismokaos.<group>.<tool>` — which is what you want
inside a script or under `nohup`.

- [0. Setup](#0-setup)
- [1. Which data source](#1-which-data-source)
- [2. FDSN pull — `sk fdsn`, `sk fdsn-noise`](#2-fdsn-pull)
- [3. TDVMS campaign — `sk campaign`, `sk poll`](#3-tdvms-campaign)
- [4. Catalogue — `sk catalog`](#4-catalogue)
- [5. Stations — `sk station-select`, `station-range`, `station-loss`](#5-stations)
- [6. Windows — `sk cut-events`, `sk cut-length`](#6-windows)
- [7. Manifest repair — `sk distances`](#7-manifest-repair)
- [8. Training — `sk train`, `sk models`](#8-training)
- [9. Evaluation — `sk magprofile`, `cascade_eval`](#9-evaluation)
- [10. Continuous false alarms — `sk falsealarm`](#10-continuous-false-alarms)
- [11. Reports — `sk docx`, `sk pdf`, `sk figures`](#11-reports)
- [12. What happened — `sk status`, `sk results`](#12-what-happened)
- [13. Recipes](#13-recipes)
- [14. Gotchas](#14-gotchas)

---

## 0. Setup

```bash
uv sync                 # creates .venv/, installs the package editable
direnv allow            # if you use direnv
uv run sk               # the command listing
uv run pytest -q        # 1040 passed, 1 skipped
```

**After moving or copying the checkout, delete `.venv` first** — `rm -rf .venv &&
uv sync`. A *copy* is the dangerous case: the editable `.pth` keeps pointing at
the old `src`, so imports silently resolve to the stale checkout.

**Two machines.** Development is local; runs happen on `vegs`, which holds the
archives (`afad_raw/`, 47 GB), the datasets and the GPU. Pull on vegs before
running anything. If vegs is asleep: `ssh vegsjumper 'bash ~/wol_vegs.sh'`
(~30 s to wake).

---

## 1. Which data source

The choice is by **network**, not by event-vs-continuous.

| Network | Route | Cost | Use for |
|---|---|---|---|
| KO (Kandilli) | `sk fdsn` | 24 h in ~13 s, direct HTTP | anything on KO |
| TU (AFAD) | `sk campaign` + `sk poll` | ~2 min per station-day, email queue | TU only — no FDSN node serves it |

FDSN is ~100× cheaper and returns the exact window. **TU's only advantage is
station-disjointness**: 156 of the 163 FDSN KO stations are already in the
training corpus, so FDSN cannot supply unseen stations.

**One TDVMS request returns one station.** The payload accepts a `stations` array
and the reply says "the station(s) you requested", but only one arrives.
N stations cost N queue slots.

---

## 2. FDSN pull

### `sk fdsn plan`

```
sk fdsn plan --catalog CATALOG [--network KO] [--url URL]
```

| flag | meaning |
|---|---|
| `--catalog` | event catalogue CSV to plan requests from |
| `--network` | network code, default KO |
| `--url` | FDSN base URL; defaults to the KOERI EIDA node |

Writes the request plan to stdout — redirect it.

### `sk fdsn fetch`

```
sk fdsn fetch --requests REQUESTS --out-dir OUT_DIR [--url URL]
```

Produces `OUT_DIR/<station>/event_<id>_raw.mseed`.

**The station goes in the path, never in the filename.** A previous build wrote
`event_<id>_<station>_raw.mseed`, and the consumer's non-greedy parse read the
event id as `153534_TASB` — a full 13,150-window encode wasted.

### `sk fdsn-noise`

```
sk fdsn-noise --requests REQUESTS --event-dir EVENT_DIR --out OUT
              [--per-station N] [--offset SECONDS]
```

Plans noise windows for the stations an event pull returned. Magnitude training
needs these as the negative pool.

### Example

```bash
sk fdsn plan  --catalog catalogs/catalog_current.csv > requests.csv
sk fdsn fetch --requests requests.csv --out-dir raw/fdsn_magnitude
sk fdsn-noise --requests requests.csv --event-dir raw/fdsn_magnitude \
              --out noise_requests.csv
```

---

## 3. TDVMS campaign

Credentials come from the environment only: `AFAD_IMAP_HOST`, `AFAD_IMAP_USER`,
`AFAD_IMAP_PASS`. `.env*` are gitignored and chmod 600.

### `sk campaign`

All subcommands take a shared `--ledger PATH` **before** the subcommand.

| subcommand | flags | what it does |
|---|---|---|
| `plan` | `--station S [--start D] [--end D]` | writes the chunk ledger, ~21-day windows |
| `next` | `--email A [--timeout 180] [--force]` | submits the next unclaimed chunk from one address |
| `pump` | `--email A [--email B ...] [--timeout] [--force]` | `next` for every address, then `status` |
| `paste` | `--url U [--from-file F] [--out-dir afad_raw] [--force]` | ingests a returned archive |
| `mark` | `--email A --state {pending,nodata,failed} [--note]` | corrects a stuck slot |
| `reset` | `--start D` | releases one chunk back to unclaimed |
| `status` | — | where the ledger stands |

### `sk poll`

```
sk poll [--ledger L] [--out-dir afad_raw] [--folder INBOX] [--search QUERY]
        [--claim-unknown] [--fail-log F] [--interval SEC] [--once] [--pump]
        [--dry-run]
```

Watches the mailbox, fetches links, pastes them back, and with `--pump` refills
the slot that just freed.

### Example

```bash
sk campaign --ledger mant.jsonl plan --station MANT
sk campaign --ledger mant.jsonl pump --email you+a1@x.com --email you+a2@x.com
sk poll     --ledger mant.jsonl --out-dir afad_raw \
            --search '(UNSEEN FROM "tdvms@afad.gov.tr")' --pump
```

**Why plus-addresses.** TDVMS keys its one-request-at-a-time limit on the literal
address string, so `you+a1@x` and `you+a2@x` are separate slots. Several pollers
can share one mailbox: each leaves mail addressed to a slot its own ledger never
submitted from *unread*, for the poller that owns it.

**Never chain submission onto a monitor or a foreground command.** A 15-minute
monitor timeout once killed the refill before it ran, and the campaign sat idle
with zero requests in flight. `pump` is safe to re-run — `next` refuses to
double-claim.

**Being listed is not being served.** `GetStations` lists 390 TU stations for
free; a listed station can still return nodata at every date. Probe before
committing slots.

---

## 4. Catalogue

```
sk catalog --out OUT [--start D] [--end D] [--min-magnitude M]
           [--box MINLAT MAXLAT MINLON MAXLON] [--page-days N] [--retries N]
```

```bash
sk catalog --out catalogs/catalog_afad_$(date +%F).csv --min-magnitude 1.5
ln -sfn catalog_afad_$(date +%F).csv catalogs/catalog_current.csv
```

The AFAD API is public, unauthenticated and unthrottled — 413,785 events in
under 30 s. Nothing to do with the TDVMS waveform queue.

**The catalogue here was never KOERI's.** All three older local catalogues were
AFAD data despite being labelled KOERI, and the one in use was missing 1,688 of
5,770 regional events including 253 at M ≥ 4.0 — essentially the whole
February 2025 Santorini–Amorgos swarm. Re-deriving labels moved a base rate from
25.1% to 39.9%. Any result predating the 2026-08-30 rebuild needs re-deriving.
The *waveforms* genuinely are KOERI.

Parsing: `pd.to_datetime(..., format="ISO8601")` is required (pre-2010 records
omit fractional seconds), and the box must reach lat 44.5 or six Black Sea
events are clipped.

---

## 5. Stations

### `sk station-select`

```
sk station-select --events EVENTS --stations STATIONS
                  [--network TU] [--radius 500] [--min-magnitude M]
                  [--start D] [--end D] [--top N]
```

Ranks stations by how much of the catalogue they cover.

### `sk station-range`

```
sk station-range --zips 'GLOB' --station S --stations-csv CSV --catalog CSV
                 --out OUT [--max-distance 500] [--min-magnitude M]
                 [--freqmin 1.0] [--freqmax 45.0]
```

Per-event SNR at one station. **Not optional** — it is what makes recall a
statement about the detector rather than about the catalogue. Only 27.5% of
catalogued events reach SNR 3 at MANT (median 1.39); scored against all of them
the event AUC is 0.67–0.73, which describes the catalogue's reach.

### `sk station-loss`

```
sk station-loss --stations CSV --broken CSV --truth CSV
                [--radius R] [--min-magnitude M] [--network N] [--top N]
                [--out-csv OUT]
```

What a station's catalogue misses relative to a truth catalogue.

---

## 6. Windows

### `sk cut-events`

```
sk cut-events --zips 'GLOB' --station S --stations-csv CSV --catalog CSV
              --out-dir DIR
              [--snr-csv CSV] [--snr-min 3.0] [--max-distance 500]
              [--fs 100] [--pre 2.0] [--window-seconds S [S ...]]
              [--noise-offset SEC]
              [--anchor-csv CSV] [--anchor-column COL] [--anchor-lag SEC]
              [--limit-chunks N]
```

| flag | meaning |
|---|---|
| `--window-seconds` | one or more lengths **in a single pass** — this is how 6/10/20 s corpora were cut over identical events |
| `--pre` | seconds before the arrival the window starts |
| `--snr-csv` / `--snr-min` | exclude events the station did not record |
| `--anchor-csv` | anchor on a CSV of times instead of predicted arrivals |
| `--anchor-column` | which column holds the epoch (e.g. `alarm_epoch`) |
| `--anchor-lag` | shift applied to each anchor |

Produces `OUT_DIR/<tag>/<station>/{eq,noise}/`.

**Catalogue-anchored:**

```bash
sk cut-events --zips 'afad_raw/MANT/*.zip' --station MANT \
    --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv \
    --snr-csv mant_range.csv --window-seconds 6 10 20 \
    --out-dir raw/continuous_windows
```

**Alarm-anchored** — this is what a cascade actually sees:

```bash
sk falsealarm timing --scores 'scores_mant/6s/*.npz' --station MANT \
    --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv \
    --window-seconds 6.0 --threshold 0.9971 --out mant_alarm_times.csv

sk cut-events --zips 'afad_raw/MANT/*.zip' --station MANT \
    --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv \
    --anchor-csv mant_alarm_times.csv --anchor-column alarm_epoch \
    --window-seconds 10 --out-dir raw/alarm_windows
```

**Record which arm and which threshold produced an anchor CSV.** The existing
`mant_alarm_times.csv` does not, so a dataset built from it cannot be compared
with one cut from a different detector.

### `sk cut-length`

```
sk cut-length --src SRC --seconds N --out OUT [--limit N]
```

Re-cuts existing windows to another length without re-reading archives.

---

## 7. Manifest repair

```
sk distances --manifest M --station-coords CSV --catalog CSV
             [--tolerance 5.0] [--report-above 1.0] [--write]
```

Recomputes `distance_km` for **every** row and refuses to write unless it
reproduces the values already present within `--tolerance`. Always dry-run first.

```bash
sk distances --manifest dataset_magreg_fdsn_10s/manifest.csv \
    --station-coords ../seismic_cli/catalogs/station_coords.csv \
    --catalog catalogs/catalog_current.csv
sk distances ... --write
```

**Missing `distance_km` is not benign.** The trainer mean-imputes NaN aux after
standardizing, so every row from a station with no distance trains against one
fixed wrong value — silently. It also breaks the `ridge(log_snr, log_distance)`
floor every magnitude model is judged against. On the FDSN corpus, 42 of 161
stations had none for any row.

---

## 8. Training

### Finding the task

```bash
sk train                      # 25 tasks, grouped by what the label answers
sk train --predicts forecast  # one group
sk train detect --label       # what this task's label actually is
sk train detect --help        # the trainer's own flags
```

### The model registry

```bash
sk models                     # every architecture, by input shape
sk models dual-channel        # one model's branches, flags, defaults
sk models --spec DIR          # the spec saved beside those checkpoints
sk models --family sequence   # one family
```

### Common flags on the dual-channel trainers

| flag | meaning |
|---|---|
| `--dataset-dir` | dataset root, containing `manifest.csv` and `train/ val/ test/` |
| `--save-dir` | where checkpoints and `model.json` go |
| `--channels` | `all,1d,2d,aux,1d+aux,2d+aux,1d+2d` — which branches are active |
| `--model-branch` | 1D front end: `lstm`, `cnn`, `cnn-lstm` (`--branch-1d` still accepted) |
| `--split-by` | `auto,event,station,both,detector` — see below |
| `--seed` | model initialisation |
| `--seed-split` | **which partition** — vary this, not just `--seed` |
| `--detector-manifest` | required by `--split-by detector` |
| `--epochs 80` `--patience 12` `--lr 1e-3` `--batch-size` | training loop |

### `--split-by`, which decides what the number means

| value | what it does | when |
|---|---|---|
| `auto` | `both` when >1 station, else `event` | default; prints which it chose |
| `event` | the generator's own split | only when one station forces it |
| `station` | stations disjoint, shared event can leak | rarely — prefer `both` |
| `both` | station-**and**-event disjoint | the honest protocol; costs ~85% of val/test on small corpora |
| `detector` | copies a detector's station partition | mandatory for cascade evaluation |

**Vary `--seed-split`, not just `--seed`.** Which stations land in test dominates:
partition variance was measured at 2.4× model-seed variance.

**A cascade's stage 2 must inherit stage 1's partition.** Without
`--split-by detector --detector-manifest PATH`, 77% of detector test stations
were regressor *training* stations.

**Non-aux is the magnitude default** (`1d+2d`). The aux vector is
`(log_snr, log_distance)` and log_distance needs a hypocentre a fresh detection
does not have, so `all` trains a model that cannot run in the cascade it exists
for. Pass `--channels all` explicitly for an aux ablation.

Every registry-wired run writes `model.json` (architecture) and a record under
`runs/` (argv, git commit, **whether the tree was dirty**, dataset, seeds,
metrics, checkpoints).

---

## 9. Evaluation

### `sk magprofile`

```
sk magprofile --dataset-dir DIR --ckpt CKPT
              [--split-by {event,station,both,detector}] [--seed-split N]
              [--detector-manifest PATH] [--channels C] [--hidden N]
              [--fusion-dim N] [--batch-size N] [--out CSV]
```

Where a magnitude regressor's error lives — by magnitude band, by distance.

### `cascade_eval`

```
python -m sismokaos.detection.cascade_eval \
    --detector-dir DS --detector-ckpt-dir CKPT \
    --magnitude-dir DS2 --magnitude-ckpt CKPT2
    [--threshold T] [--detector-fusion F] [--detector-branch-1d B]
    [--hidden N] [--fusion-dim N]
    [--reg-hidden N] [--reg-fusion-dim N] [--reg-channels C]
```

Both read `model.json` and **the saved spec wins over the flags** — they print
the spec used and name any flag that disagrees. `--reg-*` apply only when a
checkpoint has no `model.json`.

**Report MAE by magnitude band, always.** An aggregate is largely a readout of
the corpus's magnitude mix: 71.1% of the 6 s test set is M ≤ 2.5 where MAE is
0.1430, against 0.4085 for M > 3.

---

## 10. Continuous false alarms

Five phases; the scan is the expensive one, which is why they are separate.

### `sk falsealarm baseline`

```
sk falsealarm baseline --zips 'GLOB' --out OUT
                       [--sample-chunks 6] [--piece-seconds 3600]
                       [--fs 100] [--freqmin 1.0] [--freqmax 45.0]
```

The station's long-term (μ, σ) that every scored window is standardized against.

### `sk falsealarm scan`

```
sk falsealarm scan --zips 'GLOB' --baseline-json JSON --out-dir DIR
                   --arm SPEC [--arm SPEC ...]
                   [--channels 1d] [--fusion linear] [--hidden 48]
                   [--fusion-dim 96] [--batch-size 1024] [--block-windows 20000]
                   [--workers 6] [--limit-chunks N]
                   [--near-csv CSV] [--near-pre 30] [--near-post 90]
```

`--arm NAME:WINDOW_SECONDS:CKPT_DIR:BRANCH[:STEP_SECONDS]`, repeatable. Every arm
windows the same read, so two arms cost well under twice one. Put `stalta` in the
CKPT_DIR slot to run a classical baseline on the identical stream.

Windows are disjoint by default (step = window), so an alarm count is also a
count of independent decisions.

### `sk falsealarm report`

```
sk falsealarm report --scores 'GLOB' --station S --stations-csv CSV
                     --catalog CSV --window-seconds W --out-prefix P
                     [--snr-csv CSV] [--snr-min 3.0]
                     [--max-distance 500] [--guard-pre 10] [--guard-post 60]
                     [--signal-post 20] [--cluster-seconds 60]
```

The operating table: pick an alarm budget, read off the threshold.

**Never inherit the benchmark threshold.** The 6 s detector scores a median of
0.80 on continuous noise and flags 92.7% of a quiet station-day at 0.5 — 12,599
alarms/day, against 257 extrapolated from benchmark FPR.

### `sk falsealarm timing`

```
sk falsealarm timing --scores 'GLOB' --station S --stations-csv CSV
                     --catalog CSV --window-seconds W --threshold T --out CSV
                     [--snr-csv CSV] [--snr-min 3.0]
                     [--guard-pre 10] [--guard-post 60] [--max-distance 500]
```

Per event: when it fired, relative to P and S. Take `--threshold` from `report`.
The alarm time is the window's **end** — a detection cannot be declared before
the whole window exists.

### `sk falsealarm coincidence`

```
sk falsealarm coincidence --scores-a 'GLOB' --station-a A
                          --scores-b 'GLOB' --station-b B
                          --stations-csv CSV --catalog CSV
                          --window-seconds W --out-prefix P
                          [--coincidence-seconds S] [--vp 6.0]
                          [--snr-csv-a CSV] [--snr-csv-b CSV] [--snr-min 3.0]
                          [--max-distance 500] [--guard-pre 10]
                          [--guard-post 60] [--signal-post 20]
                          [--cluster-seconds 60]
```

What requiring two stations to agree costs, against what independence predicts.
`--coincidence-seconds` defaults to separation / Vp.

### `sk falsealarm verify`

```
sk falsealarm verify --dataset-dir DIR --ckpt-dir DIR
                     [--branch-1d cnn-lstm] [--channels 1d] [--fusion linear]
                     [--hidden 48] [--fusion-dim 96] [--limit 4000]
                     [--expect-auc X]
```

Checks the scan path still matches the training pipeline.

### Full example

```bash
sk falsealarm baseline --zips 'afad_raw/MANT/*.zip' --out mant_baseline.json

sk falsealarm scan --zips 'afad_raw/MANT/*.zip' \
    --baseline-json mant_baseline.json \
    --arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm \
    --arm ponly:3.4:trained_model_ponly_matched:cnn-lstm \
    --arm stalta:6.0:stalta:1.0-10.0 \
    --out-dir scores_mant

sk falsealarm report --scores 'scores_mant/6s/*.npz' --station MANT \
    --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv \
    --window-seconds 6.0 --snr-csv mant_range.csv --out-prefix mant_fa_6s

sk falsealarm coincidence \
    --scores-a 'scores_mant/6s/*.npz' --station-a MANT \
    --scores-b 'scores_demi/6s/*.npz' --station-b DEMI \
    --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv \
    --window-seconds 6.0 --out-prefix coinc_mant_demi
```

---

## 11. Reports

```bash
uv run --with python-docx sk docx in.md out.docx
uv run --with markdown --with weasyprint sk pdf in.md out.pdf
sk figures
```

**`sk figures` ignores `--help` and does the work instead** — invoking it at all
rewrites its output directory. Known bug.

---

## 12. What happened

```
sk status [--host HOST] [--dir DIR] [--runs DIR] [--all]
sk results [--runs-dir DIR] [--task T] [--metric M] [--best] [--failed]
           [--show RUN] [--limit N]
```

```bash
sk status --host vegs
sk results --task magnitude
sk results --metric MAE --best
```

`sk results` marks dirty-tree runs: those do not reproduce from their recorded
commit, so treat the number as provisional. An empty `runs/` means nothing was
launched through `sk train`, not that nothing ran.

---

## 13. Recipes

### Train the FDSN magnitude regressor, non-aux

```bash
for S in 0 1 2; do
  sk train magnitude \
      --dataset-dir dataset_magreg_fdsn_10s \
      --channels 1d+2d --split-by auto \
      --seed 42 --seed-split "$S" \
      --save-dir "trained_model_magreg_fdsn10s_nonaux_p$S"
done
```

Three partitions, one seed each — the spread you get is partition variance, which
is the honest error bar. Result: **MAE 0.4203 ± 0.0165**, against a
distance-having floor of 0.5481 and an information-matched `ridge(log_snr)` floor
of 0.6054.

### The cascade on continuous data

`dataset_magreg_alarm_10s` was cut from detector alarm times, so a regressor
trained on it *is* stage 2 seeing what stage 1 hands it. The cascade cost is the
paired difference against the catalogue-anchored control:

```bash
# stage 2 on what the detector actually produces
sk train magnitude --dataset-dir dataset_magreg_alarm_10s \
    --channels 1d+2d --split-by auto --seed 42 \
    --save-dir trained_model_magreg_alarm10s_nonaux

# control: same station, same window length, catalogue-anchored
sk train magnitude --dataset-dir dataset_magreg_cont_10s \
    --channels 1d+2d --split-by auto --seed 42 \
    --save-dir trained_model_magreg_cont10s_nonaux
```

Both corpora are **TU.MANT only**, so `--split-by auto` falls back to `event` and
the trainer prints that this is *that station's* estimator. It cannot claim
station transfer. The corpus is ~78% below M 2.5, so report MAE by magnitude
band, not the aggregate alone.

### Window-length experiment, controlled

`dataset_magreg_cont_{6,10,20}s` hold the identical 13,016 rows over the same
events, station and splits — only the window differs.

```bash
for W in 6 10 20; do
  sk train magnitude --dataset-dir "dataset_magreg_cont_${W}s" \
      --channels 1d+2d --split-by auto --seed 42 \
      --save-dir "trained_model_magreg_cont${W}s_nonaux"
done
```

This speaks directly to the saturation mechanism: lengthening the input is one of
the two published mitigations, and the other (more large-magnitude events) needs
a new pull.

### Reproduce a published result

`experiments/reproduce/` holds the exact runners, deliberately kept as shell
scripts — their value is being the exact thing that was run.

```bash
bash experiments/reproduce/run_ponly_matched.sh
```

---

## 14. Gotchas

| Symptom | Cause |
|---|---|
| Tests skip with "no catalogue found" | indistinguishable from a machine with no data. The sibling checkout is `seismic_cli` locally, still `data_downloader` on vegs. |
| Imports resolve to the wrong code after a move | stale editable `.pth`. `rm -rf .venv && uv sync`. |
| A recall figure looks implausibly high | check the background alarm rate beside it. One arm reported "97.8% recall, fired 7 s before P" at a 95% background rate — arithmetic, not detection. |
| A ratio to the floor moved between runs | the floor moved, not the model. On FDSN the floor's spread is 3.5× the model's. Print both. |
| Two detectors rank differently by AUC and by recall | expected. AUC integrates the whole ROC; deployment lives at FPR ~7×10⁻⁴. |
| A merge silently grew a frame | left join on a non-unique key. DEMI's SNR table has 269 duplicate event ids; use `load_snr`, which keeps the max. |
| A single-station corpus reports a great MAE | it is that station's estimator. The trainer says so; do not quote it as general. |
| A close margin reversed on re-run | single-seed comparison. Differences under ~0.01–0.02 are not established effects here. |
| A scan seems hung | it announces before the read; a fragmented chunk can take 17 min inside one obspy call. |
