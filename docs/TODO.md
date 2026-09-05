# TODO — the single list

Consolidated 2026-08-31 from `TOMORROW.md`, `TOMORROW_DETECTOR.md` and
`IDEAS.md`, which are deleted. **This is the only planning file.** Anything not
here is either done (see §3) or was stale enough not to survive the merge.

Status was re-verified against the code at consolidation time, not copied
forward — several items the old files listed as open had already been finished.

---

## 1. Decisions waiting on you

These change stated conclusions or need judgement I don't have. Nothing below
proceeds without a call from you.

### 1.1 Whether to keep spending requests on more stations

The MANT and GCAM campaigns are **done** (see §3.5). The open question is no
longer whether to run one, but whether a third station is worth 40 more
requests — and GCAM is the reason to hesitate. It returned 189 usable days
against MANT's 747 for the same 40 links, because the station stops recording
at 2024-12-18 and nothing but the requests themselves revealed that.

**My recommendation: probe uptime before committing.** A handful of
single-day requests spread across a candidate's span costs ~6 links and
answers in one afternoon what GCAM took a full campaign to tell us.

**Probed 2026-09-05. Both probes are done and the answer is encouraging.**

*Station availability* (`scr.jsonl`, one day each at 2025-06-01): INCE, KZIL,
PASA and UZP all returned data — four viable candidates for 117 MB of probe
traffic. **BAKC and IRLI are not in the TDVMS station list at all**, which is a
result in itself: the station catalogue this repo plans from contains stations
TDVMS will not serve, so any plan built from it should be checked against
`_device_code` before the requests are spent.

*Archive depth at MANT* (`depth.jsonl`, one day per year): data returned for
2012, 2019, 2021 and 2023; 2016-06-01 came back "no waveform at source".
**MANT's archive reaches back to at least 2012**, roughly ten years earlier
than the campaign has used, with at least one interior gap. That changes the
scale of what a MANT campaign could pull, and is worth weighing before spending
requests on a fourth station.

Two bugs surfaced during the probes, both fixed:

- Two pollers watching one mailbox ran the same `(UNSEEN FROM ...)` search and
  each consumed the other's links. The station poller took `+dep3` and `+dep5`
  and burned them as permanent failures while `depth.jsonl` waited for mail
  that no longer existed. A poller now leaves mail addressed to a slot its own
  ledger never submitted from unread.
- `next` wrote its claim before looking the station's device code up, so an
  unlistable station left the row `claimed` forever, holding a queue slot
  against a request never made. That is what BAKC and IRLI actually did for
  most of an afternoon, looking exactly like lost mail. The claim is now
  released and the row retired.

---

## 2. Open work

### 2.1 Regenerate the report deliverables

**`tubitak_rapor_v2.md` is done** — freshened 2026-08-31 (attribution corrected,
GPD/STEAD results folded in, §6 forecasting added) and now version-controlled in
`docs/tubitak/` with symlinks left on the Desktop.

Remaining: the `.docx` renders are stale against their sources.
`scripts/md2docx.py` regenerates them — `docs/report.docx` and
`docs/tubitak/*.docx`. Not done automatically because the renders are the thing
that actually goes to reviewers and should be regenerated deliberately.

### 2.2 Network coincidence detection

Require 2-of-N stations within a short window; independent FPs at 1.78% become
~0.03% at 2-of-2 — the order of magnitude continuous deployment actually needs.
**Lead with the limitation:** verified 2026-08-19, only 1,184 of 6,459 test
events have a second station available, so this is measurable on a subset and
not on the corpus as a whole.

The campaign changed what is available here. All 189 GCAM days sit inside
MANT's unbroken 2024-05-01..2025-08-06 run, and the two stations are ~130 km
apart, so there is now a genuine two-station **continuous** window to test
coincidence on — no longer a subset of cut event windows. That is the setup
§2.3's scan was built for, and it reuses the same scores.

**In progress 2026-09-05.** GCAM is being scanned with the same three arms as
MANT (`scores_gcam/`), which was the missing input — MANT had 36 chunks scored
and GCAM had none. `sk falsealarm coincidence` then prices the rule.

**The 1.78% → 0.03% figure above assumes the two stations' false alarms are
independent, and that is the thing to measure rather than assume.** Two
stations 130 km apart share weather, share the regional noise field, and share
whatever drives the 1.7–2.1× day/night ratio §2.3 found. The tool therefore
reports the measured 2-of-2 rate beside what two independent streams of the
same rates would produce, and their ratio; the ratio is the result, and the
reduction on its own is arithmetic.

Two constraints it enforces, because every mistake available here removes false
alarms faster than the method does. Only the span **both** stations recorded is
scored — GCAM stops at 2024-12-18, and counting MANT alarms after that as
suppressed would read as a near-total reduction and be nothing but missing
data. And recall is asked only of events reaching SNR at both stations, since
no network rule can confirm an event one station never recorded.

First numbers from the scan already say the two stations are not
interchangeable: on GCAM the 6 s arm puts **10.55%** of windows over 0.5,
against MANT's 92.7%. Whatever amplitude hole §2.6 describes, MANT sits in it
and GCAM does not.

### 2.3 Continuous-data / P-wave picking

**In progress 2026-09-04**, `scripts/continuous_false_alarms.py` on 747 days of
MANT, three arms (6 s, P-only mined, P-only natural). First 195 days:

| arm | event AUC, SNR>=3 | recall @ 10 alarms/day | background median |
|---|---|---|---|
| 6s | 0.872 | 0.653 | **0.797** |
| ponly | 0.938 | 0.597 | 0.429 |
| pnat | 0.937 | 0.585 | 0.301 |

Three things that only continuous data could show. The benchmark's 0.5 threshold
is meaningless here (12,891 alarms/day for the 6 s arm) and thresholds must come
from measured background. Scored against *every* catalogued event the AUC is
0.53–0.62, because only 10.7% of them reach SNR 3 at MANT and the median is 1.09
— that number measures the catalogue's reach, not the model. And the 6 s arm's
whole background spans 0.797–0.839, so a 0.016 seasonal drift would move its
alarm rate by an order of magnitude; §2.6 is the response to why.

Partially addressed earlier by the GPD baseline work
([`experiment_gpd_baseline_2026-08-27.md`](experiment_gpd_baseline_2026-08-27.md)),
which put this detector against four published pickers on our own windows.

### 2.4 CNN-GRU waveform branch

Swap the BiLSTM in `ConvSeqBranch` for a BiGRU, re-run the branch-1d grid as a
fourth arm (`--branch-1d cnn-gru`). Motivated: the 2026-08-19 grid put
`cnn-lstm` (0.9896) above `cnn` (0.9843) with non-overlapping per-seed ranges,
so recurrence is load-bearing here rather than decorative. Cheap.

### 2.5 Cascade false-positive handling

Stage 2 consumes `aux = (log_snr, log_distance)`, and `log_distance` needs a
catalogued hypocentre a false positive does not have. Options: a `--channels 2d`
stage 2, a waveform-derived distance estimate, or propagated uncertainty. See
`src/detection/cascade_eval.py`'s module docstring.

### 2.6 Retrain with continuous-background negatives

**Motivated by a measured failure, not a hunch.** On continuous MANT the 6 s
detector scores a *median of 0.83 on noise* — 92% of a quiet station-day clears
the benchmark's 0.5. Feeding it real training noise windows scaled down and
nothing else changed, its alarm rate goes 1.9% at the training median amplitude,
86% at a tenth of it, 100% at a hundredth. `P(event | amplitude)` is U-shaped:
amplitude mining put a floor under the negatives and physics puts one under the
positives, so below ~0.1 sigma the model has training data of **neither** class
and extrapolates to "earthquake". Continuous background sits at 0.11 sigma.

Both P-only models are monotone and do not do this, so it is not a property of
mining as such — `ponly` (mined) and `pnat` (natural) are indistinguishable on
continuous data (AUC 0.9376 vs 0.9370 at SNR>=3). Something about the 6 s build
specifically; the S-inclusive positives reaching 581 sigma are the suspect.

Zhu & Beroza said this in 2019 and it went untested here: *"In order to apply
PhaseNet for detection on continuous data, a new data set that includes more
non-seismic signals should be used for training."*

The fix is where negatives come from, not how many:

- **Sample negatives from the continuous archive at natural amplitude.** 747
  days of MANT now exist; the training pool never contained a window as quiet as
  a typical station-minute. This is the part that fills the hole.
- **Add self-mined hard negatives.** The scan produces ~800 unexplained alarm
  clusters per 98 days — real waveforms that already fool this detector. Same
  instinct as TransQuake's FilterPicker-derived negatives, from our own record.
  They are an upper bound on false alarms (some are uncatalogued events), so
  screen them before training on them as noise.
- **Evaluate at natural imbalance.** TransQuake trains balanced and *tests* at
  its natural 11:1, which is why its precision (0.712) means something; every
  benchmark here is balanced at test time, so no precision figure in this repo
  is comparable to a deployed one.

**Do the free control first.** Class ratio changes calibration, and calibration
is already obtainable without retraining — either a prior correction on the
logit, or `continuous_false_alarms.py report`, which sets the threshold from
measured background. If a ratio change is all that is tried, expect it to move
the operating point and nothing else. The amplitude coverage is the experiment.

### 2.7 Housekeeping

- The 3 s dataset overflows fp16 (max 1.21e6). Published results there are
  2B-only and unaffected, but any future `1d`/`all` run on it needs `asinh`.
- Pre-fix checkpoints remain quarantined in `trained_model_branch1d_stale_prefix/`.
  Safe to delete once nothing references them.
- Çizelge 7's 2B hard-negative figure was 0.9892 in the report but no log
  produces it; replaced with the measured 0.9882.

---

## 3. Closed — kept for the record

Outcomes, not just names, so nothing gets re-tried by accident.

### 3.1 Forecasting: answered, and bounded

**The signal is in the catalogue, not the seismogram.** Waveform-derived
features lose to a persistence floor across every architecture tried — chaotic
features (below floor on all four model variants, 0 of 10 context/horizon
cells) and three sequence models (LSTM 0.5244, GRU 0.5709, TCN 0.5204 against
0.5823). Catalogue-derived features beat it. See
[`experiment_neural_forecasters_2026-08-30.md`](experiment_neural_forecasters_2026-08-30.md)
and [`experiment_chaos_forecast_2026-08-27.md`](experiment_chaos_forecast_2026-08-27.md).

This supersedes `TOMORROW.md` §1–§4 in full: the fused-vs-catalog-only
comparison, the RFE rerun, the GBM update and the 10 Hz preprocessing question
were all overtaken by the catalogue audit and the re-derivations that followed.

### 3.2 Catalogue rebuilt (2026-08-30)

Every local catalogue was AFAD's, not KOERI's, and the copy in use was missing
~29% of regional events including nearly all of the February 2025
Santorini–Amorgos swarm. Rebuilt via `scripts/fetch_afad_catalog.py`
(one request, <30 s). Moved results in **both** directions — chaos forecasting
worse (the persistence floor rose faster than the model), per-zone forecasting
better (AEGEAN +0.173, CENTRAL +0.222 at block level, CENTRAL no longer at
chance). Detection essentially unaffected: 3 contaminated noise windows in
55,595.

### 3.3 Detector work finished since `TOMORROW_DETECTOR.md` was written

- `seq_transform` **is** in `run_tag` (`cnn_lstm_classify.py:399`); the
  filename-collision hazard is gone.
- `norm_all_cnn-lstm` **did** run: 0.9692 against the 0.9205 img floor
  (+0.0488). It was listed as "never started".
- STEAD cross-corpus: done, and reciprocal — each model wins at home
  ([`experiment_stead_reciprocal_2026-08-27.md`](experiment_stead_reciprocal_2026-08-27.md)).

### 3.4 Other closed items

| item | outcome |
|---|---|
| Operating-envelope analysis | Done 2026-08-19. Recall is SNR-governed, not magnitude-governed. |
| Calibration and threshold selection | Done (`c94c5ed`). Fusion ensemble is **under**-confident (fitted T=0.476); temperature scaling cuts ECE 0.0863 → 0.0216, Brier 0.0325 → 0.0269. MCE rises 0.190 → 0.335 because sharpening empties the middle bins. Temperature fitted on validation, never test. |
| Amplitude ablation (loudness vs shape) | Done. |
| Chaotic features as a forecaster | Done — negative, see §3.1. |
| Gated fusion | Done. |
| fp16 overflow re-check on the original benchmark | Done. |
| `catalog_forecast_report.md` pre-correction tables | Done 2026-08-31 — supersede note added in place, including that CENTRAL is no longer at chance so its near-Poisson diagnosis does not survive for that zone. |
| §4.2 finding (i) overturn | **Already resolved in the report itself.** `tubitak_rapor_v2.md` §4.8.6 carries the normalised-dataset measurement (Çizelge 27–31) and §5.4 states outright that the stronger claim "did not survive" — including the both-floors fix. Was listed as an open decision only because a 2026-08-20 note was carried forward without checking it against the current draft. |

### 3.5 AFAD waveform campaign (finished 2026-09-04)

Two stations pulled through the TDVMS email queue, ledgered in
`afad_campaign_ledger.jsonl` and `gcam_ledger.jsonl` on vegs.

| | chunks | days | size | span |
|---|---|---|---|---|
| MANT | 36/40 | 747 | 27.3 GB | 2024-05-01 .. 2026-08-10, longest unbroken run 462 d |
| GCAM |  9/40 | 189 |  6.5 GB | 2024-05-01 .. 2024-12-18 |

The misses are station outages, not failed requests: TDVMS answers a window it
has nothing for with a 269-byte archive holding a Turkish notice, and
re-requesting cannot recover it. MANT lost 4 windows that way; GCAM lost 31 and
stops dead at 2024-12-18.

Two things worth carrying forward. **Plus-addressing multiplies queue slots** —
TDVMS enforces one request at a time per literal address string, so `+a1..a6`
on one Gmail account turns a serial queue into six parallel ones. And
**multi-station requests silently truncate**: the portal accepts a station list,
returns 109, emails a link, and the archive holds only the first station.

