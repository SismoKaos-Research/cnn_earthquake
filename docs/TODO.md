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

### 1.1 The §4.2 finding (i) overturn — TÜBİTAK report

**Still open.** The report claims the 1D branch "only re-learns the scalar it
was denied." On the per-window-normalised `dataset_specdual_6s` that scalar does
not exist — the `seq` std floor is exactly 0.5000 — and the branch still works:

| arm | ensemble | vs seq floor 0.7088 | vs strongest floor 0.9205 |
|---|---|---|---|
| `cnn-lstm` | **0.9309** | +0.2221 | **+0.0104** |
| `cnn` | 0.9146 | +0.2058 | −0.0059 |
| `lstm` (the report's row) | ~0.9144 | +0.2056 | −0.0061 |

Reproduction confirmed: seed 42 of the control run scored 0.9139 against the
report's 0.9144, so the row reproduces and this is not a measurement artefact.
The original result measured the branch's *reach*, not the data's information
content.

Separable second point: Çizelge 7 judges a waveform-only model against `img`
mean dB (0.9205), a **spectrogram** scalar the model never sees. Finding (i)
asks "does it learn more than amplitude?" but scores against "is it deployable
alone?". Suggested fix — print both floors in that row and split the claim.

**Why it is yours:** it changes a stated conclusion in a document going to
reviewers.

### 1.2 Whether to run the AFAD station campaign

Fully planned in [`PLAN_afad_queue.md`](PLAN_afad_queue.md) — station set,
request arithmetic, the 185 GB-vs-166 GB disk constraint, and what the runner
needs to do. **My recommendation is not to**, for now: it buys more independent
episodes, which would mainly serve forecasting questions that §3.1 has closed.
Its one live justification is that the feature-model evaluation is
episode-starved (~15–20 independent episodes in a 2-year archive).

Blocked on one cheap fact either way: the 14-day and 21-day TDVMS bracket
probes were queued on 2026-08-29 and their results were never recorded. Chunk
size — and therefore 240 vs 714 requests — turns on them.

---

## 2. Open work

### 2.1 Regenerate the report deliverables

`docs/report.docx` is stale against `report.md`; `scripts/md2docx.py` regenerates
it. The Turkish drafts on the Desktop (`tubitak_rapor_v2.md`,
`tubitak_rapor_v2_taslak.md`, `tubitak_rapor_bolum_2_5.md`) have **not** been
touched and I don't know which is current — tell me which and I'll fold in the
catalogue corrections.

### 2.2 Network coincidence detection

Require 2-of-N stations within a short window; independent FPs at 1.78% become
~0.03% at 2-of-2 — the order of magnitude continuous deployment actually needs.
**Lead with the limitation:** verified 2026-08-19, only 1,184 of 6,459 test
events have a second station available, so this is measurable on a subset and
not on the corpus as a whole.

### 2.3 Continuous-data / P-wave picking

The right framing is "does this detector survive continuous data, and what does
it cost in false alarms?" — not "slide the 6 s classifier over a continuous
stream", which is the wrong shape for three separate reasons set out in the old
IDEAS entry. Partially addressed by the GPD baseline work
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

### 2.6 Housekeeping

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
