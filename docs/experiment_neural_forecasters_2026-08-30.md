# Neural forecasters re-run on the corrected catalogue

2026-08-30. Three neural forecasters, each run twice with identical settings and
identical fixed seeds — once against the archived `data_large.csv` that produced
the published numbers, once against the rebuilt AFAD catalogue. Only the labels
(and, for the catalog model, the features) differ.

Fixed seeds rather than the original's random draw: per-seed AUC spread on this
data is ~0.17, far larger than any plausible catalogue effect, so an unpaired
comparison would measure seed noise.

**Reproduce:** `scripts/rerun_neural_forecasters.sh [stage ...]`

---

## 1. `catalog_mlp` — improved, and for a reason

`cnn_lstm_catalog_waveform_fusion.py --channels catalog`, the configuration
behind the 0.5886 headline: 4 RFE-selected features, `--catalog-span 2000-01-01
2026-08-12`, `--horizon-days 14`, 2 folds, seeds 42–46.

| | fold 1 | fold 2 | mean | fold SD | floor | headroom captured |
|---|---|---|---|---|---|---|
| old catalogue | 0.6136 | 0.5861 | 0.5998 | 0.0138 | 0.5714 | 6.64% |
| **new catalogue** | 0.6780 | 0.6052 | **0.6416** | 0.0364 | 0.5690 | **16.84%** |

Beats its own fold's floor 2/2 in both arms. The baseline arm reproduces the
published run's `n` (57,138/fold) and floor (0.5714) exactly, which is what
confirms the configuration is right.

**Why this went up when the chaos features went down.** This model reads the
catalogue for its *inputs* as well as its labels — event counts, mean background
magnitude, inter-event coefficient of variation, Gutenberg–Richter magnitude
deficit. Completing the catalogue improves what it sees. The chaos features come
from waveforms, which the catalogue cannot touch, so those got only harder
labels and a stronger persistence floor.

**Caveat.** Fold spread widened (0.0138 → 0.0364) and the gain sits mostly in
fold 1 (+0.064 vs +0.019). Two folds is thin support for the magnitude, though
the direction is consistent.

---

## 2. `feature_lstm` and `feature_gru_tcn` — not evaluable, in either arm

These could not be compared, and the reason is worth more than the comparison
would have been.

### 2a. They had never actually run

`seismolib.catalog.load_hourly_features` read `Zaman_Dk` as minutes since the
Unix epoch. That is the Rust engine's convention; Sismokaos-featureExtract
writes minutes **within the containing hour** (range 3.33–62.5), with the date
in `Pencere_ID`. All 1,238,672 windows therefore mapped into one hour of
1970-01-01, the archive loaded as **2 hourly feature vectors**, every split came
out empty, and both scripts printed `[ERROR] Not enough hourly data` while
exiting 0 — so a runner trusting exit status recorded them as successes.

Fixed (commit `26827ce`): prefer `Pencere_ID`, fall back to `Zaman_Dk`. The
archive now loads as 17,211 hourly vectors, 2024-05-01 → 2026-08-10, in 2.0 s.
Covered by `tests/test_feature_loader.py`.

### 2b. With real data loaded, the evaluation design is degenerate

Per-fold split composition, `feature_lstm`, 30-day horizon:

| fold | | old: train / val / test | | new: train / val / test |
|---|---|---|---|---|
| 1 | | 0.641 / **0.000** / 0.317 | | 0.640 / **0.000** / 0.719 |
| 2 | | 0.332 / 0.317 / 0.647 | | 0.331 / 0.719 / **1.000** |
| 3 | | 0.327 / 0.647 / 0.887 | | 0.458 / **1.000** / 0.981 |
| 4 | | 0.406 / 0.887 / 0.688 | | 0.591 / 0.981 / 0.687 |
| 5 | | 0.501 / 0.688 / **0.000** | | 0.668 / 0.687 / **0.000** |

A single-class split gives an undefined AUC. Validation AUC was `nan` on 60 of
240 epochs (old) and 111 of 264 (new) for the LSTM; 111 of 560 and 273 of 614
for the GRU/TCN. **Early stopping therefore had no signal at all**, while
training AUC hit 1.0000 by epoch 3 and validation loss climbed monotonically
(3.1 → 9.3) — memorisation with no usable stopping criterion.

### 2c. No horizon fixes it

Test-split positive rate across horizons (label composition only, no training):

| horizon | old | new |
|---|---|---|
| 0.25 d | 0.00 0.00 0.00 0.00 0.00 | 0.00 0.00 0.00 0.00 0.00 |
| 1 d | 0.00 0.02 0.04 0.07 0.01 | 0.00 0.15 0.07 0.07 0.01 |
| 3 d | 0.00 0.05 0.11 0.16 0.03 | 0.00 0.23 0.18 0.16 0.03 |
| 7 d | 0.00 0.12 0.24 0.34 0.03 | 0.00 0.37 0.42 0.34 0.03 |
| 14 d | 0.00 0.24 0.44 0.60 0.03 | 0.00 0.60 0.75 0.60 0.03 |
| 30 d | 0.03 0.42 0.71 0.88 0.03 | 0.03 0.94 0.99 0.88 0.03 |

Short horizons starve the positive class; long ones saturate it. Folds 1 and 5
are degenerate at **every** horizon.

### 2d. The real constraint is the archive, not the model

M≥4.5 Aegean events falling inside the waveform archive window
(2024-05-01 → 2026-08-10):

| | in catalogue | in archive window |
|---|---|---|
| old | 261 | **35** |
| new | 327 | **93** |

The corrected catalogue nearly triples the count — but **54 of the 93 fall in
February 2025 alone**, the Santorini–Amorgos swarm. Declustered, this is on the
order of 15–20 independent episodes across a two-year window. That cannot
support a chronological 5-fold evaluation at any horizon, and the corrected
catalogue makes the design *worse* by concentrating the added positives into one
17-day burst, which is exactly why folds 2 and 3 saturate to 0.94 and 0.99.

**This is an effective-sample-size problem, not a model problem.** The honest
statement is that these two forecasters cannot be evaluated on a two-year
archive, and no number from them — before or after the catalogue fix — should be
reported.

**What would change it:** more independent episodes, which means a longer or
wider waveform archive. That is precisely what the station campaign in
`docs/PLAN_afad_queue.md` would buy, and it is a better-motivated reason to run
it than anything in the forecasting results so far.
