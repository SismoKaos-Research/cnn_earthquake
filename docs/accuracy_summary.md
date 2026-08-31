# Model Accuracy Summary

*Every figure is a held-out test measurement from `report.md`. Floors are shown
next to each task because several models in this project lost to theirs.*

---

## Task 1 — Detection: earthquake vs. noise (6 s windows, 9,548 test windows)

| Model | Test AUC | MCC | Accuracy |
|---|---|---|---|
| **Spectrogram CNN alone** | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram-dual, stacked fusion | 0.9743 | 0.871 | 93.54 % |
| Spectrogram + aux, stacked | 0.9758 | 0.868 | 93.37 % |
| Spectrogram-dual, gated fusion | 0.9761 | 0.850 | 92.51 % |
| Spectrogram-dual, linear fusion | 0.9646 | 0.8122 | 90.61 % |
| RAM + aux, stacked | 0.9557 | 0.781 | 89.07 % |
| RAM-dual + aux, linear fusion | 0.9514 | 0.7790 | 88.95 % |
| Raw waveform + aux (`1d+aux`) | 0.9501 | 0.7675 | 88.37 % |
| RAM + aux, no LSTM (`2d+aux`) | 0.9468 | 0.7775 | 88.84 % |
| RAM + aux, no dual architecture | 0.9230 | 0.7018 | 84.79 % |
| Raw waveform only (`1d`) | 0.9216 | 0.6849 | 84.22 % |
| RAM-dual, linear fusion, no aux | 0.9144 | 0.6042 | 79.57 % |
| RAM CNN alone, no aux | 0.8356 | 0.5339 | 76.70 % |
| RAM image only (`2d`) | 0.8408 | 0.5288 | 76.42 % |
| *floor:* STA/LTA, correctly parameterised | *0.8194* | — | *74.60 %* |
| *floor:* STA/LTA, library default (broken) | *0.5093* | — | *56.88 %* |

**Best: plain spectrogram CNN, 0.9793 AUC.** Every CNN configuration beats a
correctly-parameterised STA/LTA. No architectural addition beat the simplest
model.

---

## Task 2 — Magnitude from a short window (classification: 3s; regression: 3s and 6s)

| Model | Accuracy | AUC | MCC |
|---|---|---|---|
| **CNN (spectrogram + aux)** | **79.78 %** | **0.8550** | **+0.566** |
| *floor:* logistic on log-SNR + log-distance | *71.36 %* | *0.7486* | *+0.372* |
| *floor:* majority class | *61.35 %* | — | — |

**The encoded window adds real signal here** — +0.106 AUC over amplitude and
distance alone.

**Continuous magnitude regression, dual-channel extension (report.md §7.5).**
A related question on the same data: does pairing the spectrogram with a
raw-waveform LSTM+attention branch (this project's dual-channel detection
architecture, retargeted) beat the single-channel spectrogram+aux regressor
above? 3-seed comparison (MAE, lower is better):

| Model | MAE (mean of 3 seeds) |
|---|---|
| Single-channel spectrogram + aux | 0.205 |
| Dual-channel (spectrogram + raw waveform + aux) | 0.202 |
| *floor:* ridge(log_snr, log_distance) | 0.308 |

Branch ablation (single seed) shows why the tie happens: **spectrogram alone
(no LSTM branch) is the best model tested, at 0.197 MAE** — adding the
raw-waveform LSTM branch makes it slightly worse (0.202), and the
raw-waveform branch alone is the weakest (0.250). Same conclusion as the
detection work: no architectural addition beats the single best branch.

**Pushing on 0.197, still at 3 seconds (report.md §7.7).** Two more levers
were tried against this same spectrogram-only model: finer spectrogram time
resolution (`hop_length` 64→32→16, decoupled from `n_fft` for the first
time) and per-component amplitude aux (log_snr per Z/N/E instead of one
averaged scalar). The 0.197 MAE figure above was single-seed; it is now
3-seed confirmed at 0.1973 MAE. The best configuration found (hop=32)
reaches 0.1960 MAE — nominally better in all 3 paired seeds, but by only
0.001–0.002 MAE, the same size as normal seed-to-seed spread. Per-component
aux made no further difference (0.196), and the LSTM branch still lost with
the richer aux (0.205). **Neither lever produced a real improvement**;
0.1960–0.1973 MAE is the ceiling *at 3 seconds* — see below, this is not the
ceiling for the task overall.

**Window length, 3s vs. 6s (report.md §7.8).** Widening the same
architecture's window from 3s to 6s (`window_post_6s_anchored`, same event
anchoring used for detection) is a different lever from either of the two
above — more waveform, not a better encoding of the same waveform — and it
is the one that actually worked:

| window | MAE (3-seed mean) | seed spread | *floor:* ridge(log_snr, log_dist) |
|---|---|---|---|
| 3s | 0.1973 | 0.197–0.198 | 0.308 |
| **6s** | **0.1817** | **0.181–0.182** | 0.318 |

An 0.014–0.016 MAE gap, roughly ten times the size of anything the
hop-length/aux sweep produced, and this is not an easier test population
doing the work — both floors (predict-the-mean and ridge) get *worse* at
6s, while the model's margin over the ridge floor still widens
(+0.111 → +0.136). A useful internal control: 6s at the default hop
produces the same 10 time-frame count as 3s at hop=32, and still wins by
0.014 MAE, isolating window *length* as the active ingredient rather than
finer time resolution (already ruled out just above). **6s is now this
project's best magnitude-regression result at any window length tested**
(event-disjoint split) — see below for what survives a stricter split.

**Station-disjoint verification (report.md §7.9).** Both 3s and 6s above
use event-disjoint splits with most stations shared across train/test
(175/181 and 148/152), the exact setup that inflated a different task's
headline via site memorisation elsewhere in this project (§13.8). Checked
here with the same doubly-disjoint method (3 independent station
partitions per window length): **site memorisation is ruled out** — every
partition at both window lengths still beats its own ridge floor by a wide
margin on stations never seen in training. What does **not** survive is the
6s-vs-3s margin specifically: doubly-disjoint MAE is 0.226 (range
0.201–0.247) at 3s vs. 0.218 (range 0.195–0.236) at 6s — the 0.016 gap
above is smaller than the spread within either window length under this
noisier, few-test-station protocol. Read together: **the model's skill at
both window lengths is real and generalizes past specific sites; whether
6s specifically beats 3s is established under the clean event-disjoint
comparison and unresolved (not refuted) under this stricter one.**

---

## Task 3 — Three-class risk: noise / M<4 / M≥4 (2,970 test windows)

| Model | Accuracy | Macro-AUC | MCC |
|---|---|---|---|
| Flat gradient boosting *(inflated — see note)* | *91.72 %* | *0.9792* | *+0.8559* |
| **Two-stage scalar model, no image** | **82.83 %** | **0.9273** | **+0.7039** |
| CNN (image + aux) | 73.64 % | 0.9277 | +0.5990 |
| *floor:* majority class | *52.96 %* | — | — |

**The best model uses no image at all**, beating the CNN by ~9 points — and the
CNN had the same scalars as input. The 91.72 % row is *not* this task's result:
roughly ten of those points come from `distance_km` being undefined for noise,
so "distance is missing" identifies the noise class by construction.

---

## Task 4 — Catalog forecasting: M ≥ 4.5 within 30 days, per fault zone

Full detail in `catalog_forecast_report.md`. The previously-reported
logistic-regression / gradient-boosting scalar forecaster is retired; this is
the same validated dense target under the project's dual-channel
CNN+LSTM+attention architecture (`cnn_lstm_forecast.py`), 3 seeds.

**Pooled, window-level** (test set, 2,414 windows, positive rate 0.589):

| Model | AUC (mean of 3 seeds) | seed spread |
|---|---|---|
| **Dual-channel CNN+LSTM+attention** | **0.7331** | 0.7229–0.7398 |
| *retired:* logistic regression (scalar) | *0.7228* | — |
| *floor:* persistence | *0.5945* | — |
| *floor:* base rate | *0.5000* | — |

> **Superseded 2026-08-31.** Everything above this line was measured against a
> catalogue missing ~29% of AFAD's events for the region, including nearly all
> of the February 2025 Santorini–Amorgos swarm. Re-derived below. Also note the
> pooled window-level figures are **not** the honest sample size: consecutive
> windows overlap 11–46×, inflating AUC by +0.25 to +0.35. Report block level.

**Block level (30-day disjoint blocks), corrected catalogue, 3 seeds per arm.**
Both catalogues span 2000–2026, so only completeness differs:

| zone | n blocks | base rate | old catalogue | **corrected** | Δ |
|---|---|---|---|---|---|
| **AEGEAN** | 43 | 0.581 | 0.5190 ±0.0150 | **0.6918 ±0.0165** | **+0.173** |
| **CENTRAL** | 43 | 0.395 | 0.3960 ±0.0335 | **0.6176 ±0.0346** | **+0.222** |
| EAFZ | 47 | 0.596 | 0.6615 ±0.0173 | 0.6667 ±0.0323 | +0.005 |
| NAFZ | 42 | 0.381 | 0.4643 ±0.0346 | 0.4103 ±0.0011 | −0.054 |

**CENTRAL is no longer at chance.** The retired report's physical diagnosis —
that CENTRAL is near-Poisson (CV ≈ 1) and therefore unforecastable — does not
survive the corrected catalogue: it reaches 0.618, six times the seed spread
above its old 0.396. NAFZ remains at chance and that diagnosis stands for it.

The gains fall exactly where the catalogue defect was: the missing events were
overwhelmingly offshore Aegean, and AEGEAN plus adjacent CENTRAL move while EAFZ
far to the east does not. Full audit in
`experiment_neural_forecasters_2026-08-30.md` §4.

**Magnitude, alongside "when" (catalog_forecast_report.md §5).** The full
deliverable needs both when *and* how big. A magnitude head was added to the
same network (shared trunk, second output) and tested three ways — higher
loss weight, more training patience, twice the training data — all three
lost to a simple ridge floor on recent-activity statistics (MAE 0.29–0.34
model vs. 0.24–0.25 floor, 3-seed confirmed), the same pattern Task 3 already
found. **Recommended system: the network above for "when," `ridge(max_mag,
mean_mag, b_value, log_rate)` for "how big"** — two tools for two
sub-questions, not a compromise. Combined prediction:
`src/catalog_forecast_predict.py`.

**Waveform features do not help (2026-08-30).** `catalog_forecast_report.md`
listed folding in `Sismokaos-featureExtract`'s continuous features as future
work. Done, and negative: at an operating point where the evaluation is valid
(M≥4.0, 14 d — at M≥4.5/30 d the folds are degenerate and two AUCs undefined),
all three sequence architectures lose to a 0.5823 persistence floor (LSTM
0.5244, GRU 0.5709, TCN 0.5204). The chaotic-feature suite agrees: below floor
on all four model variants, 0 of 10 context/horizon cells. **The forecasting
signal in this project is in the catalogue, not the seismogram.**

**Where this is actually good: AEGEAN.** Per-zone, the recommended system
gets AUC 0.79 for "when" and magnitude MAE 0.215 (beats the 0.247 pooled
floor) — both halves working, on the one zone this project has consistently
found real signal in (scalar or neural, §3/§4). The other 3 zones don't
share it: NAFZ/CENTRAL sit below chance on "when" (near-Poisson, unchanged
diagnosis), so a magnitude number attached to an unreliable event
probability isn't a useful forecast there regardless of its own MAE.

---

## Reading these numbers

Most close-margin comparisons rest on a **single random seed**. Two were re-run
at three seeds and one **reversed sign entirely**, so differences under roughly
0.01–0.02 AUC are not established effects. The three tasks are related but not
interchangeable: they use different window lengths, different splits, and
different floors, so accuracy is not comparable *across* tables — only within.
