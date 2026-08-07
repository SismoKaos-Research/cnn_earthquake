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

## Task 2 — Magnitude class from one 3-second window (M ≥ 2.5 vs. below; 5,594 test)

| Model | Accuracy | AUC | MCC |
|---|---|---|---|
| **CNN (spectrogram + aux)** | **79.78 %** | **0.8550** | **+0.566** |
| *floor:* logistic on log-SNR + log-distance | *71.36 %* | *0.7486* | *+0.372* |
| *floor:* majority class | *61.35 %* | — | — |

**The encoded window adds real signal here** — +0.106 AUC over amplitude and
distance alone.

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

## Task 4 — Catalog forecasting

Two formulations were tried. The first is a clean negative result; the second
works, but only in some zones.

**4a. Time to next mainshock** (3 balanced classes, 264-event LOEO) — *abandoned*

| Model | Accuracy | Kappa |
|---|---|---|
| Gradient boosting on 9 seismicity indicators | 31.49 % | −0.028 |
| *floor:* chance (3 balanced classes) | *33.33 %* | *0.000* |

At chance with negative kappa. Declustering removes aftershocks — the most
predictable part of seismicity — leaving mainshock timing that is near-Poisson
(gap CV 0.67–1.17). Near-unlearnable by construction.

**4b. M ≥ 4.5 within 30 days, per fault zone** — *the working forecaster*

Medians over **12 rolling origins**, not a single cut. The bar is
`max(chance, persistence)`, since persistence is *below* chance in two zones.

| Zone | CV | Persistence | **Model (median)** | IQR | Clears both floors |
|---|---|---|---|---|---|
| **AEGEAN** | 1.56 | 0.540 | **0.661** | [0.532, 0.763] | 6/12 |
| **EAFZ** | 1.46 | 0.446 | **0.650** | [0.523, 0.724] | 7/12 |
| NAFZ | 1.04 | 0.341 | 0.447 | [0.334, 0.571] | 3/12 |
| CENTRAL | 1.02 | 0.183 | 0.369 | [0.311, 0.589] | 4/12 |

**Forecastable in the two clustered zones, not in the two near-Poisson ones.**
Forecastability tracks clustering: where CV ≈ 1 the process is memoryless and no
model of this kind can work.

> **These numbers replace a single-cut headline of 0.798 for AEGEAN.** The
> rolling-origin backtest showed that figure sat near the *top* of a
> distribution whose IQR spans 0.23 AUC. It also promoted EAFZ from "not
> forecastable" (single-cut 0.570) to the most consistent zone of the four.
> The effect is real but weak — AEGEAN clears both floors at exactly half its
> origins. See `catalog_report.md` §4.4.

> Two floors here are traps, not baselines. The per-fold majority-class figure
> both LOEO scripts originally reported (8.53 %) is *anti-predictive*: with
> balanced classes, removing a fold tips the pool away from that fold's own
> dominant class, so the "majority" is the class it has *least* of — matching
> the fold's true mode in 2 of 264 folds. And for 4b, the base rate is far
> weaker than **persistence** — but persistence itself scores 0.18–0.34 in NAFZ
> and CENTRAL, i.e. *below chance*, so beating it there proves nothing. Counted
> against persistence alone, CENTRAL "wins" 9 of 12 origins while sitting at
> AUC 0.369. The bar has to be `max(chance, persistence)`. See
> `catalog_report.md` §4.4.

---

## Reading these numbers

Most close-margin comparisons rest on a **single random seed**. Two were re-run
at three seeds and one **reversed sign entirely**, so differences under roughly
0.01–0.02 AUC are not established effects. The three tasks are related but not
interchangeable: they use different window lengths, different splits, and
different floors, so accuracy is not comparable *across* tables — only within.
