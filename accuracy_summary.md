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

## Task 4 — Catalog forecasting: time to next mainshock (3 balanced classes, 264-event LOEO)

| Model | Accuracy | Kappa |
|---|---|---|
| Gradient boosting on 9 seismicity indicators | 31.49 % | −0.028 |
| Dual-channel CNN+LSTM | *run in progress* | — |
| *floor:* chance (3 balanced classes) | *33.33 %* | *0.000* |

**The scalar model is below chance with negative kappa** — the nine seismicity
indicators (b-value, Lyapunov exponent, energy release rate, …) carry no usable
signal for this target under leave-one-event-out evaluation. See
`catalog_report.md`.

> The per-fold majority-class baseline that both LOEO scripts originally
> reported (8.53 %) must **not** be used as the floor: with balanced classes,
> removing a fold tips the training pool away from that fold's own dominant
> class, so the "majority" is the class the fold has *least* of — it matched the
> fold's true mode in 2 of 264 folds. Beating it is evidence of nothing.

---

## Reading these numbers

Most close-margin comparisons rest on a **single random seed**. Two were re-run
at three seeds and one **reversed sign entirely**, so differences under roughly
0.01–0.02 AUC are not established effects. The three tasks are related but not
interchangeable: they use different window lengths, different splits, and
different floors, so accuracy is not comparable *across* tables — only within.
