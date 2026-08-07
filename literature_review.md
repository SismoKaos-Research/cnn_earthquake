# Literature Review

**Seven papers, read against this project's own results**

Sources: `~/Downloads/cnn-lstm/`. Three of these already underpin work here; four
were read to decide what, if anything, to adopt for the catalog forecaster.

Each entry answers three questions: **what it claims**, **what it actually
evaluated against**, and **what transfers**. The third is the one that matters,
and the answer is usually "nothing" — which is itself the finding.

---

## The through-line

**Not one of the seven runs a non-neural floor in its own experiments.** Every
paper compares neural models to other neural models, or to nothing.

That matters because this project has computed such floors four times, and the
simple model won three of them:

| Task | Best neural | Non-neural floor | Winner |
|---|---|---|---|
| Detection (`report.md` §6) | spectrogram CNN 0.9793 AUC | STA/LTA 0.8194 | CNN |
| Magnitude class (§7) | CNN 0.8550 AUC | logistic on 2 scalars 0.7486 | CNN |
| Three-class risk (§8) | CNN 73.64 % | two-stage scalar **82.83 %** | **scalar** |
| Catalog forecast (`catalog_report.md`) | — | logistic beats gradient boosting | **simpler** |

A published architecture improvement in this field is therefore not evidence
that the architecture is doing the work. That prior shaped every decision in
`catalog_report.md`, and it is the main thing this batch of papers reinforced.

---

## 1. Kaushal, Gupta & Sehgal (2025) — hybrid RNN-LSTM, 98 % accuracy
*Soil Dynamics and Earthquake Engineering* 195:109432

**The closest published work to our task**, and the one that should be trusted
least.

**Claims:** 98 % accuracy for a hybrid RNN-LSTM on earthquake prediction, vs
97.4 % LSTM, 95 % AdaBoost, 93 % ANN. Global NEIC catalog, M ≥ 5.5, 1965–2023.

**What it actually evaluated against:** this cannot be determined, because
**the paper never defines its classification target.** The methods section's
complete description of the labelling step is:

> "Prediction task on the significant earthquake dataset. Labels the datasets."

It trains categorical cross-entropy on that undefined label. Three further
problems compound it:

- **Every row in the dataset is an earthquake.** The catalog is filtered to
  M ≥ 5.5 events. There is no negative class described anywhere, so it is not
  clear what a "prediction" is being made about.
- **The 80/20 split is never stated to be chronological.** On a time series
  spanning 1965–2023, a random split lets the model interpolate between
  neighbouring events.
- **PCA is run over a feature set that explicitly includes "date and time of
  occurrence."** Time is an input feature. Combined with a possibly-random
  split, this is textbook leakage.

Its own Table 1 surveys twelve prior works and reports a floor for none of them.

**What transfers:** nothing. The 98 % is not a target to chase — it is a worked
example of why `forecast_eval.py` refuses to print a model score without a floor
beside it. Cited in `catalog_report.md` for exactly that purpose.

---

## 2. Yılmaz, Akıllı & Akı (2023) — entropy and chaos before an earthquake
*Chaos, Solitons and Fractals* 173:113585

**Same region as this project** — Marmara / North Anatolian fault, the M5.7
İstanbul 2019 and M5.2 Düzce 2021 events.

**Claims:** four complexity measures — windowed scalogram entropy, windowed
scale index, sample entropy, and Lyapunov exponents — track changes in
low-amplitude continuous waveform in the minutes before an earthquake, and
"can be valuable in… contributing to the development of earthquake forecasting
techniques."

**What it actually evaluated against:** nothing. There is no classifier, no
floor, and **N = 2 events**. The paper is descriptive time-series analysis, not
prediction, and is careful enough not to claim otherwise in its results — but
the framing invites the stronger reading.

**The useful part is a confound the authors report as a strength.** From the
abstract:

> "Lyapunov exponents and sample entropy appear more effective in their response
> to the change in complexity and chaotic characteristics **due to the change in
> the signal amplitude**."

That is this project's dominant finding, arrived at from the other direction and
not recognised as a problem. `log_snr` — mean log amplitude ratio — is the
single largest effect measured anywhere in `report.md` (+0.0874 AUC on the RAM
representation). If Lyapunov and sample entropy respond to amplitude, then a
"chaotic precursor" that rises before an event is not shown to be separable from
the ground simply shaking harder. The paper never tests that separation.

**What transfers:** sample entropy over the *catalog* sequence is a cheap
candidate feature, and `max_lyapunov_rosenstein` already exists at
`catalog.py:174`. But this paper is **not evidence for it** — it computes both
on waveforms, and moving them to catalog space changes the object entirely. Any
such feature has to earn its place against the rolling-origin distribution in
`catalog_report.md` §4.4, not against a citation.

---

## 3. Başar & Çelik (2026) — CNN-LSTM on high-rate GNSS
*Sensors* 26(1):519

**The methodologically strongest paper of the seven**, and the only one with a
practice worth copying.

**Claims:** a hybrid CNN-LSTM detects seismic events in 5 Hz GNSS velocity time
series, trained on pseudo-synthetic data and tested on an independent real GNSS
setting.

**What it actually evaluated against:** itself, honestly. It is the only paper
here that:

- **Calibrates its operating point on validation** instead of thresholding at
  0.5, and reports results at both fixed and calibrated points.
- **Reports its own degradation.** The headline finding is that performance
  *drops* on real data relative to pseudo-synthetic training, with the vertical
  component degrading worst. Most papers in this batch would have reported the
  synthetic number.
- **Separates frame-level from event-level evaluation** and compares several
  aggregation schemes, rather than picking the flattering one.

**What transfers: the protocol, not the model.** We have no GNSS data, and the
architecture is unremarkable. But this project was thresholding at 0.5 while
holding an unused validation split. Adopted in `forecast_backtest.py`: the
threshold is now selected per origin on validation, and it turns out to be
**0.137, not 0.5** — a large correction that changes precision/recall
substantially.

Its MMD distribution-shift diagnostic was considered and **not** adopted: it
would measure a shift already nameable from the catalog (the Kahramanmaraş
sequence sits in the test era, moving the positive rate from 0.414 to 0.782).
Measuring a shift you can already point to does not change what you do next.

---

## 4. Shen, Hou, Lu & Li (2025) — real-time magnitude estimation
*Applied Sciences* 15(5):2587

**Parallels `report.md` §7** (magnitude class from a 3-second window).

**Claims:** DCRNNAmp reaches MAE 0.287, RMSE 0.397, R² 0.737 in the first 3 s
after P-wave arrival, over K-NET/KiK-net — 8,144 events, 297,099 three-component
records.

**What it actually evaluated against:** four neural models (MagNet+Bi-LSTM,
DCRNN, DCRNNAmp, Exams). Its data handling is genuinely good — the split is by
**event** and **chronological** ("according to the moment of epicenter
generation"), giving 5,365 train / 728 validation / 2,051 test events. That is
the same discipline `regression.py` uses and better than most of this batch.

But the classical parameters it surveys at length in the introduction — Pd, τc,
τp-max, Sp, the whole empirical-relation literature — are **never run as a
baseline in its own experiments.** The one comparison to "traditional
mathematical modeling" is cited from prior work, not measured here.

**What transfers:** confirmation, not method. Its own discussion notes that
epicentral distance correlates with magnitude estimation quality and that better
SNR speeds convergence — the same two scalars (`log_snr`, `log_distance`) that
beat our CNN outright in `report.md` §8.5. A paper reporting R² 0.737 without
testing amplitude-plus-distance alone has not established that its network is
doing the work.

---

## 5. Wang & Zhao (2025) — 1D2D-EDL for bearing fault diagnosis
*Applied Soft Computing* — S1568494625002005

**Already replicated in this project.** Source of `DualChannelRiskNet`'s
architecture: a 1D LSTM-with-self-attention branch and a 2D CNN branch fused as
`F' = a·F_1D + b·F_2D`.

**What transferred, and what happened:** the architecture ported cleanly to
seismic windows. It did not help. `report.md` §6 measured the plain spectrogram
CNN at 0.9793 AUC and **no dual-channel configuration beat it** — linear fusion
0.9646, gated fusion 0.9761, stacked fusion 0.9743. The two additional fusion
mechanisms tried here (per-example gating, late-fusion stacking) are extensions
beyond the paper and also did not beat the single-channel baseline.

This is the cleanest example of the through-line above: an architecture that
works on its own domain, transplanted faithfully, adding nothing once a floor is
present.

---

## 6. Nurtas et al. (2025) — CNN-BiLSTM PGA forecasting
*ACDSA 2025*

**Reviewed in full; implementation shelved** (plan retained at
`~/.claude/plans/let-s-go-over-the-foamy-elephant.md`).

**Claims:** validation MAE 2.61 gal, R² 0.714 predicting peak ground
acceleration from the first 3 s of three-component waveform. Input tensor
`(300, 3)` — identical in shape to our existing `window_post_3s_anchored`.

**What it actually evaluated against:** three neural models (ANN, LSTM,
CNN-BiLSTM). No physical baseline at all. PGA is an amplitude quantity and the
input window contains amplitude, so regressing log-PGA on the peak amplitude of
the first 3 s would plausibly reach much of R² 0.714 unaided. It is also
reported on **validation**, the same split used for early stopping and
checkpointing — there is no test set. Training on MSE in log space while
reporting R² in linear space is the likely cause of its ANN scoring R² −10.08.

**What transfers:** the target is reachable with our data, but our sensors are
the wrong class — 100 % of our channels are `HH*` high-gain broadband
*velocity* seismometers, where K-NET is a strong-motion accelerometer network.
Direct measurement during feasibility work found large events under-recorded
(a M7.7 record gave 42.96 gal against a M6.2's 71.06 gal, consistent with
clipping) and sentinel-value contamination producing 10,350 gal from an M2.3
event. Shelved on the user's instruction to finish the forecaster first.

---

## 7. Development of an LSTM-Based Statistical Model for Earthquake Forecasting in Central Asia

Read earlier in this project. Catalog-based LSTM forecasting, closest in spirit
to `forecast.py`'s target. Reinforced the same structural point that drove the
reformulation in `catalog_report.md` §3: **dense targets ("will an event of
magnitude M occur in the next N days") are learnable in a way that sparse
time-to-next-event targets are not**, because the latter, after declustering,
approaches a memoryless process where P(wait | history) = P(wait).

---

## What was adopted

| From | Adopted | Where |
|---|---|---|
| Başar & Çelik | validation-calibrated operating point | `forecast_backtest.py`, threshold 0.137 not 0.5 |
| Başar & Çelik | report degradation rather than the best slice | `catalog_report.md` §4.4 |
| Kaushal | *(cautionary)* never report a model score without a floor | already project policy; reinforced |
| Yılmaz | *(cautionary)* complexity measures may be amplitude proxies | gates any future entropy feature |
| Wang & Zhao | dual-channel fusion architecture | `DualChannelRiskNet` — replicated, did not win |

**Nothing in this batch supplied a forecasting method worth adopting.** Two
supplied warnings, one supplied an evaluation practice, one supplied an
architecture that had already been tried and had already lost to a simpler
model.
