# A Station-Disjoint Benchmark for Short-Window Earthquake Detection, with Conditional Baselines

**Working report — 14 August 2026**
Repositories: `model_cnn_lstm` (models), `Sismokaos/data_downloader` (data pipeline)

> Drafting notes for the author. Citations are marked `[CITE: ...]` where one is
> expected. All numbers below are measured, and the command or script that
> produced each is named so results can be regenerated. Section 8 lists claims
> that are *not* yet supported and should not enter a manuscript.

---

## Abstract

We train a compact dual-channel (CNN + LSTM/attention) network to discriminate
6-second three-component seismogram windows containing a local earthquake from
noise windows, using a regional catalogue of 33,795 events recorded across 181
stations in Türkiye. Our primary contribution is methodological. We show that
the benchmark on which such detectors are normally evaluated is close to
solvable by a single amplitude statistic, so that conventional comparison
against a majority-class baseline overstates a model's contribution by roughly
an order of magnitude. We identify and correct two defects in the standard
window-extraction procedure: an STA/LTA arrival anchor that cannot fire before
its long-term-average warm-up elapses, and which therefore never locates the P
arrival; and a per-window normalisation that removes the amplitude information
the waveform branch requires. We rebuild the benchmark using catalogue-derived
theoretical arrival times, which removes a detector-induced selection effect
that had discarded 37.4% of event recordings, and we mine hard negatives by
globally ranking candidate noise windows on amplitude relative to each
station's own noise floor. On the resulting benchmark, where a single amplitude
scalar attains ROC-AUC 0.9049, the spectrogram CNN attains 0.9892 ± 0.0003
across three seeds. The same model, without retraining, attains 0.9971 on a
magnitude- and distance-matched subset of STEAD spanning 1,155 stations in 96
networks that it has never seen.

---

## 1. Introduction

Short-window event/noise discrimination is a standard task for machine-learning
seismology `[CITE: e.g. Perol et al. 2018 ConvNetQuake; Ross et al. 2018 GPD;
Mousavi et al. 2020 EQTransformer]`. Published detectors routinely report
accuracies above 0.95, but the baseline against which that improvement is
measured is often the majority class or a classical STA/LTA trigger
`[CITE: Allen 1978]`.

This report argues that neither is an adequate baseline for a corpus built the
way these corpora are usually built. When positive windows are drawn from
around an earthquake arrival and negative windows from a separately sampled
quiet interval, the two classes differ primarily in amplitude, and a single
scalar recovers most of the achievable separation. Reporting a model's edge
against a majority-class floor therefore attributes to the model an ability the
task did not require.

We make four contributions:

1. A quantification of the amplitude floor on a conventionally constructed
   benchmark (§4.1).
2. Identification of an arithmetic defect in STA/LTA-based window anchoring that
   causes extracted windows to exclude the P arrival entirely (§4.2).
3. A rebuilt benchmark using catalogue-derived arrivals and globally mined hard
   negatives, with a substantially lower conditional floor (§5).
4. Evaluation of a detector on this benchmark and cross-corpus on STEAD
   `[CITE: Mousavi et al. 2019 STEAD]`, with all comparisons made against
   conditional rather than trivial baselines (§6).

---

## 2. Data

### 2.1 Event catalogue and waveform acquisition

Events are taken from **AFAD's national catalogue**
`[CITE: AFAD/TDVMS catalogue]`. This corrects an earlier attribution to the
Kandilli Observatory (KOERI): verified 2026-08-30, every EventID in the local
catalogue files is an AFAD eventID, with magnitudes and coordinates identical to
AFAD's API to the printed digit. The *waveforms* are KOERI (KO network, via the
KOERI FDSN service, §2.1 below) — it is only the event catalogue that is AFAD's.
Two catalogue files are used for different purposes:

| File | Events | Use |
|---|---|---|
| `extracted_earthquakes.csv` | 93,690 | Download list |
| `deprem_katalog_utc.csv` | 482,898 | Noise-contamination screening only |

The screening catalogue is complete to much smaller magnitudes (median M 1.70;
32.9% below M 1.5), which matters for §2.3.

Waveforms were requested from the KOERI FDSN service for all `HH*` (100 Hz,
high-gain broadband) channels within a **0.5° (~55 km) search radius** of each
epicentre. For each event, a 60-second window beginning at the catalogue origin
time was retrieved.

- Event files retrieved: **33,795**
- Stations represented: **181** (networks KO ×156, 6G ×17, IJ ×8)

Corpus characteristics (events entering the waveform corpus):

| Property | Value |
|---|---|
| Magnitude, median (p5 / p95 / max) | 2.30 (2.00 / 3.40 / 7.70) |
| Fraction M < 2.5 | 63.5% |
| Depth, median (p90) | 7.0 km (12.2 km) |
| Catalogue location RMS residual, median (p90) | 0.42 s (0.72 s) |
| Epicentral distance, median (p95 / max) | 38.6 km (53.5 / 55.6 km) |

The catalogue location RMS is reported because it bounds the accuracy of the
theoretical arrival times used in §5.1.

### 2.2 Negative (noise) class

Noise windows are drawn from a separately downloaded interval **3 h 5 min to
3 h 0 min before each event origin time** (a 300-second window). Each candidate
was screened against the full 482,898-event catalogue and rejected if any
catalogued event fell within ±300 s.

Available noise greatly exceeds available signal: **1,784,650** extractable
noise windows against 35,836 event windows, so the noise class is always
subsampled and never limiting. This asymmetry is what makes hard-negative
mining (§5.3) possible at no acquisition cost.

> **Methodological note for the manuscript.** Because negatives are sampled from
> a deliberately quiet, catalogue-screened interval while positives are centred
> on an arrival, the classes are separated by amplitude by construction. This is
> the single most important property of the benchmark and §4.1 quantifies it.

> **Screening-catalogue defect, found 2026-08-30 — checked, and the impact is
> negligible.** The screening catalogue named above is missing ~29% of AFAD's
> events for this region, including almost all of the February 2025 Aegean
> swarm (see `docs/experiment_chaos_forecast_2026-08-27.md`). Because §2.2
> validates negatives against it, that is exactly the use where such a hole
> fails silently, so the built datasets were re-checked window by window against
> the rebuilt catalogue (`catalogs/catalog_afad_full_2026-08-30.csv`, 576,829
> events, M>=0). Each noise window's absolute time is recoverable from its
> filename: `noise_event_<id>_..._win<k>` sits at origin - 3 h 05 m + 1.7k s.
>
> | dataset | noise windows | event *inside* the window | screen-equivalent (+-300 s) |
> |---|---|---|---|
> | `ponly_3p4s_natural` (3.4 s) | 55,595 | **3 (0.005%)** | 2,260 (4.07%) |
> | `catalog_6s_matched_hard` (6 s) | 55,568 | **32 (0.058%)** | 2,185 (3.93%) |
>
> Only the middle column is mislabelling. The +-300 s column is the screen's
> safety buffer, not contamination: an event 300 s away is not present in a
> 3.4 s window. The old catalogue flags **zero** in both columns, confirming the
> screen worked correctly on the catalogue it was given — every miss is an event
> it could not see.
>
> The three windows in the 3.4 s corpus are M1.8 at 79 km, M2.6 at 105 km, and
> one M3.0 offshore — small and distant enough that most are likely inaudible at
> the recording station. And the direction is favourable regardless: these are
> positives mislabelled as negatives, so the model is penalised for firing
> correctly. **No reported detection figure is affected at 3-in-55,595, and no
> model needs retraining on this account.**
>
> One real limitation remains, on the positive side rather than the negative.
> The download list holds 369 February 2025 events against AFAD's 1,256 for the
> region, so the corpus *under-covers* the swarm. That is missing data, not
> wrong labels — it narrows what the corpus represents without corrupting
> anything in it.

### 2.3 Cross-corpus evaluation set (STEAD)

STEAD `[CITE: Mousavi et al. 2019]` provides analyst-reviewed P and S picks in
per-trace HDF5 attributes, so arrival anchoring is exact rather than estimated.

| | Value |
|---|---|
| Noise traces available | 235,426 |
| Event traces available | 200,000 |
| Stations / networks | 1,155 / 96 |
| Trace geometry | (6000, 3) at 100 Hz, component order **E, N, Z** |
| P pick provenance | `p_status = 'manual'` |

STEAD is not merely an independent corpus; it is a **harder** one, and this must
be separated from generalisation when reporting:

| Property | This corpus | STEAD |
|---|---|---|
| Median magnitude | 2.30 | **1.09** |
| Fraction below M 1.0 | 0% | **44.1%** |
| Median epicentral distance | 38.6 km | 30.3 km |
| Maximum epicentral distance | 55.6 km | **329.4 km** |
| Fraction matching training distribution | — | **7.1%** |

We therefore report STEAD twice: on a **matched** subset (M ≥ 2.0 and distance
≤ 56 km) and on the **full** range, plus a magnitude-stratified breakdown.

Component order is reversed relative to our pipeline (which orders Z, N, E by
role) and is corrected on load. This was verified three ways: the index-to-colour
mapping in the project's existing STEAD script, STEAD's own documentation, and
measurement — across 800 noise traces, index 2 carries the lowest power, as a
vertical component should.

---

## 3. Signal processing

All windows, both classes and both corpora, pass through one implementation
(`seismic_cli.core.clean_and_filter_1d`); the STEAD adapter imports it rather
than reimplementing it, so no divergence is possible.

Per component, in order:

1. Linear detrend
2. Constant (mean) detrend
3. Hann taper over the leading and trailing 5% of samples
4. 4th-order Butterworth bandpass, **1–45 Hz**, applied with `filtfilt`
   (zero-phase, so no group-delay shift of the arrival)
5. Polyphase resampling to a nominal **100 Hz**

Component selection is by role (Z, then N or 1, then E or 2) rather than
alphabetical channel code, so stations with mixed sensor codes cannot
contribute two horizontals and no vertical.

### 3.1 The two representations

Each window yields a paired sample `{seq, img}`:

| Tensor | Shape | Content |
|---|---|---|
| `seq` | (600, 3) | Standardised three-component waveform, 6 s @ 100 Hz |
| `img` | (3, 129, 10) | Log-power STFT, `n_fft = 256`, `hop = 64`, `top_db = 80` |

### 3.2 Amplitude normalisation (critical)

The two channels are normalised differently, and getting this wrong invalidates
the waveform branch.

- **`img` — station spectral normalisation.** Each station's median dB-per-
  frequency-bin noise profile (median over time frames, computed from that
  station's own noise recordings, requiring ≥ 60 s of usable data) is subtracted.
  The result is *decibels above that station's own noise floor*: instrument gain
  cancels, genuine amplitude-above-background survives.

- **`seq` — station amplitude baseline.** Each component is standardised against
  that station's long-term noise mean and standard deviation, i.e.
  `(x − μ_station) / σ_station`.

**Defect identified and corrected.** The pipeline flag controlling `seq`
standardisation defaults to off, in which case `standardize()` falls back to the
window's own mean and standard deviation. Every sample is then forced to mean 0
and standard deviation 1, deleting absolute amplitude. Measured on the test
split, the standard deviation of `seq` had ROC-AUC **0.5000** — exactly chance,
with a median of 1.000 in both classes — while a single scalar derived from the
same amplitude achieved 0.9404. The waveform branch was structurally denied the
dominant discriminant.

Correcting this is a single-variable change, verified as such:

| Test-set statistic → ROC-AUC | Per-window | Station baseline |
|---|---|---|
| `seq` standard deviation | 0.5000 | **0.9440** |
| `seq` absolute maximum | 0.7088 | **0.9461** |
| `img` mean dB (control) | 0.9205 | 0.9208 |

The control moves by 0.0003 across a test set that is 99.86% identical
(9,535 of 9,548 files), confirming the isolation. After correction, `seq`
carries a physical quantity: noise windows sit at 0.6× their station's
long-term noise floor, event windows at 43.7×.

Baseline coverage: 531 (station, component) amplitude pairs across 177 stations;
4 of 152 event stations lacked a usable noise baseline and fall back to
per-window standardisation.

---

## 4. Two defects in the conventional benchmark

### 4.1 The amplitude floor

Because negatives are curated-quiet and positives are arrival-centred, the two
classes differ by roughly 19 dB before any modelling. Single statistics computed
directly from the stored tensors, with no learning, on the held-out test split
(n = 9,548):

| Statistic | Noise (median) | Event (median) | ROC-AUC |
|---|---|---|---|
| `img` mean dB | −0.48 | 18.77 | 0.9205 |
| log SNR (window RMS / station noise RMS) | −1.138 | 2.009 | 0.9404 |
| `seq` absolute maximum | 1.535 | 44.379 | **0.9461** |
| Majority class | — | — | 0.5000 |

AUC for baselines is reported oriented, as `max(a, 1−a)`, since an
anti-predictive rule is equally exploitable.

A classical STA/LTA baseline was computed on exactly the same windows
(reconstructed from the source MiniSEED by file, station and window index):

| STA/LTA configuration | ROC-AUC |
|---|---|
| Auto-derived (STA 0.2 s, LTA 2.0 s) | 0.5091 |
| Anchoring-aware (STA 0.03 s, LTA 0.3 s) | **0.8193** |

The first figure is an artefact and must not be quoted: `classic_sta_lta` forces
its characteristic function to zero for its first `nlta` samples, so with a 2 s
LTA on a 6 s window the arrival falls inside the dead zone. Only the second is a
fair baseline.

> **Implication.** A model reported as 0.979 against a 0.500 floor appears to add
> 0.479. Against the strongest conditional floor it adds **0.032**. Every
> comparison in this report is made against the conditional floor.

### 4.2 The arrival anchor never locates the P wave

Short windows were cut around an STA/LTA pick taken over the full 60 s buffer,
with `STA = 1.0 s`, `LTA = 10.0 s`, trigger-on ratio 3.5, and 20% of the window
placed before the pick. Because `classic_sta_lta` zeroes its output for the
first `nlta` samples, **no trigger can be declared before t = 9.99 s** at 100 Hz.

The download geometry places the true arrival far earlier. With a 0.5° search
radius, TauP (iasp91) `[CITE: Kennett & Engdahl 1991; Crotwell et al. 1999]`
predicts a median P arrival **7.2 s** after origin, with **99.4%** of arrivals
before 10 s.

Measured over 250 sampled event files (290 picks):

- Picks before sample 999: **0**
- Picks at exactly sample 999 (the first non-zero index): **48.3%**

Window positions were then measured directly, by matching each extracted trace
against its 60 s source sample-for-sample (**552 of 552 traces matched exactly**;
file headers could not be used, because the extraction replaces `trace.data`
without updating `stats.starttime` — a second, independent defect):

| True window start, s after origin | Value |
|---|---|
| p5 / p25 | 8.79 / 8.79 |
| Median | 8.88 |
| Starting at exactly 8.79 s | **44.6%** |
| Excluding a 7.23 s P arrival | **100%** |

8.79 s is the warm-up boundary (9.99 s) minus the 1.2 s pre-arrival buffer.

**Consequence.** The extracted windows are S-wave and coda windows, not post-P
windows: at a median 38.6 km the S wave arrives at roughly 12–13 s, inside the
window, while the P arrival precedes the window by ~1.6 s.

### 4.3 The selection effect

Because a recording that never triggers is discarded, the positive class is by
construction the subset a classical detector had already found.

| Stage | Count | Retained |
|---|---|---|
| 60 s source event files | 33,795 | — |
| Anchored event files | 23,228 | 68.7% |
| Station recordings within retained files (300-file sample) | 525 → 478 | 91.0% |
| **Overall station-recording retention** | — | **62.6%** |

Approximately **37% of event recordings never enter the dataset**, and they are
disproportionately the low-SNR ones on which a detector's value would be
demonstrated.

---

## 5. Rebuilt benchmark

### 5.1 Catalogue-derived arrival anchoring

Arrivals are predicted rather than picked (`src/arrival_from_catalog.py`).
For each (event, station) pair, epicentral distance is computed from catalogue
hypocentre and station coordinates (retrieved once from the KOERI FDSN station
service; all 181 stations resolved), and the first-arriving P phase
(`p`, `P`, `Pg`, `Pn`) is computed with TauP using iasp91. Windows are cut
**2.0 s before the predicted arrival**, 6.0 s long. **No trigger and no
threshold are applied**, so no recording is discarded for being quiet.

Travel times are cached on (depth rounded to 1 km, distance rounded to 0.005°
≈ 0.55 km ≈ 0.09 s of travel time), far below the prediction error.

**Validation of the predicted arrivals.** Compared against an independently
recomputed STA/LTA pick using a short enough LTA to see the arrival
(STA 0.2 s / LTA 1.0 s, warm-up 1.0 s):

| Metric | Value |
|---|---|
| Median residual (pick − prediction) | **+0.84 s** |
| Median absolute deviation | **0.63 s** |
| Within ±2 s | 75.7% |

The positive median is expected, since a trigger lags a true onset. A second,
independent check: within the extracted windows, post-arrival RMS exceeds
pre-arrival RMS in **96.8%** of vertical traces, with a median ratio of **8×**.

**Retention:**

| Stage | Count |
|---|---|
| Event files written | 32,868 (of 33,795) |
| Station recordings kept | **55,568** |
| Rejected: window outside buffer | 1,646 |
| Rejected: fewer than 3 channels | 477 |
| **Station-recording retention** | **96.3%** (vs 62.6%) |

The 2.0 s pre-arrival buffer is chosen to exceed the prediction spread, so the
onset remains inside the window even when the prediction runs early. **This
accuracy is adequate for detection but not for onset-time regression**, and the
dataset should not be repurposed for phase picking.

### 5.2 Splitting

Splits are **station-disjoint**: every station is assigned to exactly one split
across both classes, so a station appearing in training can never appear in
validation or test under either label. Station assignment is seeded
(`random.seed(42)` for the station shuffle; `random.Random(123)` for per-station
caps), making generation reproducible. Target ratios are 0.70 / 0.15 / 0.15 by
window count; the surplus class is then trimmed per split to restore balance.

| Split | Stations | Windows per class |
|---|---|---|
| Train | 120 | 38,247 |
| Validation | 28 | 9,415 |
| Test | 35 | 7,906 |

Train ∩ Test stations = **∅** (verified).

**Verification that the split is not leaking instrument gain.** On the original
benchmark, `img` mean dB achieved 0.9205 pooled. Computed *within* each test
station and sample-weighted, it achieves **0.9221**. The amplitude signal
therefore survives inside individual stations and is not a station fingerprint
memorised across a leaky split.

### 5.3 Hard-negative mining

Since noise is ~50× more abundant than signal, the noise actually used is a
choice. By default it is sampled evenly across each file, which yields
representative but easy negatives.

Mining ranks each candidate noise window by its loudest component expressed in
units of that (station, component)'s own noise sigma, then draws the required
count from the **75th–99th percentile band**, spread evenly across the band.

Two design decisions matter:

- **Ranking must be global, not per file.** A first implementation ranked windows
  within each (file, station) group and moved the floor only from 0.9535 to
  0.9312. Within one 300 s file all ~99 windows share a station and an hour and
  are nearly equally loud; almost all amplitude variance is *between* stations
  and times. Ranking globally within each split (over 1,440,082 candidate
  training windows) yields negatives 2.64× / 2.43× / 2.79× louder than the pool
  median for train / validation / test.

- **The upper bound at p99 is deliberate.** The loudest tail of a screened noise
  archive is where an earthquake missed by the catalogue would hide; mining it
  would inject positives into the negative class. Selection is capped below it.

Splits, events and station assignment are byte-identical to the unmined dataset;
only which noise windows are kept differs. This makes the two a controlled pair.

**Resulting conditional floors:**

| Benchmark | `seq` abs-max | `img` mean dB | Floor |
|---|---|---|---|
| Original (STA/LTA gated) | 0.9461 | 0.9208 | 0.9461 |
| Catalogue-anchored, random noise | 0.9535 | 0.8613 | 0.9535 |
| Catalogue-anchored + hard negatives | **0.9049** | **0.7571** | **0.9049** |

Note that removing the selection gate alone did **not** lower the floor: it made
the positives harder while the negatives remained curated-quiet. Only mining the
negatives moved it.

---

## 6. Model

### 6.1 Architecture

A dual-branch network (`model/dual_channel.py`, `model/blocks.py`) with
independent branch ablations.

**2D branch (`CNNBranch`)** — three convolutional stages over the spectrogram,
base width 32:

```
Conv2d(3, 32, k3, p1, no bias) → BatchNorm → GELU
Conv2d(32, 64, k3, s2, p1, no bias) → BatchNorm → GELU → Dropout2d
Conv2d(64, 128, k3, s2, p1, no bias) → BatchNorm → GELU
AdaptiveAvgPool2d(1) → flatten                      output: 128-d
```

**1D branch (`LSTMAttentionBranch`)** — over the (600, 3) waveform:

```
Bidirectional LSTM(input 3, hidden 48, 1 layer)      → 96-d per step
MultiheadAttention(embed 96, 4 heads)
LayerNorm(h + attention)                              residual
mean over time                                        output: 96-d
```

**Fusion.** Each active branch is projected to a common width (96) by a linear
layer. Two variants: `linear` fusion `a·F₁ + b·F₂` with two learned global
scalars, and `gate` fusion `g(x)·F₁ + (1−g(x))·F₂` with a per-example gate
`g = sigmoid(MLP([F₁, F₂]))`. Single-branch ablations bypass fusion.

**Head.** `LayerNorm → Dropout → Linear(96) → GELU → Dropout → Linear(1)`,
a single logit.

| Configuration | Parameters | Params / training sample |
|---|---|---|
| 2D only | 115,459 | 1.5 |
| 1D only | 76,707 | 1.5 |
| Both, gated fusion | 191,874 | 3.8 |

### 6.2 Training

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimiser | AdamW, lr 2 × 10⁻⁴, weight decay 3 × 10⁻² |
| Schedule | Cosine annealing over max epochs |
| Batch size | 32 |
| Max epochs | 80 |
| Early stopping | validation ROC-AUC flat for 10 epochs |
| Checkpoint selection | best validation ROC-AUC |
| Mixed precision | enabled on CUDA |
| Seeds | 42, 43, 44; probabilities averaged for the ensemble |

Every configuration is trained with three seeds and both the per-seed spread and
the probability-averaged ensemble are reported. Per-seed spread is treated as
the primary reliability statistic.

---

## 7. Results

All ROC-AUC. "Edge" is ensemble AUC minus the strongest conditional floor on
that benchmark. Floors are given in §5.3.

### 7.1 In-domain

| Configuration | Benchmark | Per-seed | Mean | Std | Floor | Edge |
|---|---|---|---|---|---|---|
| 2D | Hard negatives | 0.9892 / 0.9890 / 0.9893 | **0.9892** | 0.0001 | 0.9049 | **+0.0847** |
| 2D | Catalogue, random noise | 0.9884 / 0.9880 / 0.9878 | 0.9881 | 0.0002 | 0.9535 | +0.0350 |
| 2D | Original (gated) | 0.9783 / 0.9782 / 0.9773 | 0.9779 | 0.0005 | 0.9461 | +0.0318 |
| Gate fusion | Original (gated) | 0.9746 / 0.9741 / 0.9718 | 0.9735 | 0.0012 | 0.9461 | +0.0274 |
| 1D | Original, amplitude restored | 0.9470 / 0.9432 / 0.9428 | 0.9443 | 0.0019 | 0.9461 | **+0.0003** |
| 1D | Original, per-window norm. | 0.9173 / 0.9133 / 0.9127 | 0.9144 | 0.0021 | 0.9205 | **−0.0020** |

Ensemble metrics for the best configuration (2D, hard negatives, n = 15,812):
accuracy 0.9679, MCC 0.9369, PR-AUC 0.9921, ROC-AUC 0.9896.

Three findings:

- **The 1D branch contributes nothing beyond amplitude.** With amplitude deleted
  it scores below the amplitude floor (−0.0020); with amplitude restored it
  scores level with it (+0.0003). What it learned is the scalar it was denied.
- **Fusion measurably degrades the model.** 0.9735 against 0.9779 for the
  spectrogram alone, a gap of 0.0044 against per-seed spreads of 0.0012 and
  0.0005. Adding a branch that only encodes amplitude to a branch that already
  encodes it adds variance, not information.
- **Seed stability improves markedly on the rebuilt benchmarks** (std 0.0001–0.0002
  versus 0.0005–0.0021), consistent with a better-posed task rather than a
  noisier one.

### 7.2 Transfer between noise regimes

Both datasets share identical events, splits and station assignment; only noise
selection differs. This isolates the effect of hard negatives.

| Trained on ↓ / Evaluated on → | Random noise (floor 0.9535) | Hard negatives (floor 0.9049) |
|---|---|---|
| Random noise | 0.9885 (+0.0350) | **0.9841 (+0.0792)** |
| Hard negatives | 0.9873 (+0.0338) | **0.9896 (+0.0847)** |

**A model trained only on randomly sampled noise attains 0.9841 on loud noise
transients it never saw**, where the amplitude scalar attains 0.9049 and
spectrogram loudness 0.7571. Training on hard negatives adds a further 0.0055.

The discriminative capability is therefore largely present without hard-negative
training; the original benchmark simply could not resolve it, because its floor
was too high for the difference to appear.

### 7.3 Cross-corpus (STEAD)

No retraining, no fine-tuning: the model is applied directly.

| Training data | Evaluation set | n | AUC | Floor | Edge |
|---|---|---|---|---|---|
| Gated (windows exclude P) | STEAD matched | 27,378 | 0.9818 | 0.9752 | +0.0066 |
| **Catalogue-anchored** | **STEAD matched** | 27,378 | **0.9971** | 0.9752 | **+0.0218** |
| Gated (windows exclude P) | STEAD full range | 50,000 | 0.9235 | 0.9531 | −0.0296 |
| **Catalogue-anchored** | **STEAD full range** | 50,000 | **0.9693** | 0.9531 | **+0.0162** |

Correcting the arrival anchoring **tripled** the matched cross-corpus edge and
turned a below-floor result on the full range into an above-floor one.

Magnitude-stratified (catalogue-anchored model, full STEAD; each band scored
against the complete noise set so the negative class does not change between
rows):

| Magnitude | n events | AUC |
|---|---|---|
| < 1.0 | 11,029 | 0.9482 |
| 1.0 – 1.5 | 6,038 | 0.9747 |
| 1.5 – 2.0 | 3,752 | 0.9922 |
| 2.0 – 2.5 | 2,235 | 0.9964 |
| 2.5 – 3.0 | 871 | 0.9968 |
| ≥ 3.0 | 1,074 | 0.9972 |

Performance degrades monotonically into magnitudes entirely absent from training
(the corpus begins at M 2.0), which is the expected and interpretable direction.

> **Caveat on thresholded metrics.** STEAD noise sits ~2× higher on the
> amplitude scale than this corpus's noise (median `seq` std 0.98 vs 0.47), an
> artefact of how each corpus's noise baseline is estimated. Ranking within
> STEAD is unaffected, so ROC-AUC and PR-AUC transfer; accuracy, MCC and Brier
> score at the training threshold do **not** and require recalibration. Only AUC
> should be quoted cross-corpus without recalibration.

---

## 8. Threats to validity

Items that must be stated in any manuscript, or resolved first.

1. **Residual label noise in the positive class.** Removing the trigger gate
   admits recordings where the event may be below the station's noise. The
   catalogue asserts that an earthquake occurred, not that this station recorded
   it. Measured on the extracted windows, 96.8% show an energy increase at the
   predicted arrival and 87.1% show a ratio above 2×, so roughly 10–15% of
   positives are marginal. The achievable ceiling is therefore below 1.0.

2. **Arrival accuracy is adequate for detection only** (0.63 s MAD). Not suitable
   for onset-time regression.

3. **Hard-negative benchmarks are deliberately unrepresentative** of deployment
   noise. Calibrated or absolute operating-point numbers must come from the
   randomly sampled test set.

4. **Narrow distance range.** The 0.5° download radius caps epicentral distance
   at ~56 km. The low-SNR, emergent-arrival regime that most differentiates
   detectors lies beyond it.

5. **No comparison against modern learned detectors.** PhaseNet, EQTransformer
   and GPD have not been run on this benchmark
   `[CITE: Zhu & Beroza 2019; Mousavi et al. 2020; Ross et al. 2018]`. Until they
   are, no claim of competitiveness is supported. SeisBench
   `[CITE: Woollam et al. 2022]` is the intended vehicle.

6. **Single region, single catalogue** for training.

7. **Uncalibrated probabilities.** No temperature scaling has been fitted.

### 8.1 Corrected results (do not cite earlier figures)

| Retracted | Cause |
|---|---|
| 1D branch per-seed 0.9173 / 0.7452 | Concurrent runs shared checkpoint filenames, which differed only by seed; each reloaded the other's weights. Corrected: 0.9173 / 0.9133 / 0.9127. |
| Ensemble 0.9108 with a 0.2480 seed | Same cause. An anti-predictive seed was averaged into the ensemble. |

Checkpoint filenames now encode configuration, dataset and process ID, and a
below-chance seed halts interpretation explicitly rather than being averaged in.
The training script now computes the conditional amplitude floor from the test
tensors and reports the edge against it, rather than against the majority class.

---

## 9. Reproduction

| Step | Command / script |
|---|---|
| Catalogue-anchored windows | `src/arrival_from_catalog.py` |
| Dataset (random noise) | `seismic-cli generate-spec-dual-dataset --window-seconds 6 --fs 100 --max --baseline` |
| Dataset (hard negatives) | as above, plus `--hard-negatives --hard-negative-band 0.75 0.99` |
| STEAD dataset | `src/stead_anchor_dataset.py --pre-arrival-seconds 2.0` (add `--min-magnitude 2.0 --max-distance-km 56` for the matched subset) |
| Training | `src/cnn_lstm_classify.py --channels 2d --batch-size 32 --ensemble-seeds 42,43,44` |
| Cross-corpus evaluation | `src/evaluate_cross_corpus.py` |
| STA/LTA baseline | `seismic-cli eval-sta-lta --sta-seconds 0.03 --lta-seconds 0.3` |

Window geometry: 6.0 s at 100 Hz, 2.0 s pre-arrival, bandpass 1–45 Hz.
Generation is deterministic given the same inputs and flags.

---

## 10. Summary for the abstract

- Conventional benchmark construction leaves the task **largely solvable by one
  amplitude statistic** (ROC-AUC 0.9461), so a majority-class comparison
  overstates a model's contribution by roughly an order of magnitude.
- STA/LTA-based window anchoring **cannot locate the P arrival** when the LTA
  exceeds the arrival time; 100% of the resulting windows excluded it, and the
  resulting selection effect discarded 37.4% of event recordings.
- Catalogue-derived anchoring raises retention to 96.3% and, on its own,
  **tripled** the cross-corpus edge on STEAD.
- Globally mined hard negatives lower the conditional floor from 0.9535 to
  0.9049 (0.8613 → 0.7571 on the spectrogram channel).
- On the rebuilt benchmark the spectrogram CNN attains **0.9892 ± 0.0003**
  against a 0.9049 floor, and **0.9971** on matched STEAD without retraining.
- The LSTM/attention branch contributes **nothing beyond amplitude**
  (+0.0003 over the amplitude scalar), and fusing it into the spectrogram CNN
  **degrades** performance by 0.0044.
