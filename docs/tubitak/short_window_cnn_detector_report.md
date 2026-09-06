# Short-Window CNN Event/Noise Classifier — Pipeline, Architecture, Results

**Consolidated technical report, 18 August 2026.**
Repositories: `model_cnn_lstm` (models, training, evaluation), `Sismokaos/data_downloader`
(`seismic_cli` acquisition + dataset generation), `sismokaos-cli` (Rust preprocessor, not
used by this task).

**Scope note, read first.** This report covers the *detection* task — is there an earthquake
in this short three-component window, or is it noise. Every detection result below was
measured on **6-second** windows. The **3-second** window length appears in this project in
three places, and none of them is a detection result: it is the window used for the magnitude
classifier (report §7), for the ground-motion regressor (report §13), and it is analysed
*structurally* for why the RAM encoding degrades as windows shorten (§3.4 here). No 3 s
event/noise detector has been trained under the corrected pipeline. §9 states what would be
needed to produce one and what it would likely show.

Two prior documents are consolidated here, with their measured numbers reproduced rather than
restated from memory: `docs/report.md` §§2–6 (the original dual-channel investigation) and
`src/sismokaos/detection/REPORT_event_noise_detector.md` (the rebuilt benchmark, 14 Aug). Where the two
disagree, the later one governs and the difference is marked.

---

## 1. Summary

A compact dual-branch network (2D CNN over a spectrogram; 1D bidirectional
LSTM + self-attention over the raw waveform) classifies 6 s, three-component, 100 Hz
seismogram windows as earthquake or noise. The corpus is 33,795 AFAD-catalogue events
recorded across 181 stations in Türkiye, on KOERI FDSN waveforms.

The headline is **ROC-AUC 0.9892 ± 0.0003** (three seeds) on the rebuilt benchmark, against a
**conditional amplitude floor of 0.9049**, and **0.9971** on a magnitude- and distance-matched
subset of STEAD — 1,155 stations across 96 networks — with no retraining and no fine-tuning.

The substantive contribution is not the accuracy number. It is that the benchmark on which
detectors of this kind are normally evaluated is **largely solvable by a single amplitude
statistic**, so the conventional majority-class comparison overstates a model's contribution
by roughly an order of magnitude. Three defects were found and corrected in the standard
construction (§4), and every number in this report is quoted against a conditional floor.

Two negative findings are as well-supported as the positive one:

- **The 1D waveform branch contributes nothing beyond amplitude.** With amplitude deleted by
  normalisation it scores *below* the amplitude scalar (−0.0020); with amplitude restored it
  scores *level* with it (+0.0003). What it learned was the scalar it had been denied.
- **Fusing the two branches degrades the model.** 0.9735 gated vs 0.9779 for the spectrogram
  CNN alone, a gap of 0.0044 against per-seed spreads of 0.0012 and 0.0005.

The best configuration in the entire investigation is the simplest one tested: the plain
spectrogram CNN, no LSTM branch, no auxiliary input, no fusion.

---

## 2. Data acquisition

### 2.1 Catalogue and waveforms

Events come from **AFAD's** event catalogue. (Both files below were long referred to in
this repo as KOERI's; they are not. 100% of their EventIDs are AFAD eventIDs, and their
magnitudes and coordinates match AFAD's API to the printed digit — see
`katalog_kusuru_raporu.md`. The **waveforms** are genuinely KOERI KO.* , requested from
the KOERI FDSN service; only the catalogue attribution was wrong.) Two catalogue files
serve different purposes, and conflating them would break the noise screening:

| File | Events | Use |
|---|---|---|
| `extracted_earthquakes.csv` | 93,690 | download list |
| `deprem_katalog_utc.csv` | 482,898 | noise-contamination screening only |

The screening catalogue is complete to much smaller magnitudes (median M 1.70; 32.9 % below
M 1.5). That matters: a noise window must be screened against everything the network
recorded, not against the subset large enough to be worth downloading.

Waveforms were requested from the KOERI FDSN service for all `HH*` channels (100 Hz,
high-gain broadband) within a **0.5° (~55 km)** search radius of each epicentre. Station
metadata lookups are cached on a ~1.1 km coordinate grid so co-located events share one query,
and all windows for an event are fetched in a single bulk request and sliced in memory.

- Event files retrieved: **33,795**
- Stations represented: **181** (networks KO ×156, 6G ×17, IJ ×8)

| Corpus property | Value |
|---|---|
| Magnitude — median (p5 / p95 / max) | 2.30 (2.00 / 3.40 / 7.70) |
| Fraction below M 2.5 | 63.5 % |
| Depth — median (p90) | 7.0 km (12.2 km) |
| Catalogue location RMS residual — median (p90) | 0.42 s (0.72 s) |
| Epicentral distance — median (p95 / max) | 38.6 km (53.5 / 55.6 km) |

The location RMS is recorded because it bounds the accuracy of the theoretical arrival times
the rebuilt benchmark depends on (§5.1).

### 2.2 The negative class

Noise is drawn from a separately downloaded interval **3 h 05 m to 3 h 00 m before** each
event origin (a 300 s slice; a second slice at −6 h also exists). Each candidate is screened
against the full 482,898-event catalogue and rejected if any catalogued event falls within
**±300 s**. The window is deliberately wide, because coda from a large event persists for
minutes. The check is purely temporal, so an event 500 km away also vetoes a candidate — over-
conservative, and it discards noise that is otherwise scarce.

Noise is nonetheless not the binding constraint: **1,784,650** extractable noise windows
against 35,836 event windows, roughly 50:1. The noise class is always subsampled, which is
precisely what makes hard-negative mining (§5.3) free.

> **This asymmetry is the origin of the benchmark's central problem.** Negatives are sampled
> from a deliberately quiet, catalogue-screened interval; positives are centred on an arrival.
> The two classes are therefore separated by amplitude *by construction*, before any model is
> involved. §4.1 quantifies exactly how much.

### 2.3 Cross-corpus set (STEAD)

STEAD supplies analyst-reviewed P and S picks in per-trace HDF5 attributes, so anchoring is
exact rather than estimated.

| | Value |
|---|---|
| Noise traces available | 235,426 |
| Event traces available | 200,000 |
| Stations / networks | 1,155 / 96 |
| Trace geometry | (6000, 3) at 100 Hz, component order **E, N, Z** |
| P pick provenance | `p_status = 'manual'` |

STEAD is not merely independent, it is **harder**, and the two must not be conflated when
reporting transfer:

| Property | This corpus | STEAD |
|---|---|---|
| Median magnitude | 2.30 | **1.09** |
| Fraction below M 1.0 | 0 % | **44.1 %** |
| Median epicentral distance | 38.6 km | 30.3 km |
| Maximum epicentral distance | 55.6 km | **329.4 km** |
| Fraction matching the training distribution | — | **7.1 %** |

STEAD is therefore reported twice — a **matched** subset (M ≥ 2.0, distance ≤ 56 km) and the
**full** range — plus a magnitude-stratified breakdown (§7.3).

Component order is reversed relative to this pipeline, which orders Z, N, E by role. It is
corrected on load, and the correction was verified three ways: the index-to-colour mapping in
the project's existing STEAD script, STEAD's own documentation, and measurement — across 800
noise traces, index 2 carries the lowest power, as a vertical component should.

---

## 3. Signal processing and dataset generation

### 3.1 The filter chain

All windows — both classes, both corpora — pass through one implementation,
`seismic_cli.core.clean_and_filter_1d`. The STEAD adapter imports it rather than
reimplementing it, so the two corpora cannot silently diverge.

Per component, in order:

1. Linear detrend
2. Constant (mean) detrend
3. Hann taper over the leading and trailing 5 % of samples
4. 4th-order Butterworth bandpass, **1–45 Hz**, applied with `filtfilt` — zero-phase, so the
   arrival is not group-delay shifted
5. Polyphase resampling to a nominal **100 Hz**

Component selection is by **role** (Z, then N or 1, then E or 2), not alphabetical channel
code, so a station with mixed sensor codes cannot contribute two horizontals and no vertical.

### 3.2 The two representations

Each window yields a paired sample `{seq, img}` written as a single `.pt` file:

| Tensor | Shape | Content |
|---|---|---|
| `seq` | (600, 3) | standardised three-component waveform, 6 s @ 100 Hz |
| `img` | (3, 129, 10) | log-power STFT, `n_fft = 256`, `hop = 64`, `top_db = 80` |

At 3 s the same STFT settings give `1 + 300//64 = 5` time frames rather than 10 — a point that
matters for the magnitude work and would matter for a 3 s detector (§9).

A legacy alternative for the 2D channel is the **RAM (relative angle matrix)** image, retained
because the source paper (Wang & Zhao 2025, 1D2D-EDL) uses it. Both encoders are supported by
the same training script; the loader is representation-agnostic and takes whatever the dataset
directory holds. `dataset_specdual_*` is spectrogram, `dataset_dual_*` is RAM.

### 3.3 Amplitude normalisation — the single most consequential setting

The two channels are normalised differently, and getting this wrong silently invalidates the
waveform branch.

- **`img` — station spectral normalisation.** Each station's median dB-per-frequency-bin
  noise profile (median over time frames, computed from that station's own noise recordings,
  requiring ≥ 60 s of usable data) is subtracted. The result is *decibels above that station's
  own noise floor*: instrument gain cancels, genuine amplitude-above-background survives.
- **`seq` — station amplitude baseline.** Each component is standardised against that
  station's long-term noise mean and standard deviation, `(x − μ_station) / σ_station`.

**Defect found and corrected.** The pipeline flag controlling `seq` standardisation defaults
to *off*, in which case `standardize()` falls back to the window's own mean and standard
deviation. Every sample is then forced to mean 0, standard deviation 1 — **deleting absolute
amplitude entirely**. Measured on the test split, the standard deviation of `seq` had ROC-AUC
**0.5000**, exactly chance, with a median of 1.000 in *both* classes, while a single scalar
derived from the same amplitude achieved 0.9404. The waveform branch was structurally denied
the dominant discriminant.

The fix is a single-variable change, and was verified as such:

| Test-set statistic → ROC-AUC | Per-window norm. | Station baseline |
|---|---|---|
| `seq` standard deviation | 0.5000 | **0.9440** |
| `seq` absolute maximum | 0.7088 | **0.9461** |
| `img` mean dB (control) | 0.9205 | 0.9208 |

The control moves by 0.0003 across a test set that is 99.86 % identical (9,535 of 9,548
files), which is what confirms the change was isolated. After correction `seq` carries a
physical quantity: noise windows sit at 0.6× their station's long-term noise floor, event
windows at 43.7×.

Baseline coverage: 531 (station, component) amplitude pairs across 177 stations. 4 of 152
event stations lack a usable noise baseline and fall back to per-window standardisation.

### 3.4 Why the RAM encoding fails at short windows

The RAM transform reshapes an *m*-sample window into *n* feature vectors of length
`d = ⌈m/n⌉` and takes pairwise angles. With *n* fixed at 64:

| Window | Samples *m* | *d* (samples/vector) | Segment duration | Zero-padding |
|---|---|---|---|---|
| 60 s | 6000 | 94 | 0.94 s | 0.3 % |
| 10 s | 1000 | 16 | 0.16 s | 2.3 % |
| **6 s** | **600** | **10** | **0.10 s** | **6.2 %** |
| **3 s** | **300** | **5** | **0.05 s** | **6.2 %** |

At 3 s each angle is between two **five-dimensional** vectors. Cosine similarity of short
random vectors has high variance, so the angles become sampling-noise dominated. This is a
geometric limitation of the encoding, independent of model capacity or dataset size. There is
no clean fix inside the design: raising *d* requires lowering *n*, and at 3 s reaching
60 s-equivalent segment lengths would require *n* = 3.

A second, independent reason RAM underperforms: **the transform is scale-invariant**, so it
cancels exactly the amplitude ratio that separates the classes. This was measured directly by
comparing self-standardised against baseline-standardised images at the same station:

| Event strength | σ_window / σ_noise | Mean pixel difference | Max |
|---|---|---|---|
| Weak (SNR ≈ 2) | 1.84 | 0.63 / 255 levels | 3 |
| Strong (SNR ≈ 20) | 19.19 | 0.27 / 255 levels | 1 |

The differences are sub-level and, decisively, **do not grow with the amplitude ratio** — a
20× stronger event produces a *smaller* image difference. The `--baseline` flag therefore
cannot deliver its intended effect on a RAM dataset: the amplitude information it preserves is
eliminated by the very next pipeline step. This also explains why STA/LTA was competitive
against early RAM models — its entire discriminative signal is amplitude against a long-term
baseline, precisely the quantity RAM structurally cannot represent.

### 3.5 Dataset-generation constraints

`seismic-cli generate-spec-dual-dataset` enforces five constraints, each of which was added
in response to a measured failure:

**Station-disjoint splits, unified across classes.** Each station is assigned to exactly one
of train/validation/test, and both its earthquake *and* noise windows follow that assignment.
About 97 % of earthquake stations also contribute noise, so allocating the classes
independently would let nearly every station appear in training under one label and in test
under the other.

**Per-window station caps** (`--max-windows-per-station`), enforced by giving each
(station, file) pair an evenly spaced quota rather than dropping whole files. Filenames retain
the original window index *w*, so the source sample range `[w·s, w·s + T)` stays recoverable
for baseline reconstruction even after subsampling.

**Maximum-size balanced mode** (`--max`). Assigns every usable station to the split with the
largest relative deficit against ratio-proportional targets, then balances classes per split
by trimming the surplus class via largest-remainder proportional rounding over per-file
quotas. Without it, generation stops once global targets are filled and silently discards
every remaining station — costly exactly where station diversity is already scarce.

**Gap rejection.** Traces are merged without interpolation fill so gaps stay masked; gaps are
then linearly filled for filtering while a boolean mask records which samples are synthetic.
Any window whose worst channel exceeds **5 %** synthetic samples is rejected.

**Per-station sampling rates.** Window sizing uses each station's own recorded sampling rate
from the manifest, not the first trace's rate in the file.

Output is a `manifest.csv` (`split, class_name, station_key, file_path, filename, fs`)
sufficient to reconstruct the exact source samples behind any encoded window — which is what
makes the like-for-like STA/LTA baseline in §4.1 possible.

---

## 4. Three defects in the conventional benchmark

### 4.1 The amplitude floor

Because negatives are curated-quiet and positives are arrival-centred, the classes differ by
roughly **19 dB** before any modelling. Single statistics, read directly off the stored
tensors with no learning, on the held-out test split (n = 9,548):

| Statistic | Noise (median) | Event (median) | ROC-AUC |
|---|---|---|---|
| `img` mean dB | −0.48 | 18.77 | 0.9205 |
| log SNR (window RMS / station noise RMS) | −1.138 | 2.009 | 0.9404 |
| `seq` absolute maximum | 1.535 | 44.379 | **0.9461** |
| Majority class | — | — | 0.5000 |

Baseline AUC is reported **oriented**, as `max(a, 1−a)`, since an anti-predictive rule is
equally exploitable. This is implemented as `safe_auc(..., oriented=True)` in
`sismokaos/metrics.py` and computed at run time by `trivial_amplitude_floor()` in
`detection/cnn_lstm_classify.py`, so the floor travels with every result rather than being
looked up.

A classical STA/LTA baseline was computed on **exactly the same windows**, reconstructed from
the source MiniSEED by file, station and window index:

| STA/LTA configuration | ROC-AUC |
|---|---|
| Auto-derived (STA 0.2 s, LTA 2.0 s) | 0.5091 |
| Anchoring-aware (STA 0.03 s, LTA 0.3 s) | **0.8193** |

The first figure is an artefact and must not be quoted — see §4.2.

> **Implication.** A model reported as 0.979 against a 0.500 floor appears to add 0.479.
> Against the strongest conditional floor it adds **0.032**.

### 4.2 The STA/LTA anchor never locates the P wave

Short windows were originally cut around an STA/LTA pick over the full 60 s buffer, with
STA = 1.0 s, LTA = 10.0 s, trigger-on 3.5, and 20 % of the window placed before the pick
(`pre_arrival_fraction = 0.2`). `classic_sta_lta` forces its characteristic function to
**exactly zero for its first `nlta` samples** — no long-term average exists yet — so at 100 Hz
**no trigger can be declared before t = 9.99 s**.

The download geometry places the true arrival far earlier. With a 0.5° radius, TauP (iasp91)
predicts a median P arrival **7.2 s** after origin, with **99.4 %** of arrivals before 10 s.

Measured over 250 sampled event files (290 picks):

- Picks before sample 999: **0**
- Picks at exactly sample 999, the first non-zero index: **48.3 %**

Window positions were then measured directly, by matching each extracted trace against its
60 s source sample-for-sample (**552 of 552 traces matched exactly**). File headers could not
be used, because the extraction replaces `trace.data` without updating `stats.starttime` — an
independent second defect:

| True window start, s after origin | Value |
|---|---|
| p5 / p25 | 8.79 / 8.79 |
| Median | 8.88 |
| Starting at exactly 8.79 s | **44.6 %** |
| **Excluding a 7.23 s P arrival** | **100 %** |

8.79 s is the warm-up boundary (9.99 s) minus the 1.2 s pre-arrival buffer.

**Consequence.** The extracted "earthquake" windows are S-wave and coda windows, not post-P
windows. At a median 38.6 km the S arrives around 12–13 s, inside the window, while the P
arrival precedes the window by ~1.6 s.

The same arithmetic breaks the *evaluation* baseline. `derive_sta_lta_params` computes
`LTA = min(10, T/3)` and `STA = max(0.05, LTA/10)`, giving 0.2 s / 2.0 s at 6 s and
0.1 s / 1.0 s at 3 s. But `LTA = T/3` exceeds the pre-arrival buffer `0.2·T` for **every** T,
so on any anchored window the arrival sits inside the forced-zero dead zone. Measured on one
representative window (Z channel, KO.KULA, event 696188): amplitude jumps from σ ≈ 1313 to
σ ≈ 13111 at sample 120 — a clean, strong onset — yet the characteristic function is exactly
0.0 through sample 199 and peaks at only 1.21 at sample 356. That is the difference between
AUC 0.5093 and 0.8194 on identical data.

The corrected parameters were selected on the **validation** split before any test number was
looked at (LTA ∈ {0.10 … 0.50} s; best 0.30 s at val AUC 0.8212), then evaluated once on test:
**AUC 0.8194**, accuracy 74.60 %, precision 73.90 %, recall 76.08 % at the oracle Youden's-J
threshold. The auto-derivation formula was left unchanged for un-anchored windows — 60 s
results depend on it reproducing 1.0/10.0 exactly — and now emits a runtime warning when the
derived LTA exceeds 15 % of the window.

### 4.3 The selection effect

Because a recording that never triggers is discarded, the positive class is by construction
*the subset a classical detector had already found*.

| Stage | Count | Retained |
|---|---|---|
| 60 s source event files | 33,795 | — |
| Anchored event files | 23,228 | 68.7 % |
| Station recordings within retained files (300-file sample) | 525 → 478 | 91.0 % |
| **Overall station-recording retention** | — | **62.6 %** |

About **37 % of event recordings never enter the dataset**, and they are disproportionately
the low-SNR ones on which a learned detector's value would actually be demonstrated.

---

## 5. The rebuilt benchmark

### 5.1 Catalogue-derived arrival anchoring

Arrivals are **predicted, not picked** (`seismic_cli/src/arrival_from_catalog.py`). For each
(event, station) pair, epicentral distance is computed from the catalogue hypocentre and the
station coordinates — retrieved once from the KOERI FDSN station service, all 181 stations
resolved — and the first-arriving P phase (`p`, `P`, `Pg`, `Pn`) is computed with TauP using
iasp91. Windows are cut **2.0 s before the predicted arrival**, 6.0 s long. **No trigger and
no threshold are applied**, so no recording is discarded for being quiet.

Travel times are cached on (depth rounded to 1 km, distance rounded to 0.005° ≈ 0.55 km ≈
0.09 s of travel time), far below the prediction error.

**Validation of the predicted arrivals**, against an independently recomputed STA/LTA pick
using an LTA short enough to actually see the arrival (STA 0.2 s / LTA 1.0 s, warm-up 1.0 s):

| Metric | Value |
|---|---|
| Median residual (pick − prediction) | **+0.84 s** |
| Median absolute deviation | **0.63 s** |
| Within ±2 s | 75.7 % |

The positive median is expected — a trigger lags a true onset. A second, independent check:
within the extracted windows, post-arrival RMS exceeds pre-arrival RMS in **96.8 %** of
vertical traces, median ratio **8×**.

| Retention stage | Count |
|---|---|
| Event files written | 32,868 (of 33,795) |
| Station recordings kept | **55,568** |
| Rejected: window outside buffer | 1,646 |
| Rejected: fewer than 3 channels | 477 |
| **Station-recording retention** | **96.3 %** (was 62.6 %) |

The 2.0 s pre-arrival buffer is chosen to exceed the prediction spread, so the onset stays
inside the window even when the prediction runs early. **This accuracy is adequate for
detection but not for onset-time regression** — the dataset must not be repurposed for phase
picking.

### 5.2 Splitting

Splits are **station-disjoint**: every station is assigned to exactly one split across both
classes. Assignment is seeded (`random.seed(42)` for the station shuffle,
`random.Random(123)` for per-station caps), so generation is reproducible. Target ratios are
0.70 / 0.15 / 0.15 by window count; the surplus class is then trimmed per split to restore
balance.

| Split | Stations | Windows per class |
|---|---|---|
| Train | 120 | 38,247 |
| Validation | 28 | 9,415 |
| Test | 35 | 7,906 |

Train ∩ Test stations = **∅**, verified.

**Verification that the split is not leaking instrument gain.** On the original benchmark
`img` mean dB achieved 0.9205 pooled. Computed *within* each test station and sample-weighted,
it achieves **0.9221**. The amplitude signal therefore survives inside individual stations and
is not a station fingerprint memorised across a leaky split.

### 5.3 Hard-negative mining

Since noise is ~50× more abundant than signal, *which* noise is used is a free choice. By
default it is sampled evenly across each file, which yields representative but easy negatives.

Mining ranks each candidate noise window by its loudest component, expressed in units of that
(station, component)'s own noise sigma, then draws the required count from the
**75th–99th percentile band**, spread evenly across it.

Two design decisions matter, and the first was found the hard way:

- **Ranking must be global, not per file.** A first implementation ranked within each
  (file, station) group and moved the floor only from 0.9535 to 0.9312. Within one 300 s file
  all ~99 windows share a station and an hour and are nearly equally loud; almost all
  amplitude variance is *between* stations and times. Ranking globally within each split —
  over 1,440,082 candidate training windows — yields negatives 2.64× / 2.43× / 2.79× louder
  than the pool median for train / validation / test.
- **The p99 upper bound is deliberate.** The loudest tail of a screened noise archive is
  exactly where an earthquake the catalogue missed would hide; mining it would inject
  positives into the negative class.

Splits, events and station assignment are **byte-identical** to the unmined dataset; only
which noise windows are kept differs. That makes the two a controlled pair, which §7.2
exploits.

**Resulting conditional floors:**

| Benchmark | `seq` abs-max | `img` mean dB | Floor |
|---|---|---|---|
| Original (STA/LTA gated) | 0.9461 | 0.9208 | 0.9461 |
| Catalogue-anchored, random noise | 0.9535 | 0.8613 | 0.9535 |
| Catalogue-anchored + hard negatives | **0.9049** | **0.7571** | **0.9049** |

Note that removing the selection gate alone did **not** lower the floor — it made the positives
harder while the negatives stayed curated-quiet. Only mining the negatives moved it.

---

## 6. Model architecture

Source: `sismokaos/model/dual_channel.py` (`DualChannelTrunk`, `DualChannelNet`) and
`sismokaos/model/blocks.py` (`CNNBranch`, `LSTMAttentionBranch`, `GatedFusion`). The binary
detector is `DualChannelBinaryNet` in `detection/cnn_lstm_classify.py`.

### 6.1 The 2D branch — `CNNBranch`

Three convolutional stages over the (3, 129, 10) spectrogram, base width 32:

```
Conv2d(3,   32,  k3, p1, bias=False) → BatchNorm2d → GELU
Conv2d(32,  64,  k3, s2, p1, bias=False) → BatchNorm2d → GELU → Dropout2d
Conv2d(64,  128, k3, s2, p1, bias=False) → BatchNorm2d → GELU
AdaptiveAvgPool2d(1) → flatten                              → 128-d
```

Deliberately compact. The images are small, so a four-stage SE-ResNet — which exists in this
repo as `model/trunk2d.py` and is used by the image-only classifiers — is heavily
over-provisioned here. §6.5 of the original report confirms this empirically: a 4×-capacity
variant scored *lower* on validation than the default.

### 6.2 The 1D branch — `LSTMAttentionBranch`

Over the (600, 3) standardised waveform:

```
LSTM(input 3, hidden 48, 1 layer, bidirectional)    → 96-d per timestep
MultiheadAttention(embed 96, 4 heads)
LayerNorm(h + attn)                                  residual, transformer-style
mean over time                                       → 96-d
```

This is the paper's Sec. 3.3.1 design — self-attention over the *raw* waveform, not over a
reshaped version of the 2D image. An earlier implementation made the latter mistake; the
correction is documented in `seismic_cli/ram_dual.py`.

### 6.3 Fusion

Each active branch is projected to a common width (96) by a linear layer. Two mechanisms:

- **`linear`** (the paper's default): `F = a·F₁ + b·F₂`, two learned global scalars.
- **`gate`**: `F = g(x)·F₁ + (1−g(x))·F₂` with a per-example gate
  `g = sigmoid(MLP([F₁, F₂]))`, conditioned on both branches' features for *this* example.

Single-branch ablations bypass fusion; the scalars remain as a harmless global rescale that
the optimiser settles near 1.

`--channels` selects the active set: `all`, `1d`, `2d`, `aux`, `1d+aux`, `2d+aux`. The `aux`
branch is a small vector of amplitude scalars concatenated *after* fusion, added specifically
to restore what the RAM transform discards (§3.4).

### 6.4 Head

```
LayerNorm → Dropout → Linear(→96) → GELU → Dropout → Linear(96 → 1)
```

A single logit, trained with `BCEWithLogitsLoss`.

### 6.5 Parameter counts

| Configuration | Parameters | Params / training sample |
|---|---|---|
| 2D only | 115,459 | 1.5 |
| 1D only | 76,707 | 1.5 |
| Both, gated fusion | 191,874 | 3.8 |

At roughly 1.5 parameters per training sample the model is not capacity-limited, which is one
reason the hyperparameter sweeps in §8.2 found nothing.

### 6.6 Training

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss`, label smoothing 0 → 0.1, 1 → 0.9 |
| Diagnostic | unsmoothed BCE logged alongside, so the loss stays comparable across runs |
| Optimiser | AdamW, lr 2 × 10⁻⁴, weight decay 3 × 10⁻² |
| Schedule | cosine annealing over max epochs |
| Gradient clipping | max-norm 1.0 |
| Batch size | 32 (headline runs) / 64 (script default) |
| Dropout | 0.4 |
| Max epochs | 80 |
| Early stopping | validation ROC-AUC flat for 10 epochs |
| Checkpoint selection | best validation ROC-AUC |
| Mixed precision | enabled on CUDA |
| Seeds | 42, 43, 44; probabilities averaged for the ensemble |
| Decision threshold | fixed 0.50 for all thresholded metrics |

Every configuration is trained with three seeds, and **per-seed spread is treated as the
primary reliability statistic**, reported alongside the probability-averaged ensemble.

---

## 7. Results

All figures are ROC-AUC. "Edge" is ensemble AUC minus the strongest conditional floor on that
benchmark (§5.3).

### 7.1 In-domain, rebuilt benchmark

| Configuration | Benchmark | Per-seed | Mean | Std | Floor | Edge |
|---|---|---|---|---|---|---|
| **2D** | **hard negatives** | 0.9892 / 0.9890 / 0.9893 | **0.9892** | 0.0001 | 0.9049 | **+0.0847** |
| 2D | catalogue, random noise | 0.9884 / 0.9880 / 0.9878 | 0.9881 | 0.0002 | 0.9535 | +0.0350 |
| 2D | original (gated) | 0.9783 / 0.9782 / 0.9773 | 0.9779 | 0.0005 | 0.9461 | +0.0318 |
| Gate fusion | original (gated) | 0.9746 / 0.9741 / 0.9718 | 0.9735 | 0.0012 | 0.9461 | +0.0274 |
| 1D | original, amplitude restored | 0.9470 / 0.9432 / 0.9428 | 0.9443 | 0.0019 | 0.9461 | **+0.0003** |
| 1D | original, per-window norm. | 0.9173 / 0.9133 / 0.9127 | 0.9144 | 0.0021 | 0.9205 | **−0.0020** |

Ensemble metrics for the best configuration (2D, hard negatives, n = 15,812): **accuracy
0.9679, MCC 0.9369, PR-AUC 0.9921, ROC-AUC 0.9896.**

Three findings:

- **The 1D branch contributes nothing beyond amplitude.** Denied amplitude it scores below the
  amplitude floor; given amplitude it scores level with it. What it learned *is* the scalar.
- **Fusion measurably degrades the model** — 0.9735 vs 0.9779, against per-seed spreads of
  0.0012 and 0.0005. Adding a branch that only encodes amplitude to a branch that already
  encodes it adds variance, not information.
- **Seed stability improves markedly on the rebuilt benchmarks** (std 0.0001–0.0002 vs
  0.0005–0.0021), which is consistent with a better-posed task rather than a noisier one.

### 7.2 Transfer between noise regimes

Both datasets share identical events, splits and station assignment; only noise selection
differs. This isolates the effect of hard negatives cleanly.

| Trained on ↓ / Evaluated on → | Random noise (floor 0.9535) | Hard negatives (floor 0.9049) |
|---|---|---|
| Random noise | 0.9885 (+0.0350) | **0.9841 (+0.0792)** |
| Hard negatives | 0.9873 (+0.0338) | **0.9896 (+0.0847)** |

**A model trained only on randomly sampled noise attains 0.9841 on loud noise transients it
never saw**, where the amplitude scalar attains 0.9049 and spectrogram loudness 0.7571.
Training on hard negatives adds a further 0.0055.

The discriminative capability is therefore largely present *without* hard-negative training.
The original benchmark simply could not resolve it, because its floor was too high for the
difference to appear.

### 7.3 Cross-corpus (STEAD) — no retraining, no fine-tuning

| Training data | Evaluation set | n | AUC | Floor | Edge |
|---|---|---|---|---|---|
| Gated (windows exclude P) | STEAD matched | 27,378 | 0.9818 | 0.9752 | +0.0066 |
| **Catalogue-anchored** | **STEAD matched** | 27,378 | **0.9971** | 0.9752 | **+0.0218** |
| Gated (windows exclude P) | STEAD full range | 50,000 | 0.9235 | 0.9531 | −0.0296 |
| **Catalogue-anchored** | **STEAD full range** | 50,000 | **0.9693** | 0.9531 | **+0.0162** |

Correcting the arrival anchoring **tripled** the matched cross-corpus edge, and turned a
below-floor result on the full range into an above-floor one. That is the strongest single
piece of evidence that §4.2 was a real defect and not a bookkeeping detail.

Magnitude-stratified (catalogue-anchored model, full STEAD; each band scored against the
complete noise set, so the negative class does not change between rows):

| Magnitude | n events | AUC |
|---|---|---|
| < 1.0 | 11,029 | 0.9482 |
| 1.0 – 1.5 | 6,038 | 0.9747 |
| 1.5 – 2.0 | 3,752 | 0.9922 |
| 2.0 – 2.5 | 2,235 | 0.9964 |
| 2.5 – 3.0 | 871 | 0.9968 |
| ≥ 3.0 | 1,074 | 0.9972 |

Performance degrades monotonically into magnitudes entirely absent from training — the corpus
begins at M 2.0 — which is the expected and interpretable direction.

> **Caveat on thresholded metrics.** STEAD noise sits ~2× higher on the amplitude scale than
> this corpus's noise (median `seq` std 0.98 vs 0.47), an artefact of how each corpus's noise
> baseline is estimated. Ranking *within* STEAD is unaffected, so ROC-AUC and PR-AUC transfer;
> accuracy, MCC and Brier score at the training threshold do **not**, and require
> recalibration on a held-out STEAD split. Only AUC should be quoted cross-corpus.

---

## 8. The earlier investigation (original benchmark)

These results predate the rebuilt benchmark. They are retained because the *comparative*
findings — RAM vs spectrogram, fusion mechanisms, the amplitude auxiliary — have not been
re-run and remain the only evidence on those questions. All are single-seed unless noted, and
on this dataset the single-seed noise band is about **0.01 AUC / 0.02 MCC** (§8.3).

Common dataset: 6 s arrival-anchored windows, `--max`, 71,672 windows (35,836 per class,
balanced), station-disjoint (82/30/40 earthquake and 104/35/38 noise stations across
train/val/test).

### 8.1 Headline comparison

| Model | Params | Test AUC | MCC | Accuracy |
|---|---|---|---|---|
| **Spectrogram CNN only (`2d`)** | 115,459 | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram-dual, fused linear (`all`) | 182,563 | 0.9646 | 0.8122 | 90.61 % |
| RAM-dual + aux, fused linear (`all`) | 182,759 | 0.9514 | 0.7790 | 88.95 % |
| RAM-dual + aux, no LSTM (`2d+aux`) | 115,655 | 0.9468 | 0.7775 | 88.84 % |
| RAM + aux, no dual architecture | 309,777 | 0.9230 | 0.7018 | 84.79 % |
| RAM-dual, raw waveform only (`1d`) | 76,707 | 0.9216 | 0.6849 | 84.22 % |
| RAM-dual, fused linear, no aux (`all`) | 182,563 | 0.9144 | 0.6042 | 79.57 % |
| RAM-dual, RAM image only (`2d`) | 115,459 | 0.8408 | 0.5288 | 76.42 % |
| RAM CNN only, no aux | 309,713 | 0.8356 | 0.5339 | 76.70 % |
| **STA/LTA, correctly parameterised** | — | **0.8194** | — | 74.60 % |
| STA/LTA, library default (broken) | — | 0.5093 | — | 56.88 % |

**Every CNN configuration tested beats a correctly-parameterised STA/LTA, including the
weakest** (0.8356 vs 0.8194). The margin is 0.16 AUC against the best configuration.

### 8.2 The amplitude auxiliary input

On the **RAM** classifier this is the single largest effect measured anywhere in the
investigation. Architecture-matched, single-variable (`RamAuxCNN`, differing only in whether
`aux` is concatenated):

- AUC 0.8356 → **0.9230** (+0.0874)
- MCC 0.5339 → **0.7018** (+0.1679)
- Accuracy 76.70 % → **84.79 %** (+8.09 pp)

On the **1D waveform branch** the same fix helps for the same underlying reason — the raw
waveform is standardised before entering the LSTM, which removes absolute amplitude exactly as
RAM's internal standardisation does:

- AUC 0.9216 → **0.9501** (+0.0285), MCC 0.6849 → 0.7675, accuracy 84.22 % → 88.37 %

On the **spectrogram** 2D branch it does **not** repeat, and mildly hurts:

| Configuration | AUC | MCC | Accuracy |
|---|---|---|---|
| Spectrogram 2D only, no aux | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram + aux, no LSTM (`2d+aux`) | 0.9749 | 0.8626 | 93.02 % |

A station-normalised spectrogram already encodes amplitude as a function of time *and*
frequency; appending two collapsed scalars adds estimation noise, not information. This is
also why the amplitude-corrected RAM branch (0.9468) still falls well short of the plain
spectrogram (0.9793): the correction restores what RAM discarded, but two scalars are a far
thinner representation than amplitude-as-a-function-of-time-and-frequency.

**Hyperparameter sweeps found nothing.** Six configurations of the RAM+aux classifier fell
within a validation-AUC band of 0.9268–0.9307 (spread 0.0039); five configurations of
`LSTMAttentionBranch` (depth, heads, hidden width) within 0.9565–0.9588 (spread 0.0023). In
both sweeps the nominal winner beat the default by less than the noise floor, and in the LSTM
sweep the default had the **best** test MCC and accuracy of all five despite ranking fourth of
five on validation. Selection was fixed to validation before training in both cases. Neither
sweep found a productive direction; the defaults should be retained, which locates any
remaining gap in feature representation rather than architecture search.

### 8.3 Fusion mechanisms, and the seed-repeated correction

**Linear fusion underperforms the best single branch on both 2D representations.** With RAM,
the 1D branch alone (0.9216) beats the fused model (0.9144); with a spectrogram, the 2D branch
alone (0.9793) beats the fused model (0.9646). A fixed pair of scalars trained jointly cannot
suppress a weaker branch on the specific examples where it is wrong, and joint training lets a
noisy branch degrade the stronger branch's own representation.

**Late-fusion stacking recovers what joint fusion lost.** Freezing both checkpoints and
fitting a two-input logistic regression on their logits:

| Base checkpoints | 1d alone | 2d alone | Naive logit average | Stacked |
|---|---|---|---|---|
| RAM-dual | 0.9229 / 0.688 | 0.8408 / 0.530 | 0.9136 / 0.692 | 0.9203 / 0.688 / 84.31 % |
| Spectrogram-dual | 0.9229 / 0.688 | 0.9793 / 0.867 | 0.9697 / 0.866 | 0.9743 / 0.871 / 93.54 % |

This confirms the fusion problem originates in **joint training**, not in combining the two
branches per se.

**Two close-margin claims were re-run at three seeds, and one of them did not survive.**

Gated vs linear fusion (spectrogram-dual, no aux):

| Seed | Linear (AUC/MCC/Acc) | Gated (AUC/MCC/Acc) | Δ AUC |
|---|---|---|---|
| 42 | 0.9646 / 0.812 / 90.61 % | 0.9761 / 0.850 / 92.51 % | +0.0115 |
| 1 | 0.9719 / 0.848 / 92.32 % | 0.9753 / 0.851 / 92.53 % | +0.0034 |
| 2 | 0.9746 / 0.834 / 91.68 % | 0.9720 / 0.849 / 92.36 % | **−0.0026** |

The AUC advantage reverses sign at seed 2. Accuracy and MCC are more consistent — gated wins
on both at all three seeds, with much lower spread (accuracy spread 0.17 pp vs linear's
1.71 pp). The corrected claim is *"gated fusion gives more consistent decisions at the
operating threshold, with a real but small and noisy ranking advantage"* — materially weaker
than the original single-seed framing.

Amplitude aux vs no aux (spectrogram-dual, linear fusion):

| Seed | No aux | With aux | Δ AUC |
|---|---|---|---|
| 42 | 0.9646 | 0.9733 | +0.0087 |
| 1 | 0.9719 | 0.9707 | **−0.0012** |
| 2 | 0.9746 | 0.9705 | **−0.0041** |

This does not merely shrink, it **reverses and averages to about zero** (+0.0011). The
seed-42 result was not representative. This does not undermine the amplitude fix generally —
the RAM-alone effect (+0.087) and the `1d+aux` effect (+0.029) are an order of magnitude above
this ~0.01 noise band — but it establishes that band, and **none of the other close-margin
claims in §8 has been checked against more than one seed.**

---

## 9. On 3-second windows

There is no 3 s detection result under the corrected pipeline. What is known:

1. **The pipeline supports it directly.** `--window-seconds 3` flows through anchoring,
   generation and the STA/LTA baseline; `derive_sta_lta_params` yields 0.1 s / 1.0 s at 3 s
   (and is subject to the same warm-up defect as at 6 s — pass explicit
   `--sta-seconds 0.03 --lta-seconds 0.3`-style overrides, per §4.2).
2. **The spectrogram would halve in time resolution**: `1 + 300//64 = 5` frames instead of 10
   at the default `hop = 64`. On the magnitude task, halving the hop to recover 10 frames from
   a 3 s window did **not** recover the performance of a genuine 6 s window — 6 s beat 3 s by
   0.014 MAE at matched frame count. Finer sampling of a shorter window is not the same as a
   longer window.
3. **The RAM branch would be structurally worse** (§3.4): five-dimensional feature vectors,
   noise-dominated angles. If a 3 s detector is built, use spectrograms.
4. **The 2.0 s pre-arrival buffer would consume two-thirds of the window.** The buffer exists
   because the TauP prediction has 0.63 s MAD; shrinking it to keep more post-arrival signal
   trades directly against the risk of cutting the onset out. This is the design decision a
   3 s detector turns on, and it has not been measured.

Expected outcome, stated as a prediction and not a result: a 3 s spectrogram CNN should retain
most of the 6 s performance on the M ≥ 2.0 in-corpus range, where the SNR is high, and lose
disproportionately on the low-magnitude STEAD bands where the arrival is emergent and 6 s of
context is doing real work. That prediction is worth one run to test; it is not evidence.

---

## 10. Threats to validity

These must be stated in any manuscript, or resolved first.

1. **Residual label noise in the positive class.** Removing the trigger gate admits recordings
   where the event may be below the station's noise. The catalogue asserts that an earthquake
   occurred, not that *this station* recorded it. Measured on the extracted windows, 96.8 %
   show an energy increase at the predicted arrival and 87.1 % show a ratio above 2×, so
   roughly **10–15 % of positives are marginal**. The achievable ceiling is below 1.0.
2. **Arrival accuracy is adequate for detection only** (0.63 s MAD). Not suitable for
   onset-time regression or phase picking.
3. **Hard-negative benchmarks are deliberately unrepresentative** of deployment noise.
   Calibrated or absolute operating-point numbers must come from the randomly sampled test set.
4. **Narrow distance range.** The 0.5° download radius caps epicentral distance at ~56 km. The
   low-SNR, emergent-arrival regime that most differentiates detectors lies beyond it.
5. **No comparison against modern learned detectors.** PhaseNet, EQTransformer and GPD have
   not been run on this benchmark. Until they are, no claim of competitiveness is supported.
   SeisBench is the intended vehicle.
6. **Single region, single catalogue** for training.
7. **Uncalibrated probabilities.** No temperature scaling has been fitted.
8. **§8's comparative findings are single-seed** except where noted, on a dataset whose
   measured single-seed noise band is ~0.01 AUC.

### 10.1 Retracted figures — do not cite

| Retracted | Cause |
|---|---|
| 1D branch per-seed 0.9173 / 0.7452 | Concurrent runs shared checkpoint filenames differing only by seed; each reloaded the other's weights. Corrected: 0.9173 / 0.9133 / 0.9127. |
| Ensemble 0.9108 with a 0.2480 seed | Same cause — an anti-predictive seed was averaged into the ensemble. |
| §8.2's claim that aux improves the fused linear spectrogram model | Did not survive seed repetition; reverses sign and averages to ~0. |
| Any 60 s RAM figure (89.61 % acc / STA/LTA AUC 0.7777) | Predates fixes to cross-class station leakage, the station cap, origin-anchored short windows, and the STA/LTA DC-offset defect. Not re-run. |

Checkpoint filenames now encode configuration, dataset identity and process ID
(`best_cnnlstm_classify_{channels}_{fusion}_{dataset}_pid{PID}_seed{seed}.pth`), making
collision impossible even for two identical commands launched at once, and a below-chance seed
now halts interpretation explicitly rather than being silently averaged in. The training
script computes the conditional amplitude floor from the test tensors at run time and reports
the edge against it, rather than against the majority class.

---

## 11. Reproduction

| Step | Command / script |
|---|---|
| Catalogue-anchored windows | `seismic_cli/src/arrival_from_catalog.py` |
| Dataset (random noise) | `seismic-cli generate-spec-dual-dataset --window-seconds 6 --fs 100 --max --baseline` |
| Dataset (hard negatives) | as above, plus `--hard-negatives --hard-negative-band 0.75 0.99` |
| STEAD dataset | `seismic_cli/src/stead_anchor_dataset.py --pre-arrival-seconds 2.0` (add `--min-magnitude 2.0 --max-distance-km 56` for the matched subset) |
| Training (headline) | `python src/sismokaos/detection/cnn_lstm_classify.py --dataset-dir dataset_specdual_6s --channels 2d --batch-size 32 --ensemble-seeds 42,43,44` |
| Gated fusion variant | `... --channels all --fusion gate` |
| Cross-corpus evaluation | `python src/sismokaos/detection/evaluate_cross_corpus.py` |
| STA/LTA baseline | `seismic-cli eval-sta-lta --sta-seconds 0.03 --lta-seconds 0.3` |

Window geometry: 6.0 s at 100 Hz, 2.0 s pre-arrival, bandpass 1–45 Hz. Generation is
deterministic given the same inputs and flags. Prefer `--random-seeds N` over fixed
`--ensemble-seeds 42,43,44` when the question is run-to-run variance: fixed seeds sample it
exactly once and then hide it.

---

## 12. What to say about this work in one paragraph

Conventional short-window detection benchmarks are largely solvable by a single amplitude
statistic (ROC-AUC 0.9461 here), so comparison against a majority-class baseline overstates a
model's contribution by roughly an order of magnitude. STA/LTA-based window anchoring cannot
locate the P arrival when the LTA exceeds the arrival time — 100 % of the resulting windows
excluded it — and the induced selection effect discarded 37.4 % of event recordings.
Catalogue-derived anchoring raises retention to 96.3 % and, on its own, tripled the
cross-corpus edge on STEAD. Globally mined hard negatives lower the conditional floor from
0.9535 to 0.9049. On the rebuilt benchmark a 115k-parameter spectrogram CNN attains
0.9892 ± 0.0003 against that floor, and 0.9971 on matched STEAD without retraining. The
LSTM/attention branch contributes nothing beyond amplitude (+0.0003 over the amplitude
scalar), and fusing it into the spectrogram CNN degrades performance by 0.0044.
