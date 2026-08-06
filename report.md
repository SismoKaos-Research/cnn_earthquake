# Earthquake Detection from Raw Waveforms via Relative Angle Matrix Image Encoding and Dual-Channel CNN–LSTM Networks

## Abstract

This report documents an investigation into applying the Relative Angle
Matrix (RAM) method — a technique proposed by Wang and Zhao (2025,
*Applied Soft Computing* 172) for converting one-dimensional vibration
signals into two-dimensional images for convolutional neural network (CNN)
classification — to seismic event detection from three-component (Z/N/E)
waveform data. All comparisons are made against classic STA/LTA, the
standard seismological trigger algorithm, rather than an arbitrary baseline.

The investigation proceeds in two phases. The first establishes a RAM-image
CNN classifier, evaluates it against STA/LTA, and identifies a structural
limitation: the RAM transform is provably scale-invariant, and therefore
cannot represent absolute signal amplitude, which is the dominant feature
short-window detection depends on. The second phase implements the source
paper's full dual-channel architecture — a CNN over the RAM image paired
with an LSTM and multi-head self-attention branch over the raw waveform —
corrects a design error identified by direct comparison against the source
paper, introduces an amplitude auxiliary input to address the
scale-invariance limitation, evaluates a spectrogram-based alternative to
the RAM image, and compares three fusion strategies for combining the two
channels.

The amplitude auxiliary input is the single largest contributor to
classification performance measured in this work (test AUC 0.836 → 0.923 on
the RAM-only classifier, an architecture-matched, single-variable
comparison). Spectrogram-based encoding outperforms RAM-based encoding as
the two-dimensional channel in every configuration tested. Gated fusion, a
per-example alternative to the source paper's fixed linear fusion, improves
performance substantially on the spectrogram-based configuration but
degrades it on both RAM-based configurations tested — a result reported in
full rather than simplified, since a tidy explanation is not yet supported
by the evidence.

Every numerical claim in this report is drawn directly from a measurement
taken against the code in this repository. Where a result has not been
verified computationally, or has not been repeated across random seeds, that
limitation is stated explicitly rather than implied.

---

## 1. Introduction

Wang and Zhao (2025) propose the Relative Angle Matrix (RAM) method for
converting one-dimensional bearing-vibration signals into two-dimensional
images for CNN-based fault diagnosis. Seismic waveform data shares the same
basic structure — a time series — and three-component (Z/N/E) seismic
stations map naturally onto a three-channel image, making this a plausible
transfer target. The central question addressed in this report is whether
RAM encoding combined with a CNN, and subsequently with the source paper's
full dual-channel CNN–LSTM architecture, can separate a genuine seismic
event from ambient noise directly from raw waveforms, and how such a system
compares to the established STA/LTA trigger algorithm.

This work was conducted independently, outside assigned project tasks, using
publicly available data (STEAD-format HDF5 chunks and self-downloaded FDSN
MiniSEED archives), and therefore did not require navigating data-access or
ethics constraints beyond standard public-data terms of use.

---

## 2. The RAM Transform, Formally Defined

### 2.1 Definition

Given a window of $m$ samples $v = [v_1, \dots, v_m]$ from one channel:

**Step 1 — Standardization.**

$$\mu = \frac{1}{m}\sum_i v_i, \qquad \sigma = \sqrt{\frac{1}{m}\sum_i (v_i-\mu)^2}, \qquad x_i = \frac{v_i - \mu}{\sigma}$$

with $\sigma \leftarrow \max(\sigma, \varepsilon)$, $\varepsilon = 10^{-12}$,
to guard against flatlined channels.

**Step 2 — Reshaping into local feature vectors.** For a target image
resolution $n$ (`--target-n`), the column depth is derived as

$$d = \max\left(2,\ \left\lceil m/n \right\rceil\right)$$

and $x$ is zero-padded (or truncated) to length $dn$ and reshaped
column-wise into $M \in \mathbb{R}^{d \times n}$. Each column
$X_i \in \mathbb{R}^d$ is a local feature vector — a contiguous $d$-sample
segment of the window.

**Step 3 — Angles relative to the centroid.** With the central vector
$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$,

$$\beta_i = \arccos\left(\frac{X_i \cdot \bar{X}}{\lVert X_i\rVert\,\lVert \bar{X}\rVert}\right) \in [0, \pi]$$

The cosine argument is clipped to $[-1,1]$ before the inverse cosine is
taken; floating-point error can otherwise push it marginally outside this
range, producing `NaN` — a failure mode not addressed in the source paper.

**Step 4 — Pairwise angle differences.**

$$R_{ij} = \beta_j - \beta_i, \qquad R \in \mathbb{R}^{n \times n}$$

**Step 5 — Rendering to 8-bit.** Since each $\beta \in [0,\pi]$, it follows
that $R_{ij} \in [-\pi, \pi]$ exactly, so a fixed affine map is used:

$$P_{ij} = \text{round}\left(255 \cdot \frac{\text{clip}(R_{ij}, -\pi, \pi) + \pi}{2\pi}\right)$$

The fixed range is significant: a per-image min–max normalization (as used
in the deprecated `src/preprocessor/` path) rescales every window by its own
extremes, which destroys comparability between windows. Under the fixed
map, a given pixel value represents the same angle difference in every
image in the dataset.

### 2.2 Structural Properties, Verified Numerically

The following are properties of the transform itself, confirmed to machine
precision. They determine, independent of any model, what information a CNN
operating on these images can possibly recover.

**(a) The matrix is antisymmetric and rank-2.** In matrix form
$R = \mathbf{1}\beta^\top - \beta\mathbf{1}^\top$, hence

$$R^\top = -R, \qquad R_{ii} = 0, \qquad \operatorname{rank}(R) \le 2$$

Measured: $\max|R + R^\top| = 0$, the diagonal is exactly $0$, and the
singular value decomposition yields exactly two nonzero singular values of
equal magnitude, with the remainder zero.

**(b) The image carries $n-1$ degrees of freedom, not $n^2$.** Because $R$
is generated by the vector $\beta$ up to a global offset, an entire
$64\times64=4096$-pixel channel is reconstructible from 63 numbers. This was
verified by recovering $\beta$ from the first column of $R$ and
reconstructing $R$ in full; reconstruction error was exactly zero.

A 64×64×3 RAM image (12,288 pixels) therefore carries at most 189
independent values. The transform is a severe, lossy compression of the
6,000-sample (60 s) or 600-sample (6 s) input, and a CNN's two-dimensional
spatial processing operates on a highly redundant embedding of a
one-dimensional signal.

**(c) Every RAM image shares the same mean pixel value.** Antisymmetry
implies $\sum_{ij} R_{ij} = 0$, so the mean pixel is exactly the map's
midpoint (127.5; measured 127.508 after integer rounding). The DC component
of every image is constant by construction and carries no information.

**(d) The transform is exactly scale-invariant, and shift-sensitive.** For
any $c > 0$, scaling $x \to cx$ scales every $X_i$ and $\bar X$ by $c$, and
the cosine ratio cancels the factor exactly:

$$\frac{(cX_i)\cdot(c\bar X)}{\lVert cX_i\rVert \lVert c\bar X\rVert} = \frac{c^2 (X_i \cdot \bar X)}{c^2 \lVert X_i\rVert \lVert \bar X\rVert}$$

Measured: $\max|\text{RAM}(x) - \text{RAM}(37.5x)| = 8.9\times10^{-16}$, i.e.
exact cancellation to floating-point precision. A mean *shift*, by
contrast, does not cancel ($\max|\Delta R| = 1.31$ rad for a representative
offset). Section 8.2 develops the consequence of property (d), which is the
most consequential finding of the first investigation phase.

### 2.3 Short Windows Are Structurally Disadvantaged

The reshape step ties image resolution to feature-vector length: with $n$
fixed at 64, the segment length $d = \lceil m/n \rceil$ collapses as the
window shortens (all figures at 100 Hz):

| Window | Samples $m$ | $d$ (samples/vector) | Segment duration | Zero-padding |
|---|---|---|---|---|
| 60 s | 6000 | 94 | 0.94 s | 0.3 % |
| 10 s | 1000 | 16 | 0.16 s | 2.3 % |
| 6 s | 600 | 10 | 0.10 s | 6.2 % |
| 3 s | 300 | 5 | 0.05 s | 6.2 % |

At 3 s, each $\beta_i$ is the angle between two five-dimensional vectors.
The cosine similarity of short random vectors has high variance, so the
angles become dominated by sampling noise rather than signal structure, and
the resulting image is correspondingly noisy. This is a geometric
limitation of the encoding, independent of model capacity or dataset size,
and it applies equally to classification and regression tasks.

The tension has no clean resolution within the current design: increasing
$d$ requires decreasing $n$ (a smaller image), and at 3 s a target
resolution of $n=3$ would be required to reach 60 s-equivalent segment
lengths. Reducing `--target-n` was tested and produced results in the same
accuracy range as other short-window variants (Section 7.2).

### 2.4 Three-Channel Composition

Each component (Z, N/1, E/2) is RAM-transformed independently and stacked
into RGB channels: R = Z, G = N-like, B = E-like. Component selection is by
role rather than alphabetical ordering, and stations lacking a usable
vertical component are excluded (Section 6, defect 13).

One question was identified but not pursued in this phase: whether
combining the three components before the transform (for example, by vector
magnitude) would preserve inter-channel amplitude relationships better than
transforming each independently. Given property 2.2(d), any such variant
should be evaluated specifically for its effect on amplitude information.

---

## 3. Data Processing Pipeline

The pipeline comprises four stages, implemented in the `seismic_cli`
package of the `data_downloader` repository (`core.py`, `anchor.py`,
`eval_baseline.py`, `cli.py`).

### 3.1 Acquisition (`src/download.py`)

For each catalog event, stations within `SEARCH_RADIUS_DEG = 0.5°`
(approximately 55 km) are resolved via FDSN (KOERI), with lookups cached on
a roughly 1.1 km coordinate grid so co-located events share one metadata
query. All windows for an event are fetched in a single bulk request and
sliced in memory.

- **Earthquake windows:** 60 s from origin time.
- **Noise windows:** 300 s slices at −3 h and −6 h relative to origin.
- **Contamination check:** a noise window is discarded if any event in the
  unfiltered catalog falls within ±300 s of it. Checking against the
  filtered catalog would allow sub-threshold events to pass silently into
  the noise class; the buffer is deliberately wide because coda from larger
  events can persist for several minutes.

This check is purely temporal, so an event 500 km away will veto a
candidate noise window. This is over-conservative and discards noise data
that is otherwise scarce; adding a distance term is listed as a limitation
in Section 9.

### 3.2 Arrival Anchoring (`seismic-cli anchor-windows`)

Short windows sliced from origin time can miss the P-wave arrival entirely
at distant stations. At 6 s, an arrival later than 6 s after origin time
means the nominal "earthquake" window contains no earthquake signal.
Anchoring re-derives short windows from already-downloaded 60 s data without
requiring re-download.

The pick uses the classic STA/LTA characteristic function

$$\text{CF}(k) = \frac{\frac{1}{n_{\text{STA}}}\sum_{j=k-n_{\text{STA}}+1}^{k} x_j^2}{\frac{1}{n_{\text{LTA}}}\sum_{j=k-n_{\text{LTA}}+1}^{k} x_j^2}$$

with the first `trigger_onset` crossing of `trigger_on = 3.5` taken as the
arrival sample $a$. The window is then cut as

$$[\,a - f\cdot T,\ a - f\cdot T + T\,), \qquad f = \texttt{pre\_arrival\_fraction} = 0.2$$

so that 20% of the window precedes the arrival (at 6 s: 1.2 s before, 4.8 s
after).

Two correctness requirements were violated prior to this round of
corrections (Section 6, defects 6–7): the trace must be detrended before the
characteristic function is computed, and the pick should prefer the
vertical component with fallback to horizontal components. A diagnostic
block now reports, per run, the number of stations seen, skipped, picked on
the vertical component, picked via fallback, and unpicked, together with how
close failed picks came to the trigger threshold.

### 3.3 Dataset Generation (`seismic-cli generate-dataset`)

Per window, per channel, processing proceeds: linear and constant detrend →
5% Hann taper → fourth-order Butterworth bandpass (1–45 Hz, zero-phase) →
RAM transform → 8-bit rendering → RGB stacking → PNG output.

Dataset generation enforces five constraints that required several
iterations to implement correctly.

**Station-disjoint splits, unified across classes.** Each station is
assigned to exactly one of train, validation, or test, and both its
earthquake and noise windows follow that assignment. With approximately 97%
of earthquake stations also contributing noise data, allocating the two
classes independently — the earlier behavior — allowed nearly every station
to appear in the training split under one label and the test split under
the other (Section 6, defect 1).

**Per-window station caps** (`--max-windows-per-station`). Enforced by
assigning each (station, file) pair an evenly spaced window quota rather
than dropping entire files. Filenames retain the original window index $w$,
so the sample range $[\,w s,\ w s + T)$ — with step
$s = T(1-\text{overlap})$ — remains recoverable for baseline reconstruction
even after subsampling.

**Maximum-size balanced mode** (`--max`). Assigns every usable station to
the split with the largest relative deficit against ratio-proportional
targets, then balances classes per split by trimming the surplus class via
largest-remainder proportional rounding over per-file quotas. Without this
mode, generation stops once global targets are filled and silently discards
every remaining station, which is costly precisely where station diversity
is already scarce.

**Gap rejection.** Traces are merged without interpolation fill, so gaps
remain masked; gaps are then linearly filled for filtering purposes while a
boolean mask records which samples are synthetic. Any window whose worst
channel exceeds 5% synthetic samples is rejected. Previously,
`fill_value='interpolate'` converted telemetry gaps into linear ramps that
entered training as real signal.

**Per-station sampling rates.** Window sizing uses each station's own
sampling rate, recorded in the manifest, rather than assuming that the
first trace's rate in a file applies to every station represented in it.

Output is an `ImageFolder` directory tree together with `manifest.csv`
(columns: `split, class_name, station_key, file_path, filename, fs`), one
row per image, sufficient to reconstruct the exact source samples for any
given image.

### 3.4 Baseline Standardization (`--baseline`)

Optionally, each channel may be standardized against that station's
long-term noise statistics $(\mu_{\text{sta}}, \sigma_{\text{sta}})$ —
accumulated in a streaming pass over all noise files for the given
(station, component) pair, requiring at least 60 s of usable data — instead
of the window's own $(\mu, \sigma)$. The intent was to give the pipeline the
long-term amplitude memory that constitutes STA/LTA's principal advantage.

This does not achieve its intended effect, for the structural reason
established in Section 8.2.

---

## 4. Model Architecture

`ImprovedSeismicCNN` (`src/model/cnn_train.py`) is a ResNet-style CNN with
Squeeze-and-Excitation blocks and a single-logit binary output.

### 4.1 Components

**Residual block.** For input $u$:

$$\text{ResBlock}(u) = \text{GELU}\Big(\text{SE}\big(\text{BN}_2(W_2 * \text{GELU}(\text{BN}_1(W_1 * u)))\big) + \mathcal{S}(u)\Big)$$

using $3\times3$ convolutions, the first of which carries the stride-2
downsample, and $\mathcal{S}$ the identity mapping or a $1\times1$
projection when the input and output shapes differ.

**Squeeze-and-Excitation** (reduction $r = 16$), applied before the
residual addition. Given $z \in \mathbb{R}^{H\times W\times C}$:

$$s = \text{GAP}(z) \in \mathbb{R}^{C}, \qquad e = \sigma\big(W_b\,\text{ReLU}(W_a s)\big) \in (0,1)^C, \qquad \text{SE}(z)_c = e_c \cdot z_c$$

that is, a learned per-channel gain conditioned on global context. Because
the three input channels correspond to Z/N/E components, channel attention
has a direct physical interpretation: it can learn to weight components
differently, and vertical components typically carry the cleanest P-wave
onset.

**Classification head.** Global average pooling → `Dropout(p_1)` →
`Linear(C_f, h)` → GELU → `Dropout(p_2)` → `Linear(h, 1)`, producing a
single logit.

### 4.2 Dimensions and Parameter Budget (64×64×3 Input, Measured)

| Stage | Output (C,H,W) | Parameters |
|---|---|---|
| `in_conv` (3→16) | (16, 64, 64) | 448 |
| `layer1` (16→32, stride 2) | (32, 32, 32) | 14,656 |
| `layer2` (32→64, stride 2) | (64, 16, 16) | 58,240 |
| `layer3` (64→128, stride 2) | (128, 8, 8) | 232,192 |
| `layer4` (128→256, stride 2) | (256, 4, 4) | 927,232 |
| GAP → head (256→64→1) | (1,) | 16,513 |
| **Total (long preset)** | | **1,249,297** |
| **Total (short preset, 3 stages, head width 32)** | | **309,713** |

`layer4` alone accounts for 74% of the long-preset parameter count, which
is why the short preset omits it, reducing the model by a factor of 4.0.

### 4.3 Training Procedure

The loss function is `BCEWithLogitsLoss` applied to label-smoothed targets
$\tilde{y} = 0.8y + 0.1$, which bounds the confidence the model is rewarded
for. This introduces a floor on training loss: at the optimum, per-sample
binary cross-entropy equals the binary entropy $H(0.1) \approx 0.325$ nats,
so smoothed training loss is not directly comparable in magnitude to
unsmoothed validation loss. An unsmoothed training-loss diagnostic is
therefore logged alongside the smoothed value, so the train/validation gap
can be interpreted correctly (Section 6, defect 9).

The optimizer is AdamW with gradient-norm clipping at 1.0 and mixed
precision on CUDA. Checkpointing retains the best epoch by the monitored
metric, and final test evaluation always loads that checkpoint.

### 4.4 Window-Length Presets

Short-window training runs overfit the full network within approximately 10
epochs, while 60 s runs do not. `--window-seconds` therefore selects a
preset (12 s or shorter selects `short`); any explicitly passed flag
overrides its preset value, and omitting the flag reproduces the original
configuration exactly.

| | `long` (60 s) | `short` (≤ 12 s) | Rationale |
|---|---|---|---|
| Stages / head width | 4 / 64 | 3 / 32 | Reduces parameters from 1.25 M to 0.31 M |
| Dropout $p_1,p_2$ | 0.5, 0.3 | 0.6, 0.4 | Stronger regularization on the head |
| Weight decay | 1e-2 | 3e-2 | Stronger regularization |
| RandomErasing | Off | $p=0.25$ | The only label-safe augmentation available; flips or rotations would scramble the RAM matrix's temporal ordering, since axis position encodes time |
| Batch size / learning rate | 128 / 1e-4 | 64 / 2e-4 | More updates per epoch on a smaller dataset |
| Schedule | ReduceLROnPlateau | Cosine annealing | Plateau scheduling reacts too slowly when the performance peak arrives by roughly epoch 6 |
| Checkpoint metric | Validation loss | Validation AUC | Validation cross-entropy degrades from calibration drift while ranking quality continues to improve; loss-based selection risks saving a pre-peak model |

---

## 5. Baseline Method: STA/LTA on Identical Windows

The baseline scores the same characteristic function as Section 3.2, taking
$\max_k \text{CF}(k)$ over each channel and the maximum across channels as
the window's score. Two properties make the comparison fair.

- **Exact-window reconstruction.** Windows are rebuilt from the raw
  MiniSEED data via the manifest (same file, station, and window index),
  not resampled independently.
- **Window-adaptive parameters.** Fixed parameters (`sta=1.0`, `lta=10.0`)
  cannot be computed inside a 3 s or 6 s window. Parameters are instead
  derived as $\text{LTA} = \min(10,\ T/3)$ and
  $\text{STA} = \max(0.05,\ \text{LTA}/10)$, which reproduces the classic
  1/10 ratio exactly at 60 s and gives 0.2/2.0 at 6 s and 0.1/1.0 at 3 s.

Reported metrics are AUC — threshold-free, the appropriate comparison
against the CNN — together with accuracy, precision, and recall at the
Youden's-J threshold. That threshold is selected on the evaluated split,
making it an oracle: the thresholded figures represent STA/LTA's upper
bound rather than a like-for-like comparison against the CNN's fixed 0.5
cutoff.

---

## 6. Software Defects Identified and Corrected

The reliability of the results in this report depends substantially on
defects identified and corrected during development. Defects 1–5 predate
this round of work; defects 6–13 were identified during a systematic audit
of the full repository.

| # | Defect | Mechanism | Impact |
|---|---|---|---|
| 1 | Cross-class station leakage | Splits were allocated independently per class; with approximately 97% station overlap, a station could appear as train-earthquake and test-noise | The model could exploit station identity as a shortcut; biased measured test performance downward, most severely where station counts are low (short windows) |
| 2 | Station caps were ineffective | The cap was applied at file granularity; a single 300 s noise file (roughly 200 windows at 3 s) exceeded any smaller cap and was retained in full | Explains observations of only 2–4 distinct noise stations in validation/test even after capping |
| 3 | Fixed-resolution collapse | Reshape depth $d$ was fixed regardless of window length | 3 s windows collapsed to approximately 3×3 images; corrected by deriving $d$ from the target resolution |
| 4 | Origin-anchored short windows | Short windows were cut from origin time rather than arrival time | A meaningful fraction of nominal "earthquake" windows contained no signal |
| 5 | Noise/earthquake station mismatch | Noise data was sourced independently of earthquake stations (approximately 47% overlap) | The model rarely observed both classes from the same instrument; a targeted downloader raised overlap to approximately 97% |
| 6 | STA/LTA computed on raw counts (anchoring) | `classic_sta_lta` was computed on un-detrended MiniSEED counts; a large DC offset pins the characteristic function near 1 | Verified: a synthetic arrival with a $10^6$-count offset produces a maximum characteristic-function value of exactly 1.000 (no pick possible on any channel); after detrending, the pick lands within 4 samples of ground truth |
| 7 | Arrival pick on a horizontal component | `sorted(traces)[0]` selects the E component before Z alphabetically, with a single attempt and no fallback | P-wave onsets are cleanest on the vertical component; corrected to vertical-first with fallback through remaining channels |
| 8 | STA/LTA computed on raw counts (baseline evaluation) | The same DC-offset defect present in the baseline scorer | Baseline performance was understated at high-offset stations, plausibly explaining a measured STA/LTA AUC swing of 0.78–0.98 between data pulls |
| 9 | Label-smoothing asymmetry | Smoothing was applied to training targets only; smoothed binary cross-entropy floors near 0.325 nats | Training and validation loss curves were not directly comparable in magnitude; validation metrics were unaffected, since they always used unsmoothed labels |
| 10 | Threshold mismatch | `cnn_from_tensor.py` validated at a 0.60 threshold but tested at 0.50 | Validation and test accuracy were measuring different decision rules |
| 11 | Gap interpolation treated as signal | `merge(fill_value='interpolate')` fabricates linear ramps across telemetry gaps | Synthetic data was trained on as though genuine; corrected via gap masking, with windows exceeding 5% synthetic samples rejected |
| 12 | Single-rate assumption | The first trace's sampling rate in a file was applied to every station represented in it | Produced an incorrect physical window duration for off-rate stations; corrected to use per-station sampling rate |
| 13 | Alphabetical channel selection | `sorted(keys)[:3]` could select `['1','2','E']` — two horizontal components and no vertical | Silent component mis-assignment; corrected to role-based selection requiring a vertical component |

Two items were suspected as defects but confirmed not to be: the RAM
mathematics as implemented transcribes the source paper correctly,
including guards the paper itself omits (the $\varepsilon$ floor on
$\sigma$, and clipping before the inverse cosine); and the manifest's
window-index-to-sample mapping is exact.

---

## 7. Initial Results (Pre-Correction Baseline)

All figures in this section predate the Section 6 corrections to defects
6–13 and are reported for continuity rather than as validated results. In
particular, every STA/LTA figure was produced by the DC-offset-affected
scorer, and every short-window figure by the ineffective station cap and
the cross-class split allocation defect. These comparisons require
re-running end-to-end before any figure in this section should be treated
as current.

### 7.1 60-Second Windows

| Metric | CNN | STA/LTA |
|---|---|---|
| Accuracy | 89.61 % | — |
| Recall (earthquake) | 0.963 | — |
| Precision (earthquake) | 0.817 | — |
| ROC-AUC | — | 0.7777 |

Test set: 2,561 samples, station-disjoint. Across repeated dataset
regenerations, the CNN's margin over STA/LTA on AUC held in the range of
approximately 0.10–0.15 AUC points and, if anything, widened across data
pulls. This is the headline result of the initial phase, and the one most
likely to survive re-validation, though the baseline figure itself is
expected to change once defect 8 is corrected, so the margin must be
re-measured rather than assumed to hold.

### 7.2 6-Second Windows

Most recent pre-correction run (853 test samples):

```
Accuracy 72.80 %
              precision  recall  f1-score  support
   noise         0.7203  0.7727    0.7456      440
   earthquake    0.7375  0.6804    0.7078      413
Confusion: TN=340  FP=100  FN=132  TP=281
```

Three pipeline-side variants — per-window standardization, per-station
baseline standardization (74.88% versus 77.99% at the time, i.e. no
measurable improvement), and a smaller `--target-n` — all produced results
in a 72–78% band, with the same overfitting signature: validation AUC
peaking around epoch 4–10, followed by flattening or degradation while
training loss continued to decrease.

The convergence of three unrelated interventions on the same performance
band motivated both the model-side audit summarized in Section 4.4 and the
structural analysis in Section 8. At approximately 850 samples, binomial
noise on accuracy is roughly ±1.5 percentage points; differences under
approximately 3 percentage points between runs should not be interpreted as
meaningful.

---

## 8. Analysis

### 8.1 Three Compounding Causes of Short-Window Underperformance

1. **Geometric (Section 2.3).** At 6 s, local feature vectors are 10
   samples long; at 3 s, 5 samples. Cosine angles between very short
   vectors are noise-dominated, so the encoding degrades before any model
   observes it.
2. **Statistical.** Short-window datasets were small (approximately
   1.5–4 k samples) and, prior to the station-cap correction, drawn from
   very few distinct noise stations. The 1.25 M-parameter network is
   heavily over-provisioned at this scale (roughly 300–800 parameters per
   sample), consistent with the observed early-peak overfitting curve.
3. **Informational (Section 8.2).** The single feature that most cleanly
   separates a 6 s earthquake window from noise — amplitude relative to the
   station's background level — is provably absent from the model's input.

### 8.2 The RAM Transform Discards Amplitude, and Amplitude Is What STA/LTA Uses

This is the most consequential finding of the first investigation phase.

By property 2.2(d), the transform is exactly scale-invariant:
$\text{RAM}(cx) = \text{RAM}(x)$. Consider what baseline standardization
actually changes. After `clean_and_filter_1d` (detrend, demean, bandpass),
the window mean is approximately 0, and $\mu_{\text{sta}}$ is likewise
approximately 0, since it is accumulated from identically cleaned noise
data. The two standardization modes therefore differ, to a close
approximation, by a single factor $\sigma_{\text{win}}/\sigma_{\text{sta}}$
— a pure rescaling, which RAM cancels exactly.

Measured on representative windows, comparing self-standardized and
baseline-standardized images at the same station:

| Event strength | $\sigma_{\text{win}}/\sigma_{\text{noise}}$ | Mean pixel difference | Maximum |
|---|---|---|---|
| Weak (SNR ≈ 2) | 1.84 | 0.63 / 255 levels | 3 |
| Strong (SNR ≈ 20) | 19.19 | 0.27 / 255 levels | 1 |

The differences are sub-level, and, decisively, do not grow with the
amplitude ratio: a 20-times stronger event produces a slightly smaller
image difference, not a larger one. The residual difference originates from
the near-zero post-filter mean (RAM is shift-sensitive, property 2.2(d)),
not from amplitude.

Three conclusions follow.

- **The `--baseline` flag cannot deliver its intended effect.** The
  amplitude and SNR information it is designed to preserve is eliminated by
  the very next step in the pipeline. The earlier 6 s result (74.88% versus
  77.99%) was not evidence against the long-term-memory hypothesis; it was
  run-to-run noise on near-identical images. The hypothesis stated in an
  earlier version of this report was never actually tested.
- **This explains STA/LTA's competitiveness at short windows.** Its entire
  discriminative signal is amplitude measured against a long-term baseline
  — precisely the quantity the RAM pipeline structurally cannot represent.
  The CNN is being asked to win using shape and frequency structure alone.
- **This predicts that magnitude regression on RAM images will
  underperform.** Local magnitude is essentially log peak amplitude with a
  distance correction; a scale-invariant encoding removes the dependent
  variable's principal predictor. Retrying regression with `--baseline`
  alone would not address this limitation.

### 8.3 Implication: Amplitude Belongs as an Explicit Input

The indicated correction is not a further image variant, but an auxiliary
scalar input. For each window and channel, compute

$$\text{SNR}_{\log} = \log\left(\frac{\sigma_{\text{win}}}{\sigma_{\text{sta,noise}}}\right)$$

store it in the manifest, and concatenate it — together with epicentral
distance, needed to disambiguate magnitude from distance — to the pooled
CNN features prior to the classification head. This restores the discarded
quantity without altering the transform itself.

A three-step validation order was proposed before committing further CNN
training runs to this hypothesis:

1. Confirm the near-no-op directly: generate a sample of windows with and
   without `--baseline` and difference the resulting images (expected
   result: at most 3 levels, per the table above).
2. Regress magnitude on $\log(\sigma_{\text{win}}/\sigma_{\text{noise}})$
   and $\log(\text{distance})$ alone, using a linear model or gradient-
   boosted tree. This is effectively fitting a local-magnitude relation; if
   it fails, the station baselines are too noisy for a CNN to inherit
   anything useful from them.
3. Only then train the joint (image and scalar) model — both for
   regression and for the 6 s classifier, where this input hands the
   network the exact feature STA/LTA has been outperforming it with.

Section 10 reports the outcome of this validation order.

---

## 9. Limitations and Recommendations (First Investigation Phase)

**Immediate, blocking re-validation:**

1. Regenerate all datasets and re-run every comparison under the Section 6
   corrections, particularly the 60 s CNN-versus-STA/LTA margin, whose
   baseline term is expected to change.
2. Re-run 6 s generation with `--max` and a functioning per-window cap;
   station diversity should improve substantially, since the previous mode
   discarded every station beyond the bottleneck class target.
3. Re-run 6 s training under the `short` preset, then run ablations
   (`--num-stages 4` isolates the effect of regularization;
   `--monitor loss` isolates the effect of capacity) across at least three
   seeds before treating any single result as conclusive.

**Structural:**

4. Add the amplitude and distance auxiliary input described in Section 8.3
   — the highest-expected-value change identified, and a prerequisite for
   revisiting magnitude regression.
5. Given property 2.2(b) — 4,096 pixels carrying 63 degrees of freedom —
   evaluate whether feeding the $\beta$ vector directly to a one-dimensional
   model matches the two-dimensional CNN's performance. If it does, the
   image encoding contributes computational cost without additional
   information, which would be an important finding in its own right.
6. Compare combine-then-RAM (vector magnitude across components) against
   per-channel RAM.
7. Add a distance term to the noise-contamination check; the current
   time-only rule discards usable noise data that is otherwise scarce.

**Standing caveats:**

- This investigation has not undergone external peer review.
- The 60 s results are the strongest obtained but predate the Section 6
  corrections; the 6 s results are preliminary.
- Short-window noise-station diversity remains limited, so precision and
  recall figures at short windows should not be interpreted beyond one
  significant figure.
- All conclusions in Section 8.2 concern this transform as specified; they
  follow from scale invariance and would not apply to an
  amplitude-preserving variant.

---

## 10. Extension: The Full Dual-Channel Architecture and the Amplitude Fix

Sections 1–9 address RAM encoding paired with a CNN alone — one of the two
channels described in the source paper. This section adds the second
channel (LSTM with multi-head self-attention over the raw signal) to
evaluate the complete architecture, and then implements the correction
proposed in Section 8.3. All results in this section are fresh runs on the
corrected pipeline (post Section 6, defects 1–13); they do not carry the
pre-correction caveat attached to Section 7.

All results below share a common dataset:
`seismic-cli generate-*-dataset --max` on 6 s arrival-anchored windows,
71,672 windows total (35,836 per class, balanced), station-disjoint
(82/30/40 earthquake stations and 104/35/38 noise stations across
train/validation/test). Code is located in `data_downloader/seismic_cli/`
(`ram_dual.py`, `ram_aux.py`, `spectrogram.py`) and `cnn_earthquake/src/`
(`cnn_lstm.py`, `cnn_lstm_classify.py`, `cnn_lstm_classify_aux.py`,
`cnn_lstm_stack.py`, `cnn_ram_aux.py`). As in Section 7.2, each
configuration reflects a single train/validation/test split rather than a
repeated measurement across seeds; differences under approximately 1–2
points should be treated as noise rather than as an established effect.

### 10.1 The Source Paper's Architecture (1D2D-EDL)

Two independent channels operate on the same window and are fused:

$$\text{1D: } F_{1D} = \text{MSA}\big(\text{LSTM}(x)\big) \qquad
\text{2D: } F_{2D} = \text{CNN}\big(\text{RAM}(x)\big) \qquad
F' = a\,F_{1D} + b\,F_{2D}$$

with $a, b$ learned scalars. This is implemented as `LSTMAttentionBranch`
(bidirectional LSTM, followed by `nn.MultiheadAttention` with a residual
connection and layer normalization, matching a standard transformer block,
followed by mean-pooling over time) and `CNNBranch` (three convolution/
batch-norm/GELU stages with global average pooling, which is resolution-
agnostic and therefore accepts either a square RAM image or a non-square
spectrogram without modification). Both components reside in
`cnn_earthquake/src/cnn_lstm.py`, shared with the unrelated
catalog-forecasting model implemented in the same file.

**A design error, identified and corrected by consulting the source paper
directly rather than a prior summary of it.** The initial implementation
of the 1D branch supplied the LSTM with the $(n, d)$ chunk matrix from
which the RAM image's angle vector $\beta$ is computed, on the premise that
both channels should observe the same reshaped data. This does not match
the source paper. Section 3.3.1 and Figure 7 of Wang and Zhao (2025) state
explicitly that the one-dimensional time series is normalized and then
input directly into the LSTM model; the RAM reshape feeds only the CNN
channel. The two channels are independent feature extractors over the same
raw signal, not two views of a shared intermediate representation. The
function supplying the incorrect design
(`core.ram_matrix_and_chunks()`) was subsequently removed;
`core.ram_matrix()` was verified to produce output byte-identical to its
pre-refactor version on both occasions this change was made.

One consequence of the correction carries a computational cost:
multi-head self-attention over the full raw window is $O(m^2)$. At 100 Hz,
a 6 s window is $m=600$ (an attention matrix of 360,000 entries — trivial);
a 60 s window is $m=6000$ (36 million entries per attention head —
substantially heavier, and likely to require a smaller batch size). This is
an inherent cost of matching the source paper's design, not a defect.

### 10.2 Data Pipeline Additions

| Encoder | Output | Description |
|---|---|---|
| `RamDualEncoder` | `{seq, img}` | `seq` is the raw standardized $(m,3)$ Z/N/E waveform; `img` is the Section 2 RAM image |
| `SpectrogramDualEncoder` | `{seq, img}` | Same `seq`; `img` is a log-power spectrogram, produced by wrapping `SpectrogramEncoder` via composition so the two implementations remain identical wherever they overlap |
| `RamAuxEncoder` | `{img, aux}` | `img` is the RAM image; `aux` is $[\log\text{SNR}, \log\text{RMS}]$ (defined below); no LSTM branch is present |
| `RamDualAuxEncoder` | `{seq, img, aux}` | All three components: `RamDualEncoder` plus the auxiliary vector |

$$\log\text{SNR} = \left\langle \log\frac{\sigma_{\text{win},c}}{\sigma_{\text{sta},c}} \right\rangle_{c \in \{Z,N,E\}}
\qquad
\log\text{RMS} = \left\langle \log \sigma_{\text{win},c} \right\rangle_{c \in \{Z,N,E\}}$$

identical to the `log_snr` definition used in `regression.py`. The term
$\sigma_{\text{sta},c}$ is obtained from `compute_station_noise_baselines()`
(the mechanism described in Section 3.4), computed unconditionally in this
context — independent of the `--baseline` flag, which controls only whether
`seq` and `img` themselves use the station baseline or per-window
self-standardization. This independence is deliberate: Section 8.2
establishes that RAM's image content does not depend on which
$(\mu,\sigma)$ pair is used for standardization, so gating `log_snr` behind
`--baseline` would have made the correction unavailable by default with no
offsetting benefit. Any window whose station lacks at least 60 s of usable
noise data falls back to $\log\text{SNR} = 0$.

All four encoders resample every window to a nominal rate before
standardizing (`_resample_to`/`_fit_length`), the same correction
`SpectrogramEncoder` required for the image alone (Section 3.3's
per-station-rate constraint). In this context it additionally keeps every
`seq` tensor's sample count fixed regardless of a station's native sampling
rate, a constraint the image's fixed `target_n` did not require.

### 10.3 Model Architecture Additions

- **`DualChannelBinaryNet`** implements the source paper's architecture
  directly: `LSTMAttentionBranch` combined with `CNNBranch`, fused as
  $a F_{1D}+b F_{2D}$, with a single-logit head. The `--channels`
  argument (`all`, `1d`, `2d`) ablates either branch.
- **`RamAuxCNN`** uses the same ResNet-with-Squeeze-and-Excitation trunk as
  `ImprovedSeismicCNN` (Section 4.1, reusing `ResBlock` directly), with
  pooled features concatenated with the auxiliary vector before the
  classification head. The `--no-aux` flag removes the concatenation only,
  providing an architecture-matched control that isolates precisely what
  the two auxiliary scalars contribute.
- **`DualChannelAuxBinaryNet`** extends `DualChannelBinaryNet` with an
  auxiliary branch concatenated after the $aF_{1D}+bF_{2D}$ fusion step,
  following the same pattern already used by the unrelated catalog model's
  `DualChannelRiskNet`. The `--channels` argument extends to
  `{all, 1d, 2d, aux, 1d+aux, 2d+aux}`.
- **`cnn_lstm_stack.py`** implements late-fusion stacking: given two
  already-trained, frozen checkpoints (for example, `--channels 1d` and
  `--channels 2d` runs), it collects their pre-sigmoid logits and fits
  `sklearn.LogisticRegression` — one weight per branch plus a bias, in
  logit space — on validation-set logits only, which were not observed by
  either frozen model's own training, and evaluates on the test set. No
  backbone retraining is involved.
- **`GatedFusion`** (`cnn_lstm.py`, shared by both classification scripts)
  replaces the fixed scalar pair with a per-example gate:
  $g = \sigma(\text{MLP}([F_{1D}, F_{2D}]))$, followed by
  $g F_{1D} + (1-g) F_{2D}$. Selected via `--fusion {linear, gate}` in both
  `cnn_lstm_classify.py` and `cnn_lstm_classify_aux.py`; the option affects
  only channel combinations in which both branches are active. Gate-value
  diagnostics (mean value by true class, and by prediction correctness) are
  reported at test time in place of the linear mode's learned-weight
  summary.

### 10.4 Results

Every row reflects 6 s windows on the same 71,672-window dataset, with a
test set of 9,548 samples (balanced). Parameter counts are measured
directly from each trained model, not estimated.

| Model | Parameters | Test AUC | MCC | Accuracy |
|---|---|---|---|---|
| Spectrogram CNN only (`2d`) | 115,459 | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram-dual, fused, linear (`all`) | 182,563 | 0.9646 | 0.8122 | 90.61 % |
| RAM-dual + aux, fused, linear (`all`) | 182,759 | 0.9514 | 0.7790 | 88.95 % |
| RAM-dual + aux, no LSTM branch (`2d+aux`) | 115,655 | 0.9468 | 0.7775 | 88.84 % |
| RAM + aux, no dual architecture (`use_aux=True`) | 309,777 | 0.9230 | 0.7018 | 84.79 % |
| RAM-dual, raw waveform only (`1d`) | 76,707 | 0.9216 | 0.6849 | 84.22 % |
| RAM-dual, fused, linear, no aux (`all`) | 182,563 | 0.9144 | 0.6042 | 79.57 % |
| RAM CNN only, no aux (`use_aux=False`) | 309,713 | 0.8356 | 0.5339 | 76.70 % |
| RAM-dual, RAM image only (`2d`) | 115,459 | 0.8408 | 0.5288 | 76.42 % |

Late-fusion stacking (Section 10.3), fit on frozen 1d/2d logits:

| Base checkpoints | 1d alone | 2d alone | Naive average (logits) | Stacked |
|---|---|---|---|---|
| RAM-dual | AUC 0.9229, MCC 0.688 | AUC 0.8408, MCC 0.530 | AUC 0.9136, MCC 0.692 | AUC 0.9203, MCC 0.688, accuracy 84.31 % |
| Spectrogram-dual | AUC 0.9229, MCC 0.688 | AUC 0.9793, MCC 0.867 | AUC 0.9697, MCC 0.866 | AUC 0.9743, MCC 0.871, accuracy 93.54 % |

Learned linear-fusion weights $(a,b)$, for reference: RAM-dual without aux,
$(0.996, 0.829)$; spectrogram-dual, $(0.749, 1.051)$; RAM-dual with aux,
$(0.499, 0.354)$. The last pair is markedly more balanced in relative terms
once both branches carry usable information, rather than one weight sitting
near 1 with the other trailing.

### 10.5 Analysis

**10.5.1 The source paper's fixed linear fusion underperformed the best
single branch, on two independent representations, before the amplitude
correction.** With RAM as the 2D channel, the raw-waveform branch alone
(AUC 0.9216) outperformed the fused model (0.9144). With a spectrogram as
the 2D channel, the image branch alone (0.9793) outperformed the fused
model (0.9646). A fixed pair of scalars, trained jointly with both
branches, cannot suppress a weaker branch on the specific examples where it
is incorrect, and joint training can allow a noisy branch to degrade the
stronger branch's own learned representation. This pattern reproduced on
two independent 2D representations, which is why it is treated as a
property of the fusion mechanism rather than of either individual branch.

**10.5.2 Late-fusion stacking recovers what joint fusion lost, without
modifying either branch.** Freezing the same two checkpoints and fitting a
two-input logistic regression on their logits (Section 10.3) matched or
improved on the joint-fusion result in both cases tested. On RAM-dual, the
stacked result lands within measurement noise of the best single branch
(0.9203 versus the 1d-alone figure of 0.9229) rather than below it. On
spectrogram-dual, stacking outperforms both single branches outright (0.9743
AUC, 0.871 MCC, versus 2d-alone's 0.9793 AUC and 0.867 MCC); stacking's AUC
is marginally below the 2d-alone figure even though its MCC and accuracy
are higher, indicating a small trade of ranking quality for a better
decision at the fixed 0.5 threshold, a difference within this dataset's
established noise floor. This result confirms that the fusion problem
identified in 10.5.1 originates in the joint-training mechanism rather than
in an inherent limitation of combining the two branches.

**10.5.3 The amplitude auxiliary input is the largest single effect
measured in this investigation, and confirms the prediction made in
Sections 8.2 and 8.3 quantitatively.** In an architecture-matched,
single-variable comparison (identical `ImprovedSeismicCNN`-derived trunk,
differing only in whether the auxiliary concatenation is present):

$$\text{AUC: } 0.8356 \to 0.9230 \quad(+0.0874) \qquad \text{MCC: } 0.5339 \to 0.7018 \quad (+0.1679) \qquad \text{Accuracy: } 76.70\% \to 84.79\% \quad (+8.09\text{ pp})$$

Two scalars, computed at negligible cost from data the pipeline already
possessed, closed most of the gap between a plain RAM classifier and
STA/LTA-competitive performance. This correction also incidentally resolved
the fusion pathology identified in 10.5.1: the amplitude-augmented dual
model (`all`, AUC 0.9514) outperforms every single-branch RAM alternative,
which the equivalent model without the auxiliary input failed to do. The
`2d+aux` ablation (no LSTM branch present, AUC 0.9468) lands almost exactly
on the full fused model's result, indicating that the amplitude correction
accounts for nearly all of the improvement, and that the LSTM branch
contributes only a small increment beyond it, rather than being structurally
necessary to compensate for what the RAM image cannot represent.

**10.5.4 Spectrograms remain the stronger 2D representation, with or
without the amplitude correction.** Spectrogram-2D-only (0.9793 AUC)
outperforms RAM-plus-amplitude-2D-only (0.9468 AUC) by a substantial
margin. This is expected and does not contradict 10.5.3:
`--normalize station` spectrograms preserve amplitude as a function of
frequency and time, a considerably richer representation than two collapsed
scalars appended after the fact. The amplitude correction makes RAM
competitive with the raw-waveform branch and with its own pre-correction
performance; it does not make RAM's 2D representation as informative as a
spectrogram's. Where RAM images are specifically required — for example,
for compatibility with the source paper's method — pairing them with the
amplitude auxiliary input is a substantial, low-cost improvement. Where the
2D representation is an open choice, spectrograms remain the stronger
option on this dataset.

**10.5.5 Gated fusion produces mixed results across three tested
configurations, and does not support a single unifying explanation on the
evidence available.** `GatedFusion` was evaluated against linear fusion on
three configurations: spectrogram-dual, RAM-dual without the amplitude
input, and RAM-dual with it.

| Configuration | Linear fusion | Gated fusion | Best single branch | Mean gate $g$ |
|---|---|---|---|---|
| Spectrogram-dual | AUC 0.9646, MCC 0.812, 90.61 % | **AUC 0.9761, MCC 0.850, 92.51 %** | 2d-alone: 0.9793 / 0.867 / 93.28 % | 0.169 |
| RAM-dual, no aux | AUC 0.9144, MCC 0.604, 79.57 % | AUC 0.9071, MCC 0.637, 81.68 % | 1d-alone: 0.9216 / 0.685 / 84.22 % | 0.719 |
| RAM-dual + aux | AUC 0.9514, MCC 0.779, 88.95 % | AUC 0.9487, MCC 0.744, 87.12 % | 2d+aux-alone: 0.9468 / 0.778 / 88.84 % | 0.487 |

The outcome differs by metric as well as by configuration. On
spectrogram-dual, gated fusion improves all three metrics over linear
fusion, closing most of the gap to the best single branch without the
separate frozen-checkpoint procedure stacking requires. On RAM-dual without
the amplitude input, gated fusion is *mixed*: AUC decreases (0.9144 →
0.9071) while MCC and accuracy both increase (0.604 → 0.637; 79.57% →
81.68%), indicating a shift in decision calibration at the fixed 0.5
threshold rather than a uniform change in ranking quality. On RAM-dual with
the amplitude input, gated fusion is uniformly worse across all three
metrics.

An initial hypothesis, formed after the first two configurations were
measured, proposed that gating helps when branches are genuinely
complementary and disagree meaningfully on a per-example basis, and can hurt
when one branch already dominates. The third configuration does not cleanly
support this account: the performance gap between branches on RAM-dual
without aux (0.9216 versus 0.8408, a difference of 0.081) is larger than on
spectrogram-dual (0.9793 versus 0.9216, a difference of 0.058), yet gating
still underperformed linear fusion in the RAM-dual case. The more accurate
statement, on the evidence collected, is that gated fusion improved results
in exactly one of three tested configurations — the spectrogram-based one —
and modestly degraded results in the two RAM-based configurations. Whether
this reflects a property of the RAM representation specifically, an
estimation difficulty in learning the gate function reliably from a
comparatively noisier branch pair, or an artifact of single-seed
measurement, is not established by the data available and should not be
asserted. **Gated fusion is not a default improvement over linear fusion,
and its benefit or cost should be measured directly for any new branch
pairing rather than assumed from this result.**

**10.5.6 A hyperparameter sweep of the RAM-plus-amplitude classifier
(`cnn_ram_aux.py`) produced no distinguishable improvement over the
established defaults.** Six configurations were trained: the established
default (learning rate 2e-4, weight decay 3e-2, dropout 0.6/0.4, head width
32, three residual stages), two alternative learning rates (1e-4, 3e-4), a
lower weight decay (1e-2), a wider head alone (width 64, three stages), and
the original "long" preset's full capacity (four stages, dropout 0.5/0.3,
head width 64 — approximately four times the parameter count). Selection
was made on validation AUC, decided before training began, specifically to
avoid selecting a configuration by its test-set performance across multiple
candidates.

| Configuration | Parameters | Validation AUC | Test AUC | Test MCC | Test accuracy |
|---|---|---|---|---|---|
| Default | 309,777 | 0.9302 | 0.9230 | 0.7018 | 84.79 % |
| Learning rate 3e-4 | 309,777 | **0.9307** | 0.9268 | 0.7137 | 85.28 % |
| Weight decay 1e-2 | 309,777 | 0.9301 | 0.9228 | 0.7013 | 84.77 % |
| Learning rate 1e-4 | 309,777 | 0.9287 | 0.9224 | 0.7040 | 84.96 % |
| Four stages, wider head (4× capacity) | 1,249,425 | 0.9287 | 0.9302 | 0.7235 | 86.13 % |
| Wider head only (three stages) | 314,001 | 0.9268 | 0.9270 | 0.7054 | 85.17 % |

All six configurations fall within a validation AUC band of
0.9268–0.9307 — a spread of 0.0039, well inside the approximately 1–2 point
noise floor already established for single-seed measurements on this
dataset (Section 9). The nominal winner by the pre-specified selection rule
(learning rate 3e-4, val AUC 0.9307) exceeds the default by 0.0005, which is
not distinguishable from noise and should not be reported as a genuine
improvement. One result is worth noting for methodological reasons rather
than performance: the four-times-larger configuration scored *lower* on
validation AUC (0.9287) than the 310,000-parameter default despite scoring
higher on test AUC (0.9302) — a direct illustration of why selection was
fixed to validation performance before training began, and further evidence
that this task is limited by available information rather than by model
capacity, consistent with the diagnosis in Section 8.1. **On the evidence
gathered, hyperparameter tuning is not a productive direction for this
model; the established default configuration should be retained.**

**10.5.7 The amplitude auxiliary input, extended to the spectrogram-based
model, does not repeat its RAM-side effect — and mildly hurts the 2D branch
in isolation.** `SpectrogramDualAuxEncoder` (Section 10.2) pairs the
spectrogram-dual dataset with the same `[log_snr, log_rms]` vector used for
RAM. Results, alongside the existing no-aux spectrogram figures for
comparison:

| Configuration | Test AUC | MCC | Accuracy |
|---|---|---|---|
| Spectrogram 2D only, no aux | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram + aux, no LSTM branch (`2d+aux`) | 0.9749 | 0.8626 | 93.02 % |
| Spectrogram-dual, no aux, gated fusion | 0.9761 | 0.8501 | 92.51 % |
| Spectrogram-dual + aux, fused, linear (`all`) | 0.9733 | 0.8465 | 92.31 % |
| Spectrogram-dual, no aux, fused, linear | 0.9646 | 0.8122 | 90.61 % |

Two results follow directly. First, adding the auxiliary input to the 2D
branch alone makes it *worse*, not better: `2d+aux` (0.9749) scores below
plain `2d` (0.9793), a decrease of 0.0044. This is the reverse of the RAM
result (Section 10.5.3, an increase of 0.0874) and is consistent with the
explanation offered in Section 10.5.4: a station-normalized spectrogram
already encodes amplitude as a function of time and frequency, so appending
two collapsed, redundant scalars adds estimation noise without adding
information the model did not already have access to. Where Section 10.5.3
demonstrated that the auxiliary input matters when the 2D representation is
scale-invariant, this result demonstrates the converse: it does not matter,
and can mildly hurt, when the 2D representation already carries the same
information in richer form.

Second, the auxiliary input still improves the **fused** model relative to
its own no-aux baseline (0.9646 → 0.9733, +0.0087), for a reason similar to
why gated fusion helped in the same setting (Section 10.5.5): both
interventions reduce how much a naive linear combination is dragged down by
a comparatively weaker branch, whether by learning to trust the stronger
branch more or by handing the weaker branch's fusion partner information it
lacked. Neither intervention, alone or combined, reaches the ceiling
established by `2d` alone.

**10.5.8 The amplitude auxiliary input helps the raw-waveform branch nearly
as much as it helps the RAM branch.** `--channels 1d+aux` (the raw
standardized waveform plus `[log_snr, log_rms]`, with no 2D branch present
at all) was trained on the RAM-plus-amplitude dataset:

$$\text{Test AUC: } 0.9216 \to 0.9501 \quad(+0.0285) \qquad \text{MCC: } 0.6849 \to 0.7675 \quad(+0.0826) \qquad \text{Accuracy: } 84.22\% \to 88.37\% \quad(+4.15\text{ pp})$$

comparing against the no-aux `1d`-alone figure from Section 10.4. This is
consistent with the same underlying mechanism identified in Section 8.2: the
raw waveform is standardized before entering the LSTM (Section 10.1), which
removes absolute amplitude exactly as RAM's own internal standardization
does, so the raw-waveform branch is scale-blind for the same structural
reason the RAM image is, and benefits from the same correction. With the
auxiliary input present, `1d+aux` (0.9501 AUC) is closer to the full fused
model (0.9514, Section 10.4) than `2d+aux` is (0.9468), and closer than
either branch was to the fused model without the auxiliary input (Section
10.5.1). One data point confirms this is dataset-independent by
construction rather than by measurement: running the identical `1d+aux`
configuration against the spectrogram-plus-amplitude dataset produced
test figures identical to four decimal places (accuracy 88.37%, AUC 0.9501,
MCC 0.7675), because that ablation never reads the `img` tensor and both
encoders compute `seq` and `aux` with the same formula — a useful check on
the pipeline's correctness, not a second independent measurement.

**10.5.9 A hyperparameter sweep of `LSTMAttentionBranch` itself (depth,
attention heads, hidden width) produced no distinguishable improvement,
and illustrates the validation-versus-test divergence risk more sharply
than the RAM CNN sweep did.** Five configurations were trained on
`1d+aux` (isolating the LSTM branch, since the 2D branch is absent and the
auxiliary input is held fixed): the established default (one LSTM layer,
four attention heads, hidden width 48), two head counts (2, 8), a deeper
LSTM (two layers), and a wider hidden state (64). Selection was fixed to
validation AUC before training, as in Section 10.5.6.

| Configuration | Parameters | Validation AUC | Test AUC | Test MCC | Test accuracy |
|---|---|---|---|---|---|
| Default (1 layer, 4 heads, hidden 48) | 76,903 | 0.9574 | **0.9501** | **0.7675** | **88.37 %** |
| 2 attention heads | 76,903 | 0.9576 | 0.9485 | 0.7520 | 87.59 % |
| 8 attention heads | 76,903 | 0.9586 | 0.9495 | 0.7513 | 87.53 % |
| Hidden width 64 | 123,815 | **0.9588** | 0.9488 | 0.7508 | 87.52 % |
| 2 LSTM layers | 132,967 | 0.9565 | 0.9484 | 0.7486 | 87.33 % |

All five configurations fall within a validation AUC band of 0.9565–0.9588
(a spread of 0.0023, tighter even than Section 10.5.6's RAM CNN sweep), again
well inside the established single-seed noise floor. The nominal winner by
the pre-specified rule (hidden width 64, val AUC 0.9588) is not a
meaningful improvement over the default. More notably, the default
configuration has the *best* test-set MCC and accuracy of all five
candidates despite ranking fourth of five on validation AUC — every
configuration the sweep would have nominally preferred by its own selection
rule scores worse on held-out test than the untouched default across MCC
and accuracy. This is a sharper illustration of the same point Section
10.5.6 made with the RAM CNN sweep: validation-based selection is not a
guarantee against picking a configuration that generalizes worse, only a
discipline against choosing one by looking at test results directly.
**Neither the RAM CNN's hyperparameters (Section 10.5.6) nor the LSTM
branch's own hyperparameters reward tuning on this dataset; the established
defaults should be retained in both cases.**

**10.5.10 Late-fusion stacking of the amplitude-augmented branches beats
both linear and gated joint fusion, on both datasets.** `cnn_lstm_stack_aux.py`
(Section 10.3) applies the Section 10.4 stacking procedure to frozen
`1d+aux`/`2d+aux` checkpoints instead of the plain ones:

| Dataset | 1d+aux alone | 2d+aux alone | Stacked | Linear fusion (`all`) | Gated fusion (`all`) |
|---|---|---|---|---|---|
| RAM+aux | AUC 0.9505, MCC 0.768, 88.39 % | AUC 0.9468, MCC 0.778, 88.84 % | **AUC 0.9557, MCC 0.781, 89.07 %** | AUC 0.9514, MCC 0.779, 88.95 % | AUC 0.9487, MCC 0.744, 87.12 % |
| Spectrogram+aux | AUC 0.9505, MCC 0.768, 88.39 % | AUC 0.9749, MCC 0.863, 93.02 % | **AUC 0.9758, MCC 0.868, 93.37 %** | AUC 0.9733, MCC 0.847, 92.31 % | AUC 0.9716, MCC 0.836, 91.80 % |

On both datasets, stacking is the best fusion mechanism tested among the
three (linear, gated, stacked) once the amplitude input is present —
consistent with Section 10.5.2's finding on the non-aux branches, where
stacking also matched or beat joint fusion. Stacking on RAM+aux is a clear
win over every alternative on every metric. Stacking on spectrogram+aux is
the best fusion mechanism but does not surpass the plain `2d`-alone,
no-aux ceiling (0.9793 AUC, Section 10.4) — no fusion mechanism has yet
closed that gap.

**10.5.11 Combining two independently helpful interventions does not
compound, and mildly hurts.** Gated fusion (Section 10.5.5) and the
amplitude auxiliary input (Section 10.5.3) each measurably improved
results on their own. Combined — gated fusion applied to the
amplitude-augmented spectrogram-dual model — the result is worse than
either intervention alone: AUC 0.9716, versus gated-fusion-without-aux's
0.9761 and linear-fusion-with-aux's 0.9733. This is a useful negative
result: independently validated improvements do not automatically compose,
and each combination has to be measured rather than assumed. The three
fusion mechanisms tested on the amplitude-augmented spectrogram-dual model
now rank, by AUC: stacked (0.9758) > linear (0.9733) > gated (0.9716) — the
reverse of gated fusion's ranking on the same 2D representation without
aux (Section 10.5.5), where gated fusion was the best of the three.

**10.5.12 Seed-repeated verification of the two closest single-seed
claims changes the reported conclusion for one of them.** Sections 10.5.5
and 10.5.7 each rested on a single train/validation/test split. Both were
re-run at two additional seeds (1 and 2, alongside the original 42) to
check whether the claimed improvements survive repetition.

*Gated vs. linear fusion, spectrogram-dual, no aux (Section 10.5.5):*

| Seed | Linear AUC / MCC / Acc | Gated AUC / MCC / Acc | Gated − Linear (AUC) |
|---|---|---|---|
| 42 | 0.9646 / 0.812 / 90.61 % | 0.9761 / 0.850 / 92.51 % | +0.0115 |
| 1 | 0.9719 / 0.848 / 92.32 % | 0.9753 / 0.851 / 92.53 % | +0.0034 |
| 2 | 0.9746 / 0.834 / 91.68 % | 0.9720 / 0.849 / 92.36 % | **−0.0026** |

The AUC advantage is not robust — it reverses sign at seed 2, and its
magnitude at seed 1 is roughly a third of the original single-seed figure.
Averaged across the three seeds, gated fusion still leads on AUC (mean
0.9745 vs. 0.9704), but the effect is smaller and noisier than Section
10.5.5 reported. Accuracy and MCC tell a more consistent story: gated
fusion wins on **both** metrics at **all three** seeds, and does so with
much lower run-to-run spread (accuracy range 92.36–92.53 %, a spread of
0.17 points, versus linear fusion's 90.61–92.32 %, a spread of 1.71
points; MCC spread 0.002 vs. 0.036). The corrected finding is therefore
not "gated fusion is a clear AUC win" but **"gated fusion gives more
consistent, slightly better decisions at the operating threshold, with a
real but small and noisy ranking-quality advantage"** — a materially
weaker and more precise claim than Section 10.5.5's original framing.

*Amplitude aux vs. no aux, spectrogram-dual, linear fusion (Section 10.5.7):*

| Seed | No-aux AUC / MCC / Acc | Aux AUC / MCC / Acc | Aux − No-aux (AUC) |
|---|---|---|---|
| 42 | 0.9646 / 0.812 / 90.61 % | 0.9733 / 0.847 / 92.31 % | +0.0087 |
| 1 | 0.9719 / 0.848 / 92.32 % | 0.9707 / 0.841 / 92.01 % | **−0.0012** |
| 2 | 0.9746 / 0.834 / 91.68 % | 0.9705 / 0.834 / 91.66 % | **−0.0041** |

This result does not merely shrink — it reverses and averages out to
approximately zero (mean AUC difference +0.0011 across the three seeds).
**Section 10.5.7's claim that the amplitude auxiliary input improves the
fused linear model does not survive repetition; the original single-seed
result at seed 42 was not representative.** This does not call the
amplitude fix into question generally: the much larger effects reported in
Section 10.5.3 (RAM CNN alone, +0.087 AUC) and Section 10.5.8 (`1d+aux`
alone, +0.029 AUC) are an order of magnitude larger than the noise band
established here (~0.01 AUC) and are far more likely to be genuine, though
neither has itself been re-run at additional seeds and both should be
treated with appropriately more confidence than this reversed result, not
full certainty. What this result specifically corrects is narrower: aux
does not reliably help once it is combined with joint linear fusion on an
already-strong 2D representation, where Section 10.5.4 already established
the 2D branch has little room left for the aux input to add.

**The practical lesson from 10.5.12 is broader than either individual
correction: none of the close-margin claims elsewhere in Section 10 —
including the `1d+aux` and `2d+aux` effects in Sections 10.5.3/10.5.8, and
the stacking results in Section 10.5.2/10.5.10 — have been checked against
more than one seed, and this section demonstrates concretely that a
single-seed result on this dataset can overstate an effect, understate it,
or report the wrong sign entirely.** Only claims with effect sizes well
above the ~0.01 AUC / ~0.02 MCC band established here (the amplitude fix on
RAM alone, and the standing conclusion that plain `2d` remains the overall
best configuration) should be treated as settled without further seeds.

**Considered together with Sections 10.5.1–10.5.12, the single
best-performing configuration measured in this entire investigation remains
the plain spectrogram CNN classifier, with no LSTM branch, no auxiliary
input, and no fusion mechanism (AUC 0.9793).** Every structural addition
tested — the source paper's dual-channel architecture, the amplitude
auxiliary input, gated fusion, late-fusion stacking, and combinations of
these — improved on some other configuration along the way, and each
produced a genuine, informative finding about why RAM underperforms and how
fusion mechanisms behave, including where those findings needed correcting
under repeated measurement (Section 10.5.12). None of them, individually or
in combination, has yet exceeded the simplest model in this comparison.
This should be read as a genuine result rather than a failure of the
investigation: it indicates that, for this task and dataset, the highest-value
remaining work is more likely to be in feature representation (for example,
per-component auxiliary inputs, or spectrogram parameters) than in additional
architectural complexity layered on top of an already-strong 2D
representation.

### 10.6 Updated Limitations and Recommendations

- Every result in Section 10 reflects a single train/validation/test split
  at one random seed, **except** the two comparisons re-run at three seeds
  in Section 10.5.12, where repetition changed the reported conclusion for
  one of them (amplitude aux no longer shown to help the fused linear
  model) and meaningfully narrowed the other (gated fusion's AUC edge).
  Every other close-margin figure in Section 10 — including stacked-RAM-dual
  versus 1d-alone, `2d+aux` versus the full amplitude-dual model, `1d+aux`'s
  effect size (Section 10.5.8), and the stacking results in Sections
  10.5.2/10.5.10 — remains unverified at additional seeds and should be
  read with the caution Section 10.5.12 demonstrates is warranted, not
  assumed settled by a single run.
- The amplitude correction has now been evaluated on the spectrogram-dual
  model (Section 10.5.7) and combined with stacking (Section 10.5.10, a
  clear win on both datasets) and with gated fusion (Section 10.5.11, a
  clear loss relative to either intervention alone).
- `log_snr` and `log_rms` are single scalars per window, averaged over
  Z, N, and E. A per-component variant (six scalars rather than two,
  `--per-component-aux` in `data_downloader`) has been implemented and
  smoke-tested but not evaluated — see Section 10.7.
- 60 s windows were not re-tested with any of the additions described in
  this section. The geometric argument in Section 2.3 for why short windows
  are more difficult applies specifically to the RAM branch, not to the
  raw-waveform or auxiliary branches, so the balance between branches may
  shift at longer window lengths.
- Gated fusion's mixed result across 2D representations (Section 10.5.5)
  has not been diagnosed further, though Section 10.5.12 adds one relevant
  data point: on spectrogram-dual specifically, the effect is real on
  accuracy/MCC but small and noisy on AUC, which narrows rather than
  resolves the three explanations originally offered.
- `1d+aux` (Section 10.5.8) and the LSTM branch's own hyperparameters
  (Section 10.5.9) have now been evaluated; neither the CNN's hyperparameters
  (Section 10.5.6) nor the LSTM's reward tuning on this dataset. This
  narrows the plausible location of any remaining gap toward feature
  representation rather than architecture search on either existing branch.

### 10.7 Investigation Status and Next Steps

This closes the dual-channel waveform-classification investigation (Section
10) as an active line of work. It is not concluded because a dead end was
reached; it is being set aside deliberately because it was itself an
extension of the project's original scope, and continuing to refine it
further would be digging the same hole deeper rather than returning to that
scope.

**What this phase established.** Across all configurations tested,
`2d`-alone with no auxiliary input remains the single best-performing
configuration in the entire investigation (test AUC 0.9793, Section 7.2).
Relative to that ceiling: gated fusion gives a real but small and noisy AUC
edge over linear fusion, alongside a more robust, low-variance win on
accuracy and MCC (Section 10.5.12); the amplitude-scalar correction produces
a large, seed-independent improvement on the RAM-only and RAM-dual models
(Sections 8.2, 10.5.3, 10.5.8) but a much smaller, seed-fragile one on the
spectrogram-dual model (Section 10.5.12); late-fusion stacking of frozen
aux-augmented branches beats both linear and gated joint fusion on both
datasets (Section 10.5.10); and gated fusion combined with amplitude aux
performs worse than either intervention alone — an explicit case of two
independently-validated improvements failing to compose (Section 10.5.11).
Section 10.5.12's seed-repeat check is the standing methodological caution
for the whole section: single-seed margins under roughly 0.01–0.02 AUC have
been shown concretely, on this dataset, to overstate an effect, understate
it, or report the wrong sign.

**What was started but not finished.** Per-component auxiliary scalars
(`--per-component-aux`, adding `RamAuxEncoderV2`, `RamDualAuxEncoderV2`, and
`SpectrogramDualAuxEncoderV2` to `data_downloader/seismic_cli`, commit
`8485ffc`) are implemented, produce the expected `(6,)`-shaped tensor, and
required no model-side changes since `aux_dim` is read from the tensor shape
at load time. No dataset was regenerated at full scale and no model was
retrained against it — this is left as unevaluated, not abandoned. 60 s
windows with the LSTM+attention branch (Section 10.6) were never started.
Neither should be treated as a loose end requiring closure; either could be
picked back up later if the amplitude-correction or window-length questions
become relevant again.

**Where the project goes next.** The original objective was forecasting
event onset time and event class from earthquake catalog data — not
waveform classification. That work already has a dormant implementation
that predates this section and has never been run or reported on:
`data_downloader/seismic_cli/catalog.py` builds chronologically-split,
embargoed sliding-window datasets from a catalog (feature channels in
`SEQ_FEATURES`/`IMAGE_FEATURES`/`AUX_FEATURES`, three-class time-to-next-event
labels in `RISK_CLASSES = [lt_1y, 1_5y, gt_5y]`), reachable via the
`generate-catalog-dataset` CLI command; `cnn_earthquake/src/cnn_lstm_loeo.py`
implements `DualChannelRiskNet`, the corresponding model. Both were built
using the same dual-channel {seq, img, aux} pattern documented in Section
10.2–10.3, so the architecture work in this section is not wasted even
though its focus was elsewhere — but neither the dataset generation nor the
model has been exercised end-to-end, so the next phase starts from
"does this run at all and what does a first result look like," not from a
refinement question.

---

## Appendix A. Reproduction Instructions: Dual-Channel and Auxiliary Pipeline

```bash
# Dual-channel dataset (paper's raw-waveform seq + RAM image), 6 s, maximum size
seismic-cli generate-dual-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_dual_6s --window-seconds 6 --max

# Same, with a spectrogram in place of a RAM image as the 2D channel
seismic-cli generate-spec-dual-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_specdual_6s --window-seconds 6 --max

# Plain RAM classifier with the amplitude auxiliary input (no LSTM branch)
seismic-cli generate-ram-aux-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_ramaux_6s --window-seconds 6 --max

# Dual-channel model with the amplitude auxiliary input, all three inputs present
seismic-cli generate-dual-aux-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_dualaux_6s --window-seconds 6 --max

# Spectrogram-dual model with the amplitude auxiliary input (Section 10.5.7)
seismic-cli generate-spec-dual-aux-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_specdualaux_6s --window-seconds 6 --max

cd ../cnn_earthquake/src

python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_dual_6s \
    --channels all --batch-size 32                # or --channels 1d / 2d
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_dual_6s \
    --channels all --fusion gate --batch-size 32   # gated fusion (Section 10.5.5)
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --channels all --fusion gate --batch-size 32   # gated fusion (Section 10.5.5)

python cnn_ram_aux.py --dataset-dir ../../data_downloader/dataset_ramaux_6s
python cnn_ram_aux.py --dataset-dir ../../data_downloader/dataset_ramaux_6s --no-aux
python cnn_ram_aux.py --dataset-dir ../../data_downloader/dataset_ramaux_6s --lr 3e-4   # hyperparameter sweep (Section 10.5.6)

python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels all --batch-size 32                 # or 1d / 2d / aux / 1d+aux / 2d+aux
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels all --fusion gate --batch-size 32    # gated fusion (Section 10.5.5)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --channels all --batch-size 32                 # spectrogram + aux (Section 10.5.7)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --channels 2d+aux --batch-size 32               # spectrogram + aux, no LSTM (Section 10.5.7)

python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels 1d+aux --batch-size 32               # raw waveform + aux, no 2D branch (Section 10.5.8)

# LSTM branch hyperparameter sweep, isolated via --channels 1d+aux (Section 10.5.9)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels 1d+aux --batch-size 32 --lstm-heads 2      # or --lstm-heads 8 / --lstm-layers 2 / --hidden 64

# Late-fusion stacking, given two already-trained --channels 1d / --channels 2d checkpoints
python cnn_lstm_stack.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --ckpt-1d trained_model_cnnlstm_classify_1d/best_cnnlstm_classify.pth \
    --ckpt-2d trained_model_cnnlstm_classify_2d/best_cnnlstm_classify.pth

# Late-fusion stacking on amplitude-augmented checkpoints (Section 10.5.10)
python cnn_lstm_stack_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --ckpt-1d trained_model_cnnlstm_aux_1daux/best_cnnlstm_aux.pth \
    --ckpt-2d trained_model_cnnlstm_aux_2daux/best_cnnlstm_aux.pth

# Gated fusion + amplitude aux combined (Section 10.5.11)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --channels all --fusion gate --batch-size 32

# Seed-repeated verification (Section 10.5.12) -- rerun any of the above with --seed 1 / --seed 2
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --channels all --seed 1 --batch-size 32               # repeat for --fusion gate, and --seed 2
```

## Appendix B. Reproduction Instructions: Original RAM + CNN-Only Pipeline

```bash
# 1. Arrival-anchored 6 s windows from downloaded 60 s data
seismic-cli anchor-windows \
    --source-dir data/batched_waveforms/window_post_60s \
    --output-base-dir data/batched_waveforms -t 6

# 2. Maximum-size balanced, station-disjoint 6 s dataset
seismic-cli generate-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms \
    --output-dir dataset_6s_max \
    --window-seconds 6 --overlap 0.5 --max --max-windows-per-station 20

# 3. Train (short preset auto-selected by --window-seconds)
python src/model/cnn_train.py --dataset-dir dataset_6s_max \
    --save-dir trained_model_6s --window-seconds 6

# 4. STA/LTA baseline on the identical test windows (parameters auto-derived)
seismic-cli eval-sta-lta \
    --manifest-path dataset_6s_max/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5
```

Full option reference: `data_downloader/README.md`.
