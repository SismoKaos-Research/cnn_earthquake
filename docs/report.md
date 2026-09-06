# Earthquake Detection and Magnitude Classification from Raw Waveforms via Relative Angle Matrix Image Encoding and Dual-Channel CNN–LSTM Networks

## Abstract

This report documents an investigation into applying the Relative Angle
Matrix (RAM) method — a technique proposed by Wang and Zhao (2025,
*Applied Soft Computing* 172) for converting one-dimensional vibration
signals into two-dimensional images for convolutional neural network (CNN)
classification — to seismic event detection and magnitude classification
from three-component (Z/N/E) waveform data. All detection comparisons are
made against classic STA/LTA, the standard seismological trigger algorithm,
rather than an arbitrary baseline.

The investigation has four parts. The first establishes a RAM-image CNN
classifier, evaluates it against STA/LTA, and identifies a structural
limitation: the RAM transform is provably scale-invariant and therefore
cannot represent absolute signal amplitude, the dominant feature short-window
detection depends on. The second implements the source paper's full
dual-channel architecture — a CNN over the RAM image paired with an LSTM and
multi-head self-attention branch over the raw waveform — corrects a design
error found by direct comparison against the source paper, introduces an
amplitude auxiliary input to address the scale-invariance limitation,
evaluates a spectrogram-based alternative to the RAM image, and compares
three fusion strategies for combining the two channels. The third asks a
narrower, related question: can *event class* (a coarse magnitude bin) be
predicted directly from a single 3-second window, reusing the same amplitude
auxiliary machinery. The fourth records the project's status and its
redirection back toward its original objective — forecasting event onset
time and class from catalog data, not waveform classification — which is
carried out separately, and covered in Section 11 below. A fifth part
(Section 13) returns to waveforms for one bounded question: predicting peak
ground motion from a 3-second window, as a replication of Nurtas et al.
(2025) supplied with the non-neural floor that paper omits.

Seven findings carry the report:

- **The amplitude auxiliary input is the single largest contributor to
  classification performance measured in this work** (test AUC 0.836 → 0.923
  on the RAM-only classifier, an architecture-matched, single-variable
  comparison).
- **Once STA/LTA is correctly parameterized for arrival-anchored windows — a
  defect found and fixed during this rewrite — it scores AUC 0.82 at 6
  seconds, and every tested CNN configuration, including the weakest one,
  beats it.** The library's own auto-derived default parameters silently
  score AUC 0.51 (random) on the same data, because the default long-term
  average window is longer than the pre-arrival buffer the anchoring scheme
  provides, so classic STA/LTA's mandatory warm-up period swallows the P-wave
  arrival before the characteristic function ever sees it.
- **Spectrogram-based encoding outperforms RAM-based encoding as the
  two-dimensional channel in every configuration tested**, and the single
  best-performing configuration in the entire detection investigation remains
  the plain spectrogram CNN alone — no LSTM branch, no auxiliary input, no
  fusion mechanism (AUC 0.9793).
- **Late-fusion stacking of independently-trained branches consistently
  matches or beats joint (linear or gated) fusion**, on every branch pairing
  tested, with or without the amplitude input.
- **A single-seed measurement on this dataset can overstate an effect,
  understate it, or report the wrong sign entirely** — demonstrated directly
  by re-running two close-margin claims at three seeds each, which reversed
  one conclusion and substantially narrowed another. Neither the amplitude
  fix's largest effects nor the overall best-configuration finding above are
  in that fragile category — they are an order of magnitude larger than the
  noise band this check established — but most other close-margin
  comparisons in this report have not been re-seeded and should be read with
  that caution.
- **Event class (magnitude ≥ M2.5 vs. below) can be predicted directly from a
  single 3-second window** (test accuracy 79.78%, AUC 0.855, MCC +0.566),
  clearly beating both a majority-class floor and a fitted amplitude/distance
  baseline — at a single seed and threshold, not yet re-verified.
- **On the three-class risk task (noise / M<4 / M≥4), the best model uses no
  image at all.** A two-stage gradient-boosted model over two physical
  scalars reaches 82.83% accuracy and MCC +0.704, against the CNN's 73.64%
  and +0.599 — and the CNN had those same scalars as inputs. The
  investigation behind that number found three defects (Section 8.4), each
  of which would have caused a wrong number to be *reported* rather than a
  crash: a stuck instrument supplying 58% of one class's errors, a
  validation split too station-poor to rank models (it selected the worse
  model by 0.25 MCC), and a missing-value pattern worth ten inflated
  accuracy points. This report's most transferable finding may be
  methodological rather than seismological: on this data, the failure mode
  is almost never a crash.
- **On peak ground motion, the network wins — and the paper's own target
  definition is why that is not obvious.** A Conv1D–BiLSTM–attention model
  beats the strongest scalar floor (amplitude + distance + station) by 0.075
  MAE_log at ~50× the seed spread, in both metric spaces, and the recurrent
  branch earns its parameters — the first architectural addition in this
  project to do so. But the paper's target window *contains* its own input
  window: the input's peak amplitude is a mathematical lower bound on that
  target in 100% of rows and exactly equals it in a third of them. On that
  degenerate target the network adds almost nothing (+0.011). The value
  appears on the corrected target and evaporates on the original. Under a
  doubly station- and event-disjoint split the margin survives in all three
  station partitions (mean +0.071), though roughly a quarter of the headline
  proved to be site familiarity, and partition variance was six times seed
  variance (Section 13).

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
transfer target. The central question this report answers is whether RAM
encoding, combined with a CNN and subsequently with the source paper's full
dual-channel CNN–LSTM architecture, can separate a genuine seismic event from
ambient noise directly from raw waveforms, how such a system compares to the
established STA/LTA trigger algorithm, and whether the same encoded-window
machinery extends to a related but distinct task: classifying an event's
magnitude directly from a short window rather than merely detecting its
presence.

This work was conducted independently, outside assigned project tasks, using
publicly available data (STEAD-format HDF5 chunks and self-downloaded FDSN
MiniSEED archives), and therefore did not require navigating data-access or
ethics constraints beyond standard public-data terms of use.

**Roadmap.** Sections 2–4 lay out the RAM transform, the data pipeline, and
the model architectures used throughout, so later results can be read without
backtracking to definitions. Section 5 defines the STA/LTA baseline,
including a parameterization defect found while preparing this report.
Section 6 is the detection investigation's results and analysis, organized by
research question rather than by the order experiments were run. Section 7
extends the same encoded-window machinery to magnitude classification.
Section 8 extends this to a three-class risk task (noise / low-risk /
high-risk) and reports where the encoded window stops helping at all.
Section 9 discusses the results as a whole; Section 10 states limitations
and concrete future work; Section 11 records the project's present status
and where it goes next. Section 12 is the full changelog of software defects
found and corrected, kept as a single reference list rather than scattered
through the narrative. The appendix collects every command needed to
reproduce every numbered result in the report.

---

## 2. The RAM Transform: Definition and Structural Properties

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
reconstructing $R$ in full; reconstruction error was exactly zero. A
64×64×3 RAM image (12,288 pixels) therefore carries at most 189 independent
values — a severe, lossy compression of the 6,000-sample (60 s) or
600-sample (6 s) input, on which a CNN's two-dimensional spatial processing
then operates on a highly redundant embedding of a one-dimensional signal.

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
offset). Property (d) is the most consequential fact in this report: Section
6.2 traces its consequences for detection, and Section 5.2 shows an
unrelated tool (STA/LTA's parameterization) breaking for a structurally
similar reason — a warm-up requirement colliding with a fixed buffer length.

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
the resulting image is correspondingly noisy. This is a geometric limitation
of the encoding, independent of model capacity or dataset size, and it
applies equally to classification and regression tasks. The tension has no
clean resolution within the current design: increasing $d$ requires
decreasing $n$ (a smaller image), and at 3 s a target resolution of $n=3$
would be required to reach 60 s-equivalent segment lengths. Reducing
`--target-n` was tried informally during early short-window work and stayed
within the same accuracy range as other short-window variants; no controlled
comparison at fixed seed exists, so this is reported as a qualitative
observation, not a numbered result.

### 2.4 Three-Channel Composition

Each component (Z, N/1, E/2) is RAM-transformed independently and stacked
into RGB channels: R = Z, G = N-like, B = E-like. Component selection is by
role rather than alphabetical ordering (Section 12, defect 13), and stations
lacking a usable vertical component are excluded. Whether combining the
three components before the transform (for example, by vector magnitude)
would preserve inter-channel amplitude relationships better than
transforming each independently was identified as an open question but not
pursued (Section 10).

---

## 3. Data Processing Pipeline

The pipeline comprises four stages, implemented in the `seismic_cli`
package of the `data_downloader` repository (`core.py`, `anchor.py`,
`eval_baseline.py`, `cli.py`). It is described here as it currently stands,
after the corrections in Section 12.

### 3.1 Acquisition (`seismic_cli/src/download.py`)

For each catalog event, stations within `SEARCH_RADIUS_DEG = 0.5°`
(approximately 55 km) are resolved via FDSN (KOERI), with lookups cached on
a roughly 1.1 km coordinate grid so co-located events share one metadata
query. All windows for an event are fetched in a single bulk request and
sliced in memory.

- **Earthquake windows:** 60 s from origin time.
- **Noise windows:** 300 s slices at −3 h and −6 h relative to origin.
- **Contamination check:** a noise window is discarded if any event in the
  unfiltered catalog falls within ±300 s of it — deliberately wide, since
  coda from larger events can persist for several minutes, and checked
  against the unfiltered catalog so a sub-threshold event cannot silently
  pass into the noise class. This check is purely temporal, so an event
  500 km away will veto a candidate noise window; this is over-conservative
  and discards noise data that is otherwise scarce (Section 10).

### 3.2 Arrival Anchoring (`seismic-cli anchor-windows`)

Short windows sliced from origin time can miss the P-wave arrival entirely
at distant stations — at 6 s, an arrival later than 6 s after origin time
means the nominal "earthquake" window contains no earthquake signal.
Anchoring re-derives short windows from already-downloaded 60 s data without
requiring re-download. The pick uses the classic STA/LTA characteristic
function

$$\text{CF}(k) = \frac{\frac{1}{n_{\text{STA}}}\sum_{j=k-n_{\text{STA}}+1}^{k} x_j^2}{\frac{1}{n_{\text{LTA}}}\sum_{j=k-n_{\text{LTA}}+1}^{k} x_j^2}$$

with the first `trigger_onset` crossing of `trigger_on = 3.5` taken as the
arrival sample $a$. The window is then cut as

$$[\,a - f\cdot T,\ a - f\cdot T + T\,), \qquad f = \texttt{pre\_arrival\_fraction} = 0.2$$

so that 20% of the window precedes the arrival (at 6 s: 1.2 s before, 4.8 s
after). This 20% figure resurfaces in Section 5.2: it is the buffer a
downstream STA/LTA *evaluation* has to respect, and the library's own
default parameterization does not.

A diagnostic block reports, per anchoring run, the number of stations seen,
skipped, picked on the vertical component, picked via fallback, and
unpicked, together with how close failed picks came to the trigger
threshold.

### 3.3 Dataset Generation (`seismic-cli generate-dataset` and variants)

Per window, per channel, processing proceeds: linear and constant detrend →
5% Hann taper → fourth-order Butterworth bandpass (1–45 Hz, zero-phase) →
encoding (RAM transform, spectrogram, or both, per Section 4) → output.
Dataset generation enforces five constraints:

**Station-disjoint splits, unified across classes.** Each station is
assigned to exactly one of train, validation, or test, and both its
earthquake and noise windows follow that assignment. With approximately 97%
of earthquake stations also contributing noise data, allocating the two
classes independently would let nearly every station appear in the training
split under one label and the test split under the other.

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
every remaining station — costly precisely where station diversity is
already scarce.

**Gap rejection.** Traces are merged without interpolation fill, so gaps
remain masked; gaps are then linearly filled for filtering purposes while a
boolean mask records which samples are synthetic. Any window whose worst
channel exceeds 5% synthetic samples is rejected.

**Per-station sampling rates.** Window sizing uses each station's own
sampling rate, recorded in the manifest, rather than assuming the first
trace's rate in a file applies to every station represented in it.

Output is a manifest.csv (columns vary slightly by encoder, but always
include `split, class_name or magnitude, station_key, file_path, filename,
fs`) sufficient to reconstruct the exact source samples behind any given
encoded window.

### 3.4 Baseline Standardization (`--baseline`)

Optionally, each channel may be standardized against that station's
long-term noise statistics $(\mu_{\text{sta}}, \sigma_{\text{sta}})$ —
accumulated in a streaming pass over all noise files for the given
(station, component) pair, requiring at least 60 s of usable data — instead
of the window's own $(\mu, \sigma)$. The intent was to give the pipeline the
long-term amplitude memory that constitutes STA/LTA's principal advantage.
Section 6.2 shows this does not achieve its intended effect, for a
structural reason tied directly to property 2.2(d).

---

## 4. Model Architectures

This section describes every model variant used in the report once,
comprehensively, rather than introducing pieces as results are reported.

### 4.1 `ImprovedSeismicCNN`: The Base Detector

A ResNet-style CNN with Squeeze-and-Excitation blocks and a single-logit
binary output (`src/sismokaos/detection/cnn_train.py`).

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

**Parameter budget** (64×64×3 input, measured directly):

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
is why the short preset omits it — a factor-of-4.0 reduction.

**Training procedure.** `BCEWithLogitsLoss` on label-smoothed targets
$\tilde{y} = 0.8y + 0.1$, which bounds the confidence the model is rewarded
for and introduces a floor on training loss (at the optimum, smoothed
per-sample cross-entropy equals $H(0.1) \approx 0.325$ nats, so it is not
directly comparable in magnitude to unsmoothed validation loss — an
unsmoothed training-loss diagnostic is logged alongside it). AdamW with
gradient-norm clipping at 1.0 and mixed precision on CUDA. Checkpointing
retains the best epoch by the monitored metric; final test evaluation always
loads that checkpoint.

**Window-length presets.** Short-window training runs overfit the full
network within roughly 10 epochs, while 60 s runs do not, so
`--window-seconds` selects a preset (12 s or shorter selects `short`; an
explicit flag always overrides the preset):

| | `long` (60 s) | `short` (≤ 12 s) | Rationale |
|---|---|---|---|
| Stages / head width | 4 / 64 | 3 / 32 | Reduces parameters from 1.25 M to 0.31 M |
| Dropout $p_1,p_2$ | 0.5, 0.3 | 0.6, 0.4 | Stronger regularization on the head |
| Weight decay | 1e-2 | 3e-2 | Stronger regularization |
| RandomErasing | Off | $p=0.25$ | The only label-safe augmentation available; flips or rotations would scramble the RAM matrix's temporal ordering, since axis position encodes time |
| Batch size / learning rate | 128 / 1e-4 | 64 / 2e-4 | More updates per epoch on a smaller dataset |
| Schedule | ReduceLROnPlateau | Cosine annealing | Plateau scheduling reacts too slowly when the performance peak arrives by roughly epoch 6 |
| Checkpoint metric | Validation loss | Validation AUC | Validation cross-entropy degrades from calibration drift while ranking quality continues to improve; loss-based selection risks saving a pre-peak model |

### 4.2 The Source Paper's Dual-Channel Architecture (1D2D-EDL)

Two independent channels operate on the same window and are fused:

$$\text{1D: } F_{1D} = \text{MSA}\big(\text{LSTM}(x)\big) \qquad
\text{2D: } F_{2D} = \text{CNN}\big(\text{RAM}(x)\big) \qquad
F' = a\,F_{1D} + b\,F_{2D}$$

with $a, b$ learned scalars. This is implemented as `LSTMAttentionBranch`
(bidirectional LSTM, followed by `nn.MultiheadAttention` with a residual
connection and layer normalization, matching a standard transformer block,
followed by mean-pooling over time) and `CNNBranch` (three convolution/
batch-norm/GELU stages with global average pooling, which is
resolution-agnostic and therefore accepts either a square RAM image or a
non-square spectrogram without modification). Both reside in
`cnn_earthquake/src/cnn_lstm.py`, shared with the unrelated
catalog-forecasting model (Section 11).

**A design error, found and corrected by consulting the source paper
directly rather than a prior summary of it.** The initial implementation
fed the LSTM branch the $(n, d)$ chunk matrix the RAM image's angle vector
$\beta$ is computed from, on the premise that both channels should observe
the same reshaped data. This does not match the source paper: Section 3.3.1
and Figure 7 of Wang and Zhao (2025) state that the one-dimensional time
series is normalized and then input directly into the LSTM; the RAM reshape
feeds only the CNN channel. The two channels are independent feature
extractors over the same raw signal, not two views of a shared intermediate
representation. `core.ram_matrix_and_chunks()`, which supplied the incorrect
design, was subsequently removed; `core.ram_matrix()` was verified to
produce output byte-identical to its pre-refactor version.

One consequence of the correction carries a computational cost:
multi-head self-attention over the full raw window is $O(m^2)$. At 100 Hz,
a 6 s window is $m=600$ (an attention matrix of 360,000 entries — trivial);
a 60 s window is $m=6000$ (36 million entries per attention head —
substantially heavier, likely requiring a smaller batch size). This is an
inherent cost of matching the source paper's design, not a defect.

### 4.3 Encoders (`data_downloader/seismic_cli/`)

| Encoder | Output | Description |
|---|---|---|
| `RamDualEncoder` | `{seq, img}` | `seq` is the raw standardized $(m,3)$ Z/N/E waveform; `img` is the Section 2 RAM image |
| `SpectrogramDualEncoder` | `{seq, img}` | Same `seq`; `img` is a log-power spectrogram, wrapping `SpectrogramEncoder` by composition |
| `RamAuxEncoder` | `{img, aux}` | `img` is the RAM image; `aux` is $[\log\text{SNR}, \log\text{RMS}]$ below; no LSTM branch |
| `RamDualAuxEncoder` | `{seq, img, aux}` | `RamDualEncoder` plus the auxiliary vector |

$$\log\text{SNR} = \left\langle \log\frac{\sigma_{\text{win},c}}{\sigma_{\text{sta},c}} \right\rangle_{c \in \{Z,N,E\}}
\qquad
\log\text{RMS} = \left\langle \log \sigma_{\text{win},c} \right\rangle_{c \in \{Z,N,E\}}$$

$\sigma_{\text{sta},c}$ comes from `compute_station_noise_baselines()`
(Section 3.4's mechanism), computed unconditionally here — independent of
`--baseline`, which controls only whether `seq`/`img` themselves use the
station baseline or per-window self-standardization. This is deliberate:
Section 6.2 establishes that RAM's image content does not depend on which
$(\mu,\sigma)$ pair standardizes it, so gating `log_snr` behind `--baseline`
would make the correction unavailable by default with no offsetting benefit.
A station lacking 60 s of usable noise data falls back to
$\log\text{SNR} = 0$. All encoders resample every window to a nominal rate
before standardizing, so every `seq` tensor has a fixed sample count
regardless of a station's native sampling rate.

A per-component variant of `aux` (six scalars — `log_snr`/`log_rms` for Z,
N, E separately, instead of averaged — `RamAuxEncoderV2` /
`RamDualAuxEncoderV2` / `SpectrogramDualAuxEncoderV2`, behind
`--per-component-aux`) is implemented and produces the expected `(6,)`-shaped
tensor, requiring no model-side change since `aux_dim` is read from the
tensor shape at load time. It has not been evaluated (Section 11).

### 4.4 Fusion and Auxiliary Model Variants

- **`DualChannelBinaryNet`** implements the source paper's architecture
  directly: `LSTMAttentionBranch` combined with `CNNBranch`, fused as
  $a F_{1D}+b F_{2D}$, with a single-logit head. `--channels {all,1d,2d}`
  ablates either branch.
- **`RamAuxCNN`** uses the same ResNet-with-Squeeze-and-Excitation trunk as
  `ImprovedSeismicCNN`, with pooled features concatenated with the
  auxiliary vector before the classification head. `--no-aux` removes only
  the concatenation, giving an architecture-matched control that isolates
  exactly what the two auxiliary scalars contribute.
- **`DualChannelAuxBinaryNet`** extends `DualChannelBinaryNet` with an
  auxiliary branch concatenated after the $aF_{1D}+bF_{2D}$ fusion step,
  the same pattern used by the unrelated catalog model's
  `DualChannelRiskNet` (Section 11). `--channels` extends to
  `{all, 1d, 2d, aux, 1d+aux, 2d+aux}`.
- **`GatedFusion`** replaces the fixed scalar pair with a per-example gate:
  $g = \sigma(\text{MLP}([F_{1D}, F_{2D}]))$, followed by
  $g F_{1D} + (1-g) F_{2D}$. Selected via `--fusion {linear, gate}`; affects
  only channel combinations where both branches are active. Gate-value
  diagnostics (mean by true class, and by prediction correctness) are
  reported at test time.
- **`cnn_lstm_stack.py` / `cnn_lstm_stack_aux.py`** implement late-fusion
  stacking: given two already-trained, frozen checkpoints (for example,
  `--channels 1d` and `--channels 2d` runs), collect their pre-sigmoid
  logits and fit `sklearn.LogisticRegression` — one weight per branch plus a
  bias, in logit space — on validation-set logits only, then evaluate on
  test. No backbone retraining is involved; the `_aux` variant does the same
  for `1d+aux`/`2d+aux` checkpoints, using TRAIN-only auxiliary
  standardization statistics carried over from each frozen checkpoint's own
  training run.

### 4.5 `RegressionSeismicCNN`: The Magnitude Classifier

Built for continuous magnitude regression (`cnn_earthquake/src/
cnn_regression.py`) and reused, unchanged, for magnitude *classification*
in Section 7. Shared CNN trunk (identical to `ImprovedSeismicCNN`), pooled
features concatenated with two auxiliary scalars —
$\log\text{SNR}$ and $\log(\text{distance}_{\text{km}})$, the classical
local-magnitude relation's two physical predictors — before a head ending in
a single un-squashed `nn.Linear(hidden\_dim, 1)`. `use_aux=False` reproduces
an image-only model, the honest ablation for "does the encoded window carry
magnitude information at all." Because the head is already a bare logit,
reusing this architecture for binary classification (Section 7) requires no
architectural change at all — only the loss function and reported metrics
differ.

---

## 5. Baseline Method: STA/LTA

### 5.1 Scoring Procedure

The baseline scores the same characteristic function as Section 3.2, taking
$\max_k \text{CF}(k)$ over each channel and the maximum across channels as
the window's score, with two properties that make the comparison fair:

- **Exact-window reconstruction.** Windows are rebuilt from the raw
  MiniSEED data via the manifest (same file, station, and window index),
  not resampled independently.
- **Window-adaptive parameters.** Fixed parameters (`sta=1.0`, `lta=10.0`)
  cannot be computed inside a 3 s or 6 s window. `derive_sta_lta_params`
  instead computes $\text{LTA} = \min(10,\ T/3)$ and
  $\text{STA} = \max(0.05,\ \text{LTA}/10)$, reproducing the classic 1/10
  ratio exactly at 60 s and giving 0.2 s/2.0 s at 6 s and 0.1 s/1.0 s at 3 s.

Reported metrics are AUC — threshold-free, the appropriate comparison
against the CNN — together with accuracy, precision, and recall at the
Youden's-J threshold, selected on the evaluated split, making those
thresholded figures an oracle upper bound rather than a like-for-like
comparison against the CNN's fixed 0.5 cutoff.

### 5.2 A Parameterization Defect, Found While Preparing This Report

`eval-sta-lta` had never actually been run to completion against any of the
Section 6 dual-channel datasets: its filename regex matched only
`_winNNN.png` and silently skipped every `_winNNN.pt` row the dual-channel
encoders write, producing "0 scores computed" without an obvious error.
Fixing the regex (Section 12, defect 14) let it run — and the first result
was AUC 0.5093 on the 6 s anchored test set, statistically indistinguishable
from random, which is implausible on its face given every other result in
this report treats STA/LTA as a strong short-window competitor.

The cause is a second, more consequential defect in the same function.
`classic_sta_lta`'s characteristic function is defined as exactly 0 for its
first `nlta` samples — no long-term average exists yet to divide by. The
auto-derived default at 6 s gives $\text{LTA}=2.0\text{s}=200$ samples. But
`anchor.py`'s default `pre_arrival_fraction=0.2` places the P-wave arrival at
only 1.2 s (120 samples) into the window — *inside* that forced-zero warm-up
region. The arrival is invisible to the characteristic function by
construction; the reported "max ratio" instead reflects whatever the
function sees later, after its own long-term average has already been
contaminated by the earthquake's own elevated amplitude, which pulls the
ratio back down toward 1. Measured directly on one representative window
(Z-like channel, KO.KULA, event 696188): amplitude jumps from
$\sigma\approx1313$ (pre-arrival) to $\sigma\approx13111$ (post-arrival) at
sample 120, a clean, strong onset — yet the characteristic function is
exactly 0.0 through sample 199, and its eventual maximum of only 1.21 occurs
at sample 356 (3.56 s in), long after the true onset and well into the
degraded region.

This is a general problem, not specific to this one window: for *any*
anchored window, $\text{LTA}_{\text{auto}} = T/3$ exceeds
$f \cdot T = 0.2T$ (the pre-arrival buffer) whenever $T/3 > 0.2T$, which is
true for every $T$ — the auto-derivation formula was simply never designed
with arrival-anchored windows in mind. It only escapes the problem once the
10 s cap on LTA takes over, at $T \gtrsim 50\text{s}$: at 60 s,
$\text{LTA}=10\text{s} < 0.2\times60\text{s}=12\text{s}$, so the buffer is
just barely sufficient — and separately, 60 s windows in this pipeline are
sliced from origin time (Section 3.1), not arrival-anchored at all, so the
question does not even arise there. Every *anchored* short window (3 s, 6 s,
10 s) is affected.

**Corrected measurement.** Following this project's own established
discipline of selecting configuration on validation and reporting only the
selected configuration's test performance, seven `(STA, LTA)` pairs with
$\text{LTA} \in \{0.1,\dots,0.5\}\text{s}$ were scored on the 6 s dataset's
*validation* split (never test) before any test-set number was examined:

| LTA (s) | Val AUC |
|---|---|
| 0.10 | 0.637 |
| 0.15 | 0.650 |
| 0.20 | 0.801 |
| 0.25 | 0.801 |
| **0.30** | **0.821** |
| 0.35 | 0.821 |
| 0.40 | 0.810 |
| 0.50 | 0.798 |

$\text{LTA}=0.3\text{s}$, $\text{STA}=0.03\text{s}$ (val AUC 0.8212) was
selected and evaluated once on test:

$$\textbf{STA/LTA, corrected: Test AUC 0.8194}, \quad \text{accuracy } 74.60\%,\ \text{precision } 73.90\%,\ \text{recall } 76.08\%\ \text{(oracle threshold)}$$

against the broken default's Test AUC 0.5093 — a difference of 0.31 AUC
between a working and a silently broken baseline on identical data. The
formula itself was left unchanged for un-anchored windows (Section 12,
defect 14 discussion) rather than auto-corrected, since 60 s results
depend on it reproducing its historical 1.0/10.0 output exactly; the
function now prints a runtime warning instead when its derived LTA exceeds
15% of the window, pointing at explicit `--sta-seconds`/`--lta-seconds`
overrides. Section 6.1 places this corrected number in the full
detection-results table.

---

## 6. Results: Detection (Earthquake vs. Noise)

All results in this section share a common dataset: `seismic-cli
generate-*-dataset --max` on 6 s arrival-anchored windows, 71,672 windows
total (35,836 per class, balanced), station-disjoint (82/30/40 earthquake
stations and 104/35/38 noise stations across train/validation/test), built
under every correction in Section 12. Code is in
`data_downloader/seismic_cli/` (`ram_dual.py`, `ram_aux.py`,
`spectrogram.py`, `eval_baseline.py`) and `cnn_earthquake/src/`
(`cnn_lstm.py`, `cnn_lstm_classify.py`, `cnn_lstm_classify_aux.py`,
`cnn_lstm_stack.py`, `cnn_lstm_stack_aux.py`, `cnn_ram_aux.py`). Unless
stated otherwise (Section 6.6), every configuration reflects a single
train/validation/test split at one random seed; differences under
approximately 1–2 points should be treated as noise rather than an
established effect.

An earlier phase of this investigation trained a plain RAM CNN at 60 s and
6 s before the corrections in Section 12 were made (89.61% accuracy /
STA/LTA AUC 0.7777 at 60 s; a 72–78% accuracy band across several 6 s
variants). Those runs predate fixes to cross-class station leakage, an
ineffective station cap, origin-anchored short windows, and the STA/LTA
DC-offset defect (Section 12, defects 1, 2, 4, 8), and are superseded by
this section; 60 s has not been re-run under the corrected pipeline, so no
current 60 s figure is reported. The three unrelated 6 s variants converging
on the same 72–78% band did, however, motivate both the model-capacity audit
in Section 4.1 and the structural analysis in Section 6.2.

### 6.1 Headline Comparison

| Model | Parameters | Test AUC | MCC | Accuracy |
|---|---|---|---|---|
| **Spectrogram CNN only (`2d`)** | 115,459 | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram-dual, fused, linear (`all`) | 182,563 | 0.9646 | 0.8122 | 90.61 % |
| RAM-dual + aux, fused, linear (`all`) | 182,759 | 0.9514 | 0.7790 | 88.95 % |
| RAM-dual + aux, no LSTM branch (`2d+aux`) | 115,655 | 0.9468 | 0.7775 | 88.84 % |
| RAM + aux, no dual architecture (`use_aux=True`) | 309,777 | 0.9230 | 0.7018 | 84.79 % |
| RAM-dual, raw waveform only (`1d`) | 76,707 | 0.9216 | 0.6849 | 84.22 % |
| RAM-dual, fused, linear, no aux (`all`) | 182,563 | 0.9144 | 0.6042 | 79.57 % |
| RAM-dual, RAM image only (`2d`) | 115,459 | 0.8408 | 0.5288 | 76.42 % |
| RAM CNN only, no aux (`use_aux=False`) | 309,713 | 0.8356 | 0.5339 | 76.70 % |
| **STA/LTA, correctly parameterized (Section 5.2)** | — | **0.8194** | — | 74.60 % |
| STA/LTA, library default (broken, Section 5.2) | — | 0.5093 | — | 56.88 % |

**Every CNN configuration tested beats a correctly-parameterized STA/LTA
baseline on this dataset, including the weakest one** (plain RAM, no
amplitude correction, 0.8356 vs. 0.8194). This is a materially different,
and more clear-cut, headline finding than the pre-correction phase's — where
STA/LTA's competitiveness at short windows was a live open question — because
that question rested partly on a baseline implementation that, it turns out,
was itself badly broken. The margin is modest against the weakest CNN
configurations and substantial (0.16 AUC) against the best one.

### 6.2 Why RAM Underperforms at Short Windows

Three causes compound:

1. **Geometric (Section 2.3).** At 6 s, local feature vectors are 10
   samples long; at 3 s, 5 samples. Cosine angles between very short vectors
   are noise-dominated, so the encoding degrades before any model observes
   it.
2. **Statistical.** Short-window datasets were small in early runs
   (roughly 1.5–4k samples) and, before the station-cap fix, drawn from very
   few distinct noise stations. A 1.25M-parameter network is heavily
   over-provisioned at that scale (roughly 300–800 parameters per sample),
   consistent with the early-peak overfitting observed then.
3. **Informational — the most consequential finding of the first
   investigation phase.** The single feature that most cleanly separates a
   short earthquake window from noise — amplitude relative to the station's
   background level — is provably absent from the model's input, by property
   2.2(d).

Consider what baseline standardization (Section 3.4) actually changes.
After `clean_and_filter_1d` (detrend, demean, bandpass), the window mean is
approximately 0, and $\mu_{\text{sta}}$ is likewise approximately 0, since it
is accumulated from identically cleaned noise data. The two standardization
modes therefore differ, to a close approximation, by a single factor
$\sigma_{\text{win}}/\sigma_{\text{sta}}$ — a pure rescaling, which RAM
cancels exactly. Measured on representative windows, comparing
self-standardized and baseline-standardized images at the same station:

| Event strength | $\sigma_{\text{win}}/\sigma_{\text{noise}}$ | Mean pixel difference | Maximum |
|---|---|---|---|
| Weak (SNR ≈ 2) | 1.84 | 0.63 / 255 levels | 3 |
| Strong (SNR ≈ 20) | 19.19 | 0.27 / 255 levels | 1 |

The differences are sub-level and, decisively, do not grow with the
amplitude ratio — a 20-times stronger event produces a slightly smaller
image difference, not a larger one. The residual originates from the
near-zero post-filter mean (RAM is shift-sensitive, property 2.2(d)), not
from amplitude.

Two conclusions follow. First, **`--baseline` cannot deliver its intended
effect** — the amplitude information it is designed to preserve is
eliminated by the very next pipeline step. Second, **this explains STA/LTA's
competitiveness**: its entire discriminative signal is amplitude measured
against a long-term baseline, precisely the quantity RAM structurally cannot
represent. Section 6.3 develops the fix.

### 6.3 The Amplitude Auxiliary Input

The correction is not a further image variant but an auxiliary scalar input
(Section 4.3's `aux`), concatenated to pooled CNN features before the
classification head, restoring the discarded quantity without altering the
transform itself.

**On the RAM classifier alone, this is the single largest effect measured in
this entire investigation.** Architecture-matched, single-variable
comparison (`RamAuxCNN`, differing only in whether `aux` is concatenated):

$$\text{AUC: } 0.8356 \to 0.9230 \quad(+0.0874) \qquad \text{MCC: } 0.5339 \to 0.7018 \quad (+0.1679) \qquad \text{Accuracy: } 76.70\% \to 84.79\% \quad (+8.09\text{ pp})$$

Two scalars, computed at negligible cost from data the pipeline already
possessed, closed most of the gap between a plain RAM classifier and
STA/LTA-competitive performance (and, per Section 6.1, comfortably past it).

**On the raw-waveform LSTM branch, the same fix helps almost as much, for
the same underlying reason.** `--channels 1d+aux` (raw standardized
waveform plus `aux`, no 2D branch):

$$\text{Test AUC: } 0.9216 \to 0.9501 \quad(+0.0285) \qquad \text{MCC: } 0.6849 \to 0.7675 \quad(+0.0826) \qquad \text{Accuracy: } 84.22\% \to 88.37\% \quad(+4.15\text{ pp})$$

The raw waveform is standardized before entering the LSTM, which removes
absolute amplitude exactly as RAM's own internal standardization does — the
branch is scale-blind for the same structural reason the RAM image is, and
benefits from the same correction. Running the identical `1d+aux`
configuration against the spectrogram-plus-amplitude dataset (which shares
the same `seq`/`aux` computation and never reads `img`) reproduces this
result to four decimal places (88.37% / 0.9501 / 0.7675) — a pipeline
correctness check, not a second independent measurement.

**On the spectrogram-based 2D branch, the fix does not repeat its RAM-side
effect, and mildly hurts in isolation.**

| Configuration | Test AUC | MCC | Accuracy |
|---|---|---|---|
| Spectrogram 2D only, no aux | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram + aux, no LSTM branch (`2d+aux`) | 0.9749 | 0.8626 | 93.02 % |

A station-normalized spectrogram already encodes amplitude as a function of
time and frequency; appending two collapsed, redundant scalars adds
estimation noise without adding information the model did not already have.
Where the RAM result showed the auxiliary input matters when the 2D
representation is scale-invariant, this shows the converse: it does not
matter, and can mildly hurt, when the 2D representation already carries the
same information in richer form. This is also why the amplitude-corrected
RAM 2D branch (0.9468 AUC, `2d+aux`) still falls well short of the plain
spectrogram (0.9793): the correction restores what RAM structurally
discarded, but two collapsed scalars are a much thinner representation than
a station-normalized spectrogram's amplitude-as-function-of-time-and-frequency.
Where RAM images are specifically required (for example, to match the
source paper's method), pairing them with the amplitude auxiliary input is
a substantial, low-cost improvement; where the 2D representation is an open
choice, spectrograms remain the stronger option on this dataset.

A seed-repeated check of the aux effect on the *fused* spectrogram-dual
model changes this section's original single-seed conclusion; see Section
6.6.

### 6.4 Dual-Channel Fusion Mechanisms

**Linear fusion underperforms the best single branch, on two independent 2D
representations.** With RAM: the raw-waveform branch alone (AUC 0.9216)
outperforms the fused model (0.9144). With a spectrogram: the image branch
alone (0.9793) outperforms the fused model (0.9646). A fixed pair of
scalars, trained jointly with both branches, cannot suppress a weaker branch
on the specific examples where it is wrong, and joint training can let a
noisy branch degrade the stronger branch's own learned representation. This
reproduced on two independent 2D representations, so it is treated as a
property of the fusion mechanism, not of either branch.

**Late-fusion stacking recovers what joint fusion lost, without modifying
either branch.** Freezing the same two checkpoints and fitting a two-input
logistic regression on their logits matched or improved on joint fusion in
both cases:

| Base checkpoints | 1d alone | 2d alone | Naive average (logits) | Stacked |
|---|---|---|---|---|
| RAM-dual | AUC 0.9229, MCC 0.688 | AUC 0.8408, MCC 0.530 | AUC 0.9136, MCC 0.692 | AUC 0.9203, MCC 0.688, 84.31 % |
| Spectrogram-dual | AUC 0.9229, MCC 0.688 | AUC 0.9793, MCC 0.867 | AUC 0.9697, MCC 0.866 | AUC 0.9743, MCC 0.871, 93.54 % |

On RAM-dual, stacking lands within measurement noise of the best single
branch rather than below it. On spectrogram-dual, stacking outright beats
both single branches (0.9743 AUC, 0.871 MCC vs. 2d-alone's 0.9793 AUC,
0.867 MCC — AUC marginally below 2d-alone even as MCC and accuracy are
higher, a small trade of ranking quality for a better fixed-threshold
decision, within this dataset's noise floor). This confirms the fusion
problem originates in joint training, not in an inherent limitation of
combining the two branches.

**Applied to the amplitude-augmented branches, stacking beats both linear
and gated joint fusion, on both datasets:**

| Dataset | 1d+aux alone | 2d+aux alone | Stacked | Linear fusion (`all`) | Gated fusion (`all`) |
|---|---|---|---|---|---|
| RAM+aux | 0.9505 / 0.768 / 88.39 % | 0.9468 / 0.778 / 88.84 % | **0.9557 / 0.781 / 89.07 %** | 0.9514 / 0.779 / 88.95 % | 0.9487 / 0.744 / 87.12 % |
| Spectrogram+aux | 0.9505 / 0.768 / 88.39 % | 0.9749 / 0.863 / 93.02 % | **0.9758 / 0.868 / 93.37 %** | 0.9733 / 0.847 / 92.31 % | 0.9716 / 0.836 / 91.80 % |

(cells are AUC / MCC / accuracy). Stacking on RAM+aux is a clear win over
every alternative on every metric; on spectrogram+aux it is the best fusion
mechanism tested but still does not surpass the plain `2d`-alone, no-aux
ceiling (0.9793 AUC) — no fusion mechanism has closed that gap.

**Gated fusion is mixed across representations, and does not compose with
the amplitude fix.**

| Configuration | Linear fusion | Gated fusion | Best single branch | Mean gate $g$ |
|---|---|---|---|---|
| Spectrogram-dual | 0.9646 / 0.812 / 90.61 % | **0.9761 / 0.850 / 92.51 %** | 2d-alone: 0.9793 / 0.867 / 93.28 % | 0.169 |
| RAM-dual, no aux | 0.9144 / 0.604 / 79.57 % | 0.9071 / 0.637 / 81.68 % | 1d-alone: 0.9216 / 0.685 / 84.22 % | 0.719 |
| RAM-dual + aux | 0.9514 / 0.779 / 88.95 % | 0.9487 / 0.744 / 87.12 % | 2d+aux-alone: 0.9468 / 0.778 / 88.84 % | 0.487 |

On spectrogram-dual, gated fusion improves all three metrics. On RAM-dual
without aux, it is mixed: AUC drops (0.9144→0.9071) while MCC and accuracy
both rise (0.604→0.637; 79.57%→81.68%), a shift in decision calibration
rather than a uniform ranking change. On RAM-dual with aux, gated fusion is
uniformly worse. An initial hypothesis — that gating helps when branches are
genuinely complementary and can hurt when one branch dominates — does not
survive the third configuration: the branch gap on RAM-dual without aux
(0.081) is larger than on spectrogram-dual (0.058), yet gating still
underperformed there. The more accurate statement is that gated fusion
improved results in exactly one of three tested configurations — the
spectrogram-based one — and modestly degraded results in the two RAM-based
ones, with no established explanation distinguishing genuine representation
dependence from noisy gate estimation on a comparatively harder branch pair.
**Gated fusion is not a default improvement over linear fusion and should be
measured directly for any new branch pairing.**

Combining gated fusion with the amplitude input (spectrogram-dual)
demonstrates this concretely: the result (AUC 0.9716) is worse than either
intervention alone (gated-without-aux 0.9761; linear-with-aux 0.9733).
Independently-validated improvements did not compose here; the three
mechanisms rank, by AUC, stacked (0.9758) > linear (0.9733) > gated (0.9716)
on the amplitude-augmented model — the reverse of gated fusion's ranking on
the same 2D representation without aux, where it was best.

Learned linear-fusion weights $(a,b)$, for reference: RAM-dual without aux,
$(0.996, 0.829)$; spectrogram-dual, $(0.749, 1.051)$; RAM-dual with aux,
$(0.499, 0.354)$ — markedly more balanced once both branches carry usable
information, rather than one weight sitting near 1 with the other trailing.

### 6.5 Hyperparameter Sensitivity

Two sweeps, each selected on validation AUC before training began (never on
test), specifically to avoid selecting a configuration by its test
performance across candidates.

**RAM-plus-amplitude classifier** (`cnn_ram_aux.py`), six configurations —
the default, two alternative learning rates, lower weight decay, a wider
head, and the original 4×-capacity preset:

| Configuration | Parameters | Validation AUC | Test AUC | Test MCC | Test accuracy |
|---|---|---|---|---|---|
| Default | 309,777 | 0.9302 | 0.9230 | 0.7018 | 84.79 % |
| Learning rate 3e-4 | 309,777 | **0.9307** | 0.9268 | 0.7137 | 85.28 % |
| Weight decay 1e-2 | 309,777 | 0.9301 | 0.9228 | 0.7013 | 84.77 % |
| Learning rate 1e-4 | 309,777 | 0.9287 | 0.9224 | 0.7040 | 84.96 % |
| Four stages, wider head (4× capacity) | 1,249,425 | 0.9287 | 0.9302 | 0.7235 | 86.13 % |
| Wider head only (three stages) | 314,001 | 0.9268 | 0.9270 | 0.7054 | 85.17 % |

All six fall within a validation-AUC band of 0.9268–0.9307 (spread 0.0039),
inside the ~1–2 point single-seed noise floor. The nominal winner by the
pre-specified rule beats the default by 0.0005 — not distinguishable from
noise. Worth noting for methodological reasons: the 4×-capacity
configuration scored *lower* on validation (0.9287) than the 310k-parameter
default despite scoring *higher* on test (0.9302) — a direct illustration of
why selection was fixed to validation before training, and evidence this
task is limited by available information rather than model capacity
(Section 6.2).

**`LSTMAttentionBranch` itself** (depth, attention heads, hidden width),
five configurations, isolated via `1d+aux`:

| Configuration | Parameters | Validation AUC | Test AUC | Test MCC | Test accuracy |
|---|---|---|---|---|---|
| Default (1 layer, 4 heads, hidden 48) | 76,903 | 0.9574 | **0.9501** | **0.7675** | **88.37 %** |
| 2 attention heads | 76,903 | 0.9576 | 0.9485 | 0.7520 | 87.59 % |
| 8 attention heads | 76,903 | 0.9586 | 0.9495 | 0.7513 | 87.53 % |
| Hidden width 64 | 123,815 | **0.9588** | 0.9488 | 0.7508 | 87.52 % |
| 2 LSTM layers | 132,967 | 0.9565 | 0.9484 | 0.7486 | 87.33 % |

All five fall within an even tighter validation band (0.9565–0.9588, spread
0.0023). More notably, **the default has the best test MCC and accuracy of
all five candidates despite ranking fourth of five on validation AUC** —
every configuration this sweep would have nominally preferred by its own
selection rule scores worse on held-out test across MCC and accuracy. This
sharpens the RAM-CNN sweep's point: validation-based selection is a
discipline against choosing by looking at test results directly, not a
guarantee against picking a configuration that generalizes worse. **Neither
sweep found a productive direction for tuning; the established defaults
should be retained in both cases**, which narrows the plausible location of
any remaining performance gap toward feature representation rather than
architecture search on either branch.

### 6.6 Statistical Reliability: Seed-Repeated Verification

Two close-margin claims from Sections 6.3–6.4 were re-run at two additional
seeds (1 and 2, alongside the original 42) to check whether they survive
repetition.

**Gated vs. linear fusion, spectrogram-dual, no aux:**

| Seed | Linear AUC / MCC / Acc | Gated AUC / MCC / Acc | Gated − Linear (AUC) |
|---|---|---|---|
| 42 | 0.9646 / 0.812 / 90.61 % | 0.9761 / 0.850 / 92.51 % | +0.0115 |
| 1 | 0.9719 / 0.848 / 92.32 % | 0.9753 / 0.851 / 92.53 % | +0.0034 |
| 2 | 0.9746 / 0.834 / 91.68 % | 0.9720 / 0.849 / 92.36 % | **−0.0026** |

The AUC advantage reverses sign at seed 2, and is roughly a third of the
original figure at seed 1. Averaged across seeds, gated fusion still leads
on AUC (mean 0.9745 vs. 0.9704), but the effect is smaller and noisier than
Section 6.4's original framing suggested. Accuracy and MCC tell a more
consistent story: gated fusion wins on **both** metrics at **all three**
seeds, with much lower run-to-run spread (accuracy 92.36–92.53%, spread
0.17 points, vs. linear's 90.61–92.32%, spread 1.71 points; MCC spread 0.002
vs. 0.036). **The corrected finding is not "gated fusion is a clear AUC
win" but "gated fusion gives more consistent, slightly better decisions at
the operating threshold, with a real but small and noisy ranking-quality
advantage"** — materially weaker and more precise than the original framing.

**Amplitude aux vs. no aux, spectrogram-dual, linear fusion:**

| Seed | No-aux AUC / MCC / Acc | Aux AUC / MCC / Acc | Aux − No-aux (AUC) |
|---|---|---|---|
| 42 | 0.9646 / 0.812 / 90.61 % | 0.9733 / 0.847 / 92.31 % | +0.0087 |
| 1 | 0.9719 / 0.848 / 92.32 % | 0.9707 / 0.841 / 92.01 % | **−0.0012** |
| 2 | 0.9746 / 0.834 / 91.68 % | 0.9705 / 0.834 / 91.66 % | **−0.0041** |

This result does not merely shrink — it reverses and averages to
approximately zero (mean +0.0011 across seeds). **Section 6.3's claim that
the amplitude input improves the fused linear spectrogram model does not
survive repetition; the seed-42 result was not representative.** This does
not call the amplitude fix into question generally — the RAM-alone effect
(+0.087 AUC) and the `1d+aux` effect (+0.029 AUC) are an order of magnitude
larger than the ~0.01 AUC noise band this check establishes, and are far
more likely to be genuine, though neither has itself been re-seeded. What
this specifically corrects is narrower: aux does not reliably help once
combined with joint linear fusion on an already-strong 2D representation,
where Section 6.3 already showed the 2D branch has little room left for it
to add.

**The practical lesson is broader than either individual correction: none of
the other close-margin claims in this section — including the `1d+aux` and
`2d+aux` effects, and the stacking results — have been checked against more
than one seed, and this check demonstrates concretely that a single-seed
result on this dataset can overstate an effect, understate it, or report the
wrong sign entirely.** Only effects well above the ~0.01 AUC / ~0.02 MCC
band established here — the amplitude fix on RAM alone, and the overall
best-configuration finding below — should be treated as settled without
further seeds.

### 6.7 Summary

**Considered together, the single best-performing configuration measured in
this entire detection investigation remains the plain spectrogram CNN
classifier — no LSTM branch, no auxiliary input, no fusion mechanism (AUC
0.9793) — and it beats a correctly-parameterized STA/LTA baseline (0.8194)
by 0.16 AUC.** Every structural addition tested — the dual-channel
architecture, the amplitude auxiliary input, gated fusion, late-fusion
stacking, and combinations of these — improved on some other configuration
along the way and produced a genuine, informative finding about why RAM
underperforms and how fusion mechanisms behave, including where those
findings needed correcting under repeated measurement. None of them,
individually or combined, has yet exceeded the simplest model in this
comparison. This is a genuine result, not a failure of the investigation: it
indicates the highest-value remaining work is more likely in feature
representation (per-component auxiliary inputs, spectrogram parameters) than
in additional architectural complexity layered on an already-strong 2D
representation.

---

## 7. Results: Magnitude Classification from a 3-Second Window

### 7.1 Motivation

The detection task (Sections 5–6) asks whether a window contains an
earthquake at all. This section asks a narrower, related question directly
motivated by the same amplitude-invariance finding (Section 6.2): can
*event class* — a coarse magnitude bin — be read directly off a single
3-second window at the P-wave arrival? This reuses machinery already built
for a different purpose: `regression.py` and `cnn_regression.py`
(Section 4.5) implement continuous magnitude regression, complete with the
manifest already storing per-window magnitude, log-SNR, and epicentral
distance, but had never been run to completion or reported on before this
section. Since `RegressionSeismicCNN`'s head is already a bare logit,
classification requires no architecture change — only a relabeled target,
a swapped loss, and swapped metrics.

### 7.2 Data

`data/batched_waveforms/window_post_3s_anchored` holds 23,918
P-wave-anchored 3 s events, already downloaded for the detection work
(confirmed: each mseed file is exactly 300 samples at 100 Hz = 2.99 s). Of
the catalogs on disk, only `catalogs/archive_superseded_2026-08-30/deprem_katalog_utc.csv` (482,898 rows)
matches 100% of these events by EventID (the smaller, curated
`catalogs/data.csv` matches only 6.6%).

The magnitude distribution across all 23,918 events is heavily right-skewed,
as expected (Gutenberg–Richter): 85.8% are M2.0–3.0, and only 37 events
(0.2%) are M≥5. This rules out fine physically-meaningful bins (a "large
event" class would have only a few dozen examples) and weighs against pure
quantile bins (boundaries would sit within a quarter magnitude unit of each
other — balanced but not physically meaningful). A **binary** split at
**M = 2.5** was chosen: closer to balanced (62.4% below / 37.6% at-or-above)
than the more standard M = 3.0 cut (85.7%/14.3%), while remaining a single,
round, physically interpretable number rather than a value fitted to this
catalog; it is a CLI flag (`--mag-threshold`), not hardcoded. Each mseed
file corresponds to exactly one event (23,918 files, 23,918 unique
EventIDs), so `split_by="event"` (already the default in `regression.py`)
is the correct leakage guard when an event is recorded at multiple stations.

`seismic-cli generate-regression-dataset --eq-dir
window_post_3s_anchored --catalog-path deprem_katalog_utc.csv
--window-seconds 3 --encoding spectrogram --split-by event` produced 37,289
labelled windows: train 26,101 (M2.0–5.9, 36.9% positive, 16,685 events),
val 5,594 (M2.0–6.2, 35.8% positive, 3,639 events), test 5,594 (M2.0–5.5,
38.6% positive, 3,594 events). Verified: zero EventIDs appear in more than
one split.

### 7.3 Result

`cnn_magclass.py` (Section 4.5's `RegressionSeismicCNN`, `BCEWithLogitsLoss`
in place of `L1Loss`), short preset (3 residual stages, hidden width 32),
314,001 parameters, trained to early stopping (best epoch 4/80, val AUC
0.864; stopped at epoch 14 on 10 flat epochs):

| | Accuracy | AUC | MCC |
|---|---|---|---|
| Predict majority class (always "< M2.5") | 61.35% | — | — |
| Logistic regression on $\log\text{SNR}$ + $\log(\text{distance})$ alone | 71.36% | 0.7486 | +0.372 |
| **CNN (spectrogram + aux)** | **79.78%** | **0.8550** | **+0.566** |

The full model beats the majority-class floor by 18.4 points and, more
importantly, beats the amplitude/distance-only logistic baseline by +0.106
AUC — the 3-second window's spectrogram carries real magnitude-class signal
beyond what a simple "how loud, how far" rule already captures, directly
analogous to Section 6.3's finding that the encoded window contributes
beyond amplitude+distance for the *detection* task.

### 7.4 Caveats

This is a single run at one seed and one threshold (M = 2.5) — the same
caveat Section 6.6 demonstrated matters concretely for this project's other
close-margin claims; a seed-repeat check has not been run here. The
spectrogram input to the CNN was small — `(3, 129, 5)`, only 5 time frames
from a 3 s window at the default FFT settings — so most of the model's
signal is coming from very few effective time steps, not rich temporal
structure. Neither a per-component-aux variant nor a threshold-sensitivity
sweep has been tried.

### 7.5 A Dual-Channel Extension: Spectrogram + Raw-Waveform LSTM

A natural follow-up to 7.1–7.4: does pairing the spectrogram with a second
branch reading the raw waveform through an LSTM + multi-head self-attention
— this project's own dual-channel architecture (Section 4.2, Wang & Zhao
2025, already used for detection) — add anything to continuous magnitude
regression? No paper surveyed for this project combines the two for
magnitude specifically (the closest, Shen et al. 2025, is pure 1D CNN+Bi-LSTM
with no image channel at all); this retargets an architecture this project
already has at a task it had not been tried on.

`seismic-cli generate-regression-dataset --dual` reuses the SAME
event-disjoint split and window construction as 7.2, adding a `--dual` flag
that swaps in the detection pipeline's already-built `SpectrogramDualEncoder`
— which already implements the identical per-window encoder protocol the
regression orchestrator calls — instead of `SpectrogramEncoder`, so no new
dataset-generation code was needed for this, only the choice of encoder. The
comparison to 7.3 is therefore apples-to-apples except for the added
raw-waveform channel.

**Result, 3 seeds** (`cnn_lstm_regression.py`, same `LSTMAttentionBranch` and
learned-fusion pattern as `cnn_lstm.py`'s `DualChannelRiskNet`, regression
head in place of the classification one):

| Model | MAE (mean of 3 seeds) | seed range |
|---|---|---|
| Single-channel spectrogram + aux (7.3's architecture) | 0.205 | 0.204–0.206 |
| Dual-channel, all branches | 0.202 | 0.202–0.203 |
| *floor:* ridge(log_snr, log_distance) | 0.308 | — |

The dual-channel model ties (very slightly edges) the single-channel one —
a 0.003 MAE difference sitting inside the seed-to-seed spread of either
model, not an established effect (Section 6.6's standing caution about
close margins on this dataset applies here too). Both clear the ridge floor
by essentially the same margin (~0.10 MAE).

**Branch ablation, same CNN backbone, isolates what each channel
contributes** (single seed):

| Channels | MAE |
|---|---|
| 2D (spectrogram) + aux, no LSTM branch | **0.197** |
| All (1D + 2D + aux) | 0.202 |
| 1D (raw waveform) + aux, no spectrogram branch | 0.250 |

**The spectrogram-only branch is the best model tested; adding the LSTM
branch makes the result slightly worse, not better.** This is the same
conclusion Section 6's detection work reached (no architectural addition
beat the single best branch), now reproduced on magnitude regression. The
full model's own learned fusion weights already pointed this way across
every seed tried: the 2D branch's weight (0.89–0.94) consistently exceeded
the 1D branch's (0.78–0.84).

### 7.6 Caveats (7.5)

The branch ablation is a single seed, not three — reported without a
multi-seed repeat because its direction (the 2D branch alone beating the
joint-fusion model) matches every other fusion-vs-single-branch comparison
in this project (Section 6.4, Section 8.3), not because it has been
independently verified to the same standard as the 3-seed headline number
above it. The dual-channel dataset (`--dual`) writes both a spectrogram and
a raw-waveform tensor per window, roughly doubling storage and per-window
encoding cost for a model that, on this evidence, should not be built this
way in the first place — 7.3's plain single-channel spectrogram+aux CNN,
or 7.5's own 2D-only ablation, is the model to use for magnitude regression
from a short window, not the dual-channel one.

### 7.7 Pushing on 0.197: Finer Time Resolution and Per-Component Amplitude

7.5's branch ablation left an unresolved single-seed number (2D+aux,
0.197 MAE) and two structural gaps worth closing before treating it as a
ceiling: the spectrogram has almost no time resolution (`hop_length`
defaults to `n_fft // 4 = 64`, giving only 5 frames from a 3 s window,
never exposed as a CLI flag), and the aux vector collapses Z/N/E into one
averaged `log_snr` rather than the per-component form
(`ram_aux.RamAuxEncoderV2`) already validated elsewhere in this project.
Both were added as CLI flags (`--hop-length`, `--per-component-aux` on
`generate-regression-dataset`) and tested against the identical
event-disjoint split (51,408 windows; 35,984/7,711/7,713;
distance_km present for 43,341/51,408 rows either way) so the only
difference between runs is the lever under test.

**First, 7.5's own headline number gets a 3-seed confirmation it never
had** — the 0.197 MAE in the table above was a single seed:

| hop_length | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| 64 (unchanged default) | 0.197 | 0.197 | 0.198 | 0.1973 |

Confirmed: 2D+aux at the default hop length is **0.1973 MAE, 3-seed mean**,
not just a lucky single draw.

**Hop-length sweep** (`--channels 2d+aux`, single-seed screening first):

| hop_length | time frames | MAE (seed 42) |
|---|---|---|
| 64 (baseline) | ~5 | 0.197 |
| 32 | ~10 | 0.195 |
| 16 | ~19 | 0.197 |

hop=32 screened best, so it got the same 3-seed confirmation as the
baseline, paired seed-for-seed against it:

| seed | hop=64 | hop=32 | Δ |
|---|---|---|---|
| 42 | 0.197 | 0.195 | −0.002 |
| 43 | 0.197 | 0.196 | −0.001 |
| 44 | 0.198 | 0.197 | −0.001 |
| **mean** | **0.1973** | **0.1960** | **−0.0013** |

hop=32 was nominally better in all three paired seeds, but by only
0.001–0.002 MAE — a margin the same size as the within-configuration seed
spread (hop64: 0.197–0.198; hop32: 0.195–0.197). Too small to matter for
any downstream use, and not the kind of gap this project treats as an
established effect (Section 6.6). hop=16 was not carried to 3 seeds: its
single-seed screening result (0.197) already sat behind hop=32 with no
indication finer-than-32 resolution helps further.

**Per-component aux**, layered on the hop=32 dataset (single-seed
screening, since neither lever showed a 3-seed-worthy signal to build on):

| channels | aux | MAE (seed 42) |
|---|---|---|
| 2d+aux | averaged (2 scalars) | 0.195 |
| 2d+aux | per-component (4 scalars: log_snr × Z/N/E + log_distance) | 0.196 |
| all | per-component | 0.205 |

Per-component aux made no measurable difference to the 2D+aux model
(0.195 → 0.196, inside noise) — the ridge floor moved slightly
(0.308 → 0.306 MAE, reflecting that the per-component signal is real) but
the CNN was already recovering that information from the station-normalized
spectrogram itself, so giving it explicitly bought nothing further. The
`all`-channels model with the richer aux still lost to 2D+aux (0.205 vs.
0.196) — the same LSTM-branch-hurts finding from 7.5, now re-checked with a
richer aux vector in case that changed which branch earned its parameters.
It didn't.

**Conclusion: neither lever moved the needle.** The ceiling from 7.5
(0.197 MAE, single seed) is now a 3-seed-confirmed **0.1973 MAE**, and the
best configuration found in this section (hop=32, 2D+aux, averaged aux) is
**0.1960 MAE** — a real but negligible refinement, not a second win. For a
magnitude regressor from a *3-second* window, this project has not found an
architectural or spectrogram-resolution change that beats the plain
2D+aux spectrogram CNN by a margin worth acting on. 7.8 asks the same
question a different way — not finer resolution of the same 3 seconds, but
more seconds — and gets a very different answer.

### 7.8 Window Length: 3s vs 6s

7.7's hop-length sweep bought finer time *resolution* of the same 3-second
window and got nothing (0.1973 → 0.1960 MAE, inside noise). This section
asks whether more time *extent* — the window itself doubled to 6s, using
the same event-anchored data (`window_post_6s_anchored`, already used for
detection in Section 6) and the identical 2D+aux architecture from 7.5/7.7,
default `hop_length` — does any better.

A useful control falls out of 7.7 for free: 6s at `hop_length=64` produces
`1 + 600 // 64 = 10` time frames, the *same* frame count as 3s at
`hop_length=32` (`1 + 300 // 32 = 10`). If time resolution were the active
ingredient, these two configurations should score about the same. They do
not.

**Result, 3 seeds, `--channels 2d+aux`:**

| window | frames (at its hop) | seed 42 | seed 43 | seed 44 | mean MAE |
|---|---|---|---|---|---|
| 3s (hop=64, 7.5/7.7 baseline) | 5 | 0.197 | 0.197 | 0.198 | 0.1973 |
| 3s (hop=32, 7.7's best) | 10 | 0.195 | 0.196 | 0.197 | 0.1960 |
| **6s (hop=64)** | **10** | **0.182** | **0.182** | **0.181** | **0.1817** |

Same frame count as 3s/hop=32, but 6s beats it by 0.014 MAE — roughly ten
times the gain (or lack of one) that finer resolution of the same 3 seconds
produced. **The lever that works is more waveform, not a better picture of
the same waveform.** 7.7 and 7.8 together are one result: resolution was
tested and ruled out first, so the 6s gain cannot be attributed to it.

The two 3-second models and the two 6-second seeds are each internally as
tight as anything reported in this project (3-seed spread ≤0.001), so this
is not a seed-noise artifact.

**The 6-second test set is a *different*, and if anything *harder*,
population — which cuts in favour of the result, not against it.** Fewer
events have a full 6 seconds of clean post-arrival data than 3, so the
dataset shrinks (35,836 windows / 16,247 train events vs. 51,408 windows /
21,870 train events) and both floors get *worse*, not better:

| window | predict-the-mean | ridge(log_snr, log_distance) | 2D+aux model |
|---|---|---|---|
| 3s | 0.350 | 0.308 | 0.1973 |
| 6s | 0.376 | 0.318 | 0.1817 |

Both floors move against the model — the 6s test population is not simply
"easier." The model's margin over the ridge floor still widens (+0.111 at
3s vs. +0.136 at 6s; model/ridge ratio 0.641 → 0.571), a floor-controlled
comparison on each population's own split, which is the strongest evidence
here that the extra 3 seconds carries real information rather than an
easier draw of events.

**A physical candidate mechanism, not just "more data helps":** for
near-station recordings, the S arrival typically lands within a few seconds
of the P pick (S–P moveout ≈ distance / 8 km/s, the same relation Section
13's ground-motion work uses); a 3-second window anchored on the arrival
plausibly captures P and early coda only, while 6 seconds crosses the S
arrival for a meaningful fraction of stations — and S-wave amplitude is
where most of an event's radiated energy, and its correlation with
magnitude, actually is. This is offered as a plausible mechanism, not
verified here (would require per-window S–P timing, not attempted).

**Branch choice re-checked at 6s, not just carried over:** the raw-waveform
branch now sees a 600-sample sequence instead of 300 (self-attention's cost
is quadratic in that length), so 2D+aux beating the LSTM branch at 3s did
not guarantee the same at 6s. It does: `--channels all` at 6s (single seed
42) scores 0.189 MAE — worse than 2D+aux's 0.182 — with near-balanced
learned fusion weights (1D +0.904, 2D +0.916). Same conclusion, re-verified
at the new window length rather than assumed.

**Remaining caveats.** Distance coverage is proportionally similar
(30,496/35,836 = 85.1% vs. 43,341/51,408 = 84.3%), so the missing-distance
mechanism (Section 7.2) is not obviously different between the two window
lengths. Hop-length was not re-swept at 6s; 7.7's finding that finer
resolution of a fixed window does nothing does not guarantee the same
holds once the window itself is longer. `window_post_10s_anchored` already
exists in this project's data and is the obvious next point on this
curve, but the anchored-window population keeps shrinking as the window
grows, and moving further from "first few seconds" trades away the
early-warning framing that motivated this task in the first place — an
open question, not attempted here. One caveat *is* addressed next: every
number above uses an event-disjoint split under which most stations are
shared across train/val/test (175/181 at 3s, 148/152 at 6s), so part of
either headline could be site memorisation rather than waveform shape —
Section 13.8 found exactly this for a different task on this project's
data. 7.9 runs that check here.

### 7.9 Station-Disjoint Verification

Section 13.8 found that a related task's (peak ground motion) event-disjoint
headline was inflated by site memorisation — roughly a quarter of it,
recovered only once a **doubly-disjoint** split (station-disjoint, then
every val/test row whose event also appears in train dropped, so neither
the magnitude label nor site identity can leak) was tried. The same risk
applies to 7.5-7.8: event-disjoint splits leave 175/181 (3s) and 148/152
(6s) stations shared across train/val/test. This section runs the identical
check on the magnitude-regression task, re-partitioning each dataset's
manifest in memory (`cnn_lstm_regression.py --split-by station|both`,
tensors untouched on disk) rather than regenerating data — exactly
`cnn_groundmotion.py`'s `respilt`/`report_split` pattern, reused here.

Because a doubly-disjoint split holds few stations back for test (17-38 in
what follows), Section 13.8's own lesson — a single station partition is
not enough — is applied from the start: **three independent partitions**
per window length (`--seed-split 42/43/44`), one model seed each (`--channels
2d+aux`, the 7.5-7.8 architecture throughout):

| window | grouping | leaks | MAE | ridge floor | model/ridge |
|---|---|---|---|---|---|
| 3s | event (3-seed mean, 7.5-7.7) | site response | 0.197 | 0.308 | 0.641 |
| 3s | station | source term | 0.215 | 0.311 | 0.691 |
| 3s | both, p42 | **neither** | 0.230 | 0.320 | 0.719 |
| 3s | both, p43 | **neither** | 0.247 | 0.296 | 0.834 |
| 3s | both, p44 | **neither** | 0.201 | 0.384 | 0.523 |
| 6s | event (3-seed mean, 7.8) | site response | 0.182 | 0.318 | 0.571 |
| 6s | station | source term | 0.215 | 0.340 | 0.632 |
| 6s | both, p42 | **neither** | 0.195 | 0.363 | 0.537 |
| 6s | both, p43 | **neither** | 0.236 | 0.335 | 0.704 |
| 6s | both, p44 | **neither** | 0.222 | 0.358 | 0.620 |

**(a) Site memorisation does not explain the result.** Every doubly-disjoint
partition, at both window lengths, beats its own ridge floor by a wide
margin (ratio 0.52-0.83, always well under 1) on stations the network never
saw in training. Whatever the model is doing, it generalises past the
specific sites it trained on — the question this section exists to answer.

**(b) The 6s-beats-3s margin does not survive this stress test, and that is
a different finding from "it's wrong."** Doubly-disjoint means: 3s
0.226 ± 0.023 (range 0.201-0.247), 6s 0.218 ± 0.021 (range 0.195-0.236). The
7.8 gap (0.016 MAE, 3-seed, same grouping both sides) is smaller than the
spread *within* either window length here. The doubly-disjoint check is a
different, noisier protocol (17-38 test stations and one model seed per
partition, so its spread is partition variance *and* seed variance
combined, not partition variance alone — Section 13.8 could separate the
two with three seeds per partition; this does not). A 0.016 effect falling
below this protocol's resolution is not the same as the effect being
shown absent under the original, tighter one. Both statements are true at
once: 7.8's comparison is controlled and 3-seed-tight; this section's is
real but too coarse to confirm or overturn it.

**(c) Mean degradation from event- to doubly-disjoint is comparable at both
lengths** (mean ratio across the 3 partitions: 0.641→0.692 at 3s, a +0.051
shift; 0.571→0.620 at 6s, +0.049) — individual partitions swing well
above and below their own mean (point (d)), but averaged over three, site
familiarity is not disproportionately propping up the 6s number
specifically, which is the thing this section set out to rule out for 7.8.

**(d) The single-partition trap, demonstrated rather than just described.**
p42 alone would have supported a *different* conclusion: 6s's ratio
improves doubly-disjoint (0.571→0.537) while 3s's worsens (0.641→0.719),
suggesting 6s's advantage is the *more* robust one. Running p43 and p44
erases that pattern — 3s's best partition (p44, ratio 0.523) beats 6s's
worst (p43, ratio 0.704). A single partition, at either window length,
would have been reported with an error bar an order of magnitude too
small. This is Section 13.8's own lesson, reproduced on a second task.

**Conclusion.** Magnitude regression from a short window is not a
station-memorisation artifact at either window length tested — that was
the open risk, and it is closed. Whether 6s is *reliably* better than 3s
remains open: real under the clean, matched 3-seed event-disjoint
comparison (7.8), not confirmable or refutable under the noisier
doubly-disjoint one run here.

---

## 8. Results: Three-Class Risk Classification, and Why the Best Model Has No Image

### 8.1 Motivation and Task

Section 7 classified magnitude among confirmed earthquakes. This section
asks the operationally closer question: given an arbitrary 3-second window,
is it **noise**, a **low-risk** event (M < 4), or a **high-risk** event
(M ≥ 4)? It folds the detection task of Section 6 and the magnitude task of
Section 7 into one decision, on the same 3-second windows.

The threshold M = 4 was chosen as a round, physically meaningful cut rather
than a quantile fitted to this catalog. It is a CLI flag
(`--mag-threshold`), not hardcoded.

This section reports a **negative result for the CNN** and a positive one
for a model that does not use the encoded window at all. The investigation
that produced it also turned up three distinct defects, each of which would
have inflated a reported number if left alone.

### 8.2 Data, and the Limits of Fetching More

The high-risk class is intrinsically scarce: of 23,918 already-downloaded
3-second events, only 479 were M ≥ 4 (2.0%), following the
Gutenberg–Richter distribution documented in Section 7.2. An attempt was
made to close that gap from the catalog.

**The attempt largely failed, for an instructive reason.** 726 catalog-listed
M ≥ 4 events had no downloaded waveforms. Re-running the existing FDSN
downloader against exactly those events retrieved **76 of 726 (10.5%)**.
Station *metadata* resolves normally for the failures — a representative
case returned 7 stations within the search radius — but the waveform
archive itself answers `HTTP 204, no data available` for the requested
station/time. Successes clustered almost entirely in the 2023
Kahramanmaraş sequence; most other years returned nothing. **A catalog entry
does not imply a retrievable waveform**, and for this network the shortfall
is an order of magnitude, not a margin. The high-risk pool moved 479 → 556.

Separately, re-running `anchor-windows` picked up roughly 7,300
previously-downloaded but never-anchored events, taking the 3-second pool
from 23,918 to 31,325 — a larger gain than the download attempt produced,
obtained from data already on disk.

The dataset (`generate-riskclass-dataset`,
`data_downloader/seismic_cli/riskclass.py`) pools earthquake and noise
windows into one manifest with a station-disjoint split spanning all three
classes, and caps the abundant classes at `--balance-ratio` × the high-risk
count per split (default 4.0) rather than either discarding most of them at
1:1:1 or leaving the raw ~1:19:many imbalance. Final composition: 4,267
train / 736 validation / 2,970 test windows, **zero stations shared across
splits**.

### 8.3 The CNN Result

`cnn_riskclass.py` reuses `RegressionSeismicCNN` with `num_classes=3` and
`CrossEntropyLoss` (Section 4.5; the trunk is unchanged and `num_classes`
defaults to 1, so the Section 7 and regression scripts are unaffected).

| Model | Accuracy | Macro-AUC | MCC |
|---|---|---|---|
| Predict majority class | 52.96 % | — | — |
| Logistic on the two scalars | 81.52 % | 0.9476 | +0.6730 |
| **CNN (image + aux)** | **73.64 %** | **0.9277** | **+0.5990** |

**The CNN loses to a logistic regression on two scalars it was itself given
as input.** Its errors concentrate almost entirely on one boundary: 609 of
1,573 noise windows classified as low-risk. The rare high-risk class is
*not* the problem — it is recovered well (268/281, recall 0.954), which
overall accuracy alone would have hidden.

### 8.4 Three Defects Found While Investigating the Gap

The first version of this experiment showed validation accuracy 95.7 % against
test accuracy 71.7 % — a 24-point gap demanding explanation before any
number from it could be reported.

**(a) A stuck instrument, supplying 58 % of one class's errors.** Station
`6G.MADM` contributed 199 test noise windows, *all* misclassified. Reading
its MiniSEED directly: traces span ~58 counts on a ~5.38-million-count DC
offset, with ~50 unique sample values across 30,001 samples — a stuck
digitizer, not quiet ground. Its window RMS (~6 counts) against its own
station baseline (~975–3118) gives log SNR ≈ −6, far outside the training
range (train noise floor −2.99), so the CNN extrapolated into territory it
had never seen while a monotonic logistic model did not.

`--min-log-snr` (default −3.0) now rejects any window whose RMS falls below
5 % of its own station's long-term noise floor. The threshold is set by
instrument physics — genuine ambient noise does not sit 20× below a
station's own floor — corroborated by a clean gap in the pooled
distribution (5th percentile −2.67, then an isolated cluster at −6.0), and
applied uniformly to every class and split. It also removes 18 *earthquake*
windows on identical reasoning. This is deliberately not "discard whatever
the model gets wrong": the criterion was verified against the raw waveform,
and it is label-independent.

With the filter applied and both noise directories used (23,031 files
instead of 11,454), **the gap closes from 24 points to about 1** (validation
74.9 % against test 73.6 %), and the test noise log-SNR distribution returns
to tracking train (mean −1.24, sd 0.85, against v1's −1.87, sd 2.45).

**(b) The validation split cannot rank models on this task.** It holds 736
windows from just **two** noise stations. It ranked the CNN (validation MCC
0.873) above gradient boosting (0.867), when their test MCCs are 0.599 and
0.851 — it selected the *worse* model by a 0.25 MCC margin. This is the same
validation-versus-test divergence documented in Section 6.5, but far more
consequential: there it cost a fraction of a point, here it would have
picked the wrong model outright.

Selection therefore moved to **station-grouped 5-fold cross-validation** over
train and validation pooled (145 stations). Every failure diagnosed on this
task has been station generalization, so the selection procedure must see
many unseen stations. Under CV the ranking is stable and matches test:

| Model | CV MCC (mean ± sd) | Test MCC |
|---|---|---|
| Logistic | 0.776 ± 0.061 | +0.673 |
| Random forest | 0.809 ± 0.089 | — |
| Gradient boosting | 0.868 ± 0.035 | +0.851 |
| Gradient boosting (shallow) | **0.869 ± 0.037** | **+0.851** |

**(c) An artifact worth ~10 accuracy points, introduced by this report's own
dataset design.** `distance_km` is *undefined* for noise windows — there is
no event to measure from — so in a flat three-class model, "distance is
missing" separates noise almost perfectly by construction. Measured: 91.72 %
accuracy with distance against 81.55 % without. That difference is dataset
assembly, not physics.

Notably, the equivalent check on the Section 7 binary task came back
negative (log SNR alone scored within 0.3 points of the pair), so this
artifact is specific to introducing a noise class, and would not have been
caught by reusing the earlier task's reasoning.

### 8.5 The Final Model: Two-Stage, Scalars Only

The fix is structural rather than a feature deletion. Splitting the decision
confines distance to where it physically exists:

$$\text{Stage 1: noise vs. earthquake} \;\rightarrow\; \log\text{SNR only}$$
$$\text{Stage 2: low- vs. high-risk (earthquakes only)} \;\rightarrow\; \log\text{SNR} + \log(\text{distance})$$

with the two stages recombined by the chain rule into a proper
three-class distribution. Stage 2 is not two arbitrary features: observed
amplitude together with distance *is* the local-magnitude relation this
project already relies on (Section 4.5). Both stages' class-weight
exponents were selected by the station-grouped CV of 8.4(b), never on test.
Implemented in `cnn_earthquake/src/riskclass_scalar.py`.

**Test set, evaluated once after selection:**

| Model | Accuracy | Macro-AUC | MCC | Balanced acc. | High-risk recall |
|---|---|---|---|---|---|
| CNN (image + aux) | 73.64 % | 0.9277 | +0.5990 | — | 0.954 |
| Flat gradient boosting *(distance artifact)* | 91.72 % | 0.9792 | +0.8559 | 0.8805 | 0.790 |
| **Two-stage scalar (leak-free)** | **82.83 %** | **0.9273** | **+0.7039** | **0.8314** | **0.861** |

Per-class recall for the two-stage model: noise 0.866, low-risk 0.767,
high-risk 0.861. Stage-1 AUC (noise vs. earthquake) 0.9425; stage-2 AUC
(low vs. high, earthquakes only) 0.9441.

The 91.72 % figure is reported here for completeness but **should not be
quoted as this task's result**: roughly ten points of it come from the
missing-distance artifact, and it would not survive deployment, where a
single-window detector does not know the distance to an event it has not
yet decided exists. The honest number is 82.83 %, which still exceeds the
CNN by about nine accuracy points and recovers substantially more high-risk
events than the flat model does.

### 8.6 Caveats

Single seed, single magnitude threshold, single train/validation/test split;
no seed-repeat check, which Section 6.6 showed concretely can reverse a
conclusion on this dataset. Noise-station diversity remains the binding
constraint — 9 train / 2 validation / 6 test noise stations after the
filter — which is precisely why one bad station could distort the whole
picture, and why station-grouped CV rather than the held-out split is doing
the model selection. The CNN was not re-architected or re-regularized in
response to the diagnosis; the conclusion is that a scalar model wins *as
configured here*, not that no CNN could win. Notably the CNN retains the
best high-risk recall of any model tested (0.954), so a hybrid that uses it
only for that boundary is unexplored.

---

## 9. Discussion

Four threads run through this investigation. First, **absolute amplitude is
the recurring missing ingredient**: it is what RAM cannot represent
(Section 2.2(d)), what STA/LTA depends on entirely (Section 5), what its
correction restores on both the RAM image and the raw-waveform LSTM branch
(Section 6.3), what the magnitude classifier's own logistic baseline is
built from (Section 7.3), and — carried to its conclusion — the *entire*
input of the best three-class risk model, which uses no image at all
(Section 8.5). Every result in this report that involves amplitude, in
either direction, traces back to the same structural fact. The trajectory
across Sections 6 to 8 is worth stating plainly: the more directly a task
depends on amplitude, the less the encoded image contributes, until at
Section 8 it contributes negatively.

Second, **fusion mechanisms behave inconsistently enough that none should be
assumed without measurement**: linear fusion underperforms its own best
branch; gated fusion helps on one 2D representation and hurts on two others;
stacking is the one mechanism that has not underperformed a single branch in
any configuration tested, at the cost of a separate training and fitting
procedure rather than end-to-end joint training. If a single practical
recommendation follows from Section 6.4, it is to default to stacking over
joint fusion when branches are trained separately regardless, and to measure
gated fusion's effect directly rather than assume it from another
representation.

Third, **the RAM transform's core promise — extracting useful structure from
a scale-invariant, geometrically compressed image — never exceeded a
station-normalized spectrogram fed to the same CNN trunk, in any
configuration tested, with or without the amplitude fix, with or without an
LSTM partner branch.** This is not a failure specific to this
implementation of RAM; it follows directly from properties 2.2(b) and (d):
a 64×64×3 RAM image carries at most 189 independent values and no amplitude
information at all, while a spectrogram of comparable size carries
substantially more of both. Where RAM is specifically required — matching
the source paper exactly, for instance — the amplitude auxiliary input is a
necessary, low-cost companion; where the 2D representation is an open
design choice, this investigation did not find a case where RAM was the
better one.

The two bodies of work in Sections 6 and 7 are related but should not be
over-connected: both reuse the amplitude-auxiliary-input pattern and the
same CNN trunk, but they answer structurally different questions (is there
an event at all, versus how large is it), on different window lengths (6 s
versus 3 s), against different comparison baselines (STA/LTA versus a
fitted amplitude/distance relation). Their agreement on the general
principle — encoded windows carry information beyond simple amplitude
scalars — is a genuine cross-check, not a coincidence of shared code.
Section 8 is the boundary case that qualifies it: once a noise class is
introduced and the decision hinges on amplitude relative to a station's own
floor, the encoded window stops adding and starts subtracting.

Fourth, and least expected, **the dominant failure mode in this project is
a wrong number, not a crash.** Of the seventeen defects in Section 12, the
ones that mattered most were silent: STA/LTA scoring zero windows while
printing a plausible-looking summary (defect 14), a baseline crashing past
the model's own numbers so the model appeared to have no floor to beat
(defect 16), a stuck instrument entering the dataset as valid quiet ground
(defect 15), and a missing-value pattern standing in for a class label
(defect 17). Every one produced output that looked like a result. The
practices that caught them were not sophisticated — read the raw data
behind a suspicious number, keep a floor next to every headline figure, and
treat an implausibly large validation/test gap as a defect report rather
than a tuning problem — but they had to be applied deliberately, because
nothing failed loudly enough to force the issue.

---

## 10. Limitations and Future Work

**Statistical.** Every result in Sections 6, 7 and 8 reflects a single
train/validation/test split at one random seed, except the two comparisons
re-seeded in Section 6.6, where repetition changed one reported conclusion
and substantially narrowed another. Every other close-margin figure —
stacked-RAM-dual vs. `1d`-alone, `2d+aux` vs. the full amplitude-dual model,
the `1d+aux`/`2d+aux` effect sizes, the magnitude-classification result in
Section 7, and the three-class results in Section 8 — remains unverified at
additional seeds and should be read with the caution Section 6.6
demonstrates is warranted. Section 8's margin over the CNN (about nine
accuracy points) is comfortably outside that noise band; its narrower
internal comparisons are not.

**Station diversity is the binding constraint on the risk task, and
possibly on more than that.** Section 8 runs on 9 train / 2 validation / 6
test *noise* stations. That is few enough that a single faulty instrument
distorted the entire picture (Section 8.4a) and that the held-out
validation split could not rank models at all (Section 8.4b). Acquiring
more noise stations would do more for Section 8 than any modelling change
considered here, and the same concern applies, less acutely, to the
detection results in Section 6.

**Untested extensions, implemented but not evaluated.** Per-component
auxiliary scalars (Section 4.3, six scalars instead of two, commit
`8485ffc` in `data_downloader`) are implemented and smoke-tested but not
evaluated at scale — deliberately deferred (Section 11), not abandoned.

**Untested extensions, proposed but not implemented.** Feeding the RAM
angle vector $\beta$ directly to a one-dimensional model, to test whether
property 2.2(b)'s 63-degrees-of-freedom argument means the 2D CNN adds
nothing the 1D vector didn't already carry. Combining the three components
before the RAM transform (vector magnitude) rather than transforming each
independently. A distance term for the noise-contamination check
(Section 3.1), which currently discards otherwise-scarce noise data purely
on elapsed time.

**Window length.** 60 s windows have not been re-run under the corrected
pipeline (Section 6, introduction) and 60 s results from before those
corrections are not reported here at all, only referenced as superseded.
60 s windows with the LSTM+attention branch were never started. The
geometric argument for why short windows are harder (Section 2.3) applies
specifically to the RAM branch, not to the raw-waveform or auxiliary
branches, so the balance between branches may shift at longer window
lengths.

**Gated fusion's mechanism.** Its mixed result across three 2D
representations (Section 6.4) has not been diagnosed further, though
Section 6.6 adds one relevant data point: on spectrogram-dual specifically,
the effect is real on accuracy/MCC but small and noisy on AUC — narrowing
rather than resolving the explanations originally offered.

**Hyperparameters.** Neither the CNN's hyperparameters (Section 6.5) nor the
LSTM branch's own hyperparameters reward tuning on this dataset; this
narrows the plausible location of any remaining gap toward feature
representation rather than architecture search on either existing branch.

**Magnitude classification (Section 7).** Single seed, single threshold, no
per-component-aux variant tried, and a spectrogram input small enough (5
time frames) that its effective temporal resolution is worth questioning
directly.

**Risk classification (Section 8), specific open threads.** The CNN was not
re-architected or re-regularized after the diagnosis, so the finding is
that a scalar model wins *as configured*, not that no CNN could. Two
concrete follow-ups are unexplored: the CNN retains the best high-risk
recall of any model tested (0.954 against the two-stage model's 0.861), so
a hybrid using it only for that boundary may beat both; and a properly
cross-validated stack of CNN and scalars was never built, because doing it
honestly requires out-of-fold CNN predictions and therefore k-fold
retraining (the one stack that was tried fit on the 736-window validation
split and lost to the scalars outright). Richer scalars — per-component log
SNR, spectral band ratios — are cheaper to test than either.

**External validity.** This investigation has not undergone external peer
review.

---

## 11. Project Status: Original Objective and Next Steps

The project's original objective is forecasting **event onset time and
event class from earthquake catalog data** — not waveform classification.
Sections 2–10 are, in that light, a substantial but deliberate detour: they
established (a) that a specific architecture from a different domain
transfers to seismic waveforms with real, measured limitations and fixes,
(b) that the same amplitude-input pattern generalizes to a related but
distinct question (event class from a short window), and (c) that on the
three-class risk task the pattern reaches its limit — two scalars beat the
encoded window outright. All three are genuine, useful results, and they
are being set aside now — not because a dead end was reached, but because
continuing to refine them further would be extending the detour rather than
returning to the original scope.

**What this phase established**, most-to-least confident: the amplitude
auxiliary input's large effects on RAM (Sections 6.3, 6.5); that a
correctly-parameterized STA/LTA baseline is beaten by every tested CNN
configuration at 6 s (Section 6.1), a finding that only became measurable
after fixing a previously-undiscovered evaluation defect (Section 5.2);
that late-fusion stacking is the one fusion mechanism that has not
underperformed a single branch in any configuration tested (Section 6.4);
that magnitude class is predictable from a single 3-second window well
beyond an amplitude/distance floor (Section 7); that on the three-class
risk task a two-scalar model beats the CNN by about nine accuracy points
(Section 8.5); and, as standing methodological cautions rather than
specific findings, that single-seed margins under roughly 0.01–0.02 AUC on
this dataset have been shown concretely to overstate an effect, understate
it, or report the wrong sign (Section 6.6), and that a held-out split too
poor in stations can rank models backwards outright (Section 8.4b).

**What happened next — this paragraph supersedes the plan that stood here, which
described the catalog code as never having been exercised end-to-end.** That
work was carried out and, for a time, was reported separately in
`catalog_report.md`: a three-class time-to-next-event formulation measured at
chance and was replaced by a binary one — "will a M ≥ 4.5 event occur in this
fault zone within the next 30 days?" — under a logistic-regression / gradient-
boosting scalar model, reaching block-level AUC 0.62 (East Anatolian) and 0.60
(Aegean) over ~190 independent 30-day blocks, with the North Anatolian and
Cyprus zones indistinguishable from chance. That scalar forecaster has since
been retired — it does not meet this project's neural-architecture mandate —
in favor of retargeting a dual-channel CNN+LSTM+attention model already built
for this task (`cnn_lstm.py`), which had only ever been tried against the
abandoned three-class target, never the reformulated (dense, learnable) one.
Retargeted (`cnn_lstm_forecast.py`) and evaluated at 3 seeds, it ties the
retired scalar model at the pooled level (mean AUC 0.733 vs. 0.723) and
matches it zone-by-zone to within seed noise, with AEGEAN remaining the one
zone where both architectures agree real signal exists. Full results in
`catalog_forecast_report.md`; `catalog_report.md` no longer exists.

**Superseded 2026-08-31 — the figures in the preceding paragraph predate a
catalogue defect that has since been corrected.** The catalogue those runs used
was missing ~29% of AFAD's events for the region, including almost all of the
February 2025 Santorini–Amorgos swarm; see
`docs/experiment_neural_forecasters_2026-08-30.md` §4 for the audit. Re-derived
on the rebuilt catalogue (3 seeds per arm, both catalogues spanning 2000–2026 so
only completeness differs), at the **block level** — 30-day disjoint blocks, the
honest sample size, because consecutive windows overlap 11–46× and pooling at
window level inflates AUC by +0.25 to +0.35:

| zone | n blocks | old catalogue | corrected catalogue | Δ |
|---|---|---|---|---|
| AEGEAN | 43 | 0.5190 ±0.0150 | **0.6918 ±0.0165** | **+0.173** |
| CENTRAL | 43 | 0.3960 ±0.0335 | **0.6176 ±0.0346** | **+0.222** |
| EAFZ | 47 | 0.6615 ±0.0173 | 0.6667 ±0.0323 | +0.005 |
| NAFZ | 42 | 0.4643 ±0.0346 | 0.4103 ±0.0011 | −0.054 |

The gains are five to ten times the seed spread and fall **exactly where the
catalogue defect was** — the missing events were overwhelmingly offshore Aegean,
and it is AEGEAN and the adjacent CENTRAL zone that move, while EAFZ far to the
east does not and NAFZ stays at chance. A variance artefact would not respect
that geography. **CENTRAL moving from "indistinguishable from chance" to 0.618
is the largest single change in this project's forecasting results.**

**Folding in `Sismokaos-featureExtract`'s auto-extracted waveform features —
listed here as "not yet done" — was carried out on 2026-08-30, and the answer is
negative.** At an operating point where the evaluation is actually valid (M≥4.0,
14-day horizon; at the original M≥4.5/30 d the walk-forward folds are degenerate
and two AUCs undefined), all three sequence architectures lose to a persistence
floor of 0.5823: LSTM 0.5244, GRU 0.5709, TCN 0.5204. This agrees with the
chaotic-feature suite, which is below floor across all four model variants and 0
of 10 context/horizon cells.

Taken together these two results bound the question the project set out to
answer: **the forecasting signal here comes from the earthquake catalogue, not
from the seismogram**, and completing the catalogue improves the catalogue-based
model measurably. Full detail in
`docs/experiment_neural_forecasters_2026-08-30.md` and
`docs/experiment_chaos_forecast_2026-08-27.md`.

Section 13 then returns to waveforms for a bounded, specific question — peak
ground motion from a 3-second window — chosen because it tests the same "does
the encoded window beat two scalars" pattern as Section 8, on a task where the
network turns out to win.

---

## 12. Software Defects Identified and Corrected

The reliability of this report's results depends substantially on defects
identified and corrected during development. Defects 1–5 predate the
dual-channel extension (Section 4.2 onward); defects 6–13 were found during
a systematic audit of the full repository; defect 14 was found while
preparing this rewrite; defects 15–17 were found while investigating the
three-class model's validation/test gap (Section 8.4); defects 18–22 were
found during the peak-ground-motion work (Section 13). Defects 15–17 are
worth reading together: each would have caused a *wrong number to be
reported* rather than a crash, and two of them (16, 17) would have made a
losing model look like a winning one.

| # | Defect | Mechanism | Impact |
|---|---|---|---|
| 1 | Cross-class station leakage | Splits were allocated independently per class; with ~97% station overlap, a station could appear as train-earthquake and test-noise | The model could exploit station identity as a shortcut; biased measured test performance downward, most severely where station counts are low (short windows) |
| 2 | Station caps were ineffective | The cap was applied at file granularity; a single 300 s noise file (~200 windows at 3 s) exceeded any smaller cap and was retained in full | Explains observations of only 2–4 distinct noise stations in validation/test even after capping |
| 3 | Fixed-resolution collapse | Reshape depth $d$ was fixed regardless of window length | 3 s windows collapsed to ~3×3 images; corrected by deriving $d$ from the target resolution |
| 4 | Origin-anchored short windows | Short windows were cut from origin time rather than arrival time | A meaningful fraction of nominal "earthquake" windows contained no signal |
| 5 | Noise/earthquake station mismatch | Noise data was sourced independently of earthquake stations (~47% overlap) | The model rarely observed both classes from the same instrument; a targeted downloader raised overlap to ~97% |
| 6 | STA/LTA computed on raw counts (anchoring) | `classic_sta_lta` was computed on un-detrended MiniSEED counts; a large DC offset pins the characteristic function near 1 | Verified: a synthetic arrival with a $10^6$-count offset produces a maximum characteristic-function value of exactly 1.000 (no pick possible); after detrending, the pick lands within 4 samples of ground truth |
| 7 | Arrival pick on a horizontal component | `sorted(traces)[0]` selects the E component before Z alphabetically, with no fallback | P-wave onsets are cleanest on the vertical component; corrected to vertical-first with fallback |
| 8 | STA/LTA computed on raw counts (baseline evaluation) | The same DC-offset defect present in the baseline scorer | Baseline performance was understated at high-offset stations, plausibly explaining a measured STA/LTA AUC swing of 0.78–0.98 between data pulls |
| 9 | Label-smoothing asymmetry | Smoothing was applied to training targets only; smoothed binary cross-entropy floors near 0.325 nats | Training and validation loss curves were not directly comparable in magnitude; validation metrics were unaffected |
| 10 | Threshold mismatch | `cnn_from_tensor.py` validated at a 0.60 threshold but tested at 0.50 | Validation and test accuracy were measuring different decision rules |
| 11 | Gap interpolation treated as signal | `merge(fill_value='interpolate')` fabricates linear ramps across telemetry gaps | Synthetic data was trained on as though genuine; corrected via gap masking, windows exceeding 5% synthetic samples rejected |
| 12 | Single-rate assumption | The first trace's sampling rate in a file was applied to every station represented in it | Incorrect physical window duration for off-rate stations; corrected to per-station sampling rate |
| 13 | Alphabetical channel selection | `sorted(keys)[:3]` could select `['1','2','E']` — two horizontal components, no vertical | Silent component mis-assignment; corrected to role-based selection requiring a vertical component |
| 14 | STA/LTA silently unrunnable, then silently mis-parameterized, against anchored/dual-channel data | `eval_baseline.py`'s filename regex matched only `_winNNN.png`, so every `.pt`-based dual-channel manifest scored zero windows with no error; once fixed, the auto-derived LTA (window/3) was found to exceed `anchor.py`'s pre-arrival buffer (0.2×window) for any anchored window under ~50 s, putting the true arrival inside `classic_sta_lta`'s forced-zero warm-up region | `eval-sta-lta` had never actually run against the Section 6 dual-channel datasets; its default parameters, even once made runnable, silently score AUC 0.51 (random) at 6 s. Corrected via a validation-selected LTA (Section 5.2), giving AUC 0.82 |

| 15 | Stuck-instrument windows entering the risk dataset as valid "quiet noise" | `6G.MADM`'s traces span ~58 counts on a ~5.38-million-count DC offset with ~50 unique values across 30,001 samples; gap rejection catches telemetry gaps but not a digitizer stuck at a constant | 199 windows at log SNR ≈ −6, far outside the training range, supplied 58% of one class's test errors and created a 24-point validation/test gap. Corrected by `--min-log-snr` (Section 8.4a); the gap closes to ~1 point |
| 16 | Multi-class baseline crashed silently *past* the model's own numbers | `LogisticRegression(multi_class=...)` was removed in scikit-learn 1.9; the exception was caught by a broad `except` that printed a warning and returned `None`, after which the reporting code skipped the comparison line | The 3-class run reported the CNN's 71.71% accuracy with **no floor beneath it**. The floor, once computed, was 90.37% — the model was losing to two scalars, and the output as printed suggested the opposite |
| 17 | `distance_km` undefined for noise leaks the noise class | There is no event to measure distance from for a noise window, so "distance is missing" identifies the noise class by construction in a flat multi-class model | Worth ~10 accuracy points of pure inflation (91.72% vs 81.55%). Corrected structurally by the two-stage split (Section 8.5). The equivalent check on the Section 7 binary task was negative, so reusing that task's reasoning would not have caught it |
| 18 | Ground-motion label window entirely contained the model's input window | `sismokaos.groundmotion.py` originally took the peak over `[record_start + 3 s, end]`. The arrival lands ~10–12 s into a 60 s record and the input window sits at `[arrival − 0.6 s, arrival + 2.4 s]`, so the target interval enclosed the input interval completely | The model could have read its own target off its own input, with nothing to signal it. Corrected by replaying `anchor.py`'s deterministic STA/LTA pick to recover the arrival — verified to reproduce the stored anchored corpus bit-exactly on 532 stations, worst absolute sample difference 0 — and opening the label at `arrival + 2.4 s`, where the input closes |
| 19 | `remove_response` silently rejected every station on a placeholder timestamp | StationXML responses carry validity epochs. Traces were stamped `UTCDateTime(0)`, which falls outside every epoch, so `remove_response` refused the correction and raised nothing | First smoke test returned `response_ok=False` for 13 of 13 stations with no error. Had the flag not existed, the result would have been an all-NaN dataset resembling a data problem rather than a code one. Corrected by threading the real trace `starttime` through; it is now a required argument that raises on `None` |
| 20 | Instrument responses whose stage gains contradict their reported sensitivity | Across 828 cached channel-epochs the disagreement between the reported overall sensitivity and the product of stage gains is cleanly bimodal: 97.1 % agree to within 0.01 %, and 2.9 % disagree by a factor of ~690,000. All of the latter are 6G stations (ATIM, BOZM, BUYM, GBZM, IGDM, KMRM, MADM, YNKM). obspy emits a warning and continues | An amplitude wrong by six orders of magnitude would enter the dataset as a plausible number. Surfaced as a `sens_mismatch` column with an explicit tolerance rather than left to appear downstream as an inexplicable R² |
| 21 | Quality flags computed over the survivors of their own filter | Windows without a usable response produce no input tensor and were skipped with a bare `continue`, before the manifest was written. The manifest then reported `response_ok` 100.0 % and `sens_mismatch` 0.0 % | Both rates were true by construction and conveyed nothing. Worse, the 6G stations carrying defect 20's sensitivity error are largely the same ones failing the response lookup, so the flag built to expose them read a clean 0 % *because they had already been removed*. Found by reconciling the manifest's 49,680 rows against the 51,408 the anchored corpus should yield; the 1,728-row gap is entirely 6G and IJ. Corrected by counting drops per station and reporting them by network ahead of the flag table. Data was unaffected — the defect was purely in what the report claimed |
| 22 | Anchored windows carry their parent record's start time | `anchor.py` builds each sliced window with `tr.copy()` and replaces `.data`, but never advances `stats.starttime` to match the slice offset | Every window in `window_post_{3,6,10}s_anchored` is stamped with the start time of the 60 s record it came from. Harmless for the classifiers, which never used absolute time, which is why it went unnoticed; it destroys the arrival time, and is the reason the ground-motion work had to re-derive the pick rather than read it. Not corrected in place — the anchored corpus would need regenerating — but documented and routed around |

Two items were suspected as defects but confirmed not to be: the RAM
mathematics as implemented transcribes the source paper correctly,
including guards the paper itself omits (the $\varepsilon$ floor on
$\sigma$, and clipping before the inverse cosine); and the manifest's
window-index-to-sample mapping is exact.

---

## 13. Results: Peak Ground Motion from a 3-Second Window

### 13.1 Motivation

Nurtas et al. (ACDSA 2025) predict peak ground acceleration from the first
three seconds of three-component waveform, reporting validation MAE 2.61 gal
and R² 0.714 for a CNN–BiLSTM+attention model. Their input tensor is
$(300, 3)$ — identical in shape to this project's existing
`window_post_3s_anchored` windows — so the input side of a replication required
nothing new.

The paper compares its model against an ANN and an LSTM. All three are neural.
Peak ground motion is fundamentally an amplitude, and the input window contains
amplitude, so the obvious floor — regress the target on the peak amplitude of
the input window — is absent. Section 8.5 of this report found a two-scalar
model beating a CNN by nine accuracy points on a related task, and Section 12
defect 16 records what happens when a floor is computed *after* a headline
rather than before it. The floor was therefore measured and committed before
any network for this task existed.

### 13.2 The label, and two defects in defining it

Raw counts cannot serve as the label. Counts are proportional to ground motion
only within a single instrument; KOERI's HH\* channels run ~2.5×10⁹
counts/(m/s), and sensitivities differ station to station, so a model trained on
raw peaks would partly be learning which station recorded the event. Instrument
response removal converts to physical units and makes the target comparable
across the network — 154 stations here.

**Our sensors are the wrong class for a like-for-like replication, and this is
stated rather than absorbed.** Effectively all channels are HH\*, high-gain
broadband *velocity* seismometers; K-NET, the paper's source, is a strong-motion
*accelerometer* network. Obtaining acceleration requires differentiating
velocity, which amplifies high-frequency noise exactly where broadband data is
weakest. Both `pgv_cms` (physically native, and a standard early-warning
intensity measure) and `pga_gal` (for numerical comparability with the paper)
are therefore emitted.

Defining *when* the label window opens produced defect 18. The first
implementation took the peak over `[record_start + 3 s, end]`. Arrivals land
~10–12 s into a 60 s record and the input window sits at
`[arrival − 0.6 s, arrival + 2.4 s]`, so the target interval **enclosed the
input interval entirely** — the model could have read its own answer off its own
input. The arrival is not recoverable from the anchored files (defect 22), but
is recoverable by replaying `anchor.py`'s deterministic STA/LTA pick, which
reproduces the stored corpus bit-exactly across 532 stations. The label now
opens where the input closes.

Two targets are emitted rather than one being chosen silently:

* `*_fwd` — peak over `[arrival + 2.4 s, +25 s]`, strictly after everything the
  model saw. Zero overlap with the input.
* `*_full` — peak over the whole record. This is the paper's quantity, retained
  for comparability, but it overlaps the input.

### 13.3 Neither target is a clean task, and both flaws are measured

**The `_full` target is degenerate against an amplitude baseline.** The peak
amplitude of the input window is ≤ the full-record target in **100.00 %** of
rows and **exactly equal in 33–34 %**. It is a mathematical lower bound on the
target, not a predictor of it; for a third of the corpus the "baseline" *is* the
answer. Results on `_full` are reported separately and are not comparable to
`_fwd`.

**The `_fwd` target is contaminated by S–P moveout.** Fitting
$\log_{10}(\text{target}) \sim a M + b \log_{10}(\text{distance})$ over 43,091
windows:

| target | $a$ (magnitude) | $b$ (distance) | R² |
|---|---|---|---|
| `log_pgv_full` | +0.874 | **−1.455** | 0.596 |
| `log_peak_input_vel` | +0.813 | −0.761 | 0.365 |
| `log_pgv_fwd` | +0.935 | **+0.226** | 0.501 |

$b \approx -1$ is what geometric spreading predicts, and `log_pgv_full`
delivers it, confirming the station coordinates and the response correction are
sound. The `_fwd` inversion has a specific cause: the input window closes at a
fixed +2.4 s while the S wave, which carries the peak, moves out with distance.

| distance | `peak_in_input` | median peak time rel. arrival |
|---|---|---|
| 22 km | 34.0 % | −1.26 s |
| 35 km | 45.6 % | +0.42 s |
| 47 km | 16.5 % | +5.87 s |
| 53 km | 11.7 % | +7.09 s |

corr(distance, peak time) = **+0.501**. At near stations the forward window sees
only coda; at far stations it sees the whole S wave. That window-capture effect
opposes geometric spreading and, over this corpus's 5–56 km range, wins.
`pgv_cms_fwd` is therefore partly a measurement of whether S landed in the
window — a distance question. This does not invalidate the CNN-versus-scalar
comparison, since both face the identical confound, but it does mean the result
cannot be called ground-motion forecasting skill without qualification.

### 13.4 The floor

49,680 windows across 154 stations, M 2.0–7.7, event-disjoint splits (0 events
shared). Test n = 5,768 after label-independent quality rules. Predictors are
restricted to what is knowable at inference: magnitude appears only as a marked
oracle, since the point of early warning is to characterise shaking before the
source is characterised.

| target | amplitude only | + log distance | GBM (same 2) |
|---|---|---|---|
| `pgv_fwd` | +0.5694 | +0.6471 | +0.6749 |
| `pga_fwd` | +0.6705 | **+0.7318** | +0.7473 |
| `pgv_full` *(degenerate)* | +0.7606 | +0.7721 | +0.7968 |
| `pga_full` *(degenerate)* | +0.8242 | +0.8282 | +0.8468 |

*(R²_log, test split.)* On the honest forward target, **two scalars reach
0.7318** — already above the paper's 0.714 headline. On the paper's own target
definition a single scalar reaches 0.8242, but that target is the degenerate one.

**A third floor is required for attribution.** Site response is a per-station
additive term in log space, and 149 of 154 stations appear in more than one
split with ~173 training windows each. A linear model structurally cannot
express that term, so part of any network margin over it is per-station
calibration rather than waveform shape. Adding station as a categorical takes
the `pgv_fwd` floor from MAE_log 0.2816 to **0.2616**. That is the floor quoted
against below.

### 13.5 The paper's unexplained R² of −10.08, reproduced

The paper's ANN scores R² −10.08 with no explanation offered. The mechanism is
training in log space and reporting R² in *linear* space on a heavy-tailed
target, and it reproduces here exactly — the same model, the same rows, the same
predictions, scored two ways:

| model | R²_log | R²_linear |
|---|---|---|
| oracle, `pgv_fwd` | **+0.7489** *(best)* | **−23.1753** |
| oracle, `pgv_full` | **+0.8273** *(best)* | **−12.8951** |

Reporting both spaces is what makes this visible rather than mysterious. It is
also a live trap for this project's own results: an undertrained checkpoint of
the model below won in log space and lost in linear space, and would have been
reported as a win had only one space been printed.

### 13.6 The model

`cnn_groundmotion.py`: a Conv1D trunk over the $(3, 300)$ sequence, an optional
BiLSTM+attention block (`LSTMAttentionBranch`, reused from Section 4.4), and a
head over the concatenation of pooled features and auxiliary scalars. A 1D trunk
rather than `RegressionSeismicCNN`'s Conv2d stack, because the input is a
genuine time series and a $3\times3$ kernel over a height-1 tensor is mostly
padding. 210,017 parameters.

**Input normalisation is the design decision that makes the comparison sharp.**
By default each window is divided by its own peak vector magnitude, and
$\log_{10}$ of that peak is passed in as an auxiliary scalar, recomputed from
the tensor rather than read from the manifest. Under this arrangement the linear
floor is a *strict special case* of the network — ignore the convolutional
features and use the auxiliary path. Confirmed empirically: a linear model on
the auxiliary scalars alone scores R²_log 0.6471, reproducing the
amplitude+distance floor exactly. Any margin above that is therefore
attributable to waveform shape, not to amplitude the model was handed.

Test is touched once; early stopping and checkpoint selection use validation
only — which is precisely what the paper does not do, reporting its headline on
the split it early-stopped on. Three seeds throughout, with the spread printed
beside every mean, because Section 6.6 showed single-seed margins on this
project reversing sign.

### 13.7 Result

| run | target | arch | aux | MAE_log | R²_log | R²_lin |
|---|---|---|---|---|---|---|
| A main | `pgv_fwd` | cnn+lstm | 2 | **0.1864** ±0.0014 | 0.8258 | 0.5290 |
| B no LSTM | `pgv_fwd` | cnn | 2 | 0.2113 ±0.0029 | 0.7824 | 0.4723 |
| C no distance | `pgv_fwd` | cnn+lstm | 1 | 0.1848 ±0.0034 | 0.8270 | 0.5725 |
| D waveform only | `pgv_fwd` | cnn+lstm | 0 | 0.2192 ±0.0071 | 0.7608 | 0.4887 |
| E | `pga_fwd` | cnn+lstm | 2 | 0.1769 ±0.0009 | 0.8513 | 0.3647 |
| F *(degenerate)* | `pga_full` | cnn+lstm | 2 | 0.1667 ±0.0017 | 0.8668 | 0.3963 |

Floors: `pgv_fwd` 0.2616, `pga_fwd` 0.2265, `pga_full` 0.1779 (MAE_log,
amplitude + distance + station).

**(a) Waveform shape carries information beyond its own peak.** A beats the
station-augmented floor by +0.0752 MAE_log, in *both* metric spaces, at roughly
50× the seed spread. The physical reading is straightforward: after the S
arrival ground motion decays, the decay rate is visible in the window's envelope
and frequency content, and a single peak amplitude cannot express it.

**(b) The recurrent branch earns its parameters** — B loses 0.0249 without it,
~9× the seed spread. This is a genuine departure from Section 6.7, where no
architectural addition tested had ever exceeded the simplest model. It is
reported as a single-dataset result, not a general claim.

**(c) Distance is redundant given shape.** C matches A within noise, so the
network recovers distance from the waveform rather than needing it supplied —
consistent with the 3 s window containing S–P timing. This matters
operationally: a single station in an early-warning setting does not yet know
its distance.

**(d) Amplitude too.** D, with no scalars at all and raw unnormalised physical
input, still beats the floor (0.2192 vs 0.2616).

**(e) On PGA the two metric spaces disagree.** E wins in log space (+0.0496) and
is fractionally *worse* in linear space (R²_lin 0.3647 vs 0.3709). PGA is the
noisier target here — differentiating a velocity sensor amplifies high-frequency
noise — so its tail is heavier and linear R² is dominated by a few large values.
This is the same mechanism as Section 13.5, now operating on our own result.

**(f) On the paper's degenerate target the network adds almost nothing.** F
beats its floor by only +0.0112, loses linear space by −0.2815, and is *worse*
than the floor on M ≥ 3.0 (−0.0211). That is the expected shape of a task where
amplitude is already a lower bound on the answer and equals it outright for a
third of rows.

**(a) and (f) together are the finding.** The network's value appears on the
honest target and evaporates on the paper's. A replication that had adopted the
paper's target definition unexamined — and run no amplitude floor, as the paper
does not — would have reported a strong R² produced mostly by the target
containing its own input.

### 13.8 Station-disjoint verification

Section 13.7's results use event-disjoint splits, under which 149 of 154
stations appear in more than one split. Site response is a per-station term, so
part of the network's margin could be site memorisation rather than waveform
shape — and the station-augmented floor controls only for the *linear* part of
that term.

Neither obvious grouping is clean, because the label belongs to the (event,
station) pair: making stations disjoint makes *events* shared, so one earthquake
recorded at a train station and a test station leaks its source term instead.
All three groupings were therefore run, with `both` being station-disjoint *and*
event-disjoint — every val/test row whose event also appears in train is dropped
— so that neither term can leak. Because the doubly-disjoint test set holds only
23 stations, `both` was repeated over three independent station partitions.

| run | grouping | leaks | MAE_log | seed sd | floor | delta | R²_lin |
|---|---|---|---|---|---|---|---|
| A | event | site response | 0.1864 | 0.0014 | 0.2616 | +0.0752 | 0.5290 |
| G | station | source term | 0.2174 | 0.0060 | 0.2728 | +0.0554 | 0.3869 |
| H s42 | both | **neither** | 0.2302 | 0.0008 | 0.2797 | +0.0495 | 0.2678 |
| H s43 | both | **neither** | 0.2293 | 0.0042 | 0.2857 | +0.0564 | 0.4983 |
| H s44 | both | **neither** | 0.1725 | 0.0035 | 0.2790 | +0.1065 | 0.5650 |

**The margin survives.** All three doubly-disjoint partitions favour the network
in log space, mean delta +0.0708 (range +0.0495 to +0.1065). The advantage is
therefore not primarily station memorisation: it holds when the network has
never seen the test station *or* the test event.

**But roughly a quarter of the headline was site familiarity.** Absolute error
degrades from 0.1864 to ~0.23 on two of the three partitions. Section 13.7's
number should not be read as generalisation to a new site.

**Partition variance dwarfs seed variance.** Seed spread within a partition is
0.0008–0.0042; the spread of the delta *across* partitions is 0.0254, roughly
six times larger. With 23 test stations the station draw, not the seed, is the
dominant source of uncertainty — a single station-disjoint run would have been
reported with an error bar an order of magnitude too small. This is the same
lesson as Section 6.6, arriving through a different door.

**The linear-space verdict is partition-dependent** (+0.081 and +0.289 for s43
and s44, −0.048 for s42) while the log-space verdict is unanimous. Both are
reported.

The scalar floor is stable across every partition (0.2728–0.2857 against 0.2816
event-disjoint), so site response is a small term for a *linear* model —
consistent with the network exploiting more of it than a per-station intercept
can express.

### 13.9 Caveats

The `_fwd` target's S–P moveout contamination (13.3) is not removed by any of
this; both the network and the floor face it, so the *comparison* is sound while
the *interpretation* is limited — this is not demonstrated ground-motion
forecasting skill. Single architecture family, single window length, single
corpus; M ≥ 4 is 100 test windows on the event-disjoint split, so the
strong-motion regime the application cares about is thinly sampled, and the
doubly-disjoint partitions reach only M 5.3–5.9 rather than 7.7. The
back-transform $10^{\hat{y}}$ is the median rather than the mean of the implied
lognormal and is deliberately left uncorrected, since correcting it would change
what the log-space model claims.

---

## Appendix. Reproduction Instructions

> **Catalogue paths below point into `catalogs/archive_superseded_2026-08-30/`
> deliberately.** These commands reproduce the figures *as published*, and those
> were measured against the pre-2026-08-30 catalogue. Running them against
> `catalogs/catalog_current.csv` will give different — and for the forecasting
> tasks, better — numbers, because the current catalogue restores ~29% of
> regional events the old one was missing. New work should use
> `catalog_current.csv`; only reproduction should use the archive.

**Original RAM + CNN-only pipeline (Section 6.2, pre-amplitude-fix figures):**

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
python src/sismokaos/detection/cnn_train.py --dataset-dir dataset_6s_max \
    --save-dir trained_model_6s --window-seconds 6

# 4. STA/LTA baseline on the identical test windows (parameters auto-derived --
#    see Section 5.2 for why this default is wrong on anchored windows;
#    pass --sta-seconds/--lta-seconds explicitly instead, as below)
seismic-cli eval-sta-lta \
    --manifest-path dataset_6s_max/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5 \
    --sta-seconds 0.03 --lta-seconds 0.3
```

**Dual-channel and auxiliary datasets (Section 6):**

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

# Spectrogram-dual model with the amplitude auxiliary input
seismic-cli generate-spec-dual-aux-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_specdualaux_6s --window-seconds 6 --max

# STA/LTA on the corrected 6s dual-channel dataset (Section 6.1's headline number)
seismic-cli eval-sta-lta --manifest-path dataset_specdual_6s/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5 \
    --sta-seconds 0.03 --lta-seconds 0.3

cd ../cnn_earthquake/src

python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_dual_6s \
    --channels all --batch-size 32                # or --channels 1d / 2d
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_dual_6s \
    --channels all --fusion gate --batch-size 32   # gated fusion (Section 6.4)
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --channels all --fusion gate --batch-size 32   # gated fusion (Section 6.4)

python cnn_ram_aux.py --dataset-dir ../../data_downloader/dataset_ramaux_6s
python cnn_ram_aux.py --dataset-dir ../../data_downloader/dataset_ramaux_6s --no-aux
python cnn_ram_aux.py --dataset-dir ../../data_downloader/dataset_ramaux_6s --lr 3e-4   # hyperparameter sweep (Section 6.5)

python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels all --batch-size 32                 # or 1d / 2d / aux / 1d+aux / 2d+aux
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels all --fusion gate --batch-size 32    # gated fusion (Section 6.4)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --channels all --batch-size 32                 # spectrogram + aux (Section 6.3)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --channels 2d+aux --batch-size 32               # spectrogram + aux, no LSTM (Section 6.3)

python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels 1d+aux --batch-size 32               # raw waveform + aux, no 2D branch (Section 6.3)

# LSTM branch hyperparameter sweep, isolated via --channels 1d+aux (Section 6.5)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_dualaux_6s \
    --channels 1d+aux --batch-size 32 --lstm-heads 2      # or --lstm-heads 8 / --lstm-layers 2 / --hidden 64

# Late-fusion stacking, given two already-trained --channels 1d / --channels 2d checkpoints
python cnn_lstm_stack.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --ckpt-1d trained_model_cnnlstm_classify_1d/best_cnnlstm_classify.pth \
    --ckpt-2d trained_model_cnnlstm_classify_2d/best_cnnlstm_classify.pth

# Late-fusion stacking on amplitude-augmented checkpoints (Section 6.4)
python cnn_lstm_stack_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --ckpt-1d trained_model_cnnlstm_aux_1daux/best_cnnlstm_aux.pth \
    --ckpt-2d trained_model_cnnlstm_aux_2daux/best_cnnlstm_aux.pth

# Gated fusion + amplitude aux combined (Section 6.4)
python cnn_lstm_classify_aux.py --dataset-dir ../../data_downloader/dataset_specdualaux_6s \
    --channels all --fusion gate --batch-size 32

# Seed-repeated verification (Section 6.6) -- rerun any of the above with --seed 1 / --seed 2
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --channels all --seed 1 --batch-size 32               # repeat for --fusion gate, and --seed 2
```

**Magnitude classification (Section 7):**

```bash
seismic-cli generate-regression-dataset \
    --eq-dir data/batched_waveforms/window_post_3s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --catalog-path catalogs/archive_superseded_2026-08-30/deprem_katalog_utc.csv \
    --station-catalog catalogs/istasyon_katalog.csv \
    --output-dir data/dataset_magclass_3s \
    --window-seconds 3 --encoding spectrogram --split-by event

cd ../cnn_earthquake/src
python cnn_magclass.py --dataset-dir ../../data_downloader/data/dataset_magclass_3s \
    --window-seconds 3 --save-dir trained_model_magclass_3s   # --mag-threshold to change the split point
```

**Magnitude regression, dual-channel + hop-length + per-component aux
(Sections 7.5-7.7):**

```bash
# --dual adds the raw-waveform 'seq' channel; --hop-length overrides the
# default n_fft//4=64 (5 time frames from a 3s window); omit for the 7.5
# baseline. --per-component-aux emits log_snr_0/1/2 instead of one averaged
# log_snr (7.7).
seismic-cli generate-regression-dataset \
    --eq-dir data/batched_waveforms/window_post_3s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --catalog-path catalogs/archive_superseded_2026-08-30/deprem_katalog_utc.csv \
    --station-catalog catalogs/istasyon_katalog.csv \
    --output-dir data/dataset_magclass_dual_3s_hop32 \
    --window-seconds 3 --encoding spectrogram --split-by event \
    --dual --hop-length 32                      # 32/16 for the sweep, omit for hop=64

cd ../cnn_earthquake/src
python cnn_lstm_regression.py \
    --dataset-dir ../../data_downloader/data/dataset_magclass_dual_3s_hop32 \
    --channels 2d+aux --seed 42                 # --seed 43/44 for the 3-seed confirmation
```

**Magnitude regression, 6-second window (Section 7.8):**

```bash
# Same architecture and --channels 2d+aux as above; only the window and
# source directory change. window_post_6s_anchored has fewer usable events
# than the 3s directory, so this is a smaller, not-strictly-comparable
# population (7.8's caveats) -- both floors move too, which is why they are
# reported alongside the model rather than assumed constant.
seismic-cli generate-regression-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --catalog-path catalogs/archive_superseded_2026-08-30/deprem_katalog_utc.csv \
    --station-catalog catalogs/istasyon_katalog.csv \
    --output-dir data/dataset_magclass_dual_6s \
    --window-seconds 6 --encoding spectrogram --split-by event --dual

cd ../cnn_earthquake/src
python cnn_lstm_regression.py \
    --dataset-dir ../../data_downloader/data/dataset_magclass_dual_6s \
    --channels 2d+aux --seed 42                 # --seed 43/44 for the 3-seed confirmation
```

**Station-disjoint verification (Section 7.9):** same trained-on-disk
tensors as above; `--split-by` re-partitions the manifest in memory, no
regeneration needed.

```bash
python cnn_lstm_regression.py \
    --dataset-dir ../../data_downloader/data/dataset_magclass_dual_3s \
    --channels 2d+aux --seed 42 --split-by both --seed-split 42   # 43/44 for the other partitions
# swap in dataset_magclass_dual_6s for the 6s rows
```

**Three-class risk classification (Section 8):**

```bash
# Both noise directories, and the dead-instrument filter of Section 8.4(a).
# --noise-dir takes a PARENT directory: file discovery is recursive, so this
# picks up noise_pre_3h and noise_pre_6h together (23,031 files).
seismic-cli generate-riskclass-dataset \
    --eq-dir data/batched_waveforms/window_post_3s_anchored \
    --noise-dir data/batched_noise_waveforms \
    --catalog-path catalogs/archive_superseded_2026-08-30/deprem_katalog_utc.csv \
    --station-catalog catalogs/istasyon_katalog.csv \
    --output-dir data/dataset_riskclass_3s_v2 \
    --window-seconds 3 --mag-threshold 4.0 --balance-ratio 4.0 --min-log-snr -3.0

cd ../cnn_earthquake/src

# The CNN (Section 8.3) -- reported for comparison; it loses to the scalars
python cnn_riskclass.py --dataset-dir ../../data_downloader/data/dataset_riskclass_3s_v2 \
    --window-seconds 3 --save-dir trained_model_riskclass_3s_v2

# The selected model (Section 8.5): two-stage, scalars only, class weights
# chosen by station-grouped CV internally, test evaluated once at the end
python riskclass_scalar.py --dataset-dir ../../data_downloader/data/dataset_riskclass_3s_v2
```

To reproduce the fetch reality of Section 8.2, target only the M ≥ 4 events
that have no downloaded waveform (`catalogs/target_missing_m4plus.csv`,
committed) and run `seismic_cli/src/download.py` against it; expect roughly one in ten
to return data.

Full CLI option reference: `data_downloader/README.md`.
