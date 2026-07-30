# Earthquake Detection from Raw Waveforms via Relative Angle Matrix (RAM) Image Encoding and CNN

_Independent side investigation — not a project deliverable. Written up because early results looked promising enough to be worth sharing._

## 1. Motivation

A recent paper proposes a "Relative Angle Matrix" (RAM) method for converting 1D bearing-vibration signals into 2D grayscale images for a CNN classifier, originally applied to mechanical fault diagnosis. Since seismic data is also fundamentally a time-series signal, and 3-component (Z/N/E) seismic stations naturally map onto a 3-channel image, this seemed like a plausible transfer target: **can the same RAM + CNN approach distinguish a real seismic event from ambient noise, directly from raw waveforms?**

This was explored independently, outside of assigned project tasks, using publicly available data (STEAD-style earthquake/noise HDF5 chunks and self-downloaded mseed data via FDSN), so there were no data-access or ethics constraints to navigate.

## 2. Method Overview

### 2.1 RAM transform

For a 1D signal window of length _m_, samples are z-score standardized, reshaped into a `(d, n)` matrix (`d` derived from a fixed target resolution `n`, not fixed independently — see Section 5.1 for why this distinction mattered), and the cosine angle between each column vector and the mean column vector is computed. The resulting angle differences form an `n × n` matrix, rendered to grayscale.

### 2.2 Building 3-channel images

Each of the three seismic components (Z, N, E) is RAM-transformed independently and stacked as R/G/B channels into one RGB image per window. (Open question, parked for a future ablation: whether combining the three raw channels _before_ the RAM transform — e.g. via vector magnitude — might preserve inter-channel amplitude relationships better than transforming each channel independently. Not yet tested.)

### 2.3 CNN architecture

A ResNet-style CNN with Squeeze-and-Excitation blocks, trained as a binary classifier (earthquake vs. noise) on the resulting RGB images.

## 3. Dataset & Pipeline

- **Data sources:** STEAD-style HDF5 earthquake/noise chunks, and a self-built pipeline downloading raw mseed data via FDSN (obspy) for a broader Turkish-catalog dataset.
- **Window lengths tested:** 60s, 6s, 3s (shorter windows chosen specifically to test how much detection latency could be reduced).
- **Splits:** train/val/test, **70/15/15**.

## 4. Baseline for comparison

**Classic STA/LTA** (short-term/long-term average ratio), the standard classical seismic trigger algorithm, was implemented as the point of comparison — chosen because it's the real, established baseline any new detection method should be measured against, not an arbitrary strawman.

Critically, STA/LTA was always evaluated on the **exact same test windows** (same file, same station, same window index) as the CNN, reconstructed directly from the raw mseed/HDF5 data — not a separately-sampled comparison set.

## 5. The debugging journey (worth including — this is most of the actual rigor)

Several serious methodological bugs were found and fixed before any result could be trusted. Listing them is useful context for how much an initial "good-looking" number can mislead:

### 5.1 Fixed-resolution collapse at short windows

Early version used a fixed `d` (row count) for the RAM reshape regardless of window length. At short windows (e.g. 3s), this collapsed images down to as little as 3×3 pixels — destroying almost all spatial information. **Fix:** derive `d` from a fixed target output resolution instead, so every window length produces a consistently informative image.

### 5.2 Station-identity leakage

Initial dataset splits were done by _event/file_, not by _station_. Since the same physical station records many different events, this let identical instruments appear in both train and test — meaning the model could partly be learning "which station is this" rather than genuine signal detection. **Fix:** rewrote the splitting logic to allocate whole _stations_ to a single split, guaranteeing test-time stations were never seen during training.

### 5.3 Single-station dominance

Even after fixing splitting, one station's disproportionately large event count could single-handedly make up most of a split (both on the earthquake and, more severely, the noise side, where a single long trace can produce thousands of overlapping sliding windows). **Fix:** added a per-station window cap, tuned so val/test require genuine multi-station diversity to reach their target size, not just one or two big contributors.

### 5.4 Silent manifest mislabeling

A dataset manifest (used to align STA/LTA baseline comparisons with CNN test data) mislabeled earthquake-class entries as noise whenever a station legitimately belonged to both classes (common, since ~97% of earthquake stations also had matching noise recordings). **Fix:** the correct class label is now captured at the point of assignment, not reconstructed afterward from a lossy per-station lookup.

### 5.5 Origin-anchored short windows missed the actual arrival

Short windows (3s/6s) were originally sliced starting at event _origin time_, not arrival time. For stations near the edge of the search radius, the P-wave arrival can land after the whole window ends — meaning a meaningful fraction of "earthquake" short windows contained no seismic signal at all, badly corrupting the label. **Fix:** re-derived arrival-anchored short windows directly from already-downloaded longer (60s) raw data, using a coarse STA/LTA pick to locate the approximate arrival — no redownload required.

### 5.6 Noise/earthquake station mismatch

Publicly-sourced noise data only overlapped with ~47% of earthquake-recording stations, meaning the model rarely saw both classes from the same instrument. **Fix:** built a targeted downloader that fetches noise directly from the same stations that recorded earthquakes, raising overlap to ~97%.

## 6. Results

### 6.1 60-second windows — clear, validated win over STA/LTA

| Metric                 | CNN                                              | STA/LTA    |
| ---------------------- | ------------------------------------------------ | ---------- |
| Accuracy               | _[insert final confirmed number from your logs]_ | _[insert]_ |
| ROC-AUC                | _[insert]_                                       | _[insert]_ |
| Recall (earthquake)    | _[insert]_                                       | _[insert]_ |
| Precision (earthquake) | _[insert]_                                       | _[insert]_ |

Across multiple validated regenerations of the dataset (after all fixes in Section 5), the CNN consistently and substantially outperformed STA/LTA on AUC — a gap in the range of roughly **0.10–0.15 AUC points**, evaluated on a properly station-disjoint test set with no single station dominating either class. This is the headline result.

### 6.2 6-second and 3-second windows — inconclusive, still in progress

At shorter windows, results were much closer to STA/LTA and, in some iterations, worse. Two distinct causes were identified and partially addressed (see 5.5, 5.6), and the most recent runs show meaningful improvement in recall after fixes — but station diversity on the noise side at short windows remains thin (as few as 2–4 distinct noise stations contributing to val/test even after capping), so these numbers should be treated as preliminary, not final.

**Working hypothesis for why short windows are structurally harder** (independently arrived at, worth stating as the current best explanation): STA/LTA's core advantage is a long-term running baseline (the LTA) — it always has _some_ memory of what "normal" looks like at that station. The current RAM pipeline standardizes each window purely against itself, with no memory of longer-term station behavior. At 60s, a window is long enough that this self-normalization is fairly stable; at 6s or 3s, it's a much smaller, noisier sample to self-normalize against. This gap should, in principle, grow as windows shrink — consistent with the observed pattern.

## 7. Proposed next steps

1. **Incorporate a long-term per-station noise baseline** into preprocessing (analogous to STA/LTA's LTA) rather than relying purely on per-window self-standardization. This directly targets the hypothesis in 6.2.
2. **Tighten noise station diversity further** at short window lengths (lower per-station cap, or source additional noise stations) before drawing final conclusions about 6s/3s performance.
3. **Ablation: per-channel RAM vs. combine-then-RAM.** Test whether combining Z/N/E into a single signal (e.g. vector magnitude) before the RAM transform, rather than transforming each channel independently, better preserves cross-channel amplitude information.
4. **Held-out-station sanity check** at each window length, to further confirm generalization beyond the station-disjoint test split already in place.

## 8. Caveats

- This is a personal side project, not a project deliverable, and has not been reviewed by anyone else.
- 60s results are well-validated; 6s/3s results are preliminary and still being iterated on.
- Sample sizes, especially for noise-station diversity at short windows, are currently modest — worth keeping in mind when reading precision/recall to more than one significant figure.
