# Short-Window Earthquake Detector — One Page

## The model

A 2D CNN over a spectrogram. **115,459 parameters.** Input is a 6-second,
three-component, 100 Hz seismogram window; output is one logit — earthquake or
noise.

```
spectrogram (3, 129, 10)
  → Conv(3→32)  BN GELU
  → Conv(32→64,  stride 2)  BN GELU  Dropout
  → Conv(64→128, stride 2)  BN GELU
  → GlobalAvgPool → 128-d
  → LayerNorm → Linear(96) → GELU → Linear(1)
```

There is also an optional LSTM+attention branch over the raw waveform. It does
not help, so the shipped model is the CNN alone.

## The data

| | |
|---|---|
| Source | Kandilli (KOERI) regional catalogue + KOERI FDSN waveforms |
| Events | 33,795 across **181 stations** (Türkiye) |
| Magnitude | median 2.30 (63.5% below M 2.5) |
| Distance | median 38.6 km, capped at ~56 km |
| Positives | 6 s cut 2.0 s before the TauP-predicted P arrival |
| Negatives | 300 s windows from ~3 h before each event, screened against a 482,898-event catalogue |
| Splits | **station-disjoint** — 38,247 / 9,415 / 7,906 windows per class |

## How a window becomes a prediction

1. Detrend → 5% Hann taper → 1–45 Hz Butterworth bandpass (zero-phase) → resample to 100 Hz
2. STFT → log-power spectrogram, then subtract that station's own median noise profile
   (result is **dB above the station's noise floor**, so instrument gain cancels)
3. CNN → logit → sigmoid, threshold 0.50

Trained with AdamW (lr 2e-4, wd 3e-2), cosine schedule, batch 32, early stopping
on validation ROC-AUC. Three seeds, probabilities averaged.

## What it does

| | ROC-AUC |
|---|---|
| **In-domain test set** | **0.9892 ± 0.0003** |
| — single amplitude statistic, no learning | 0.9049 |
| — classical STA/LTA | 0.8193 |
| **STEAD, no retraining** (1,155 stations, 96 networks) | **0.9971** |

Accuracy 0.9679 · MCC 0.9369 · PR-AUC 0.9921.

**Read the second row.** These windows are separated largely by loudness, so the
honest bar is what a single amplitude scalar achieves — 0.9049, not the 0.5000
majority-class bar. The model's real contribution is **+0.0847**.

## Limits

- Trained on M ≥ 2.0 within ~56 km; accuracy degrades into smaller magnitudes
  (0.948 below M 1.0 on STEAD).
- ROC-AUC transfers across corpora; **thresholded metrics do not** — recalibrate
  before quoting accuracy on new data.
- Arrival timing is good to ~0.6 s. Fine for detection, not for phase picking.
- Not yet benchmarked against PhaseNet / EQTransformer / GPD.
