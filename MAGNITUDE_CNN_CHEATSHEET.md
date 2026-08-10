# Magnitude Regression CNN — Cheat Sheet

One-line summary: **a CNN over a spectrogram image of a short seismic
window, plus two physics scalars (amplitude, distance), predicts earthquake
magnitude directly.** Full detail: `report.md` §7.5–7.9. Raw numbers:
`src/cnn_lstm_regression_results.csv`.

## What it is

```
3-6 s waveform (Z, N, E)
      │
      ▼  spectrogram (log-power, per component)
(3, freq, time) image  ──┐
                          ├──► concat ──► dense head ──► magnitude
  [log_snr, log_distance]─┘
```

- **Spectrogram CNN** (`CNNBranch`): ordinary conv+pool stack, same as an
  image classifier, over the 3-channel spectrogram.
- **Aux scalars**: `log_snr` (amplitude vs. that station's noise floor) and
  `log_distance` (epicentral distance) — the two terms local magnitude is
  physically built from.
- **No LSTM.** The code (`cnn_lstm_regression.py`) is built as a *dual*-
  channel net with a raw-waveform LSTM+attention branch too, but that
  branch is switched OFF (`--channels 2d+aux`) — adding it back makes
  results worse. The waveform-LSTM branch is dead weight for this task.
- Trained with L1 loss (MAE) directly on magnitude.

## The numbers (event-disjoint, 3-seed mean)

| window | MAE | MAPE | R² | floor: guess-the-mean | floor: physics formula (ridge) |
|---|---|---|---|---|---|
| 3 s | 0.197 | 7.8% | 0.669 | 0.350 | 0.308 |
| **6 s** | **0.182** | **7.2%** | 0.744 | 0.376 | 0.318 |

- Error is ~**40–50% lower** than either floor.
- Typical prediction is within **~7% of true magnitude**; worse on bigger
  events (M≥3 MAPE ~9–10%) than smaller ones (M<3 MAPE ~7%).
- 6s is the current best; 3-seed spread is ≤0.001 both rows (tight, not luck).

## Robustness check: station memorization?

Ran doubly-disjoint splits (never-seen stations **and** never-seen events;
3 independent partitions per window length — report.md §7.9).

- **Cleared.** Every partition, both window lengths, still beats the
  physics-formula floor by a wide margin on unseen stations.
- **Caveat:** the 6s-beats-3s margin itself doesn't survive this stricter,
  noisier test (3s avg 0.226, 6s avg 0.218 — overlapping). Say: *"6s is the
  best number under the clean comparison; whether it's reliably better than
  3s needs more testing than we've done."*

## Things that were tried and did NOT help (don't re-try without a reason)

| Lever | Result |
|---|---|
| Finer spectrogram time resolution (`hop_length` 64→32→16) | 0.197→0.195, inside noise |
| Per-component aux (log_snr × Z/N/E instead of 1 averaged) | 0.195→0.196, no change |
| Re-adding the LSTM/raw-waveform branch (`--channels all`) | Worse at both 3s (0.202) and 6s (0.189) |
| Dual-channel vs. single-channel spectrogram-only | Tied (0.202 vs 0.205) |

The one lever that worked: **more seconds of window**, not a better encoding
of the same seconds (§7.8's controlled comparison — same time-frame count,
6s still won).

## Reproduce it

```bash
# from data_downloader/
seismic-cli generate-regression-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --station-catalog catalogs/istasyon_katalog.csv \
    --output-dir data/dataset_magclass_dual_6s \
    --window-seconds 6 --encoding spectrogram --split-by event --dual

# from cnn_earthquake/src/
python cnn_lstm_regression.py \
    --dataset-dir ../../data_downloader/data/dataset_magclass_dual_6s \
    --channels 2d+aux --seed 42
```

## Caveats worth remembering

- Single region (Marmara network), single corpus — no external test set.
- `distance_km` missing for ~15% of windows (station-catalog coverage gap,
  not investigated further).
- No literature benchmark comparison done (don't claim "state of the art").
- Everything above is single-architecture-family; only this CNN+aux design
  was tested at this rigor.
