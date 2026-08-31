# Hand Features + LSTM vs. Raw Waveform + CNN-LSTM — Head-to-Head

One-line summary: on the same continuous KO.GEDZ data, same target, same
split, same 3-seed ensembling — **the raw-waveform CNN-LSTM clearly beat
the hand-crafted-feature LSTM**, but there's one honest asymmetry in the
comparison worth stating up front, not glossing over.

## The setup (identical for both)

| | |
|---|---|
| Station | KO.GEDZ (Aegean zone) |
| Period | May 1, 2024 – Feb 22, 2025 (~10 months, 7,152 hours) |
| Target | Will a M≥4.5 event occur in the AEGEAN zone within 30 days? (dense, no declustering — the validated target from `cnn_lstm_forecast.py`) |
| Sequence length | 24 hours of context per prediction |
| Split | Chronological 70/15/15 — train 4,990h (34.2% positive) / val 1,069h (39.1%) / test 1,070h (89.7%) |
| Evaluation | 3-seed ensemble (mean of each seed's predicted probability) — individual seeds are too noisy to trust alone on this much data |

## The numbers

| Model | Input | Test AUC (3-seed ensemble) | Per-seed spread |
|---|---|---|---|
| LSTM + hand features | Hourly mean of Sismokaos-featureExtract's features (STA/LTA, Hjorth, permutation entropy, spectral centroid, cross-corr — 50s resolution, mean-pooled to 1 vector/hour) | **0.558** | 0.373 – 0.569 (wide, unstable) |
| **CNN-LSTM, raw waveform** | Full continuous (3, 18000)-sample hourly waveform, no hand features, no pooling | **0.787** | 0.714 – 0.791 (tight, stable) |
| *floor:* base-rate | — | 0.500 | — |
| *floor:* persistence | — | 0.343 | — |

**The raw CNN-LSTM wins by +0.23 AUC, and it's also the more stable model** — tighter seed spread (0.077 vs. 0.196), which is itself evidence this is a real learned signal rather than a lucky fit.

## Architectures

Both models share the same skeleton — a per-hour encoder feeding a 24-hour
LSTM+attention over the sequence of hourly summaries — and differ only in
what the per-hour encoder is.

**LSTM + hand features** (`feature_lstm_forecast.py`, `ForecastLSTM`, 20,257 params):

```
Hourly mean of 102 hand-crafted features
(51 stats × {E,N,Z}+cross-corr, plus 51 epoch-to-epoch deviations)
      │
      ▼  (no learned encoder here — the 102-dim vector IS the hourly summary)
Sequence of 24 hourly feature vectors, shape (24, 102)
      │
      ▼  LSTMAttentionBranch (19,648 params)
   one summary vector for the 24-hour window
      │
      ▼  head: LayerNorm → Dropout → Linear → GELU → Dropout → Linear→1 (609 params)
P(M≥4.5 in AEGEAN within 30 days)
```

The "encoder" step is just Sismokaos-featureExtract's `feature_functions.py`
— STA/LTA, Hjorth parameters, permutation entropy, spectral centroid,
cross-axis correlation, etc., computed by formula, not learned. All the
network's own parameters go into the LSTM+attention step.

**CNN-LSTM, raw waveform** (`raw_cnn_lstm_forecast.py`, `RawCNNLSTM`, 22,721 params):

```
One hour of raw waveform (3 components, 18,000 samples @ 5Hz)
      │
      ▼  RawWaveformEncoder: 4× [Conv1d → BatchNorm1d → GELU → Dropout],
      │  stride 4 each (18000→4500→1125→281→70), channels 3→16→32→32→32,
      │  then AdaptiveAvgPool1d(1)                        (11,424 params)
   one 32-dim LEARNED embedding for that hour
      │
      │  ...repeated for all 24 hours (batched: reshape (B,24,3,18000) ->
      │     (B×24,3,18000), one CNN pass, reshape back to (B,24,32))
      ▼
Sequence of 24 hourly embeddings, shape (24, 32)
      │
      ▼  LSTMAttentionBranch, same class as the other model (10,688 params)
   one summary vector for the 24-hour window
      │
      ▼  head: identical structure to the other model             (609 params)
P(M≥4.5 in AEGEAN within 30 days)
```

Here the CNN *is* the encoder — it replaces `feature_functions.py`'s
formulas with a learned compression of the raw amplitude signal into a
32-dim vector, trained end-to-end against the same forecasting label rather
than against any hand-picked statistical definition.

**The load-bearing difference:** both models spend roughly the same total
parameter budget (~20-23K) and use the *identical* `LSTMAttentionBranch`
class for the cross-hour reasoning step — the only thing that changed is
whether the hourly summary comes from hand formulas (0 learned params,
possibly discarding information via mean-pooling) or a small trained CNN
(11,424 learned params, no pooling, full within-hour signal preserved).

## The one asymmetry to say out loud

**This isn't a perfectly clean "hand features vs. learned representations" comparison — it also conflates time resolution.** The hand-feature pipeline computes features every 50 seconds (514,827 rows total) but was *mean-pooled down to one vector per hour* for the LSTM, discarding all within-hour temporal structure, purely to keep the sequence length tractable (24 steps instead of ~1,728). The raw CNN-LSTM saw the *full* continuous signal for every hour — no resolution loss at all. Some of the CNN-LSTM's advantage is plausibly "more temporal detail," not purely "CNN-learned features beat hand-picked ones." A fully controlled version of this test would feed the hand features at their native 50s resolution too (a longer, more expensive sequence) before concluding the representation itself is why one wins.

## What this does and doesn't establish

**Does:** on this one station, this one ~10-month period, this one target — a raw-waveform CNN-LSTM clearly outperforms an LSTM over hand-crafted, hourly-pooled features, and does so more stably across seeds. That's a real, presentable result.

**Doesn't (yet):** that hand-crafted feature extraction should be abandoned generally. This is one comparison, on one dataset slice, with the resolution asymmetry above unresolved, going *against* this project's own repeated pattern elsewhere (simple/hand-crafted features beating fancier models on nearly every other task this session — Task 3's scalar model, the magnitude-head ridge floor, spectrogram+aux beating dual-channel fusion). If this result holds up on a different station or period, or once the resolution asymmetry is controlled for, that's a much stronger claim. Tonight it's the first real data point in that direction, not a settled verdict.

## Reproduce it

```bash
# Hand-feature LSTM
python feature_lstm_forecast.py \
    --features-csv ../../Sismokaos-featureExtract/results/GEDZ/GEDZ_2024_05_01-2025_02_22_ENZ_features.csv \
    --catalog-path ../../data_downloader/catalogs/catalog_current.csv \
    --seq-hours 24 --hidden 16 --weight-decay 0.1 --dropout 0.5

# Raw-waveform CNN-LSTM
python raw_cnn_lstm_forecast.py \
    --data-root ../../Sismokaos-featureExtract/data/aegean_2024_2025 \
    --catalog-path ../../data_downloader/catalogs/catalog_current.csv
```

Both scripts share the same labeling/splitting/ensembling code
(`feature_lstm_forecast.py`'s `label_hours`, `load_aegean_events`,
`days_since_prev_major`, `safe_auc` — imported directly by
`raw_cnn_lstm_forecast.py`) so the comparison is apples-to-apples on
everything except the per-hour representation and its resolution.
