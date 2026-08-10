# Hand-Feature LSTM Forecaster — Cheat Sheet

One-line summary: **an LSTM trained directly on Sismokaos-featureExtract's
hand-crafted continuous features** (STA/LTA, Hjorth parameters, permutation
entropy, spectral centroid, cross-axis correlation — real KO.GEDZ data,
Aegean zone) forecasts the same validated "will a M≥4.5 event occur within
30 days" target the catalog-based model uses, but from raw continuous
waveform-derived features instead of catalog event statistics.

## What it is

```
Continuous KO.GEDZ waveform (FDSN, ~10 months: May 2024 - Feb 2025)
      │
      ▼  Sismokaos-featureExtract pipeline (preprocess + feature_functions)
Hourly hand-crafted feature vectors (STA/LTA, Hjorth, permutation entropy,
spectral centroid, cross-axis correlation — 3 components × ~17 stats + DEV)
      │
      ▼  24-hour sliding sequences
LSTM + attention (reused cnn_lstm.py's LSTMAttentionBranch, 20K params)
      │
      ▼
P(M≥4.5 in AEGEAN zone within 30 days)
```

Same target definition as `cnn_lstm_forecast.py`'s catalog-based model — the
one difference is the input: hand-crafted features on real continuous
waveform, not catalog event sequences. This is a genuinely different
feature source answering the same question.

## The numbers

3-seed ensemble (mean of each seed's predicted probability — needed because
individual seeds are noisy, see below), test set = 1,070 held-out hours:

| | AUC |
|---|---|
| base-rate (majority) floor | 0.500 |
| persistence floor | 0.343 |
| **hand-feature LSTM (3-seed ensemble)** | **0.558** |

**Beats both floors by +0.058 AUC.** Modest, not dramatic — but real,
non-degenerate, and produced by a pipeline that pulls real data, extracts
real features, and trains a properly-validated model end to end.

## The honest part: per-seed instability

| seed | test AUC |
|---|---|
| 42 | 0.569 |
| 43 | 0.373 (fails to beat the floor on its own) |
| 44 | 0.531 |

Individual seeds swing by ~0.20 AUC — too much to trust any single run.
**The ensemble (averaging 3 seeds' predicted probabilities) is the actual
result being reported**, not any one seed cherry-picked. This is a standard,
legitimate variance-reduction technique, not a workaround — report it as
"3-seed ensemble," not as a single confident number.

## Why the data needed two iterations to get right

- First attempt: 100 days (Nov 2024 - Feb 2025) — landed almost entirely
  inside one sustained earthquake swarm. Validation split came out 100%
  positive (single-class), which silently broke checkpoint selection (val
  AUC undefined every epoch). Fixed by falling back to validation loss for
  checkpoint selection when val AUC is undefined.
- Real fix: extended back to May 2024, adding two isolated events (Jun 25,
  Jul 22) and a genuine ~5-month quiet stretch before the swarm. This is
  what actually matters — genuine variety in the training signal, not just
  more days of the same pattern.
- Model size was cut down hard (20K params, 24-hour context, dropout 0.5,
  weight-decay 0.1) to match the *actual* independent information content
  of ~10 months of hourly data — a large LSTM here just memorizes and
  produces AUC far below chance (0.06 in the failed first attempt).

## Caveats worth stating plainly if asked

- Single station (KO.GEDZ), single ~10-month period. Not yet tested on a
  different station or period.
- Ensemble, not a single trustworthy model — the instability itself is a
  real finding, not just noise to average away.
- Test set skews positive (89.7%) — few negative examples, so the AUC
  estimate itself has real sampling variance on top of the training
  instability.
- Compare honestly to the catalog-based model: that one gets 0.73 pooled AUC
  (report.md/catalog_forecast_report.md) on a much larger, more diverse
  dataset (4 zones, full catalog history). This hand-feature model is not
  competing with that number — it's a first, much smaller-scale test of a
  different feature source on the same question.

## Reproduce it

```bash
# 1. Pull continuous data (Sismokaos-featureExtract/, testing branch)
.venv/bin/python raw_download_gedz.py          # Nov 2024 - Feb 2025
.venv/bin/python raw_download_gedz_extend.py   # May 2024 - Nov 2024 (the fix)

# 2. Preprocess + extract features (100 subprocess calls, then one combined pass)
.venv/bin/python run_batch_pipeline.py

# 3. Train (cnn_earthquake/src/)
python feature_lstm_forecast.py \
    --features-csv ../../Sismokaos-featureExtract/results/GEDZ/GEDZ_2024_05_01-2025_02_22_ENZ_features.csv \
    --catalog-path ../../data_downloader/catalogs/data_large.csv \
    --seq-hours 24 --hidden 16 --weight-decay 0.1 --dropout 0.5
```

All work is on the `testing` branch in `Sismokaos-featureExtract` —
`main`/production is untouched.
