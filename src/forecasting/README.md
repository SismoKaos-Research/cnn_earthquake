# Forecasting

Will an event occur in some forward window? Catalog history, raw
waveform history, and fusions of the two. This is the family with the
weakest results and the most careful baselines -- see the persistence and
Omori floors, not a majority-class bar.

## Reports

- [`FEATURE_LSTM_CHEATSHEET.md`](FEATURE_LSTM_CHEATSHEET.md) — Hand-Feature LSTM Forecaster — Cheat Sheet
- [`LSTM_FEATURES_VS_RAW_CHEATSHEET.md`](LSTM_FEATURES_VS_RAW_CHEATSHEET.md) — Hand Features + LSTM vs. Raw Waveform + CNN-LSTM — Head-to-Head
- [`catalog_forecast_report.md`](catalog_forecast_report.md) — Catalog Forecasting, Retargeted onto a Neural Architecture

## Scripts

- **`catalog_forecast_predict.py`** — Combined "when and how big" prediction: the validated dual-channel network for P(M >= threshold within horizon_days), paired with a simple ridge regre.
- **`catalog_lgbm_forecast.py`** — LightGBM directly on the same catalog features `cnn_lstm_catalog_waveform_fusion.py`'s MLP catalog branch uses -- a non-neural comparison, matching th.
- **`cnn_chunk_forecast.py`** — CNN-only classifier over non-overlapping multi-day chunks: does an M>=threshold AEGEAN event occur in the `--horizon-days` right after this chunk ends.
- **`cnn_lstm.py`** — Dual-channel CNN + LSTM/self-attention model for time-to-major-earthquake risk.
- **`cnn_lstm_catalog_waveform_fusion.py`** — Catalog-LSTM + raw-waveform-CNN fusion forecaster, trained end-to-end.
- **`cnn_lstm_daily_3class.py`** — CNN+LSTM 3-class monthly chunk classifier: does a M>=threshold AEGEAN event occur INSIDE this chunk ("event"), shortly AFTER it ("event_after"), or ne.
- **`cnn_lstm_forecast.py`** — Dual-channel CNN + LSTM/self-attention model, retargeted onto the validated dense per-zone forecasting target: will a M >= threshold event occur in th.
- **`cnn_lstm_loeo.py`** — Leave-one-event-out (LOEO) cross-validation for the dual-channel risk model.
- **`cnn_lstm_lstm_multiweek.py`** — Hierarchical CNN -> LSTM(within-week) -> LSTM(across-weeks) 3-class classifier, pooling BOTH stations (BODT + DAT) for more independent samples than e.
- **`cnn_proximity_classify.py`** — CNN-only classifier: is this hour close (in either time direction) to an M>=threshold AEGEAN event -- a regime-recognition task, not a forecast.
- **`feature_gru_tcn.py`** — GRU and TCN forecasters trained directly on Sismokaos-featureExtract's hand-crafted continuous features.
- **`feature_lstm_forecast.py`** — LSTM forecaster trained directly on Sismokaos-featureExtract's hand-crafted continuous features (STA/LTA, Hjorth, permutation entropy, spectral centro.
- **`gru_cnn.py`** — _(undocumented)_
- **`gru_cnn_train.py`** — _(undocumented)_
- **`next_event_regression.py`** — Regression to the next major event: predict days until the next M>=threshold AEGEAN event for each hour, using catalog features and optionally DWT wav.
- **`raw100hz_cnn_lstm_forecast.py`** — Same setup as raw_cnn_lstm_forecast.py (KO station, M>=4.5 dense forecast target, walk-forward CV) but reads scripts/gap_only_preprocess.py's output i.
- **`raw_cnn_lstm_forecast.py`** — Raw-waveform CNN-LSTM forecaster (or feature CSV loader for massive files).
- **`waveform_lgbm_forecast.py`** — LightGBM on the DWT/spectral waveform features from `waveform_dwt_features.py` (Bhatia et al.

## Running

Shared code lives in `seismolib`, installed editable (`uv pip install -e .`),
so scripts run from anywhere:

```bash
python3 src/forecasting/<script>.py --help
```

Project-wide results spanning several families are in [`docs/`](../../docs/):
`report.md` (the full write-up) and `accuracy_summary.md` (headline numbers per task).
