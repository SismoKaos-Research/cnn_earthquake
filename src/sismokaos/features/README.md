# Features and datasets

Feature engineering, recursive feature elimination, and the builders that
turn raw archives into model-ready arrays. Nothing here trains a model.

## Scripts

- **`build_offline_features.py`** — Offline feature builder for preprocessed Parquet waveforms.
- **`catalog_feature_processor.py`** — Extracts rolling physical features from a seismic catalog.
- **`catalog_feature_rfe.py`** — Recursive feature elimination over the 11 catalog features in `cnn_lstm_catalog_waveform_fusion.build_catalog_features`, using LightGBM gain-importanc.
- **`consolidate_hourly_raw.py`** — Consolidates per-hour .npy struct-array files (E/N/Z fields, produced by either the real 5Hz pipeline or a gap-only preprocessing step whose script is not in this repo) into one pre-.
- **`lgbm_cluster.py`** — Feature distillation and clustering on CNN-extracted feature vectors.
- **`parquet_feature_rfe.py`** — Recursive Feature Elimination operating on pre-computed Parquet features.
- **`parquet_to_memory.py`** — Converts a preprocessed Parquet time-series into a flat binary memmap.
- **`seismic_fusion_dataset.py`** — _(undocumented)_
- **`waveform_dwt_features.py`** — Hand-engineered DWT/spectral waveform features, per Tables 3-5 of Bhatia, Ahanger & Manocha (2023) "Artificial intelligence based real-time earthquake.

## Running

Shared code lives in `sismokaos`, installed editable (`uv pip install -e .`),
so scripts run from anywhere:

```bash
python3 src/sismokaos/features/<script>.py --help
```

Project-wide results spanning several families are in [`docs/`](../../docs/):
`report.md` (the full write-up) and `accuracy_summary.md` (headline numbers per task).
