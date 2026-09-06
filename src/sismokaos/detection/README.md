# Detection

Is this short window an earthquake or noise? The strongest results in the
project live here, but so does the amplitude floor: a single loudness
scalar scores ~0.90-0.95 on these benchmarks, so read every number against
it rather than against 0.5.

## Reports

- [`REPORT_event_noise_detector.md`](REPORT_event_noise_detector.md) — A Station-Disjoint Benchmark for Short-Window Earthquake Detection, with Conditional Baselines
- [`cnn_lstm_classify.md`](cnn_lstm_classify.md) — 2D Branch
- [`spectrogram_classifier_report.md`](spectrogram_classifier_report.md) — The Spectrogram Classifier

## Scripts

- **`cnn_from_tensor.py`** — Trains the seismic classifier on spectrogram tensors (.pt) produced by `seismic-cli generate-spectrogram-dataset`.
- **`cnn_lstm_classify.py`** — Dual-channel CNN + LSTM/self-attention (1D2D-EDL, Wang & Zhao 2025) for earthquake-vs-noise classification on short arrival-anchored windows, rather t.
- **`cnn_lstm_classify_aux.py`** — Dual-channel CNN + LSTM/self-attention classifier, plus the amplitude aux branch that fixed the plain RAM classifier (see cnn_ram_aux.py: test AUC 0.8.
- **`cnn_lstm_cross_station.py`** — Cross-station generalization test: train the raw-waveform CNN+LSTM forecaster entirely on one station's archive, evaluate on a second station's archiv.
- **`cnn_lstm_pre_event_classify.py`** — CNN+LSTM classifier over CURATED chunks: reuses the exact architecture and training loop from raw_cnn_lstm_forecast.py (RawCNNLSTM, train_one_seed, ru.
- **`cnn_lstm_stack.py`** — Late-fusion stacking on FROZEN 1d-only and 2d-only checkpoints.
- **`cnn_lstm_stack_aux.py`** — Late-fusion stacking on FROZEN amplitude-augmented single-branch checkpoints (`--channels 1d+aux` / `--channels 2d+aux`).
- **`cnn_ram_aux.py`** — RAM-image classifier with amplitude-scalar auxiliary inputs -- the direct fix for RAM's diagnosed blind spot.
- **`cnn_run.py`** — Runs inference with a trained seismic classifier loaded from the FULL pickled model object (`trained_model/full_model.pth`), as saved by `training.run.
- **`cnn_run_from_state.py`** — Runs inference with a trained `ImprovedSeismicCNN` loaded from its state-dict checkpoint (`trained_model/best_seismic_model.pth`, as saved by `trainin.
- **`cnn_train.py`** — Trains the seismic classifier on RAM images (PNG) produced by `seismic-cli generate-dataset`.
- **`evaluate_cross_corpus.py`** — Evaluates already-trained `cnn_lstm_classify.py` checkpoints on a dataset they were never trained on -- the cross-corpus test.

## Running

Shared code lives in `sismokaos`, installed editable (`uv pip install -e .`),
so scripts run from anywhere:

```bash
python3 src/sismokaos/detection/<script>.py --help
```

Project-wide results spanning several families are in [`docs/`](../../docs/):
`report.md` (the full write-up) and `accuracy_summary.md` (headline numbers per task).
