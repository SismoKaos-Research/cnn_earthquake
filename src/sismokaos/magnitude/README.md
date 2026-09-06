# Magnitude

How big was it, from a short window? Regression and binned classification,
scored against a ridge(amplitude, distance) physics floor.

## Reports

- [`MAGNITUDE_CNN_CHEATSHEET.md`](MAGNITUDE_CNN_CHEATSHEET.md) — Magnitude Regression CNN — Cheat Sheet

## Scripts

- **`cnn_lstm_regression.py`** — Dual-channel CNN + LSTM/self-attention magnitude regression.
- **`cnn_magclass.py`** — Magnitude-class (binary) classification from encoded seismic windows.
- **`cnn_regression.py`** — Magnitude regression from encoded seismic windows.
- **`cnn_riskclass.py`** — Three-class risk classification (noise / low-risk / high-risk) from encoded seismic windows.
- **`riskclass_scalar.py`** — Two-stage scalar risk classifier: the best-performing model found for the three-class (noise / low-risk / high-risk) task.

## Running

Shared code lives in `sismokaos`, installed editable (`uv pip install -e .`),
so scripts run from anywhere:

```bash
python3 src/sismokaos/magnitude/<script>.py --help
```

Project-wide results spanning several families are in [`docs/`](../../docs/):
`report.md` (the full write-up) and `accuracy_summary.md` (headline numbers per task).
