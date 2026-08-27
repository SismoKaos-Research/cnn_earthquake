# Tests

```bash
uv pip install --python .venv/bin/python pytest   # once
.venv/bin/python -m pytest                        # ~4 s, no GPU, no data
```

The whole suite runs on synthetic arrays and `tmp_path`. Nothing here touches
the datasets under `Sismokaos/data_downloader/`, loads a checkpoint, or needs
CUDA, so it is safe to run mid-experiment.

## What is covered, and why these things

The suite is not aimed at coverage. It is aimed at the specific failures this
project has actually had, all of which were **silent** — they produced a
plausible number rather than an error.

| File | Guards against |
|---|---|
| `test_checkpoints.py` | Ensembling checkpoints from the wrong arm. A save dir routinely holds `1d`, `2d` and `all`; `cnn` is a prefix of `cnn-lstm`; the run tag has grown over time, so old names must still resolve. |
| `test_metrics.py` | A floor reported below chance. `safe_auc(oriented=True)` is the correction five forecasting scripts were missing — an anti-correlated baseline is exactly as exploitable as a correlated one. |
| `test_catalog.py` | Leakage at a window boundary. The forward label is `(t, t+w]` and the trailing count is `(t-w, t]`, so an event exactly at `t` is past, never future. |
| `test_splits.py` | A split that lets a model see its own future, and an embargo that does not actually separate adjacent hours. |
| `test_label_sweep.py` | The cell-selection arithmetic behind M≥2.5 / 400 km / 6 h: the forward positive rate, and the orientation-corrected persistence floor. |
| `test_amplitude_bins.py` | Reading a wide amplitude bin as evidence about waveform shape. Equal-count bins are not equal-width; the top decile spans ~530×. |
| `test_imports.py` | A module that runs an experiment when imported. `label_sweep.py` used to read the 482k-row catalogue and sweep 140 cells at import time. |

## Adding to `EXECUTES_ON_IMPORT`

`test_imports.py` holds a list of flat `__main__`-only scripts. Adding an entry
should be a deliberate decision with a docstring to back it; removing one is
always an improvement. A module on that list cannot be unit-tested at all.
