# Earthquake Detection, Classification, and Forecasting from Seismic Waveforms

CNN/LSTM models for four related seismic ML tasks, all trained on real
KOERI (KO network) waveform data: earthquake-vs-noise **detection**,
short-window **magnitude classification/regression**, catalog-based event
**forecasting** ("will a M≥threshold event occur in this zone within N
days?"), and **peak ground motion** prediction. The architecture is the
dual-channel CNN+LSTM+attention design from Wang & Zhao (2025, *Applied Soft
Computing* 172) — a 1D LSTM/self-attention branch over the raw or
feature-derived waveform, a 2D CNN branch over a RAM-image or spectrogram
encoding, fused and classified/regressed — adapted, corrected, and extended
across all four tasks.

**Start here:** [`docs/report.md`](docs/report.md) is the full technical writeup —
architecture, every experiment, every defect found and fixed, and the
reproduction commands for each result. [`docs/accuracy_summary.md`](docs/accuracy_summary.md)
is a one-page table of every headline number. Both span all four tasks, which
is why they sit in `docs/` rather than under any one of them.

## Layout

Scripts are grouped by the task they serve, with that task's reports beside
them. Each directory has its own README indexing what is inside.

| Directory | | Contents |
|---|---|---|
| [`src/forecasting/`](src/forecasting/) | 18 | Catalog and raw-waveform forecasting, fusion, LOEO |
| [`src/detection/`](src/detection/) | 12 | Earthquake-vs-noise classification, stacking, cross-corpus evaluation |
| [`src/features/`](src/features/) | 9 | Feature engineering, RFE, dataset builders |
| [`src/magnitude/`](src/magnitude/) | 5 | Magnitude regression and classification |
| [`src/groundmotion/`](src/groundmotion/) | 3 | Peak ground motion |
| [`src/seismolib/`](src/seismolib/) | — | Shared library: metrics, training, catalog, splits, waveform, baselines, models |
| [`docs/`](docs/) | — | Cross-cutting write-ups and `experiment_results/` |

`seismolib` holds everything two families would otherwise each copy. Install it
once and every script resolves it regardless of where it is run from:

```bash
uv pip install -e .
python3 src/detection/cnn_lstm_classify.py --help
```

This repo (on GitHub as `cnn_earthquake`) is the model/training half of a
two-repo pipeline. The companion repo — referred to below as `Sismokaos`
(locally `../Sismokaos` or `../data_downloader`, depending on checkout) —
does the FDSN downloading, windowing, and RAM/spectrogram dataset generation
that these scripts train on; see its own README for that side.

---

## Headline findings

(Full detail and caveats in `report.md`; this is the abstract's summary.)

- The amplitude of the raw waveform is the single largest contributor to
  detection performance measured in this project — the RAM transform is
  provably scale-invariant, so an auxiliary amplitude scalar had to be added
  alongside it (test AUC 0.836 → 0.923, architecture-matched comparison).
- A correctly-parameterized STA/LTA baseline (AUC 0.82 at 6s) is beaten by
  every tested CNN configuration; the library's own auto-derived defaults
  silently score AUC 0.51 (random) on arrival-anchored windows.
- A plain spectrogram CNN, no LSTM branch, no fusion, is the best detector
  found (test AUC 0.9793) — every fusion mechanism tried underperforms or
  only marginally beats this single-branch model.
- Magnitude event-class (≥M2.5 vs. below) is predictable from a single
  3-second window (79.78% accuracy, AUC 0.855).
- On three-class risk classification, a two-scalar gradient-boosted model
  beats the CNN outright (82.83% vs. 73.64% accuracy) — the encoded window
  isn't always the right tool.
- On peak ground motion, a Conv1D-BiLSTM-attention model beats the strongest
  scalar floor by 0.075 MAE_log, and the margin survives a doubly
  station-and-event-disjoint split.
- **Forecasting signal comes from the catalogue, not the seismogram** — the
  project's original question, now bounded from both sides. Catalogue-derived
  features beat a persistence floor: per-zone block-level AUC 0.692 (Aegean) and
  0.618 (Central), the latter a zone previously diagnosed as unforecastable.
  Waveform-derived features do not, across chaotic features and three sequence
  architectures (LSTM 0.524, GRU 0.571, TCN 0.520 against a 0.582 floor).
- The event catalogue is **AFAD's, not KOERI's** (the waveforms are KOERI), and
  the copy used until 2026-08-30 was missing ~29% of events for the region —
  nearly all of the February 2025 Santorini–Amorgos swarm. Rebuilding it moved
  forecasting results in *both* directions: chaotic-feature results got worse as
  the persistence floor rose, catalogue-feature results got better. Detection is
  essentially unaffected (3 contaminated noise windows in 55,595).

---

## Repository layout

```
src/                        Python source -- every training/eval script, no data or outputs
├── model/                  Shared nn.Module building blocks and composed architectures
│   ├── blocks.py             SEBlock, ResBlock, LSTMAttentionBranch, CNNBranch, GatedFusion
│   ├── trunk2d.py            SETrunk2D -- shared SE-ResNet trunk (detection/classification/regression)
│   ├── dual_channel.py       DualChannelTrunk/Net/DualHeadNet -- shared 1D+2D(+aux) fusion
│   └── sequence.py           SequenceHeadNet -- shared LSTM+attention forecasting head
├── metrics.py               Shared *_report()/print_report() accuracy reporting, used by every script
├── training.py               Core detection training loop + ImprovedSeismicCNN
│                              (checkpoint-compatible class name/path -- see its docstring
│                              before renaming anything here)
├── cnn_lstm*.py, cnn_ram_aux.py, cnn_regression.py, cnn_riskclass.py,
│   cnn_magclass.py, cnn_groundmotion.py, cnn_run*.py, ...
│                              One script per task/architecture variant (Sections 6-8, 13 of report.md)
├── feature_lstm_forecast.py, raw_cnn_lstm_forecast.py,
│   raw100hz_cnn_lstm_forecast.py, consolidate_hourly_raw.py
│                              Catalog/continuous-waveform forecasting (Section 11), current work
└── groundmotion_baselines.py, groundmotion_summary.py, lgbm_cluster.py, riskclass_scalar.py
                               Non-neural floors/baselines each task is measured against

scripts/                    Shell drivers that run several src/ scripts back to back
                             (e.g. the full ground-motion experiment grid)

experiment_results/          Checked-in result CSVs each script/driver writes, kept for the
                             numbers report.md and the cheatsheets cite

report.md, accuracy_summary.md, catalog_forecast_report.md,
spectrogram_classifier_report.md, *_CHEATSHEET.md, report.docx
                             Documentation (see "Start here" above)
```

**Not checked in** (gitignored — regenerate or point at your own copy):
`dataset*/`, `data/`, `data-old/`, `raw/`, `trained_model*/`, `results/`.
These hold downloaded waveforms, generated tensor datasets, and trained
checkpoints — all either regenerable from `Sismokaos` or produced by running
the scripts in `src/` yourself.

---

## Installation

Requires **Python 3.12+**. Managed with [uv](https://docs.astral.sh/uv/);
`pyproject.toml`/`uv.lock` pin the exact dependency set (PyTorch,
torchvision, ObsPy, scikit-learn, LightGBM, pandas, scipy).

```bash
git clone https://github.com/hogib/cnn_earthquake.git
cd cnn_earthquake
uv sync                 # creates .venv/ and installs everything, incl. CUDA-enabled torch
source .venv/bin/activate
```

A CUDA GPU is strongly recommended — several scripts (`raw_cnn_lstm_forecast.py`,
`raw100hz_cnn_lstm_forecast.py`) train directly on continuous raw waveform and
are RAM/VRAM-heavy at full-archive scale; see their own docstrings for
`--batch-size`/`--max-days`/`--consolidated` knobs that trade memory for
speed on smaller machines.

---

## Running something

Every script in `src/` documents itself: a top-of-file comment block states
what it does, its exact CLI invocation, and (where applicable) what else
imports it, followed by Google-style docstrings on every function. Read the
script you're about to run before running it — that's the authoritative,
up-to-date reference, not this file.

For copy-pasteable, result-linked reproduction commands for every number in
`report.md`, use its own **Appendix: Reproduction Instructions** — it walks
through dataset generation (via the `Sismokaos`/`data_downloader` CLI) and
training for every task in the order the report covers them, from the RAM
detector through catalog forecasting and peak ground motion.

The ground-motion experiment grid (`report.md` Section 13) has two
one-command drivers:

```bash
scripts/run_groundmotion_experiments.sh   # the A-F configuration grid
scripts/run_groundmotion_disjoint.sh      # the station-disjoint G/H follow-up
```

Both `cd` into `src/`, write their CSVs to `experiment_results/`, and default
`PY` to this repo's own `.venv`; override with `PY=/path/to/python
scripts/run_groundmotion_experiments.sh` to use a different interpreter.
`python src/groundmotion_summary.py` then collates every
`experiment_results/groundmotion_cnn_*.csv` into one comparison table.
