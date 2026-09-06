# cnn_earthquake

CNN/LSTM models for four seismic ML targets, trained on real waveform data from
the Turkish networks: **detection** (earthquake vs noise on a short window),
**magnitude** (how big), **forecasting** (whether an event is coming, and when),
and **ground motion** (how hard it will shake). The core architecture is the
dual-channel CNN+LSTM+attention design of Wang & Zhao (2025, *Applied Soft
Computing* 172) — a 1D branch over the raw waveform, a 2D branch over a
spectrogram or recurrence-plot image, fused — adapted and extended across all
four targets, and measured against non-neural floors throughout.

This is the model half of a two-repo pipeline. The companion repo,
`seismic_cli` (a sibling checkout, `../seismic_cli`), does FDSN downloading,
windowing, and dataset generation; this one trains and evaluates.

**Where to read what.** This file is the overview: how to install it and what
lives where. [`docs/MANUAL.md`](docs/MANUAL.md) is the operations manual — every
command, every flag, worked examples, and the trap each step has actually hit. [`docs/report.md`](docs/report.md) is the
technical writeup — every experiment, every defect found, and reproduction
commands. [`docs/accuracy_summary.md`](docs/accuracy_summary.md) is the
one-page table of headline numbers. [`docs/TODO.md`](docs/TODO.md) is the only
planning file.

---

## Install

Requires **Python 3.12+**, managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:SismoKaos-Research/cnn_earthquake.git
cd cnn_earthquake
uv sync                  # creates .venv/, installs everything incl. CUDA torch
source .venv/bin/activate
```

`uv sync` installs the repo itself in editable mode, which is what puts `sk` on
your path and lets `import sismokaos` resolve from any directory. A CUDA GPU is
strongly recommended: several forecasters train directly on continuous raw
waveform and are VRAM-heavy at archive scale.

---

## `sk` — the command line

Everything runnable has one front door. `sk` on its own lists the commands
grouped by what you are trying to do; each command keeps its own `--help`.

```
acquire     campaign poll fdsn fdsn-noise plan-pull catalog
stations    station-select station-range station-loss distances
windows     cut-events cut-length
train       train
evaluate    falsealarm magprofile
report      docx pdf figures
inspect     status models results
```

Every command is exactly the underlying module's `main()`, arguments untouched,
so a command recorded in a report also runs as
`python -m sismokaos.<group>.<tool> ...`. That is deliberate: results here are
expected to be traceable to a command, and a front end that rewrote arguments
would make the recorded command and the real one diverge. A test fails if `sk`
stops listing a tool, or lists one that does not import.

### Getting data

Two sources, split by **network**, not by data type:

| | network | served by | cost |
|---|---|---|---|
| `sk fdsn` | KO (KOERI, 163 stations, archive to 2012) | FDSN | 24 h in ~13 s |
| `sk campaign` + `sk poll` | TU (AFAD) | TDVMS email queue | ~2 min per station-day |

Use FDSN wherever the station is on KO — it is roughly a hundred times cheaper.
TDVMS is for TU, whose value is that its stations are disjoint from the KO
training corpus.

```bash
sk fdsn plan  --catalog catalogs/catalog_current.csv --out requests.csv
sk fdsn fetch --requests requests.csv --out-dir raw/fdsn_magnitude --batch 40
```

The TDVMS side is a queue, not a download. `sk campaign` submits one window per
plus-address (TDVMS keys its one-request-at-a-time limit on the literal address
string, so `you+a1@x` and `you+a2@x` are separate slots), and `sk poll` watches
the mailbox, pastes each link back, and refills the slot that just freed.

```bash
sk campaign --ledger mant.jsonl plan --station MANT --start 2024-05-01
sk campaign --ledger mant.jsonl next --email you+a1@example.com
sk poll --ledger mant.jsonl --out-dir afad_raw --search '(UNSEEN FROM "tdvms@afad.gov.tr")' --pump
```

Credentials come from the environment only (`AFAD_IMAP_HOST` / `_USER` /
`_PASS`). Several pollers can share one mailbox: each leaves mail addressed to
a slot its own ledger never submitted from unread, for the poller that owns it.

### Training

`sk train` is indexed by **what you are predicting**, not by architecture,
because the trainers are named after their networks and that is the wrong axis
to search along.

```bash
sk train                            # every task, grouped by what its label answers
sk train --predicts forecast        # one group
sk train detect --label             # what this task's label actually is
sk train detect --help              # the trainer's own flags
```

Twenty-five tasks in four groups: **detect** (6), **magnitude** (4),
**forecast** (14), **shaking** (1).

Seven of them are wired to the model registry, so the architecture is a flag
rather than a choice of script:

```bash
sk train detect            --dataset-dir ds --model-branch cnn-lstm
sk train magnitude         --dataset-dir ds --model-branch cnn-lstm --fusion gate
sk train forecast-features --features-csv f.npy --catalog-path c.csv --model tcn
sk train groundmotion      --dataset-dir ds --model-branch cnn
```

`forecast-features` is the clearest case: one task, one set of labels and one
split, three architectures (`sequence-head`, `gru`, `tcn`) selected by
`--model`. Everything a registry-wired run resolves is written to `model.json`
in the checkpoint directory, so evaluation reads the architecture back instead
of having it retyped.

### The model registry

```bash
sk models                        # every architecture, grouped by what it consumes
sk models dual-channel           # one model's branches, flags and defaults
sk models --spec trained_model_x # the spec saved beside those checkpoints
```

Eleven architectures in six families, where a family is the shape of the input:
`dual` (1D waveform + 2D image + aux scalars), `image`, `sequence`,
`window`, `hierarchical`, `fusion`. `--model` picks the architecture,
`--model-branch` picks its variant — for `dual-channel` that is the 1D front
end (`lstm`, `cnn`, `cnn-lstm`), and `--branch-1d` is still accepted as an
alias so older recorded commands keep running.

### Knowing what happened

```bash
sk status                  # what is running, how far along, did the last run work
sk status --host vegs      # the box the jobs are actually on
```

Three sections: running jobs with the argument that tells them apart, recent
runs from `runs/*.json` with their headline metric, and free disk. Every
registry-wired trainer writes a run record through `sismokaos.runlog` —
argv, git commit **and whether the tree was dirty**, dataset identity, split,
seeds, metrics, checkpoints. A run that crashes leaves a record saying
`started` rather than no record at all.

---

## Layout

```
src/sismokaos/  everything importable: the library, the trainers, and the 20
                command-line tools, all reachable as `sk <command>`
experiments/    reproduce/ (the exact runners for published results)
                analyses/  (one-off analysis scripts)
tests/          the suite; `pytest -q` from the repo root
docs/           report.md, accuracy_summary.md, TODO.md, experiment records,
                tubitak/ (the Turkish deliverables)
runs/           one JSON per training run (provenance)
catalogs/       event and station catalogues
```

Not checked in (gitignored, regenerate or point at your own): `dataset*/`,
`data/`, `raw/`, `trained_model*/`, `results/`, `.env*`.

### `src/sismokaos/` — one package

Everything is one installed namespace. There used to be six top-level names —
`seismolib` plus one per family — which made `detection` and `features`
importable names owned by this repo, and made a cross-family import
(`from detection.cnn_lstm_classify import ...`) read like a third-party one.
Each family subpackage keeps its own README indexing what is inside.

| | files | |
|---|---|---|
| [`detection/`](src/sismokaos/detection/) | 23 | event vs noise, stacking, cross-corpus and cross-station evaluation, published-picker baselines |
| [`magnitude/`](src/sismokaos/magnitude/) | 6 | magnitude regression and classification, risk classes |
| [`forecasting/`](src/sismokaos/forecasting/) | 26 | catalogue and raw-waveform forecasting, fusion, chaos features, LOEO |
| [`groundmotion/`](src/sismokaos/groundmotion/) | 4 | peak ground acceleration and velocity |
| [`features/`](src/sismokaos/features/) | 11 | feature engineering, RFE, dataset builders |
| [`continuous/`](src/sismokaos/continuous/) | 12 | scoring an uninterrupted station record — chunks, spans, association, alarms, and the six `falsealarm` subcommands |
| [`acquisition/`](src/sismokaos/acquisition/) | 6 | the TDVMS ledger and its mail poller, the FDSN pull, the AFAD catalogue rebuild |
| [`stations/`](src/sismokaos/stations/) | 3 | coverage ranking, per-event SNR, what a station's catalogue misses |
| [`windows/`](src/sismokaos/windows/) | 2 | cutting arrival-anchored windows, and re-cutting them to another length |
| [`reporting/`](src/sismokaos/reporting/) | 3 | Markdown to .docx and .pdf, and the report figures |
| [`tooling/`](src/sismokaos/tooling/) | 4 | `sk train`, `sk status`, `sk models`, `sk results` |

The last five used to be `scripts/`, a directory that cannot be imported --
which is why `cut_event_windows.py` reached for its own directory to borrow six
functions from a sibling, and why the tools had a test suite of their own that
had to load them from a file path. Nothing manipulates `sys.path` to reach any
of this now, and a test fails if something starts to.

**Source of truth for which waveform service to use:** FDSN (KOERI) for the KO
network, TDVMS only for TU, which no FDSN node serves.

### The shared library

| module | |
|---|---|
| `tasks.py` | what the repo can be trained to predict, and which module trains it |
| `cli.py` | the `sk` dispatcher |
| `training.py` | the detection training loop, presets, and `ImprovedSeismicCNN` |
| `metrics.py` | the `*_report()` / `print_report()` accuracy reporting every script uses |
| `baselines.py` | conditional floors — what a trivial rule scores on the same data |
| `splits.py` | walk-forward chronological cross-validation and its diagnostics |
| `catalog.py` | catalogue loading and hourly labelling |
| `waveform.py` | hourly raw-waveform loading, and the 1D CNN encoders over it |
| `arrivals.py` | cached iasp91 travel times on a distance/depth grid |
| `checkpoints.py` | selecting the checkpoints of one training arm, and refusing to guess |
| `runlog.py` | one JSON per run, so a number can be traced to what produced it |
| `rust_io.py` | reading everything `sismokaos-cli` writes |
| `logging.py` | stdout tee for long-running training scripts |
| `data/` | the dataset classes |
| `continuous/` | continuous-record scanning; see its own docstring |

`checkpoints.py` earns its place: `run_ponly_natural.sh` writes three
architectures into one `--save-dir`, so a bare `glob("*.pth")` averages over a
mixture of models answering different questions. That has already cost two sets
of checkpoints and one seed reported at 0.2480 — an inverted model read as a
training outcome.

### `src/sismokaos/model/` — the networks

| module | |
|---|---|
| `registry.py` | every architecture, its knobs, its variants, and the `--model` flags |
| `blocks.py` | `SEBlock`, `ResBlock`, `LSTMAttentionBranch`, `ConvSeqBranch`, `CNNBranch`, `GatedFusion` |
| `trunk2d.py` | `SETrunk2D` — the SE-ResNet trunk behind the image models |
| `dual_channel.py` | `DualChannelTrunk` / `Net` / `DualHeadNet` — the 1D+2D+aux fusion |
| `sequence.py` | `SequenceHeadNet` — LSTM+attention over a sequence, optional per-step encoder |
| `recurrent.py` | `ForecastGRU` |
| `tcn.py` | `ForecastTCN` and its dilated causal blocks |

Some models still live in their trainers (`GroundMotionNet`,
`CatalogWaveformFusionNet`, the hierarchical day/week nets). The registry
imports those lazily, so listing the architectures never pulls a trainer's
argparse in behind a model.

---

## Reproducing a published result

`experiments/reproduce/` holds the exact runners, deliberately kept as scripts
rather than as `sk` commands: their value is being the precise thing that was
run. They default `PY` to this repo's `.venv`; override it to use another
interpreter.

```bash
experiments/reproduce/run_ponly_natural.sh
experiments/reproduce/run_groundmotion_experiments.sh
```

`docs/report.md`'s **Appendix: Reproduction Instructions** walks every number
in the report from dataset generation through training.

---

## Reading the code

Every module states in its first lines what it does, the exact invocation, and
what else imports it — followed by Google-style docstrings. Where a design
choice was forced by something that went wrong, the docstring says what went
wrong. Those notes are the reason several traps are not still traps; read the
file you are about to run.

```bash
pytest -q          # 414 passed, 3 skipped
```

The suite is mostly about drift rather than about correctness of the maths: it
asserts every module imports, every `sk` command points at a file with a
`main()`, every registered architecture builds and runs at the shape it
advertises, registry defaults match each class's own constructor defaults, and
every task the listing marks as taking `--model` has a parser that accepts it.
Each of those has caught a real break.
