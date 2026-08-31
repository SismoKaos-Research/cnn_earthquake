# Reproducing the P-only detector experiments

Everything here regenerates from raw miniSEED. Each stage prints a number you
can check against the "expect" line before moving on — if a stage disagrees,
stop there rather than carrying the discrepancy forward.

Timings are wall-clock on a Ryzen 5 5600H (12 threads) with a 4 GB GTX 1650.

---

## 0. What you need

### Repositories

| repo | branch | commit at time of writing |
|---|---|---|
| `model_cnn_lstm` | `main` | `673e521` |
| `data_downloader` (in `Sismokaos/`) | `main` | `b3980c5` |

`b3980c5` is the commit that adds `--match-negative-amplitude`. Anything older
cannot build the matched dataset.

### Environment

Both repos are `uv`-managed with a committed lockfile. From each repo root:

```bash
uv sync            # creates .venv from uv.lock
```

Verified versions: Python 3.12.13, torch 2.13.0+cu130, obspy 1.5.0,
scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.1.

A GPU is not required — everything runs on CPU, roughly 4–6× slower for
training. Inference stages are minutes either way.

### Data

| what | where | size |
|---|---|---|
| 60 s event waveforms | `data_downloader/raw/data/batched_waveforms/window_post_60s` | 3.0 GB, 33,795 files |
| noise pool | `data_downloader/raw/data/batched_noise_waveforms/noise_pre_3h` | 3.0 GB |
| event catalogue | `data_downloader/catalogs/archive_superseded_2026-08-30/extracted_earthquakes.csv` | 93,690 events. **Archived 2026-08-30** — retained because it is the download list these datasets were built from, so reproducing them needs this file rather than the current catalogue. New work should use `catalogs/catalog_current.csv`. |
| station coordinates | `data_downloader/catalogs/station_coords.csv` | — |

The two waveform archives are the only things that do not regenerate without
re-downloading from FDSN. Everything below is derived from them.

Set once, used throughout:

```bash
export DD=~/Projects/Sismokaos/data_downloader
export MC=~/Projects/model_cnn_lstm
```

---

## 1. Generate P-only windows  (~47 min, 8 workers)

```bash
cd $DD
.venv/bin/python src/arrival_from_catalog.py \
    --window-seconds 3.4 --pre-arrival-seconds 2.0 \
    --out-name window_post_3.4s_ponly
```

**Why the flags are decoupled.** `--pre-arrival-seconds` defaults to
`window/3`, which at 3.4 s gives a 1.13 s pre-buffer against a 0.63 s arrival
prediction MAD — onsets fall out of the window and retention collapses.
Holding it at 2.0 s is what keeps retention at the 6 s configuration's level.
Do not let it scale.

**Expect:** 32,880 event files, 55,595 station recordings, retention
**96.4%**, ~656 MB.

## 2. Verify the windows are actually P-only  (~1 min)

Do not skip this. Generation only ever computes P phases, so nothing at
generation time checks whether S landed inside the cut.

```bash
cd $MC
.venv/bin/python src/detection/verify_ponly_windows.py \
    --metadata $DD/raw/data/batched_waveforms/window_post_3.4s_ponly/window_metadata.csv \
    --window-seconds 3.4 --pre-arrival-seconds 2.0
```

**Expect:** `windows where S intrudes (S-P < 1.4 s): 0`, minimum S−P
**1.450 s**, margin **+0.050 s**.

That margin is thin, and the script prints why it matters: S−P is predicted
from a catalogue hypocentre, so it carries the catalogue's location error
(median RMS residual 0.42 s). If the prediction is off by 0.3 / 0.5 / 0.63 s,
740 / 1,445 / 2,021 recordings (1.3% / 2.6% / 3.6%) could contain S. **State
the result as "P-only under iasp91", not "zero S".**

## 3. Build the four negative regimes  (~25–30 min each, ~1.8 GB each)

Identical positives in all four; only negative selection differs. This is what
makes them a controlled set.

```bash
cd $DD
common="--eq-dir raw/data/batched_waveforms/window_post_3.4s_ponly \
        --noise-dir raw/data/batched_noise_waveforms/noise_pre_3h \
        --window-seconds 3.4 --fs 100 --max --baseline \
        --n-fft 64 --hop-length 16"

# (a) amplitude-matched -- the measurement instrument
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset $common \
    --hard-negatives --match-negative-amplitude \
    --output-dir dataset_specdual_ponly_3p4s_matched

# (b) band-mined 75-99 -- the default, and the one with the artifact
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset $common \
    --hard-negatives --hard-negative-band 0.75 0.99 \
    --output-dir dataset_specdual_ponly_3p4s_hard

# (c) wideband 0-99 -- coverage across the range
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset $common \
    --hard-negatives --hard-negative-band 0.0 0.99 \
    --output-dir dataset_specdual_ponly_3p4s_wideband

# (d) natural -- no mining; the regime a station actually sees
.venv/bin/python -m seismic_cli.cli generate-spec-dual-dataset $common \
    --no-hard-negatives \
    --output-dir dataset_specdual_ponly_3p4s_natural
```

**Expect for each:** 111,190 tensors, splits 38,271 / 9,416 / 7,908 per class
across 120 / 28 / 35 stations.

**`--n-fft 64 --hop-length 16` is not the default** and must be passed. The
defaults (256 / 64) leave a 340-sample window with almost no time frames.

**The output filename comes from the `--output-dir` basename**, so two runs
into same-named directories silently overwrite each other.

### Check the regimes came out right

```bash
cd $MC
.venv/bin/python src/detection/negative_regime_transfer.py \
    --ckpt-dir trained_model_ponly_matched \
    --datasets matched=$DD/dataset_specdual_ponly_3p4s_matched \
               band=$DD/dataset_specdual_ponly_3p4s_hard \
               wideband=$DD/dataset_specdual_ponly_3p4s_wideband \
               natural=$DD/dataset_specdual_ponly_3p4s_natural
```

(The floors print before any model is loaded, so this is usable as a build
check even before training.)

**Expect floors:**

| regime | monotone | non-monotone | gap |
|---|---|---|---|
| matched | 0.6679 | 0.6658 | −0.0021 |
| band | 0.6447 | **0.7461** | **+0.1015** ← flagged as artifact |
| wideband | 0.7927 | 0.7845 | −0.0082 |
| natural | 0.7878 | 0.7795 | −0.0082 |

The band row is *supposed* to show that gap. Under 75–99 mining the negatives
carry an amplitude floor the positives lack, which makes
`P(event | amplitude)` U-shaped — 0.67 in the quietest decile — and ROC-AUC,
which only measures monotone ranking, understates it by 0.10. That is the
defect `--match-negative-amplitude` exists to remove.

## 4. Train  (~25–35 min per arm, 3 seeds each)

```bash
cd $MC
./scripts/run_ponly_matched.sh      # 1d, 2d  on matched
./scripts/run_ponly_matched_fusion.sh
./scripts/run_ponly_natural.sh      # 1d, 2d, all  on natural
```

Sequential by design — three concurrent runs OOM a 4 GB card.

`--seq-transform asinh` is applied on top of the generation-time `--baseline`
standardization. These are different things and both are needed: `--baseline`
puts `seq` in station-sigma units (preserving absolute amplitude), and `asinh`
compresses the tail so mixed precision does not overflow — station-sigma values
reach 3.6e5 against fp16's 65,504 ceiling. asinh is strictly monotone, so it
does not move any floor.

**Expect (matched-trained, ensemble of 3 seeds):**

| arm | AUC | per-seed |
|---|---|---|
| 1D | 0.8712 | 0.8673 / 0.8709 / 0.8671 |
| 2D | 0.8602 | 0.8605 / 0.8610 / 0.8544 |
| fusion | **0.8762** | 0.8730 / 0.8746 / 0.8737 |

Seed spreads 0.0038 / 0.0066 / 0.0016. A spread much larger than that means
something is wrong; these are stable configurations.

⚠ **The training log prints `ROC-AUC ... -> +0.2033 <- the number that
matters!`. It is not the number that matters** across datasets. Raw gain grows
whenever the floor falls, so it is not comparable between regimes. Use headroom
captured, `(AUC − floor) / (1 − floor)`.

## 5. Evaluate

### Transfer matrix — every arm against every regime

```bash
.venv/bin/python src/detection/negative_regime_transfer.py \
    --ckpt-dir trained_model_ponly_matched \
    --datasets matched=... band=... wideband=... natural=...
```

**Expect (matched-trained), headroom captured:**

| arm | matched | band | wideband | natural |
|---|---|---|---|---|
| 1D | 61.1% | 62.9% | 11.7% | 13.6% |
| 2D | 57.9% | 53.4% | 14.4% | 16.2% |
| fusion | 62.7% | 65.4% | 14.4% | 16.0% |

Recall is deliberately absent: the positives are identical across regimes, so
it is fixed at ~0.638 and cannot respond to the negatives. Only false alarms
can.

### Operating envelope — what the detector finds and misses

```bash
.venv/bin/python src/detection/operating_envelope.py \
    --detector-dir $DD/dataset_specdual_ponly_3p4s_matched \
    --magnitude-dir $DD/dataset_magreg_catalog_6s \
    --ckpt-dir trained_model_ponly_matched \
    --branch-1d cnn-lstm --channels 1d --fusion linear
```

**Expect:** overall recall **0.6380**; recall by distance **0.6616 / 0.6316 /
0.6334** across 0–25 / 25–50 / 50–100 km — i.e. flat, unlike the 6 s
configuration where it ran 0.977 / 0.942 / 0.939. Recall by SNR rises
0.469 → 0.702 → 0.823 → 0.945.

### Calibration and threshold

```bash
.venv/bin/python src/detection/calibrate.py \
    --dataset-dir $DD/dataset_specdual_ponly_3p4s_matched \
    --ckpt-dir trained_model_ponly_matched \
    --channels 1d --branch-1d cnn-lstm --fusion linear --seq-transform asinh
```

**Expect ECE to get *worse*:** 0.0484 → 0.0765, Brier 0.1386 → 0.1444, at
T = 0.6008. Temperature scaling helps the 6 s configuration and hurts this one.
**Report this configuration uncalibrated.** MCC-optimal threshold is 0.77
(recall 0.6085, precision 0.9715, 141 false alarms).

### Does it read shape or loudness?  (~5 min per run)

The floor comparison alone cannot answer this — see §8 and the results doc.
This holds amplitude nearly constant and checks whether discrimination
survives.

```bash
.venv/bin/python src/detection/within_amplitude_auc.py \
    --dataset-dir $DD/dataset_specdual_ponly_3p4s_matched \
    --ckpt-dir trained_model_ponly_matched --channels all
```

**Expect** (matched test, matched-trained fusion): pooled AUC 0.8763, floor
0.6679, and narrow-bin AUCs **0.6298 / 0.7090 / 0.7781 / 0.8013 / 0.8578 /
0.9274 / 0.9689** for bins 2–8, median **0.8013** across 7 bins.

Repeat with `--dataset-dir ..._natural --ckpt-dir trained_model_ponly_natural`
for pooled 0.8410 and median narrow-bin **0.7167**.

**Read the `evidence?` column, not just the AUCs.** Bins 1, 9 and 10 are
flagged `no (too wide)` because amplitude varies 3×–530× inside them, so a high
AUC there could still be loudness. Only bins ≤ 2.5× wide support the claim.
This matters: bin 1 spans ~500× and its low AUC is often misread as evidence
that the model fails at low SNR. It is not evidence of anything.

### S-dependence of the 6 s detector (separate experiment)

```bash
.venv/bin/python src/detection/s_arrival_ablation.py \
    --detector-dir $DD/dataset_specdual_catalog_6s_matched_hard \
    --magnitude-dir $DD/dataset_magreg_catalog_6s \
    --catalog $DD/catalogs/archive_superseded_2026-08-30/extracted_earthquakes.csv \
    --ckpt-dir trained_model_branch1d_asinh --branch-1d cnn-lstm
```

**Expect:** S-present windows lose −0.0288 recall when S is masked, against
−0.0356 for the duration-matched control on S-absent windows. Removing S costs
*less* than removing an equal stretch of ordinary signal, so the 6 s detector
was not leaning on S.

---

## 6. Total cost

| stage | time | disk |
|---|---|---|
| window generation | ~47 min | 656 MB |
| 4 dataset builds | ~2 h | 7.2 GB |
| 6 training arms (3 seeds each) | ~3 h | small |
| all evaluations | ~1 h | — |

**~7 h and ~8 GB** on top of the 6 GB of raw archives.

To reproduce only the headline result, do stages 1, 2, 3(a), 4 (matched only),
and 5 — about 2.5 h.

---

## 7. Things that will trip you up

1. **`--pre-arrival-seconds` must be passed explicitly** (stage 1). The default
   scales with window length and destroys retention at 3.4 s.
2. **`--n-fft 64 --hop-length 16` must be passed explicitly** (stage 3).
3. **`--baseline` and `--seq-transform asinh` are independent.** The first is
   generation-time and preserves absolute amplitude; the second is
   training-time and only prevents fp16 overflow. `--no-baseline` is a
   different experiment entirely — it deletes absolute amplitude and drops the
   `seq` std floor to exactly 0.5000.
4. **Raw gain is not comparable across regimes.** Use headroom captured. The
   training script's own celebratory log line gets this wrong.
5. **`cascade_eval.py` globs `*.pth` unfiltered.** Point it at a directory
   holding several arms and it silently ensembles `cnn` with `cnn-lstm`. The
   other evaluation scripts anchor on the exact arm tag; this one has not been
   fixed.
6. **`Zaman_Dk`-style timestamps elsewhere in this codebase are window *end*
   times.** Not relevant to these stages, but it matters the moment anything is
   joined by time.
7. Station splits are seeded and reproduce across builds — all four regimes
   share the same 35 test stations and the same 7,908 event windows. If they do
   not, the transfer matrix is not a controlled comparison and the numbers
   above will not match.

---

## 8. What the results mean

Reported in `docs/experiment_ponly_2026-08-22.md`. The short version:

- The 6 s window contains S in **28.8%** of event windows (99.3% within 25 km).
  The detector was not leaning on it, but a detection that only works once S
  has arrived carries no early-warning value, which is why the P-only
  configuration exists.
- Cutting to P-only drops the amplitude separation from ~29× to 2.0×, and the
  honest floor from 0.9049 to 0.6679.
- Headroom captured falls **89% → 72% → 61%** across the 6 s, band-mined, and
  matched configurations. Raw gain rises across the same sequence, which is
  why it is the wrong statistic.
- On the deployment-realistic (natural) regime, a matched-trained model adds
  only ~**+0.03 AUC** over a single amplitude scalar.
- The deployable claim is **recall ~0.61–0.64** on M ≥ 2.0 events within 56 km
  from 1.4 s of P — not the AUC.
