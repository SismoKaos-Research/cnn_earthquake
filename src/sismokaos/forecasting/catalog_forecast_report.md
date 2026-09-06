# Catalog Forecasting, Retargeted onto a Neural Architecture

> **Superseded in part, 2026-08-31.** Every per-zone number below was measured
> against a catalogue missing ~29% of AFAD's regional events, including nearly
> all of the February 2025 Santorini–Amorgos swarm. Re-derived at block level on
> the rebuilt catalogue: AEGEAN 0.519 → **0.692**, CENTRAL 0.396 → **0.618**,
> EAFZ 0.662 → 0.667, NAFZ 0.464 → 0.410 (3 seeds/arm, both catalogues spanning
> 2000–2026). **CENTRAL is no longer at chance**, so this document's diagnosis
> that it is near-Poisson and therefore unforecastable does not survive; that
> diagnosis still holds for NAFZ. See
> `docs/experiment_neural_forecasters_2026-08-30.md` §4.
>
> Report **block-level** figures only: consecutive windows overlap 11–46×, so
> the pooled window-level AUCs here inflate by +0.25 to +0.35.


**Companion to the retired `catalog_report.md`. Same data, same validated
target, different model.**

---

## 1. What changed and why

The original catalog-forecasting work (`catalog_report.md`, now removed)
reformulated an abandoned three-class "days until the next major earthquake"
target (measured at chance, kappa −0.028) into a dense one: **will a M ≥ 4.5
event occur in this fault zone within 30 days?** That target measured real
signal in 2 of 4 fault zones — but only ever under a logistic-regression /
gradient-boosting scalar model. This project's own dual-channel
CNN+LSTM+attention architecture (`cnn_lstm.py`, built from Wang & Zhao 2025's
1D2D-EDL) had already been built for exactly this kind of task, but it was
only ever tried against the *abandoned* target, and abandoned early once that
target was diagnosed as unlearnable ("started and stopped at fold 69" per the
old report). **It was never retried against the target that actually works.**

That retargeting is what this report covers: a new dataset generator
(`seismic-cli generate-catalog-forecast-dataset`, in
`data_downloader/seismic_cli/catalog.py`) that attaches the validated dense
label to the same kind of {seq, img, aux} tensors the network already
consumes, and a new training script
(`cnn_earthquake/src/cnn_lstm_forecast.py`) with a binary head in place of the
old 3-class one. `cnn_lstm_loeo.py` was not retargeted: LOEO forms one fold
per discrete target *event*, and the dense target has no such event — every
window is its own horizon-bounded outcome. The single chronological split is
the only evaluation mode that applies here, which is also why the retired
scalar forecaster never had a LOEO variant either.

The dataset regenerated from the same catalog reproduces the retired report's
window counts and split positive rates exactly (NAFZ 3,828 / EAFZ 6,224 /
AEGEAN 9,193 / CENTRAL 2,094 windows; test set 2,414 windows, positive rate
0.589) — confirming the reimplementation is faithful, not a different dataset
by accident.

## 2. Method

Same architecture as `cnn_lstm.py`: LSTM + multi-head self-attention over the
per-event feature sequence (1D), a compact CNN over the window's RAM image
(2D), learned-scalar fusion, plus an auxiliary branch carrying window-level
physical scalars (the RAM transform is exactly scale-invariant, so absolute
magnitude/energy level has to enter some other way — the same reasoning
`report.md` established for every other task in this project). The binary
head replaces the 3-class one; loss is `BCEWithLogitsLoss` with a
train-derived `pos_weight`.

Floors, printed on every run: **base rate** (train-period majority class) and
**persistence** (predict positive iff a qualifying event occurred in the
previous 30 days — available directly from the `days_since_prev_major` aux
feature). Following this project's established rule (`report.md` §6.6), every
close-margin result below was run at **three seeds** (42/43/44) before being
trusted.

Evaluation is reported at two granularities: **window-level** (every sliding
window in the test split, directly comparable to the retired report's §4.1/
§4.2 single-cut headline table) and **block-level** (disjoint 30-day blocks
within the test era only, via the retired `forecast.py`'s `build_blocks` —
kept specifically for this reuse). This block-level check is *not* the same
as the retired report's §4.4/§4.7 rolling-origin backtest, which pooled ~190
blocks across the *entire* catalog span using a walk-forward re-fit; this one
covers only the single test era each seed was scored on (23–28 blocks per
zone), so it is the honest-sample-size analogue of the window-level number
above, not a replacement for a full rolling-origin study. That remains future
work.

## 3. Results

**Pooled, window-level** (test set, 2,414 windows, positive rate 0.589):

| seed | AUC | acc | MCC |
|---|---|---|---|
| 42 | 0.7398 | 0.6632 | +0.3398 |
| 43 | 0.7229 | 0.6508 | +0.3372 |
| 44 | 0.7366 | 0.6587 | +0.3367 |
| **mean** | **0.7331** | 0.6576 | +0.3379 |
| *floor:* base rate | 0.5000 | 0.4105 | — |
| *floor:* persistence | 0.5945 | 0.6172 | +0.1941 |
| *retired scalar model* | *0.7228* | — | — |

The network ties, and across the 3-seed spread (0.7229–0.7398) very slightly
exceeds, the retired logistic-regression pooled AUC — the two architectures
land in the same place at the pooled level. Neither the base-rate nor the
persistence floor is a real contest here; the pooled number is dominated by
AEGEAN, exactly as the retired report found.

**Per zone, window-level** (mean of 3 seeds; retired scalar numbers from
`catalog_report.md` §4.2 for comparison — "pooled" = one model over all
zones, "per-zone" = separately fit per zone):

| zone | NN mean AUC | NN spread | scalar (pooled model) | scalar (per-zone model) |
|---|---|---|---|---|
| AEGEAN | **0.794** | 0.020 | 0.798 | 0.777 |
| EAFZ | 0.565 | 0.009 | 0.570 | 0.565 |
| NAFZ | 0.492 | 0.061 | 0.371 | 0.409 |
| CENTRAL | 0.413 | 0.173 | 0.349 | 0.424 |

AEGEAN and EAFZ are a wash — the network matches the scalar model to within
seed noise on both. NAFZ's mean nudges above the scalar model's, but the
spread (0.061, seed range 0.471–0.532) means "the network fixed NAFZ" is not
a claim this data supports. CENTRAL's spread (0.173, range 0.352–0.525) is
the widest of any zone measured in this comparison — one seed (42) would
have supported "the network roughly matches the per-zone scalar model here,"
the other two would not.

**Per zone, block-level, single test era** (mean of 3 seeds; not the retired
report's rolling-origin figure — see §2):

| zone | NN mean AUC | NN spread | n blocks |
|---|---|---|---|
| AEGEAN | 0.655 | 0.129 | 27 |
| EAFZ | 0.487 | 0.102 | 28 |
| NAFZ | 0.390 | 0.033 | 26 |
| CENTRAL | 0.237 | 0.237 | 23 |

At the honest sample size, only AEGEAN is directionally consistent above
chance across all three seeds — the same zone the retired report's full
rolling-origin backtest also found strongest (median AUC 0.661 there). NAFZ
and CENTRAL sit *below* chance in every one of the 3 seeds at block level,
consistent with the retired report's physical diagnosis: these are the two
near-Poisson zones (inter-event-gap CV ≈ 1), where "forecastability tracks
clustering" predicts no model of this kind should work, architecture
included. CENTRAL's spread (0.237 — the full width from 0.14 to 0.38) on only
23 blocks is not something a third decimal place should be trusted from.

## 4. What this establishes, and what it doesn't

**Established:** the dual-channel network is not worse than the scalar model
at the target that's actually learnable — a real answer to a question this
project had never actually asked (§1). AEGEAN's signal is architecture-
agnostic: both the scalar model and the network find it, at both window and
block level, across every seed tried on either. The physical diagnosis from
the retired report (forecastability tracks clustering, not model choice)
survives the swap to a neural architecture intact.

**Not established:** that the network is *better* than the scalar model
anywhere. Every zone-level difference measured here is within, or close to,
the 3-seed spread — the same caution `report.md` §6.6 attached to every
close-margin comparison in this project. NAFZ and CENTRAL's window-level
means nudge upward relative to the scalar model, but their seed spreads
(0.061 and 0.173) are large enough relative to the differences involved that
"the network helps the weak zones" is not a claim this run supports.

**Not attempted here:** a rolling-origin backtest for the neural model
(matching the retired report's §4.4/§4.7, the most rigorous evaluation it
ran), and the feature-extractor integration (`Sismokaos-featureExtract`'s
auto-extracted chaos/complexity measures as additional aux inputs) — both
remain open next steps.

## 5. Adding Magnitude: When *and* How Big

§1–4 answer half of this project's actual deliverable. Confirmed directly
with the project lead: the target is a **probabilistic** forecast of both
*when* and *at what magnitude* the next earthquake will occur (not
deterministic prediction — no current method in seismology does that
reliably, grant document notwithstanding). §1–4 is the validated "when"
half. This section adds a magnitude output and reports what happened.

**Design: extend the existing network with a second head, not build a new
one.** `build_dense_windows` already computes, for the binary label, the
array of future major-event times within the horizon. It was extended to
also keep the magnitude of the first (earliest) such event —
`next_magnitude`, NaN exactly when the binary label is 0. `catalog.py`'s
`encode_and_write_dense` writes it as a manifest column (a target, not an
aux feature). `DualChannelForecastNet`'s single output head was restructured
into a shared trunk feeding **two** linear outputs: the existing binary
logit, unchanged, and a new magnitude point-estimate. Loss: the existing
`BCEWithLogitsLoss` plus a masked L1 loss on the magnitude head (masked to
positive windows only, where a next-event magnitude is defined), weighted by
`--mag-loss-weight`. Checkpoint selection stayed tied to validation AUC only
— deliberately, so the already-validated binary result's checkpoint choice
couldn't be put at risk by an unproven second objective.

**Floors, evaluated on test-set positive windows only** (n=1,423 at the
original dataset size): predict-the-mean of `next_magnitude` over TRAIN
positive windows, and a ridge regression on four aux features
(`max_mag`, `mean_mag`, `b_value`, `log_rate` — all already in
`DENSE_AUX_FEATURES`) fit on TRAIN positive windows, mirroring the
"fitted-statistical-relation" floor pattern `cnn_regression.py`'s
`report_baselines` established for a different task in this project.

**Result — three attempts, single seed, all clean losses to the floors:**

| variant | model MAE | predict-the-mean floor | ridge floor | binary AUC (unaffected) |
|---|---|---|---|---|
| default (`--mag-loss-weight 1.0`) | 0.289 | 0.239 | 0.247 | 0.7318 |
| `--mag-loss-weight 3.0` | 0.341 (worse) | 0.239 | 0.247 | 0.7323 |
| `--patience 30` (30+ epochs run; checkpoint never moved past epoch 1) | 0.289 (identical) | 0.239 | 0.247 | 0.7318 |
| 2× training windows (`--stride-events 4`, same catalog) | 0.300 | 0.239 | 0.248 | 0.7253 |

None of the three levers tried — higher magnitude-loss weight, more training
patience, twice the training data via denser striding — closed the gap, and
higher loss weight made it measurably worse (likely gradient interference
with the shared trunk). The binary head's AUC held steady across every
variant (0.725–0.732, inside the established 3-seed range from §3) — adding
and tuning the magnitude head never put the validated result at risk, but it
also never earned its own keep.

**3-seed confirmation of the default variant** (`--mag-loss-weight 1.0`,
same dataset as §3): binary AUC 0.7318 / 0.7326 / 0.7265 (mean 0.730) —
squarely inside §3's established range, confirming the two-head architecture
doesn't destabilize the validated binary result across seeds. The magnitude
head's own MAE swings much more across the same 3 seeds (0.289 / 0.248 /
0.280) — one seed nearly matched the ridge floor (0.247) while the others
didn't — which is itself an argument for the floor: it scores exactly 0.247
every time, deterministically, with no seed lottery involved.

**Per-zone breakdown of the recommended system** (seed 42 checkpoint +
ridge, `catalog_forecast_predict.py`) sharpens where "good" actually means
good:

| zone | n | positive rate | AUC (when) | n positive | MAE (how big) |
|---|---|---|---|---|---|
| **AEGEAN** | 1129 | 0.782 | **0.791** | 883 | **0.215** |
| EAFZ | 823 | 0.487 | 0.573 | 401 | 0.303 |
| NAFZ | 320 | 0.356 | 0.440 | 114 | 0.265 |
| CENTRAL | 142 | 0.176 | 0.411 | 25 | 0.399 |

**AEGEAN is where this system is genuinely good at both halves** — AUC 0.79
for "will a M≥4.5 event occur," and a magnitude MAE of 0.215, *better* than
the pooled 0.247, on the same zone. That's not cherry-picking: AEGEAN is the
same zone §3/§4 already identified as the one place this project's forecast
work (scalar or neural) has found real, seed-consistent signal — this is
that same zone's story extended to magnitude, not a new claim. EAFZ, NAFZ,
and CENTRAL don't share it: NAFZ and CENTRAL sit below-chance on "when"
(§3's near-Poisson diagnosis, unchanged), and their magnitude numbers don't
rescue that — a magnitude estimate attached to an unreliable "will it happen
at all" isn't a useful forecast, whatever its own MAE reads. CENTRAL's
n=25 positive test windows is also too few to trust its 0.399 figure at
face value either way.

**Recommendation: use the ridge floor as the actual magnitude answer,
not the neural head.** This is the same pattern this project already found
in report.md's Task 3 (three-class risk classification) — a simple
statistical model on a handful of physically-motivated scalars beats a
CNN/LSTM architecture built for the same target, repeatably, not once. The
honest, presentable system this project has for "when and how big" is
**two different tools for two different sub-questions**: the validated
dual-channel network for "will a M≥threshold event occur within the
horizon" (real skill, AUC 0.73, §3), paired with
`ridge(max_mag, mean_mag, b_value, log_rate)` for "how big will it be, given
one occurs" (MAE ≈ 0.24–0.25, beats every neural attempt at the same
question). Not a consolation prize for the magnitude half — the best
available answer, honestly arrived at.

Raw per-variant numbers: `experiment_results/cnn_lstm_forecast_maghead_results.csv`. Combined
end-to-end prediction (binary network + ridge magnitude, run together on the
same test windows): `src/sismokaos/forecasting/catalog_forecast_predict.py`.

## Reproduction

```bash
# Dataset (data_downloader/)
seismic-cli generate-catalog-forecast-dataset \
    --catalog-path catalogs/catalog_current.csv \
    --output-dir data/dataset_catalog_forecast

# Training (cnn_earthquake/src/), one per seed
python cnn_lstm_forecast.py \
    --dataset-dir ../../data_downloader/data/dataset_catalog_forecast \
    --catalog-path ../../data_downloader/catalogs/catalog_current.csv \
    --seed 42   # 43, 44
```

Raw per-seed numbers: `experiment_results/cnn_lstm_forecast_results.csv`.

**Magnitude head (§5) — same dataset command, plus `--stride-events 4` for
the denser variant:**

```bash
python cnn_lstm_forecast.py \
    --dataset-dir ../../data_downloader/data/dataset_catalog_forecast \
    --catalog-path ../../data_downloader/catalogs/catalog_current.csv \
    --seed 42 --mag-loss-weight 1.0   # 3.0 for the higher-weight variant;
                                       # --patience 30 --epochs 100 for the
                                       # longer-training variant
```

Recommended magnitude prediction, run standalone against the same test
windows: `python catalog_forecast_predict.py --dataset-dir
../../data_downloader/data/dataset_catalog_forecast`.
