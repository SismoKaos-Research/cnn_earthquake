# Catalog Forecasting, Retargeted onto a Neural Architecture

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

## Reproduction

```bash
# Dataset (data_downloader/)
seismic-cli generate-catalog-forecast-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir data/dataset_catalog_forecast

# Training (cnn_earthquake/src/), one per seed
python cnn_lstm_forecast.py \
    --dataset-dir ../../data_downloader/data/dataset_catalog_forecast \
    --catalog-path ../../data_downloader/catalogs/deprem_katalog_utc.csv \
    --seed 42   # 43, 44
```

Raw per-seed numbers: `src/cnn_lstm_forecast_results.csv`.
