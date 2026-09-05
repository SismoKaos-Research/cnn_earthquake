# What the detector costs on continuous data

2026-09-05. 728 days of MANT, three detectors and a classical baseline, scored
window by window. Everything below is measured; the score files and threshold
tables live on vegs (`scores_mant/`, `final_*.csv`), which are gitignored, so
this file is the record.

`scripts/continuous_false_alarms.py`, `--arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm`
`--arm pnat:3.4:trained_model_ponly_natural:cnn-lstm`
`--arm ponly:3.4:trained_model_ponly_matched:cnn-lstm`
`--arm stalta:6.0:stalta:0.5-10`.

## Why

Every detection number in this repo comes from a curated benchmark: balanced
classes, arrival-anchored positives, negatives mined from a noise pool. TODO
§2.3 asks what that benchmark cannot answer — what the detector costs when it
runs against an uninterrupted station record.

The benchmark's own extrapolation was the thing to beat. At threshold 0.5 the
3-seed `1d/cnn-lstm` ensemble misclassifies 141 of 7,906 test noise windows, an
FPR of 1.78%; a day holds 14,400 disjoint 6 s windows, so that predicts 257
false alarms per day.

## Setup

10,487,211 windows for the 6 s arm and 18,519,887 for the 3.4 s arms, over
2024-04-30 to 2026-08-09, from 36 chunk archives. Windows are disjoint (step =
window length), so an alarm count is also a count of independent decisions, and
none is ever built across a data gap: windows exist only inside spans where all
three components have unbroken record.

Preprocessing is the training pipeline's own operations (detrend twice, 5% Hann
taper, 4th-order 1–45 Hz bandpass, standardize against the station baseline)
applied to blocks rather than single windows. `verify` checks that rather than
asserting it: the vectorized filter against a transcription of
`clean_and_filter_1d` (5.4e-13 max relative difference), and real dataset
tensors through the scan's own scoring path (0.9899 against a published 0.9896
for the 6 s arm, 0.8742 against 0.8712 for P-only).

**Recall is asked only of events the station recorded.** `mant_range_full.csv`
measures SNR for 47,522 catalogued events within 500 km; the median is 1.39 and
27.5% reach SNR 3. Scored against all 48,428 events the event-level AUC is
0.67–0.73, but that number is about the catalogue's reach, not the detector, and
reporting it as the detector's would be wrong.

## Result 1 — the benchmark threshold does not survive

| arm | background p50 | alarms/day at 0.5 |
|---|---|---|
| 6s | **0.8019** | **12,599** |
| pnat | 0.3037 | 45 |
| stalta | 2.07 (CF) | 13,489 |

The 6 s detector scores a median of **0.80 on noise** and flags 92.7% of a quiet
station-day at the threshold its benchmark was calibrated on. Not 257 alarms per
day — 12,599. Any operating point must come from the measured background
distribution, which is what `report` now does.

## Result 2 — why, measured rather than guessed

Feeding real training **noise** windows through unchanged except a scalar
multiply:

| scale | median std (station sigma) | frac p>0.5 |
|---|---|---|
| 1.000 | 0.708 | 0.0186 |
| 0.300 | 0.212 | 0.1832 |
| 0.100 | 0.071 | 0.8619 |
| 0.010 | 0.007 | 1.0000 |

`P(event | amplitude)` is **U-shaped**. Amplitude mining puts a floor under the
negatives and physics puts one under the positives, so below ~0.1 sigma the
model has training data of neither class and extrapolates to "earthquake".
Continuous MANT background sits at 0.11 sigma.

Two things this is *not*. It is not the baseline sigma being wrong: random
windows in the actual training noise files span the same range, and several
training stations (GADA 0.049, GELI 0.088, ENEZ 0.085) sit exactly where MANT's
0.11 does. And it is not a property of amplitude mining as such: `ponly` is
mined and `pnat` is not, and **both are monotone** — the failure is specific to
the 6 s build, whose S-inclusive positives reach 581 sigma, not to mining.

Mining is in fact mildly *helpful* for the P-only model on continuous data
(§ Result 3), which was not the expected direction and is worth stating: at
195 days the two arms looked identical (0.9376 vs 0.9370) and only the full span
with corrected SNR separated them.

## Result 3 — AUC and the operating point disagree, in opposite directions

13,056 events at SNR≥3, 728 days.

| | event AUC @ SNR≥3 | recall @ 100/day | @ 10/day | @ 1/day | F1 @ 10/day |
|---|---|---|---|---|---|
| stalta | **0.9795** | **0.928** | 0.548 | 0.132 | 0.547 |
| ponly | 0.9622 | 0.903 | 0.627 | 0.183 | — |
| pnat | 0.9516 | 0.861 | 0.573 | 0.219 | 0.568 |
| 6s | 0.9403 | 0.864 | **0.741** | **0.316** | **0.690** |

The AUC column and the 10/day column are in **exactly reverse order**.

**A 1978-vintage STA/LTA wins on AUC at every SNR cut and loses at every alarm
budget a deployment would choose.** At 1 alarm/day the 6 s detector finds 2.4×
as many events.

AUC integrates the whole ROC; an operating point lives in one extreme corner of
it, here at FPR 7.4e-4. A detector can rank better overall and separate worse in
the far tail. **Any comparison of detectors reported as AUC alone can invert
under deployment conditions**, which is the strongest argument in this
experiment for reporting operating points.

Confusion at 10 alarms/day (event level; there is no TN — no negative
earthquake exists, and alarms are clustered at 60 s so one noise burst is one
declaration):

| | TP | FN | FP | precision | recall |
|---|---|---|---|---|---|
| 6s | 9,675 | 3,381 | 5,326 | 0.645 | 0.741 |
| pnat | 7,481 | 5,575 | 5,829 | 0.562 | 0.573 |
| stalta | 7,152 | 5,904 | 5,952 | 0.546 | 0.548 |

## Result 4 — the unexplained alarms are diurnal

At 10 alarms/day, unexplained alarms by local hour:

| arm | day (06–20) | night | ratio | peak |
|---|---|---|---|---|
| 6s | 368/h | 213/h | 1.73× | 12:00–15:00 |
| pnat | 387/h | 187/h | 2.07× | 12:00–15:00 |

Every false-alarm count here is formally an **upper** bound, because an alarm
matching no catalogued event is either a false positive or an event AFAD never
listed. The diurnal structure tightens that considerably: **earthquakes do not
have a working-hours peak.** A substantial share of these are cultural noise,
not missed detections. Only a run spanning whole days across two years shows
this.

## What this does not measure

- **Latency decomposition.** Only the model term is available (alarm time is the
  window's END — a detection cannot be declared before the window exists).
- **Calibrated uncertainty.** Softmax confidence is not a calibrated
  probability, and nothing here makes it one.
- **Station variance.** One station. GCAM (189 d) and DEMI (campaign started
  2026-09-05) address this.
- **Association.** Single-station, pre-association. Every operational EEW system
  suppresses false alarms by requiring stations to agree; §2.2 is that work.

## Reproduce

```bash
python3 scripts/continuous_false_alarms.py verify \
    --dataset-dir .../dataset_specdual_catalog_6s_matched_hard \
    --ckpt-dir trained_model_branch1d_asinh --branch-1d cnn-lstm
python3 scripts/continuous_false_alarms.py baseline \
    --zips 'afad_raw/MANT/*.zip' --out mant_baseline.json
python3 scripts/continuous_false_alarms.py scan \
    --zips 'afad_raw/MANT/*.zip' --baseline-json mant_baseline.json \
    --arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm \
    --arm pnat:3.4:trained_model_ponly_natural:cnn-lstm \
    --arm stalta:6.0:stalta:0.5-10 --out-dir scores_mant
python3 scripts/station_detection_range.py --zips 'afad_raw/MANT/*.zip' \
    --station MANT --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv --out mant_range_full.csv
python3 scripts/continuous_false_alarms.py report \
    --scores 'scores_mant/6s/*.npz' --station MANT --window-seconds 6.0 \
    --stations-csv catalogs/istasyon_katalog.csv \
    --catalog catalogs/catalog_current.csv --snr-csv mant_range_full.csv \
    --out-prefix final_6s
```

Scan cost ~4 h wall for three arms over 36 chunks on a 3060 Ti, dominated by
the obspy read (39–2215 s per chunk depending on fragmentation), not inference.
