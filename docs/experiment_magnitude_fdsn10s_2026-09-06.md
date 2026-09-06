# FDSN 10 s magnitude regression, waveform-only

**MAE 0.4203 ± 0.0165** over three doubly-disjoint partitions, against an
information-matched floor of 0.6054 ± 0.0508 — and it beats the floor that is
*handed the distance* in all three.

## What was run

    sk train magnitude --dataset-dir dataset_magreg_fdsn_10s \
        --channels 1d+2d --split-by auto \
        --seed 42 --seed-split {0,1,2} \
        --save-dir trained_model_magreg_fdsn10s_nonaux_p{0,1,2}

`--channels 1d+2d` withholds the aux vector. This is the deployable
configuration and now the trainer's default: aux is `(log_snr, log_distance)`,
and log_distance needs a catalogued hypocentre that a fresh detection does not
have. A model trained with it cannot run in the cascade it exists for.

`--split-by auto` resolves to `both` (161 stations): station-disjoint **and**
event-disjoint. The generator's own split is event-disjoint but shares 140 of
its 141 test stations with train, so it would leak site response.

Three **partitions**, not three model seeds. Which stations land in test
dominates — measured at 2.4× model-seed variance on the 6 s corpus — so varying
`--seed-split` is what produces an honest error bar. `--seed` is held at 42.

## Results

| | partition 0 | partition 1 | partition 2 | mean ± sd |
|---|---|---|---|---|
| test rows | 578 | 672 | 484 | 1,734 total |
| test stations (all unseen) | 25 | 17 | 27 | |
| **model** (`1d+2d`) | **0.4094** | **0.4123** | **0.4393** | **0.4203 ± 0.0165** |
| ridge(log_snr) — matched | 0.5767 | 0.6641 | 0.5754 | 0.6054 ± 0.0508 |
| ridge(log_snr, log_distance) | 0.4950 | 0.6105 | 0.5389 | 0.5481 ± 0.0583 |
| constant-mean | 0.7917 | 0.6948 | 0.6833 | 0.7233 ± 0.0595 |

Ratio to the matched floor **0.698 ± 0.072**; to the distance-having floor
0.773 ± 0.084.

## Read the floor, not the ratio

The model's spread is 0.0165. Its floor's is 0.0583 — **3.5× larger**. The
ratio therefore moves from 0.83 to 0.68 between partitions almost entirely
because the baseline got weaker on a different draw of test stations, while the
model barely moved. Quoting a ratio alone would report that as a change in the
model. It is not.

**The floor the trainer printed sees more than the model does.** `log_snr` is a
waveform statistic, so a waveform-only model has it by another route;
`log_distance` is not, and cannot be recovered from one station's 10 s window.
The matched floor is `ridge(log_snr)`, and the trainer prints both now.

The strong result survives either choice: **0.4203 beats 0.5481**, so the
waveform-only model is better than a fitted local-magnitude relation *that is
given the distance it is not allowed*. That is the comparison a deployment
cares about, because nothing has the hypocentre at alarm time.

## Not comparable to 0.2023

`docs/` records MAE 0.2023 ± 0.0051 doubly-disjoint on `dataset_magreg_catalog_6s`.
This is 2× worse and the difference is **not** a regression:

- **no aux** — that run was `2d+aux`, this one is waveform-only by design
- **a quarter the corpus** — 13,150 rows against 55,568
- **tiny test sets** — 484–672 rows, because doubly-disjoint drops ~85% of
  val/test (2,808–3,362 rows per partition) to clear shared events
- **10 s FDSN windows**, not 6 s catalogue-anchored ones

An aux-enabled run on this same corpus would isolate how much of the gap is the
withheld distance. It has not been run.

## A dataset defect fixed first

`distance_km` was missing on 1,331 of 13,150 rows (10.1%), and the missingness
was **binary per station**: 42 of 161 stations had none for any row, 119 had it
for every row — a station-key mismatch, not unknown data. That mattered twice:
the trainer mean-imputes NaN aux after standardizing, so those 42 stations
would have trained against one fixed wrong distance; and it breaks the
ridge floor above, which this project judges every magnitude model against.

All 42 were in `station_coords.csv`, which agrees with the KOERI FDSN station
service to **0.000 km across all 277 KO stations**. Recomputing the column
reproduced the generator's own values on 99.4% of rows to within 1 km (median
0.0003 km), which is what licensed the repair; a guard refused to write until
that held. It also **corrected 72 rows across 4 stations** the generator had
wrong — `KO.KIZT` by 14.4 km. `distance_km` is now 100% present, with the
original manifest kept as `manifest.csv.pre-distance-repair`.
