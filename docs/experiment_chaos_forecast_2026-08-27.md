# Chaotic features as a forecaster: a negative result, and why more data will not fix it

**Date:** 2026-08-27
**Scripts:** `src/sismokaos/forecasting/chaos_dataset.py`, `chaos_univariate_screen.py`,
`chaos_forecast.py`
**Tests:** `tests/test_chaos_dataset.py`, `tests/test_chaos_screen.py`
**Data:** `sismokaos-cli/dataset_features_chaos_q1_5hz/bodt_q1_chaos_5hz_features.parquet`
(312,626 windows × 134 columns, BODT, 2024-05-01 → 2024-10-28)

## The question

Not "can this predict earthquakes". The question is narrower and answerable:
**do chaotic features forecast anything that the timing of the previous event
does not already?** On the chosen cell the persistence rule — an event happened
recently, so another is more likely — already scores **0.5503**, and a model is
only interesting if it beats that using something else.

Cell: **M ≥ 2.5, within 400 km, 6 h horizon**, chosen 2026-08-21 by
`label_sweep.py` on statistical power. 4,343 hours, 1,092 positive (25.1%),
**232 qualifying events**.

## Step 0: the sequence gate passes

`sequence_variance_check.py`, the gate the handoff put before any training code:

| granularity | context | within-sequence variance ratio (median) |
|---|---|---|
| native 50 s | 24 × 50 s = 20 min | 37.8% |
| hourly mean | 24 × 1 h = 24 h | **56.1%** |
| *the band that made the ruled-out GRU degenerate* | | *1.2 – 9.3%* |

Both are an order of magnitude clear of the degenerate band, so the earlier
verdict does not transfer: these inputs do carry within-sequence structure.

**This also overturned a prediction made before running it.** The worry was that
collapsing 72 windows an hour to a mean would destroy the variation the idea
depends on. It does the opposite — hourly sequences vary *more* (56.1% against
37.8%), because at 24 steps the binding constraint is context length, not
granularity. Twenty minutes of chaos features barely move; twenty-four hours do.
Hourly aggregation is therefore the right choice, and mean is not a loss.

## Step 1: the univariate screen finds a whisper

Every one of 528 aggregated columns (134 features × mean/std/min/max) scored
against the label with `oriented=True`, because a candidate baseline's sign is
free.

```
floor (persistence)                       0.5503
best single feature   Z_SKEWNESS_DEV_std  0.5726   (+0.0223)
78 of 528 features beat the floor (14.8%)

best-of-528 under a block-shuffled permutation null (500 draws):
    median 0.5427    95th percentile 0.5647
observed best 0.5726 is ABOVE the 95th percentile
```

The null is a real permutation test, and two details in it matter. It scores the
**actual columns** rather than random ones — chaos features are heavily
autocorrelated, and two autocorrelated series agree spuriously far more often
than one does with white noise. And it shuffles labels in **day-long blocks**,
because one event marks six consecutive positive hours and an element-wise
shuffle would destroy that, producing a null far tighter than reality. An
earlier version of this script got both wrong and would have reported a much
easier bar.

So the screen says: marginally more association than screening 528 columns
produces by chance. Worth taking to a model — which is exactly what it was for.

## Step 2: the model does not reproduce it out of sample

Walk-forward, 4 folds, 24 h embargo (must exceed the 6 h horizon, or the last
training hours share an event with the first test hours). Every configuration
fitted twice, with and without `log1p_dsp`, because chaos features that only
help when recency is *hidden* have recovered withheld information rather than
added any.

**Mean across folds:**

| | AUC | headroom captured |
|---|---|---|
| persistence floor | 0.5423 | 0% |
| chaos only / lgbm | **0.5523** | **2.2%** |
| chaos + dsp / logreg | 0.5490 | 1.5% |
| chaos + dsp / lgbm | 0.5483 | 1.3% |
| chaos only / logreg | 0.5452 | 0.6% |

**Per fold, the sign does not hold:**

| fold | floor | chaos+dsp lgbm | chaos-only lgbm |
|---|---|---|---|
| 1 | 0.5089 | −3.2% | −0.9% |
| 2 | 0.5475 | **+12.5%** | +9.6% |
| 3 | 0.5114 | −2.2% | +5.1% |
| 4 | 0.6016 | −1.6% | −6.0% |

One fold carries everything and the other three are at or below the floor. A
+0.010 mean AUC gain that flips sign across folds is not a forecast.

The gradient-boosted trees and the L2 logistic regression land within 0.007 of
each other, which is its own answer to "would a bigger model help": when a
528-feature linear model matches boosted trees, there is no structure for extra
capacity to find.

**The fold spread is what a fold of this size produces on its own.** Each test
fold contains 34–37 qualifying events, so the effective sample is ~35 episodes,
not 700 hours. At AUC 0.55 with 35 events the Hanley–McNeil standard error is
**±0.055**, and a 95% interval spans ±0.108 — wider than every effect in the
table. Across the four folds the mean captured headroom is +1.4% with a standard
deviation of 7.5%, giving **t = 0.37 on 3 degrees of freedom** against the 3.18
needed for p < 0.05. There is nothing here to distinguish from zero.

## Why more data will not rescue it

This is the part that makes the result decision-relevant rather than merely
disappointing.

`label_sweep.py` put the **detectable edge for this cell at ±0.064 at 95%**
confidence, on 181 days. The observed effect is **+0.010**. It is not merely
non-significant — it is six times smaller than the smallest effect this design
could have detected.

Processing the full 723 days would give roughly 4× the events, which halves the
detectable edge to about **±0.032**. That is still three times larger than the
observed effect. **Finishing the extraction would not turn this result
positive**, and that is the argument against spending a day on it.

## What this licenses, and what it does not

**Licensed.**

- At this cell and this sample size, chaotic features add nothing to a
  persistence baseline that survives out-of-sample walk-forward evaluation.
- The failure is *not* the one that killed the earlier GRU experiment. That
  model degenerated because its inputs barely varied within a sequence (1.2–9.3%);
  these vary at 56.1%. The inputs are fine; the association is not there.
- Extra model capacity is not the missing ingredient — a linear model matches
  the trees.

**Not licensed.**

- **No claim that chaotic features are uninformative in general.** This is one
  station, 181 days, one label cell, and a 6 h horizon. Longer horizons, a
  tighter radius with more years behind it, or a different target could all
  differ.
- **No claim about the features' quality.** The screen found real, if marginal,
  association; what failed is its stability across time.
- **Nothing about detection**, where chaotic features have not been tried at
  all. They were extracted for forecasting, and detection is where this project
  has signal.

## Open

1. ~~Fold 2 is worth one look~~ — **checked, and it is not a swarm.** The four
   test folds contain 37 / 35 / 34 / 37 qualifying events; fold 2 is the second
   *quietest*. The busiest days in the catalogue (2011, 2017, 2020, 2025) all
   fall outside this archive's span. So fold 2's +12.5% is not a regime effect,
   it is ordinary variance — see the paragraph above, where an AUC standard
   error of ±0.055 per fold makes a swing of that size unremarkable.
2. **Longer horizons on the same features** are nearly free now that the parquet
   exists — `label_sweep.py` had 24 h and 72 h cells with more events and lower
   floors. The 6 h cell was chosen for power, not because 6 h is the target.
3. **Chaotic features on the detection task** is the untried direction, and the
   one where this project's signal actually lives.

---

# Follow-up: would a CNN + LSTM have helped?

**Date:** 2026-08-27, same day
**Added:** `--shape` and `--lags` in `chaos_dataset.py`, tested in
`tests/test_chaos_shape.py`

The proposal was to run a CNN over the raw feature stream and pass its output to
an LSTM or GRU. It targets a real gap: the aggregation above collapses 72 windows
an hour into four numbers per feature, so a feature that climbs steadily through
the hour, one that spikes at minute 40, and one that oscillates can share all
four. And the model above sees only the *current* hour, so a trajectory across
hours is invisible to it too.

Rather than build the network, both halves of the hypothesis were tested with
cheap proxies. If within-hour shape and cross-hour context carry association, a
slope term and a lag will show some of it; if they show nothing, a network
learning a fancier function of the same numbers is unlikely to.

- **CNN half → within-hour shape.** Per feature per hour: least-squares slope
  (scaled so window count does not matter), second-half minus first-half mean,
  position of the maximum, and lag-1 autocorrelation. 528 new columns.
- **LSTM half → cross-hour context.** Lagged levels and deltas at 1, 3, 6, 12
  and 24 h for the 40 strongest columns. 400 new columns.

## Result: neither half adds anything, and both slightly hurt

| configuration | columns | best AUC | headroom captured |
|---|---|---|---|
| persistence floor | — | 0.5423 | 0% |
| **summaries only (baseline)** | 528 | **0.5523** | **2.2%** |
| + within-hour shape | 1,056 | 0.5461 | 0.8% |
| + cross-hour lags | 928 | 0.5510 | 1.9% |
| + shape + lags | 1,456 | 0.5460 | 0.8% |

**The baseline is the best of the four.** Every addition left it flat or worse.

The univariate screen is blunter still. Splitting the 1,056 columns by kind:

| column kind | n | beat the floor | best | median AUC |
|---|---|---|---|---|
| summary (mean/std/min/max) | 528 | 78 (**14.8%**) | 0.5726 | 0.5267 |
| shape (slope/halfdiff/argmax/ac1) | 528 | 3 (**0.6%**) | 0.5565 | 0.5099 |

**A median oriented AUC of 0.5099 is as close to "no association" as this
measurement gets.** Not one shape column reached the top 20, and the best of
them ranks 29th overall. Doubling the column count did not move the best
feature (0.5726, a summary statistic) or the permutation null's 95th percentile
(0.5647).

**The logistic regression is the clearest signal that the additions are noise.**
It fell from +0.6% captured on summaries alone to **−5.4%** with shape and
**−7.0%** with both. An L2-penalised linear model degrades exactly like that when
uninformative columns are added; a tree model, which can ignore them, barely
moved. The two together say the new columns carry nothing rather than
something the model failed to use.

## What this does and does not settle

**Settled.** Within-hour trajectory and 1–24 h cross-hour context carry no
association with this label that the four summary statistics do not already
have. Since those are the two things a CNN encoder and a recurrent layer
respectively exist to extract, the architecture is not the missing piece here.

**Not settled.** A convolutional encoder could learn shapes these four crude
statistics do not span — a specific multi-modal profile, say. The evidence
against that is indirect: with a median shape AUC of 0.5099, the trajectories
would have to be informative in a way that is invisible to slope, half
difference, peak position *and* lag-1 autocorrelation simultaneously.

The honest correction to the earlier section stands: the ±0.064 detectable-edge
argument bounds what can be **verified** at this sample size, not what could
exist. A model producing a +0.15 effect would be detected easily. What these
proxies test is whether the raw material for such a model is present, and it
does not appear to be.

---

# Follow-up 2: sweeping context length, so the question closes

**Script:** `src/sismokaos/forecasting/chaos_config_sweep.py`
**Tests:** `tests/test_chaos_context.py`
**Output:** `logs/chaos_config_sweep.csv`

Every architecture proposal so far has been, underneath, a claim about **how
much history the model should see**. A CNN over the 50 s stream says sub-hour
shape matters. A 7-day hierarchy of 24 h embeddings says a week matters. Neither
needs to be built to be tested: make context length a parameter, aggregate the
trailing window, hand it to a model whose capacity is already known not to be
the constraint, and see whether anything clears its floor.

Grid: **5 context lengths × 3 horizons**, four walk-forward folds each,
LightGBM plus `log1p_dsp`, floor recomputed per horizon.

## Result: nothing clears the floor, anywhere

Headroom captured:

| context ↓ / horizon → | 6 h | 24 h | 72 h |
|---|---|---|---|
| 1 h (current hour) | **−2.5%** | −11.6% | n/a |
| 6 h | −5.9% | −8.5% | n/a |
| 24 h | −10.0% | −4.0% | n/a |
| 72 h | −10.8% | −23.2% | n/a |
| **168 h (the 7-day proposal)** | −9.8% | −5.1% | n/a |

**Zero of ten scoreable cells beat their floor.** The best is −2.5%. Longer
context does not help and mostly hurts, which is the signature of adding
columns that carry nothing: more to overfit, nothing to learn.

The 72 h horizon is excluded rather than reported as a failure — its positive
rate is **96.2%**, so almost every hour is within 72 h of a qualifying M ≥ 2.5
within 400 km. That is an unusable label, not evidence about the features, and
it is the same viability filter `label_sweep.py` applies.

## Why these numbers are more negative than the section above

The earlier run reached +2.2% captured at the 6 h horizon; the context-1 h cell
here reaches −2.5% at the same horizon. The two are not contradictory, they use
different feature bases: the earlier run summarised each hour with
**mean/std/min/max** (528 columns), this sweep uses **hourly means only** (132)
so that context length is the sole thing varying down a column.

That the result moves by ~5 points of headroom when the feature basis changes,
with the sign flipping, is itself the finding: **these effects are the size of
the noise.** Nothing here is stable enough to build on.

## What the whole investigation now supports

Across everything run on 2026-08-27:

| configuration tested | best captured |
|---|---|
| hourly summaries, current hour | +2.2% |
| + within-hour shape (CNN proxy) | +0.8% |
| + cross-hour lags 1–24 h (LSTM proxy) | +1.9% |
| + both | +0.8% |
| context sweep 1–168 h × horizon 6/24 h, 10 cells | **−2.5%** (best) |

Every configuration sits within noise of a persistence baseline, and the
detectable edge for this design is ±0.064. **No configuration of these features
forecasts this label**, and the failure is not architectural: capacity was ruled
out by a linear model matching boosted trees, within-window shape and cross-window
context were ruled out by direct proxies, and context length was ruled out by
sweeping it over two and a half orders of magnitude.

The remaining honest caveat is unchanged: this bounds what can be **verified** at
232 events on one station over 181 days. It is not a proof that no model could
ever work. It is a well-supported negative for this station, these features, this
label family, and this much data — which is what a reader is entitled to ask for.

---

# Follow-up 3: a second station, and what replicates

**Date:** 2026-08-28
**Script:** `src/sismokaos/forecasting/chaos_station_replication.py`
**Tests:** `tests/test_chaos_station.py`
**Data:** `dat_q1_chaos_5hz_features.parquet` (266,641 windows, DAT,
2024-05-02 → 2024-10-28, date-matched to BODT, identical config)

"You only looked at one station" is the first objection to everything above.
DAT sits 43.8 km from BODT with its own continuous archive already on disk, so
answering it cost an extraction rather than a download.

**This is not a sample-size increase.** At the 400 km label radius the two
stations share 95.3% of their events; DAT adds 3 to BODT's 232. Each station is
scored against its **own** local label, computed from its own coordinates —
sharing BODT's label across both would compare two waveforms against one label
and read the inevitable agreement as a result.

## Site character differs more than the signal does

| | BODT | DAT |
|---|---|---|
| `Z_WOLF_LYE` | 0.764 ± 0.072 | 0.812 ± 0.076 |
| `Z_CORR_DIM` | 4.482 ± 0.212 | 4.571 ± 0.437 |
| `Z_SAMP_ENT` | 1.249 ± 0.182 | **1.896** ± 0.228 |

`Z_SAMP_ENT` differs by 0.65 — over three standard deviations of either
station's own spread. Two sensors 44 km apart, in different ground, produce
materially different chaos statistics. This is why the comparison has to be over
**rankings**, not values.

## The headline feature does not replicate; the horizontals do

Overall Spearman rank correlation of the two 528-feature AUC vectors:
**+0.464**. Split by component:

| component | n | ρ | clears BODT's floor | clears DAT's floor |
|---|---|---|---|---|
| **Z** (vertical) | 168 | **−0.137** | 17.9% | **0.0%** |
| N | 168 | +0.750 | 11.9% | 2.4% |
| E | 168 | +0.798 | 16.7% | 3.0% |
| cross-component | 24 | −0.108 | 0.0% | 0.0% |

**BODT's best feature was `Z_SKEWNESS_DEV_std` at 0.5726. It ranks 325th of 528
at DAT.** Every Z feature in BODT's top 15 lands in DAT's bottom half; every E
feature lands in its top 35. Not one of the 168 vertical-component features
clears DAT's floor.

So the single-station result that looked marginally promising — one column past
a permutation null — was **vertical-component site character at BODT**, and it
does not exist 44 km away. That is a cleaner explanation of the out-of-sample
failure above than "noise", and it could not have been found without a second
station.

## The horizontal agreement survives two artifact checks

ρ = 0.75–0.80 on the horizontals is high enough to need explaining, and two
mundane explanations would produce it:

**Feature persistence.** A more autocorrelated column scores higher against an
autocorrelated label by construction, at any station, which would correlate the
rankings with no seismology involved. Measured: the partial rank correlation
controlling for each feature's own lag-1 autocorrelation is **+0.423**, barely
below the raw +0.464. And the relationship runs the *wrong way* for the
artifact — more persistent features score **lower** (ρ = −0.383 at BODT,
−0.274 at DAT).

**Diurnal cycling.** The label has a real diurnal cycle: positive rate swings
**1.65×** across the day (0.188 at 08h UTC, 0.309 at 23h), which is catalogue
completeness — more small events are detected at night when cultural noise
drops. Hour-of-day alone scores **0.5364**. But the agreeing features are not
the diurnal ones: correlation between cross-station agreement and diurnal
amplitude is **+0.113**, and the top-30 agreeing features have the same diurnal
amplitude as the full set (0.43 vs 0.45 sd).

**A prediction made here was wrong and is recorded as such.** On finding the
diurnal cycle, the expectation was that the floor should rise to include
hour-of-day, making everything above look weaker. It does not: a depth-4 tree on
`(log1p_dsp, sin h, cos h)` scores **worse** than persistence alone — −0.0025 at
BODT and −0.0236 at DAT — because it spends its split budget across three
features and overfits. The diurnal signal is real but adds nothing to
persistence out of sample.

## What the second station changes, and what it does not

**The forecasting negative is now a two-station negative, and stronger for it.**
DAT's own walk-forward persistence floor is **0.5900** — higher than the best
chaos model measured anywhere in this investigation (0.5523 at BODT). At DAT
only 1.7% of all features clear the floor, against 14.8% at BODT, and that gap
is almost entirely the vertical component.

**One thing genuinely replicates and is not yet explained.** Horizontal-component
feature *rankings* agree at ρ ≈ 0.78 across two stations, and neither
autocorrelation nor diurnality accounts for it. That is not a forecast — the
effect sizes at DAT are +0.0075 over floor at best — but it is structure, and
it is the only positive finding in this file.

## Open

1. **What drives the horizontal-component agreement?** Both stations share
   nearly the same label by construction (95% event overlap), so any weak
   genuine association would rank consistently at both. Distinguishing "weak
   real association" from a third shared artifact needs stations far enough
   apart to have *different* labels — which is what the AFAD TU network would
   provide.
2. **The vertical channel should be excluded from single-station screening**, or
   at minimum flagged. It produced this project's most promising-looking
   forecasting feature and that feature was site character.

---

# Re-run on the corrected catalogue (2026-08-30)

Everything above was computed against `deprem_katalog_utc.csv`, which held 51 of
1,256 February 2025 events in this region — almost none of the
Santorini–Amorgos swarm, the largest seismic episode in the record window. The
catalogue was rebuilt from AFAD's API (`src/sismokaos/acquisition/fetch_afad_catalog.py`) and the
whole suite re-run over the identical window, geometry and features. **Only the
labels changed.** Original logs are kept as `logs/chaos_*.log`; the new ones are
`logs/chaos_*_corrected.log`.

## The labels moved a lot

| | was | corrected |
|---|---|---|
| BODT positives (4,343 h, 6 h horizon) | 1,092 (25.1%) | **1,733 (39.9%)** |
| DAT positives (3,711 h) | 1,053 (24.2%) | **1,692 (39.0%)** |

## The answer did not change — it got firmer

The headline is that **restoring the swarm improved the baseline, not the
model.** A dense aftershock sequence is precisely what "days since the previous
event" predicts well, so the persistence floor rose faster than anything built
from waveform features:

| walk-forward, 4 folds, 24 h embargo | was | corrected |
|---|---|---|
| persistence floor | 0.5423 | **0.5713** |
| chaos + dsp / lgbm | 0.5483 (+1.3%) | 0.5687 (**−0.6%**) |
| chaos only / lgbm | 0.5523 (+2.2%) | 0.5605 (**−2.5%**) |
| chaos + dsp / logreg | 0.5490 (+1.5%) | 0.5593 (**−2.8%**) |
| chaos only / logreg | 0.5452 (+0.6%) | 0.5485 (**−5.3%**) |

Previously the best configuration cleared the floor by +2.2%. **Now every
configuration is below it.** The context/horizon sweep agrees: 0 of 10 cells
above +5% captured, best cell −0.1% (context 168 h, horizon 24 h).

## Why the univariate screen still says "worth modelling"

The screen passes more emphatically than before — best feature 0.5841 against a
block-shuffled null whose 95th percentile is 0.5628 (500 draws), and 222 of 528
features clear the floor where 78 did. That is not a contradiction. Individual
features do carry marginal association with the label; it simply is not
*complementary* to persistence. Both are reading the same underlying quantity —
how seismically active the recent past was — so once persistence is the
baseline, the features add nothing. A screen against a null and a model against
a floor are different questions, and only the second one is the forecast.

## What did replicate, and more cleanly

The one durable positive from the original run survived and sharpened. Scoring
all 528 features at both stations and correlating the two AUC vectors:

| component | was | corrected |
|---|---|---|
| Z (vertical) | −0.137 | **−0.193** |
| N (horizontal) | +0.750 | **+0.856** |
| E (horizontal) | +0.846 | **+0.846** |
| overall | +0.464 | +0.498 |

The horizontals agree strongly across two stations 40 km apart; the vertical is
mildly *anti*-correlated. It shows in the leaderboard directly — every E/N
feature in BODT's top 15 lands inside DAT's top ~45, while every Z feature lands
between 257 and 458 of 528. And 10 of BODT's 15 leaders beat DAT's own floor
against ~3.9 expected by chance.

So there is something real and station-transferable in the horizontal-component
chaos features. It is not a 6-hour forecast of whether an M>=2.5 event occurs
within 400 km — that remains below the persistence floor — but it is not noise
either, and it is the thread worth pulling.

**Reproduce:** `experiments/reproduce/rerun_chaos_corrected.sh` (62 s, sequential).
