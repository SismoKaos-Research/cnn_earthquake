# Chaotic features as a forecaster: a negative result, and why more data will not fix it

**Date:** 2026-08-27
**Scripts:** `src/forecasting/chaos_dataset.py`, `chaos_univariate_screen.py`,
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
