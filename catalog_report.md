# Catalog Forecasting

**A technical report on the project's original objective**

---

## Abstract

This is the project's stated objective — forecasting earthquake occurrence from
catalog data — as distinct from the waveform work in `report.md`, which was a
detour.

The result is a **weak but statistically real forecast in two zones, a clean
failure in the other two, and — for one zone only — probabilities worth acting
on**. Reformulated as "will a M ≥ 4.5 event occur in this fault zone within the
next 30 days?", the forecaster reaches **block-level AUC 0.62 (EAFZ) and 0.60
(AEGEAN) over ~190 independent 30-day blocks, both 95 % CIs excluding chance**.
NAFZ and the Cyprus arc are indistinguishable from chance.

Getting to that number took two corrections, each of which shrank the claim:

1. **A rolling-origin backtest (§4.4)** showed the original single-cut headline
   of 0.798 for AEGEAN sat near the **top** of a distribution spanning 0.23 AUC.
   The direction survived; the precision did not. The same backtest *promoted*
   EAFZ from "not forecastable" to the most consistent zone of the four.
2. **Block-level evaluation (§4.7)** found that consecutive windows overlap
   11–46×, so every earlier confidence statement was overstated by up to ~7× in
   its standard error. At the honest sample size the effect survives, at 0.60
   and 0.62 rather than 0.66 and 0.65.

A third finding is invisible to AUC entirely (§4.8): the raw model's
**probabilities were worse than climatology in every zone**, despite ranking
above chance. Prospective recalibration repairs this, but only **EAFZ** ends up
with probabilities that beat the base rate by a usable margin (BSS +0.032).
AEGEAN discriminates measurably yet sits within +0.005 of climatology — real
ranking, almost no value.

The reason is physical rather than architectural, and it is the same reason
throughout. **Forecastability tracks clustering.** The two zones that work are
the two with clustered inter-event gaps (CV 1.56 and 1.46); the two that fail
are the two sitting at CV ≈ 1 — exactly memoryless — where no model of this kind
can work. The single cut broke this pattern by placing EAFZ at chance; the
backtest restored it, so the physical explanation now fits all four zones.

That argument also explains the phase this replaced. The original target
(time to the next independent mainshock, in terciles) measured **at chance**:
31.49 % accuracy against a 33.33 % floor, kappa −0.028, over 264 target events.
Gardner–Knopoff declustering removes aftershocks — the *most predictable* part
of seismicity — leaving mainshock timing that is close to Poisson. The target
was near-unlearnable by construction.

Two defects were fixed along the way, each of which would have published a
wrong number rather than crashed.

---

## 1. The abandoned formulation, and why it failed

`catalog.py` labels each 64-event sliding window with the time until the next
*independent* mainshock, binned into terciles, and evaluates by
leave-one-event-out CV over 264 target events.

**Result: at chance.** Gradient boosting over the nine seismicity indicators
(b-value, Lyapunov exponent, energy release rate, …) scored 31.49 % accuracy
against a 33.33 % floor with **kappa −0.028**, and a pooled confusion matrix
that is essentially uniform:

|  | pred lt_26d | pred 26–71d | pred gt_71d |
|---|---|---|---|
| **true lt_26d** | 974 | 894 | 930 |
| **true 26–71d** | 1031 | 798 | 968 |
| **true gt_71d** | 1015 | 912 | 871 |

Per-class F1 ranged 0.296–0.335 — no class recovered above chance, and no
region carrying signal the others lacked. A 264-fold dual-channel CNN+LSTM run
was started and stopped at fold 69 once the diagnosis below made its outcome
uninteresting; per-fold accuracies to that point were scattered around the same
level.

**The structural diagnosis.** Declustering exists to stop one mainshock's
aftershock sequence masquerading as many independent targets — necessary for
target *selection*, but it removes exactly the predictable structure.
Aftershocks follow Omori decay and are the most forecastable part of a catalog.
What remains is mainshock timing, and measured inter-mainshock gaps have
CV 0.67–1.17 across regions. **CV = 1 is exactly exponential**, i.e.
memoryless, so P(wait | history) = P(wait) and no model can beat chance.

This is not a claim that catalog forecasting is impossible. It is a claim that
*this target*, on *this catalog*, was defined so as to be nearly unlearnable.

## 2. Two defects that would have published wrong numbers

**(a) Class labels wrong by an order of magnitude.**
`RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]` was hardcoded in two files while
`assign_risk_classes` derives **tercile** boundaries. On this catalog those
terciles land at 26 d and 71 d, so:

| Label says | Actually means |
|---|---|
| `lt_1y` | 0 – 25.8 days |
| `1_5y` | 25.8 – 71.3 days |
| `gt_5y` | 71.3 – 816.9 days |

Median time-to-next-mainshock is 46 days. A table reporting "`gt_5y` precision
0.81" would be read as a five-year claim and be wrong by more than an order of
magnitude. Nothing failed; only the names were wrong. Fixed by generating class
names from the boundaries in force and having both trainers read the ordinal
direction from the manifest's own `days_to_major`.

**(b) The built-in LOEO baseline is anti-predictive.**
Both scripts compared against per-fold majority-class prediction, which scores
**8.53 %** — far *below* the 33.33 % chance rate. With globally balanced classes
and highly concentrated folds (mean purity 0.748), removing a fold tips the
training pool *away* from that fold's own dominant class, so the "majority" is
systematically the class the fold has **least** of. Measured: it matched the
fold's true mode in **2 of 264** folds and its rarest class in **175**.
Comparing against it turned a chance-level model into an apparent +23-point
win. Both scripts now lead with the chance floor and flag the artifact.

## 3. The reformulation

The change is to the **target**, not the model — which is what
`catalog.report_major_events`'s own remediation advice recommends
("switch target definition: 'max magnitude in the next N days' is a dense
regression problem rather than a rare-event one").

**Target:** will a M ≥ 4.5 event occur in this fault zone within 30 days of the
window's last event? Dense (every window has one), undeclustered, so clustered
seismicity now *helps* instead of being defined away. Base rates run 0.24–0.61
by zone — neither saturated nor rare. At national scale the same target is
vacuous (M ≥ 4 occurs in essentially every 30-day window), which is why
regionalisation is load-bearing.

**Zones.** The previous pooled dataset's four regions were never recorded
anywhere and could not be reconstructed. `FAULT_ZONES` is now a committed
constant covering 92.8 % of the catalog:

| Zone | Extent (lat, lon) | Events | M≥4.5 |
|---|---|---|---|
| NAFZ — North Anatolian | 39.5–42.0, 26.0–42.0 | 30,744 | 82 |
| EAFZ — East Anatolian | 36.5–39.5, 35.0–42.0 | 50,066 | 299 |
| AEGEAN — western extension | 36.0–40.0, 25.0–30.0 | 73,703 | 200 |
| CENTRAL — Cyprus arc | 34.0–37.5, 28.0–36.0 | 16,841 | 54 |

**Two measured feature fixes.** `n_events` was in the old aux vector and is
*constant* by construction (it equals `window_events`) — one of nine features
was dead weight, now dropped. `days_since_prev_major` was absent despite being
the natural feature for a renewal process and the first indicator in comparable
published work; measured Spearman +0.129 (p = 2.6×10⁻³²) against the old
target, on par with the best feature that *was* included. Added, along with
rate- and energy-acceleration terms.

**Splitting.** Chronological with a 30-day horizon embargo. The label looks
forward exactly one horizon, so dropping one horizon of windows at each
boundary is precisely enough to stop a window being labelled by events on the
far side — and costs only 142 of 21,339 windows.

## 4. Results

### 4.1 Pooled

Test set, 2,414 windows, positive rate 0.589:

| Model | AUC | Accuracy | MCC |
|---|---|---|---|
| *floor:* base rate (majority) | 0.5000 | 41.05 % | +0.000 |
| *floor:* persistence (recency) | 0.6344 | 61.72 % | +0.194 |
| **Logistic (all features)** | **0.7228** | **63.34 %** | **+0.339** |
| Gradient boosting | 0.6927 | 60.98 % | +0.261 |

The **persistence floor** matters more than the base rate: earthquakes cluster,
so "a qualifying event happened recently, predict another" is free. A model
that cannot beat it has learned that seismicity is bursty, not how to forecast
it. AUC is the headline because positive rates run 0.24–0.61 by zone, so
accuracy is dominated by the base rate.

Logistic beating gradient boosting repeats the pattern of `report.md` §8.5 —
the simpler model wins again.

> **These are single-cut numbers. §4.4 replaces them with a distribution over
> 12 test eras, and the headline moves.** They are retained because they are
> what `forecast_eval.py` still prints, and because the gap between them and
> §4.4 is itself the finding.

### 4.2 Per zone — where the pooled number comes from

| zone | n_test | pos | persist | pooled | pool+zone | per-zone |
|---|---|---|---|---|---|---|
| NAFZ | 320 | 0.356 | 0.3799 | 0.3711 | 0.3755 | **0.4091** |
| EAFZ | 823 | 0.487 | 0.4990 | 0.5703 | 0.5689 | 0.5648 |
| AEGEAN | 1129 | 0.782 | 0.6398 | **0.7984** | 0.7929 | 0.7773 |
| CENTRAL | 142 | 0.176 | 0.2701 | 0.3491 | 0.3610 | **0.4239** |
| **MACRO** (each zone once) | | | 0.4472 | 0.5222 | 0.5246 | **0.5438** |
| **MICRO** (window-weighted) | | | 0.5356 | 0.6376 | 0.6358 | 0.6353 |

**The pooled 0.72 headline is AEGEAN's number in disguise.** Micro averaging
reproduces it because AEGEAN is the largest zone; macro averaging, where each
zone counts once, collapses it to 0.52–0.54.

Per-zone modelling is the better structure — best macro AUC, and the only
strategy beating persistence in **4 of 4** zones rather than 3 of 4 — but it
does not rescue the failing zones. NAFZ improves 0.371 → 0.409 and CENTRAL
0.349 → 0.424, both still below chance.

### 4.3 Forecastability tracks clustering

Persistence AUC differs in **sign** across zones, and tracks the coefficient of
variation of M ≥ 4.5 inter-event gaps almost monotonically:

| Zone | CV | Persistence AUC | Interpretation |
|---|---|---|---|
| AEGEAN | 1.56 | 0.640 | clustered — recent event ⇒ **more** likely |
| EAFZ | 1.46 | 0.499 | clustered, but no usable recency signal |
| NAFZ | 1.04 | 0.380 | ~Poisson — recent event ⇒ **fewer** |
| CENTRAL | 1.02 | 0.270 | ~Poisson, strongly inverted |

CV = 1 is exactly memoryless. NAFZ and CENTRAL sit there, so
P(event | history) ≈ P(event) and no model of this kind can forecast them —
the same argument that explained §1's chance-level result, now explaining
*which zones work*.

This is why a single pooled model cannot serve all four: it must represent two
opposite temporal dependences at once, so it learns the average and fails on
the minority behaviour. Per-zone models can represent both and do improve the
Poisson-like zones — but improving toward chance is not forecasting.

### 4.4 Rolling-origin backtest — the numbers above are one era

Everything in §4.1–4.3 rests on **one** chronological cut. That cannot be told
apart from *"the 2023–2026 test era happened to favour the Aegean"* — especially
since that era contains the Kahramanmaraş sequence, and CENTRAL and NAFZ are
judged on 142 and 320 windows. `forecast_backtest.py` walks the origin forward
over **12 semi-annual cuts**, refitting at each, keeping the 30-day embargo, and
reports AUC as a distribution.

**The bar is `max(chance, persistence)`, not persistence alone.** Persistence
scores 0.18–0.34 in NAFZ and CENTRAL — far *below* chance, because a recent
event there predicts *fewer* events. Beating an anti-predictive floor while
sitting below a coin flip is not a forecast; it is §2(b)'s trap in a new place.
Counted against persistence alone, CENTRAL "wins" 9 of 12 origins while its
median AUC is 0.369. Both counts are reported:

| Zone | CV | persist. median | **model median** | IQR | > persist. | **> both floors** |
|---|---|---|---|---|---|---|
| AEGEAN | 1.56 | 0.5396 | **0.6609** | [0.532, 0.763] | 8/12 | **6/12** |
| EAFZ | 1.46 | 0.4463 | **0.6500** | [0.523, 0.724] | 10/12 | **7/12** |
| NAFZ | 1.04 | 0.3414 | 0.4469 | [0.334, 0.571] | 6/12 | 3/12 |
| CENTRAL | 1.02 | 0.1830 | 0.3686 | [0.311, 0.589] | 9/12 | 4/12 |

**Three things change.**

1. **AEGEAN's 0.798 was near the top of its own distribution, not the middle.**
   The honest central estimate is **0.66, with an IQR spanning 0.23** — and it
   clears both floors at only 6 of 12 origins, which is exactly half. The
   direction of the result survives; the confidence in the specific number does
   not.
2. **EAFZ is forecastable, and §4.2 said otherwise.** Its median is 0.650 and it
   clears both floors more often than AEGEAN does (7/12). The single cut put it
   at 0.570 and §4.3 read that as "clustered, but no usable recency signal."
   That reading was an artifact of one era.
3. **NAFZ and CENTRAL are confirmed not forecastable** — both medians sit below
   chance across origins, not just in one test window.

Correction (2) **strengthens** §4.3 rather than undermining it. The clustering
story predicted that the two high-CV zones (1.56, 1.46) should work and the two
near-Poisson ones (1.04, 1.02) should not. The single cut broke that pattern by
putting EAFZ at chance; the backtest restores it. The physical explanation now
matches the data in all four zones instead of three.

Sliding vs. expanding training windows split 2–2, so there is no evidence here
that older data actively hurts.

**Cross-check: the two pipelines agree, which is what makes this diagnosis
safe.** The last origin (2025-05-28) covers roughly the same test period as
§4.2's single cut, and reproduces it:

| | n_test | positive rate | pooled AUC | per-zone AUC |
|---|---|---|---|---|
| §4.2 single cut, AEGEAN | 1129 | 0.782 | 0.7984 | 0.7773 |
| backtest origin 2025-05-28, AEGEAN | 862 | 0.881 | **0.8035** | 0.7787 |

So the earlier result was **not a bug** — it was a correct measurement of an
unrepresentative era. AEGEAN's positive rate at that origin is 0.881 against an
across-origin mean of 0.501. That is the "one lucky era" hypothesis confirmed by
measurement rather than argued from the calendar.

EAFZ's promotion survives the same check: it is not one spike but a spread —
0.762 at the 2023-03 origin (the Kahramanmaraş aftermath), 0.530 at 2024-04,
0.634 at 2025-05. The single cut sampled the low end.

**Phase-2 feature work is deliberately not being started.** The rule set before
seeing these numbers was that features would only be measured if the signal
survived a majority of origins. AEGEAN clears both floors at 6/12 — exactly the
boundary, not a majority — so `lyapunov` and sample entropy stay unmeasured.
Adding them now would mean comparing new point estimates against a distribution
this wide, which is how `report.md` §6.6's single-seed reversal happened.

### 4.5 Sensitivity to threshold and horizon

Median AUC over the same 12 origins, for M ≥ 4.0/4.5/5.0 × 15/30/60 days:

| M ≥ | 15 d | 30 d | 60 d |
|---|---|---|---|
| **AEGEAN** 4.0 / 4.5 / 5.0 | 0.597 / 0.696 / 0.605 | 0.680 / 0.661 / 0.636 | 0.661 / 0.600 / 0.524 |
| **EAFZ** | 0.624 / 0.535 / 0.549 | 0.529 / 0.650 / 0.561 | 0.668 / 0.466 / 0.599 |
| **NAFZ** | 0.529 / 0.516 / 0.395 | 0.496 / 0.447 / 0.328 | 0.476 / 0.432 / 0.359 |
| **CENTRAL** | 0.544 / 0.449 / 0.548 | 0.492 / 0.369 / 0.475 | 0.657 / 0.360 / 0.566 |

**AEGEAN clears chance in all nine cells** (0.524–0.696), so its result is not an
artifact of the M ≥ 4.5 / 30-day choice. **NAFZ clears it in none but two
marginal cells.** EAFZ is above chance in seven of nine. CENTRAL swings from
0.360 to 0.657 across cells, which is the signature of noise rather than a
horizon-dependent signal.

> **Do not read the M ≥ 5.0 rows as large wins.** At that threshold persistence
> scores *exactly* 0.0000 for CENTRAL and NAFZ — a perfectly inverted ranking,
> which happens because M ≥ 5.0 events are rare enough that
> `days_since_prev_major` is degenerate. The apparent "+0.55 over persistence"
> is a broken floor, not a good model. Those cells also rest on 7–10 usable
> origins rather than 12.

**The defensible claim**, restated to match the backtest: *M ≥ 4.5 within 30
days is forecastable in the two clustered zones — AEGEAN (median AUC 0.66,
IQR 0.53–0.76) and EAFZ (0.65, IQR 0.52–0.72) — and is not forecastable in the
near-Poisson NAFZ and Cyprus-arc zones, whose medians sit below chance. The
previously reported 0.798 for AEGEAN was the top of a wide distribution.*

### 4.6 Operating point

AUC is threshold-free; a forecast someone acts on is not. Following Başar &
Çelik (2026) — the only paper in `literature_review.md` that calibrates its
operating point rather than defaulting to 0.5 — the threshold is now chosen on
each origin's own validation slice. The selected value is **0.137, not 0.5**,
giving precision 0.317 at recall 0.679. Thresholding at 0.5, as the earlier
scripts did, was far too conservative for a target with this base rate.

### 4.7 Block-level evaluation — the honest sample size

Everything above is measured **per window**, and windows are not independent.
`forecast.py` slides a 64-event window with `stride_events=8`, so consecutive
windows share 56 of 64 events, and in AEGEAN they end **0.35 days apart while
the label looks forward 30 days** — consecutive labels share ~99 % of their
horizon.

| Zone | windows | independent 30-day blocks | inflation |
|---|---|---|---|
| AEGEAN | 9,193 | ~200 | **46×** |
| EAFZ | 6,224 | ~200 | 31× |
| NAFZ | 3,828 | ~199 | 19× |
| CENTRAL | 2,094 | ~199 | 11× |

The point estimates are not necessarily biased, but **any confidence read off
`n` is wrong** — by ~√46 ≈ 7× for AEGEAN. "n_test = 862, AUC 0.8035" rested on
roughly **ten** independent forecast opportunities.

`forecast_blocks.py` partitions each zone into disjoint 30-day blocks, giving
one forecast per block — made from the last window ending *strictly before* the
block opens, asserted on timestamps — and one outcome:

| Zone | blocks | base | **block AUC** | 95 % CI | verdict |
|---|---|---|---|---|---|
| EAFZ | 188 | 0.356 | **0.6209** | [0.529, 0.706] | **established** |
| AEGEAN | 192 | 0.432 | **0.5987** | [0.522, 0.674] | **established** |
| NAFZ | 190 | 0.274 | 0.4519 | [0.367, 0.540] | not established |
| CENTRAL | 180 | 0.189 | 0.4778 | [0.378, 0.578] | not established |

**The effect survives, smaller.** AEGEAN and EAFZ still exclude chance at
n ≈ 190, so this is a real result — but at 0.599 and 0.621, not the 0.661/0.650
window-level medians, and AEGEAN's lower bound is 0.522. NAFZ and CENTRAL now
include chance, which is a *weaker* statement than §4.4's "below chance" and the
correct one.

> **Block outcomes are read from the catalog, not inherited from window
> labels.** A window ending at `block_start + 25d` carries a horizon reaching
> `block_start + 55d`, so aggregating window labels marks a block positive for
> events happening up to a full horizon *after* it closes. A first version did
> exactly that and inflated every base rate by roughly 30 % (AEGEAN 0.620 rather
> than 0.432). Because the base rate is the reference for both Brier skill and
> information gain, that error moved the goalposts rather than the scores — and
> correcting it flipped AEGEAN's verdict in §4.8.

### 4.8 Discrimination without value: the probabilities were unusable

AUC is rank-only. Adding the CSEP-standard probabilistic scores — none of which
appears in any of the seven papers in `literature_review.md` — showed every zone
scoring **negative** Brier skill and negative information gain against its own
base rate. The raw model ranks better than chance while its probabilities are
*worse than climatology*. Reliability curves showed the cause: AEGEAN emits
0.089 where 0.410 is observed and 0.211 where 0.684 is observed — systematically
far too low.

Part of this is structural rather than a bug: the model is trained on **window**
labels but scored against **block** outcomes, which aggregate by `max` and so
carry a different base rate (AEGEAN 0.563 vs 0.620).

Prospective Platt recalibration — refit at every block on prior blocks only,
never on the block being calibrated — fixes most of it. AUC is unchanged by
construction, since Platt is monotone:

| Zone | BSS raw | BSS cal | IG raw | IG cal | usable? |
|---|---|---|---|---|---|
| **EAFZ** | +0.045 | **+0.032** | +0.014 | **+0.022** | **yes** |
| AEGEAN | −0.088 | **+0.005** | −0.109 | **+0.003** | marginally |
| NAFZ | −0.130 | −0.019 | −0.096 | −0.016 | no |
| CENTRAL | −0.116 | −0.015 | −0.106 | −0.010 | no |

**EAFZ is the only zone with clearly usable probabilities**, and it is the only
one positive even *before* recalibration. AEGEAN crosses into positive territory
only after calibration and only by +0.005 BSS — which is to say its
probabilities are, for practical purposes, the base rate. It discriminates
measurably (AUC 0.599, CI excluding chance) while carrying almost no
probabilistic value.

That gap between **discrimination and value** is completely invisible to AUC,
and is the single most useful thing this evaluation produced. A zone can rank
better than chance and still tell you nothing you did not already know from
climatology.

**Final claim, at the honest sample size:** *M ≥ 4.5 within 30 days is weakly
but measurably forecastable in the East Anatolian and Aegean zones (block AUC
0.62 and 0.60 over ~190 independent blocks, both CIs excluding chance). Only
EAFZ yields calibrated probabilities that beat climatology by a usable margin
(BSS +0.032); AEGEAN's are within +0.005 of the base rate. NAFZ and the Cyprus
arc are indistinguishable from chance.*

### 4.9 Zone-relative features — tested, not adopted

§4.3 diagnosed a problem it never fixed: a pooled model "must represent two
opposite temporal dependences at once." The features are all *absolute*
(`log_rate`, `log_total_energy`, …) while baseline seismicity differs
several-fold between zones, so the model partly learns *which zone this is*
rather than whether a zone is unusual **for itself**. The fix is a trailing
z-score — `.expanding().shift(1)` within each zone, so a window is never part of
the statistics that normalise it.

Judged by **paired bootstrap** on the same blocks (both models scored on
identical resamples, so shared variance cancels and only the difference
remains), not against §4.7's interval, which would be badly underpowered:

| Zone | feature set | block AUC | Δ vs absolute | 95 % CI on Δ | BSS cal |
|---|---|---|---|---|---|
| **EAFZ** | absolute | 0.6209 | — | — | +0.032 |
| | relative only | 0.5911 | −0.0299 | [−0.073, +0.014] | +0.009 |
| | absolute + relative | 0.6161 | −0.0048 | [−0.040, +0.029] | +0.036 |
| **AEGEAN** | absolute | 0.5987 | — | — | +0.005 |
| | relative only | 0.6211 | +0.0224 | [−0.002, +0.048] | +0.021 |
| | absolute + relative | 0.5004 | **−0.0983** | **[−0.157, −0.042]** | −0.019 |
| NAFZ | relative only | 0.4848 | +0.0329 | [−0.019, +0.084] | −0.019 |
| CENTRAL | relative only | 0.5056 | +0.0278 | [−0.025, +0.079] | −0.015 |

**Verdict: do not adopt.** The rule was fixed before running — adopt only if the
paired CI excludes zero in a zone that is *itself* established (EAFZ or AEGEAN).
No such interval does.

Three things are worth recording anyway:

1. **Relative-only beats absolute in 3 of 4 zones** (+0.033, +0.022, +0.028) and
   loses in EAFZ (−0.030). A consistent direction in three zones with no
   individually significant interval is a hint, not evidence — and this project
   has already been burned once by treating a sub-0.02 margin as real
   (`report.md` §6.6).
2. **The one significant result is that combining both sets *hurts*.** AEGEAN
   drops 0.0983 AUC with a CI excluding zero. Doubling to 22 features against
   ~190 independent blocks is straightforward overfitting, and it is the
   clearest demonstration yet of how little data this problem actually has once
   the overlap is removed.
3. **Relative-only improved AEGEAN's calibration** (BSS +0.005 → +0.021), the
   one zone whose probabilities were borderline useless in §4.8. Not significant
   on AUC, but the direction is consistent with the diagnosis in §4.3.

The honest summary is that the structural criticism in §4.3 was probably
correct and this dataset is too small to demonstrate it.

## 5. Limitations

- **Below-chance AUC in NAFZ and CENTRAL persists across all 12 origins**, so it
  is not an artifact of one era. Whether it reflects a genuinely inverted
  relationship or simply no signal plus small samples is not resolved here;
  §4.5 shows CENTRAL swinging 0.360–0.657 across grid cells, which favours
  noise.
- **Even the zones that work are wide.** AEGEAN's IQR spans 0.23 AUC and it
  clears both floors at only 6 of 12 origins. This is a weak, real effect, not
  a reliable forecast — a single origin's number should never be quoted alone,
  which is the mistake §4.1–4.3 made.
- **Kahramanmaraş is now handled structurally**, since it falls in train for
  some origins and test for others, but it still dominates EAFZ's later origins.
- **Single seed, single model family.** The backtest fixes the split, threshold
  and horizon, but every cell is one logistic fit; `report.md` §6.6 showed
  single-seed margins on this project can reverse.
- **No neural model has been run on the reformulated target.** Given that
  logistic beats gradient boosting here, and the §8.5 precedent, the priors are
  not favourable — but it is untested.
- **`lyapunov` is computed but unused, and is staying that way for now.**
  `catalog.py:174` (`max_lyapunov_rosenstein`) is in `AUX_FEATURES` but was
  dropped from `forecast.py:FEATURES` during the reformulation — an unaudited
  omission rather than a decision. It is *not* a pending TODO: per §4.4, the
  backtest did not clear the bar that was set in advance for starting feature
  work, so this and sample entropy remain unmeasured on purpose. Reopen only if
  the underlying signal strengthens.

## 6. Reproduction

```bash
cd cnn_earthquake/src
# pooled, with base-rate and persistence floors
python forecast_eval.py    --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
# per-zone vs pooled vs pooled+zone-feature, shared chronological cuts
python forecast_perzone.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
```

Zone definitions, window construction and the split rule live in
`data_downloader/seismic_cli/forecast.py`; the superseded time-to-mainshock
pipeline remains in `seismic_cli/catalog.py` with its LOEO trainers
(`cnn_lstm_loeo.py`, `catalog_scalar_loeo.py`).
