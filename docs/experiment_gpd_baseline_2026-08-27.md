# GPD on our windows: the first comparable number

**Date:** 2026-08-27
**Script:** `src/detection/pretrained_picker_baseline.py`
**Runner:** `scripts/run_picker_baseline.sh`
**Tests:** `tests/test_pretrained_picker.py`

## Why

Every headline in the P-wave detection literature is reported on its own
corpus, with its own negatives, its own split, and — as
`docs/related_work_pwave_detection.md` §2.3 records — **no conditional floor
anywhere**. "GPD reports precision >99%" and "this work reports AUC 0.8712" are
therefore not comparable quantities, and arranging them in a table does not make
them so.

There is exactly one way to get a number that compares: run their model on our
data and score it the way we score ours. That is what this does.

GPD (Ross et al. 2018) is the natural first choice, and not only because it is
foundational. Its input window is **400 samples at 100 Hz = 4.0 s**, against our
340 samples = 3.4 s. PhaseNet needs 30 s and EQTransformer 60 s, so both would
require inventing 27–57 s of context that our windows do not contain. GPD needs
0.6 s of padding.

## How the comparison was made honest

### The waveform is rebuilt from source mseed, not read from the dataset tensors

The stored `seq` is bandpass-filtered, tapered **and divided by each
(station, component)'s long-term noise sigma**. That last step is the problem:
it rescales the three components relative to each other, and GPD's own
max-normalisation cannot undo it, because that normalises the whole block
globally rather than per component.

Measured on 150 event windows, feeding the stored tensors instead of raw counts
moves GPD's P>0.5 rate from **17.3% to 6.7%**, and the two rankings agree only
at Spearman **0.60**. So the tensors are not a valid input for a foreign model,
and every window is rebuilt from the source record instead.

**The rebuild is verified, not assumed.** Applying the corpus's own filter chain
(`clean_and_filter_1d`: detrend, 5% Hann taper, Butterworth-4 bandpass 1–45 Hz)
to the reconstructed samples reproduces the stored tensor at correlation
**1.0000** on both classes — 20 events and 20 noise windows, minimum 1.0000.
That confirms the sample arithmetic: events are `win000` at offset 0, noise
windows are `win{idx}` at offset `idx × 170` (a 3.4 s window at `--overlap 0.5`).

### P is placed on GPD's prediction sample

`arrival_from_catalog.py` cuts 2.0 s before the per-station predicted P, so P
sits at **sample 200 of 340**. GPD's `pred_sample` is **200 of 400**. Padding
the 60-sample shortfall at the **end** therefore leaves P exactly where GPD
expects it, with no shifting.

This is a claim, so it is tested (`--front-pad`, and
`tests/test_pretrained_picker.py`): getting the padding backwards moves P to
sample 260 while still producing the right shape, a valid softmax and a
plausible AUC. Nothing would raise.

### Preprocessing is a choice, so both are reported

`gpd` gives the model its own documented input — a 2 Hz highpass on raw counts,
which is what `filter_args` on the pretrained weights specifies. `pipeline`
applies the corpus's 1–45 Hz bandpass instead, which is what our models saw.
Neither is "correct"; the gap between them is a measurement, not a nuisance.

## Result 1: which pretrained GPD, on identical rows

14,821 of 15,816 test windows rebuilt (7,908 event / 6,913 noise; 995 noise
windows could not be recovered from their source record). **Amplitude floor on
exactly these rows: 0.5860.** GPD's own preprocessing, P on the prediction
sample.

| weights | trained on | AUC (P-class) | headroom captured | recall @0.5 | precision @0.5 | false alarms |
|---|---|---|---|---|---|---|
| `geofon` | global | **0.7987** | **51.4%** | 0.2680 | 0.9920 | 17 |
| `original` | SCSN (Ross et al.) | 0.7710 | 44.7% | 0.1104 | 0.9876 | 11 |
| `stead` | global | 0.7499 | 39.6% | 0.0008 | 1.0000 | 0 |
| `instance` | Italy | 0.7457 | 38.6% | 0.0152 | 1.0000 | 0 |
| `scedc` | S. California | 0.7154 | 31.3% | 0.0412 | 0.9879 | 4 |

`headroom captured` = `(AUC − floor) / (1 − floor)`, the same statistic used
throughout this project, because a raw AUC difference across corpora with
different floors is not interpretable.

**The spread across weight sets is 0.083 AUC on identical data.** That is larger
than most of the differences the literature reports between *architectures*,
and it is invisible in any paper that quotes a single pretrained model. Which
GPD you download matters more than GPD-versus-something-else.

**Every weight set clears the floor, and none of them clears it by much.** The
best captures roughly half the available headroom on a corpus it never saw.

**Recall at the default threshold is very low and precision is near-perfect.**
GPD is calibrated for a different operating point than ours: it almost never
fires on our small regional events (M ≥ 2.0, median 2.3, ≤ 56 km), and when it
does it is essentially always right. Reading the recall column as "GPD is bad
here" would be wrong — reading it as "GPD's threshold does not transfer to this
magnitude range" is what the numbers support.

## Result 2: head to head, identical rows, identical floor

Both models scored on the same 14,821 windows, against the same 0.5860 floor.
This repo's detector is **re-scored here rather than quoted**: its published
0.8712 is on all 15,816 rows, and dropping the 995 unrecoverable noise windows
moves both the score and the floor.

| | AUC | headroom captured | recall @0.5 | precision @0.5 | false alarms |
|---|---|---|---|---|---|
| amplitude floor | 0.5860 | 0% | — | — | — |
| GPD `scedc` | 0.7154 | 31.3% | 0.0412 | 0.9879 | 4 |
| GPD `original` | 0.7710 | 44.7% | 0.1104 | 0.9876 | 11 |
| GPD `geofon` (best) | 0.7987 | 51.4% | 0.2680 | 0.9920 | 17 |
| **this work (fusion, 3 seeds)** | **0.8796** | **70.9%** | 0.6391 | 0.9482 | 276 |

**The margin over the best pretrained GPD is +0.081 AUC, and 19.5 points of
headroom.** That is a real gap and not a blowout, which is the honest way to
report it. A model trained on this corpus beats a foreign model on this corpus;
this says nothing about how GPD performs on the data it was built for, and
nothing about how our model would perform on SCSN.

**The operating points are not comparable, only the AUCs are.** GPD fires on
11% of events at its default threshold and is right 98.8% of the time; ours
fires on 64% and is right 94.8%. That is a threshold difference — GPD's is
tuned for a different magnitude regime — which is exactly why the
threshold-free statistic is the one in the comparison column.

## Result 3: alignment costs 0.017 AUC, and changes the operating point a lot

`--front-pad 60` moves P from sample 200 to sample 260, off GPD's prediction
sample, changing nothing else.

| | AUC | recall @0.5 | precision @0.5 | false alarms |
|---|---|---|---|---|
| P on the prediction sample | **0.7710** | 0.1104 | 0.9876 | 11 |
| P 60 samples late | 0.7540 | 0.3237 | 0.8752 | 365 |

The ranking cost is small, but the **false alarms go from 11 to 365** and
precision drops 12 points. A misaligned window still produces a valid softmax
and a plausible AUC — it just quietly becomes a different, much noisier
detector. This is the failure the padding tests exist to prevent.

## Result 4: GPD does better with its own filter than with ours

| preprocessing | AUC | recall @0.5 | precision @0.5 |
|---|---|---|---|
| GPD's own 2 Hz highpass on raw counts | **0.7710** | 0.1104 | 0.9876 |
| the corpus's 1–45 Hz bandpass | 0.7458 | 0.1004 | 0.9950 |

Handing GPD the band our own models were trained on costs it 0.025 AUC. Its
documented preprocessing is the better input, which is the expected result and
the reason the headline row uses it: the comparison should give the foreign
model its best shot, not the most convenient one.

## Result 5: the natural regime, and an independent check on the transfer matrix

Same events, natural (unmined) negatives. 15,479 rows rebuilt, floor 0.6240.

| | AUC | headroom captured | recall @0.5 | precision @0.5 | false alarms |
|---|---|---|---|---|---|
| amplitude floor | 0.6240 | 0% | — | — | — |
| GPD `original` | 0.7748 | 40.1% | 0.1104 | 0.9977 | 2 |
| **this work (natural-trained fusion)** | **0.8414** | **57.8%** | 0.6328 | 0.9709 | 150 |

Margin +0.067 AUC, 17.7 headroom points — narrower than on matched, and in the
same direction as everything else measured about the natural regime.

**GPD is nearly invariant to which negatives we selected: 0.7710 on matched,
0.7748 on natural.** It never saw either regime, so it has no reason to prefer
one, and it does not. The events are identical across the two datasets; only
the noise differs. That makes GPD a *foreign control* for the negative-regime
work in `experiment_ponly_2026-08-22.md`: our own model moves 0.8796 → 0.8414
across the same two sets while a model with no exposure to either barely moves.
The difference is a property of the negative distribution and of what training
on it teaches, not of the positives — which is what the transfer matrix
claimed, now supported by a measurement that has nothing to do with our
training pipeline.

Note GPD's recall is **identical to four decimals** across the two datasets
(0.1104, 0.0693) — as it must be, since recall is computed on the same 7,908
event windows. Only the false-alarm and precision columns can move.

## What this licenses, and what it does not

**Licensed.**

- On this corpus, with this protocol, against this floor, the local detector
  beats the best pretrained GPD by **+0.081 AUC** (matched) and **+0.067**
  (natural), capturing 70.9% vs 51.4% of available headroom.
- The literature's numbers cannot be compared to ours directly, and this is
  what the comparison costs to do properly: rebuilt waveforms, verified sample
  arithmetic, alignment on the model's own prediction sample, its own
  preprocessing, and a floor computed on the surviving rows.
- Which pretrained weights you pick moves GPD by 0.083 AUC on identical data —
  more than most published architecture comparisons.

**Not licensed.**

- **Nothing about GPD's quality on the data it was built for.** SCSN events at
  SCSN distances are not M ≥ 2.0 at ≤ 56 km in the Aegean, and a model losing
  accuracy out of distribution is the expected finding of the SeisBench
  benchmark, not a criticism of it.
- **Nothing about how our model would do on their corpus.** The reciprocal
  experiment — this detector on STEAD, against a floor computed there — was run
  the same day and **reverses the result**: on STEAD-natural GPD `geofon`
  reaches 0.9796 against our 0.9207, capturing 91.0% of headroom to our 65.1%.
  Each model wins at home, by a comparable margin. See
  `experiment_stead_reciprocal_2026-08-27.md`; the table below should never be
  quoted without it.
- **No statement about picking accuracy.** GPD is being used as a detector
  here, scored by its P-class probability. Its actual job is onset timing, and
  this measures none of it. Our task cannot produce a pick residual at all.

## Open

1. **PhaseNet and EQTransformer need 30 s and 60 s of context.** Padding our
   3.4 s windows to those lengths would be inventing 27–57 s of data. The
   correct version runs them on the original continuous records around each
   catalogued P and asks whether they pick inside our window — which also
   produces the false-alarms-per-station-day number this project has never had
   (`related_work` §4). 35 of 183 stations are held out in test; BODT cannot be
   used, being a training station.
2. **995 noise windows (12.6%) could not be rebuilt** on the matched set, 337
   (4.3%) on natural — their source records no longer supply a full
   three-component window at the recorded offset. Both models are scored on the
   survivors, so the comparison is fair, but the dropped rows have not been
   characterised.
3. **The reciprocal run** — this detector on STEAD, against a floor computed on
   STEAD — is the missing half.
