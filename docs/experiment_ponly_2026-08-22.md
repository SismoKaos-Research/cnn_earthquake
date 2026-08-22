# P-only detection windows, and what two corrections cost

2026-08-22. Everything below is measured; the logs it came from are gitignored,
so this file is the record.

## Why

The 6 s detection window is `[P − 2.0 s, P + 4.0 s]`, and generation only ever
computed P phases (`arrival_from_catalog.py`: `PHASES = ["p","P","Pg","Pn"]`).
Whether S also landed inside was never checked. TauP on the same
(distance, depth) pairs says it did — **28.8% of event windows corpus-wide,
32.5% of the detection test split, and 99.3% of windows inside 25 km**, since
S−P scales with distance.

That does not invalidate the 6 s results: the conditional amplitude floor is
computed on the same windows the model sees, so the comparison stays fair. It
matters operationally. A detection that only works once S has arrived carries
no early-warning value at that site.

## First: is the 6 s detector actually leaning on S?

Stratifying recall by S-present cannot answer this — at fixed distance S−P
varies only through depth, so "S is present" and "the event is close" are
nearly the same statement. `src/detection/s_arrival_ablation.py` intervenes
instead: zero every sample from the predicted S arrival, re-score with the
existing weights, and control for the fact that removing a tail removes signal
whether or not that signal is S.

| group | n | baseline | masked | change |
|---|---|---|---|---|
| S-present → zeroed at S | 2,567 | 0.9747 | 0.9459 | −0.0288 |
| S-absent → untouched *(sanity)* | 5,339 | 0.9358 | 0.9358 | +0.0000 |
| S-absent → same tail zeroed *(control)* | 5,339 | 0.9358 | 0.9002 | −0.0356 |

**Removing S costs less than removing an equal stretch of ordinary signal.**
The duration-matched control drops further than the S mask does. The detector
was not leaning on S.

Bounds worth stating: the control group is systematically weaker (baseline
0.9358 vs 0.9747) because S-absent means farther and lower-SNR, and lower-SNR
windows may be more fragile to truncation — which would flatter the estimate.
So S-dependence lies between 0 and −0.0288, the total drop. Zeros are also out
of distribution for a model trained on unmasked windows, so the drop is an
upper bound either way.

## The P-only dataset

`arrival_from_catalog.py --window-seconds 3.4 --pre-arrival-seconds 2.0`.

The two flags are decoupled deliberately. The default pre-buffer is `win/3`,
which at 3.4 s would be 1.13 s against a 0.63 s prediction MAD — the retention
trap the generator's own comment warns about. Holding pre at 2.0 s kept
retention at **96.4%**, matching the 6 s config's 96.3%.

Verified across all 55,595 generated recordings: **zero windows where S
intrudes**, minimum S−P 1.450 s against the 1.4 s cut.

**That guarantee is relative to iasp91, not absolute.** The closest arrival
clears by 0.050 s, well inside the corpus's own location error (median RMS
residual 0.42 s; a 5 km distance error moves S−P by ~0.6 s):

| if S−P prediction is off by | recordings that could still contain S |
|---|---|
| 0.00 s | 803 (1.44%) |
| 0.30 s | 1,621 (2.92%) |
| 0.63 s | 3,140 (5.65%) |

Write it as "P-only by construction under the velocity model", not "zero S".

## The floor collapsed, for a physical reason

**P is the weak phase.** Cutting at 1.4 s post-P removes S and coda:

| | P-only 3.4 s | 6 s |
|---|---|---|
| event median `seq abs-max` | 7.48 | 44.38 |
| noise median | 3.71 | 1.54 |
| separation | 2.0× | ~29× |
| floor | 0.6447 | 0.9049 |

35.2% of events ended up quieter than the median noise window, and
`img mean dB` fell to exactly chance (0.4990 raw) — with a 2.0 s pre-buffer,
**59% of a 3.4 s window is pre-arrival background**, so any mean-over-window
statistic is diluted.

## But 0.6447 was not the real floor

Hard-negative mining keeps the 75th–99th percentile of the noise pool, which
puts a hard amplitude floor under every negative that the positives lack.
Result:

`P(event | seq abs-max decile)` = 0.67 0.41 0.32 0.29 0.30 0.33 0.30 0.50 0.88 1.00

**U-shaped.** The quietest decile is 67% events. A model can learn
"very quiet → event" and be rewarded — an artifact of the mining, since in
continuous data quiet windows are overwhelmingly noise.

ROC-AUC only measures monotone ranking, so it understates a U-shape. A depth-4
decision tree on that *same single scalar*, fit on train and scored on test,
reached **0.7461** against the monotone 0.6447.

### This was checked everywhere, and only affects the P-only build

| dataset | monotone | non-monotone | gap |
|---|---|---|---|
| 6 s catalog+hard | 0.9049 | 0.9003 | −0.0045 |
| 3 s catalog+hard | 0.8481 | 0.8445 | −0.0036 |
| 6 s özgün (gated) | 0.7088 | 0.7086 | −0.0002 |
| 6 s catalog random | 0.9535 | 0.9477 | −0.0058 |
| 3 s catalog | 0.9308 | 0.9176 | −0.0132 |
| **P-only 3.4 s band** | **0.6447** | **0.7461** | **+0.1015** |

**Every published floor is clean.** The gaps elsewhere are negative — the tree
losing a little to discretisation, which is what a genuinely monotone
relationship looks like.

## The fix: amplitude-matched negatives

`--match-negative-amplitude` (in `data_downloader`, commit `b3980c5`) scores
the positives in the same station-sigma units and fills quantile bins of their
distribution, instead of taking a fixed percentile band.

| | band 0.75–0.99 | amplitude-matched |
|---|---|---|
| `P(event \| decile)` | 0.67 0.41 0.32 0.29 0.30 0.33 0.30 0.50 0.88 1.00 | 0.40 0.38 0.39 0.35 0.41 0.42 0.45 0.53 0.70 0.96 |
| quietest decile | 0.670 | **0.399** |
| monotone floor | 0.6447 | 0.6679 |
| non-monotone floor | 0.7461 | 0.6658 |
| gap | +0.1015 | **−0.0021** |

The U-shape flattens into a monotone rise, the quiet decile drops below base
rate (quiet now means slightly *more* likely noise — the correct direction),
and the monotone/non-monotone gap closes. **ROC-AUC is an honest floor on this
build.**

### Why it stops at 0.6679 and not 0.5

The pool runs out. Event `seq std` reaches 5,439; the loudest noise window in
2.5 M candidates is 77.8.

| percentile | event | noise | ratio |
|---|---|---|---|
| 1% | 0.106 | 0.092 | 1.15× |
| 10% | 0.328 | 0.269 | 1.22× |
| 50% | 1.685 | 0.910 | 1.85× |
| 75% | 7.03 | 1.89 | 3.73× |
| 99% | 237.4 | 15.97 | 14.9× |

Matching holds in the bottom half and degrades above it. Essentially all
remaining discrimination comes from the unmatched loud tail — which is real
physics, not mining: a window that loud *would be* an earthquake, and the
99th-percentile cap that prevents mining it exists to keep catalogue-missed
events out of the negative class.

The miner's printed "unmatchable" figure (2.4% on test) counts only events
above the *global* noise max. It does not capture how poorly the upper
quartile is matched. Read it with the table above, not alone.

## Results

| config | AUC | floor | raw gain | **headroom captured** |
|---|---|---|---|---|
| 6 s 1D | 0.9896 | 0.9049 | +0.0847 | **89.1%** |
| 6 s 2D | 0.9882 | 0.9049 | +0.0833 | 87.6% |
| P-only band 1D | 0.9291 | 0.7461 | +0.1830 | 72.1% |
| P-only band 2D | 0.9015 | 0.7461 | +0.1554 | 61.2% |
| P-only matched 1D | **0.8712** | 0.6679 | +0.2033 | **61.2%** |
| P-only matched 2D | **0.8602** | 0.6679 | +0.1923 | **57.9%** |

Per-seed: matched 1D 0.8673/0.8709/0.8671, matched 2D 0.8605/0.8610/0.8544.
Spreads 0.0038 and 0.0066 — stable, not noise.

**Raw gain grows at every step (+0.085 → +0.183 → +0.203) purely because the
floor falls.** Headroom captured moves the other way: **89% → 72% → 61%**.
Quoting the raw gain here is exactly the error §5.1 of the report warns
against. The training script's `<- the number that matters!` line prints the
raw gain; it is the wrong number across datasets with different floors.

(Band-2D and matched-1D both landing on 61.2% is coincidence — different
datasets, floors and arms.)

**1D beats 2D on both P-only builds**, reversing the 6 s near-tie
(0.9896 vs 0.9882). The spectrogram branch lost its amplitude cue *and* has
only 22 time frames, reduced to ~6 by two stride-2 convolutions, against 38→10
at 6 s. See "Open" below.

## What this means

Strip the amplitude cues and restrict the model to 1.4 s of P, and it captures
~61% of available headroom rather than ~89%. That is still a real detector
working on waveform character — clearing 0.6679 by 0.20 with no amplitude to
lean on is not trivial — but it is a materially weaker claim than the 6 s
headline implies about detection generally.

These configurations answer **different questions**, and the report should say
so rather than presenting P-only as a refinement of the same one:

- 6 s measures detection where loudness carries most of the signal.
- P-only measures early-warning-usable detection where it does not.

## Open

1. **STFT geometry for the 2D branch.** `hop=16` gives 160 ms per frame; a P
   onset transitions in tens of ms. This is the one hyperparameter chosen by
   analogy to the 3 s experiment rather than measured, and the 2D arm is the
   one underperforming. A finer hop, everything else fixed, is ~1 h.
2. **Report §3.2 discrepancy.** It states `n_fft = 256, hop = 64`,
   `img (3, 129, 10)` as the method, but the headline results come from
   `dataset_specdual_catalog_6s_matched_hard`, which is `(3, 33, 38)` —
   n_fft 64, hop 16. §4.5 acknowledges the change and quantifies it at 0.0010,
   but only in the window-length section, so a reader of §3.2 would assume
   otherwise. Two-line fix.
3. **`cascade_eval.py:91`** still does `glob("*.pth")` with no filtering, while
   `calibrate.py`, `operating_envelope.py` and `s_arrival_ablation.py` all use
   an anchored regex. Point it at a multi-arm directory and it silently
   ensembles `cnn` with `cnn-lstm`.
4. **Wideband diagnostic** (`dataset_specdual_ponly_3p4s_wideband`, band
   0.0–0.99) is built but untrained. The matched result has largely superseded
   the question it was meant to answer.
