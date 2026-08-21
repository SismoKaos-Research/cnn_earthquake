# Maybe-later list

Things worth trying that nothing currently depends on. Entries marked
[DONE] are kept for the record with their outcome, not deleted. Not a plan -- an
unordered backlog, newest first. `docs/TOMORROW.md` is the *forecasting*
task family's carried-over plan and is separate from this.

## [DONE] Operating-envelope analysis (no training -- it is a join)

**DONE 2026-08-19 (operating_envelope.py; recall is SNR-governed, not magnitude).**

Turn "0.9896 AUC" into an operational claim: recall stratified by magnitude,
by `log_snr`, and by `distance_km`.

**Feasible right now with zero GPU time.** Verified 2026-08-19: every detector
test event joins to the magnitude manifest by filename -- **7,906 of 7,906** --
which carries `magnitude`, `log_snr` and `distance_km`. The 410 missed events
from the `cnn-lstm` run are already identified. So this is a join against
existing artefacts, not a run.

**Why it is the highest value per minute available.** "0.9896 AUC" is not an
operational statement; "95% recall for M>=2.5 within 80 km" is, and it is the
form a seismology reviewer expects. It also converts the 410 misses from a
number into a characterisation: concentrated at low magnitude and low SNR is
an expected, defensible envelope; scattered across strong events means
something is wrong that a single AUC is hiding.

Feeds directly into the cascade too -- a missed event can never receive a
magnitude, so the detector's envelope IS the cascade's envelope.

## [DONE] Amplitude ablation: decompose loudness vs shape

**DONE 2026-08-20 (per-window normalised set; cnn-lstm 0,9309 vs seq floor 0,7088).**

Retrain with per-window standardisation, which deletes absolute amplitude. The
`seq` abs-max floor collapses toward 0.5, so whatever AUC survives is **pure
waveform shape**.

**Why it matters more than it looks.** It splits 0.9896 into "how much is
loudness, how much is morphology" -- the direct answer to the obvious
challenge, *isn't your detector just an amplitude threshold?* Given that this
project's whole methodological stance is about vacuous baselines and
conditional floors, this is the experiment that most strengthens the central
argument.

**State this when quoting the result, not after someone misreads it:** the
number is NOT comparable to 0.9896. Per-window standardisation changes the
dataset, not just the model. It is a separate measurement against its own
floor, and that floor must be printed alongside it. Expect a much lower AUC
that means considerably more.

## Network coincidence detection

Require 2-of-N stations to fire within a short time window. This is the
classical fix for false alarms and it is multiplicative: independent FPs at
1.78% become ~0.03% at 2-of-2 -- the order of magnitude the continuous-data
arithmetic above actually needs.

**Lead with the limitation.** Verified 2026-08-19: only **1,184 of 6,459 test
events (18.3%)** have >=2 stations (distribution: 5,275 events at 1 station,
970 at 2, 176 at 3, 27 at 4, 11 at 5). Single-station events are untouched by
coincidence logic entirely, so any measured FP reduction applies to under a
fifth of the catalogue.

So this is a promising direction **specifically** for the continuous-deployment
question, where FP/day is the metric and coincidence is the standard answer --
not a general improvement to the current detector. Testable offline on the
existing test split, no new data.

## Calibration and threshold selection

Brier is 0.0352 but nothing has checked whether the probabilities are actually
*calibrated*. For the cascade the threshold is a real decision -- it sets how
many events never receive a magnitude at all. A reliability diagram plus
temperature scaling is an afternoon's work and converts the arbitrary 0.5
threshold into a justified one. Pairs naturally with the operating-envelope
analysis above.

## Continuous-data / P-wave picking

The instinct is right -- continuous deployment is the only test that matters
for a real detector -- but "slide the existing 6s classifier over continuous
data" is the wrong shape for it, for three separate reasons. Reframe the
question as **"does this detector survive continuous data, and what does it
cost in false alarms?"**

### 1. Base rates destroy precision, and AUC hides it

The benchmark is 50/50 event/noise. Continuous data is not. `cnn-lstm`'s
false-positive rate is 141/7906 = **1.78%** (2026-08-19 run). At 6s windows
with 1s stride that is 86,400 windows/day/station:

| quantity | per station per day |
|---|---|
| false positives | ~1,540 |
| real events (optimistic) | ~10 |
| precision | **~0.6%** |

A 0.9896-AUC model becomes an alarm that is wrong 99.4% of the time. Nothing
is broken -- AUC is invariant to base rate, which is exactly why it is the
wrong metric here. Reaching even 50% precision needs an FPR near 0.012%,
about **150x lower** than current.

**Therefore: report false alarms per day at fixed recall, never AUC.**

### 2. The model is substantially an amplitude discriminator

The floor is 0.9049 from `seq` abs-max, a single scalar with no learning. The
model captures 89% of the headroom above it, so it does do more than threshold
amplitude -- but amplitude is most of the class separation (earthquake windows
reach 9.9e4, noise tops out at 1.0e2). Negatives were mined from a 75th-99th
amplitude percentile band; continuous data contains trucks, calibration
pulses, telemetry dropouts, spikes and teleseisms that band never represented.
An amplitude-dominated model fires on all of them.

### 3. Window classification is not picking -- and this is where it broke before

A window label says "an event is somewhere in these 6 seconds". A P-pick needs
an onset time to a fraction of a second. Sliding windows give time resolution
no better than the stride, and the label carries no within-window position.

The labelling ambiguity is exactly what wrecked the earlier 3s datasets:
`win002` was labelled positive while containing no onset at all (floor
0.8514), and `win000` held only 1.0s of post-arrival signal. Slicing event
recordings into overlapping windows *creates* that ambiguity by default.
Define what a coda-only window is, and what a window ending 0.5s before onset
is, BEFORE building anything -- or rebuild the same defect. See
[[project-3s-detector]].

This is why PhaseNet and EQTransformer do per-sample segmentation emitting a
probability time series rather than window classification: position comes
free and the window-label question never arises.

### Blocker to hit first: there is no continuous 100 Hz data

The detector trains at `fs=100.0` (manifest of
`dataset_specdual_catalog_6s_matched_hard`). The only continuous archive is
`aegean_dat_2024_2026_10hz` at 10 Hz. So there is currently nothing to slide
over. Either downsample the model to 10 Hz -- losing most of the P-onset band,
probably fatal for picking -- or download continuous 100 Hz waveforms.

### Recommended order

1. **Cheap diagnostic first.** Run today's `cnn-lstm` unchanged over a few
   hours of continuous 100 Hz data from one station and count false alarms
   per hour. No training, no new dataset. If it is ~60/hour as the arithmetic
   predicts, the base-rate wall is measured rather than assumed. If it is far
   lower, the model generalises better than the balanced benchmark suggests --
   also a finding worth reporting.
2. **Only then**, if picks are the goal, build a segmentation model rather
   than a window classifier -- or evaluate pretrained PhaseNet/EQTransformer
   via SeisBench on the Aegean data. A cross-corpus comparison against those
   on this network is likely a stronger contribution than a from-scratch
   picker, and it is adjacent to the STEAD evaluation already listed below.

## CNN-GRU waveform branch

Swap the BiLSTM in `ConvSeqBranch` for a BiGRU and re-run the branch-1d grid
as a fourth arm (`--branch-1d cnn-gru`).

**Why it is worth a run.** The 2026-08-19 grid put `cnn-lstm` (0.9896) above
`cnn` (0.9843) with non-overlapping per-seed ranges, so on this data the
recurrence is load-bearing rather than decorative -- the whole gap is recall
(410 misses vs 664), meaning recurrence is what recovers the marginal events
that convolution alone misses. That makes *which* recurrence a live question
rather than a detail. A GRU has ~3/4 of an LSTM's gate parameters and often
matches it on short sequences; after the conv stack downsamples 600 samples
to 75, this is a short sequence. If it matches `cnn-lstm`, it is the better
model at lower cost.

**Implementation.** `ConvSeqBranch` in `src/seismolib/model/blocks.py` already
isolates the recurrence behind a `use_lstm` flag (blocks.py:184). The change
is to generalise that to a `rnn_type` argument -- `nn.GRU` takes the same
constructor signature and the same `batch_first`/`bidirectional`/`dropout`
arguments as `nn.LSTM`, and differs only in returning `h_n` instead of
`(h_n, c_n)`, which this branch does not use. Then extend the `--branch-1d`
choices in `src/detection/cnn_lstm_classify.py:289` and the validation list in
`DualChannelTrunk`/`DualChannelNet`.

**Run it the same way or it is not comparable**: `--channels 1d`,
`--seq-transform asinh`, `--ensemble-seeds 42,43,44`, same dataset, fresh
`--save-dir`.

## [DONE] Gated fusion

**DONE 2026-08-19 (Ozgun: gated 0,9745 vs linear 0,9730; both below 2B's 0,9779).**

`--fusion gate` against `--fusion linear`, once the linear fusion arms land.
Deliberately excluded from the 2026-08-19 fusion run so the fusion mechanism
stayed constant while branch architecture varied.

## [DONE] Put `seq_transform` into `run_tag`

**DONE 2026-08-20 (commit 3baa2b8).**

`cnn_lstm_classify.py`'s checkpoint name encodes channels, fusion, branch_1d,
dataset, pid and seed -- but not `--seq-transform`. An asinh checkpoint and an
overflowed one are therefore indistinguishable by filename, and
`cascade_eval.py:stage1_scores` ensembles every `*.pth` in a directory. This
is the same class of defect as the checkpoint collision that produced the
retracted 0.9108 result.

## [DONE] Re-check the ORIGINAL benchmark for fp16 overflow

**DONE 2026-08-19 (scanned all three sets; both suspect rows reproduced -- no row affected).**

The overflow fixed in `f627d9e` was latent for as long as only `--channels 2d`
ran. Any `1d` or `all` number measured on the original benchmark before that
commit may have been produced with `inf` in the waveform input. Re-check before
citing those numbers anywhere.

## [DONE] STEAD cross-corpus evaluation

**DONE 2026-08-21 (EQTransformer bracket 0,9565-0,9989 around this project's 0,9971).**

Everything so far is single-corpus and station-disjoint. Section 5.5 (iv) of
the Turkish report marks cross-corpus generalisation as unmeasured.

## Cascade false-positive handling

Stage 2 consumes `aux = (log_snr, log_distance)`, and `log_distance` needs a
catalogued hypocentre that a false positive does not have. Options: a
`--channels 2d` stage 2, a waveform-derived distance estimate, or propagated
uncertainty. See `src/detection/cascade_eval.py`'s module docstring.
