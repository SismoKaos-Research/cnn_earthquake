# Waveform-branch architecture comparison

**Date:** 2026-08-19 · **Dataset:** `dataset_specdual_catalog_6s_matched_hard`
· **Code:** `f627d9e` (main)

## Question

The 1D branch of the dual-channel detector fed 600 raw 100 Hz samples straight
into a BiLSTM. It had **no convolutional front end at all** — the only local
structure available to it was whatever recurrence could accumulate one 10 ms
sample at a time. The detectors this project compares itself against do the
opposite: EQTransformer is CNN → BiLSTM → attention, PhaseNet is a U-Net of 1D
convolutions. Both extract local waveform features convolutionally *before* any
recurrence.

So: does a conv front end help, and is the recurrence needed once it exists?

## Design

Three arms, `--channels 1d` so the spectrogram branch is **out of the graph
entirely** and the only variable is the 1D architecture:

| arm | structure |
|---|---|
| `lstm` | BiLSTM + attention over 600 raw samples (the incumbent) |
| `cnn-lstm` | 3-stage 1D conv (600→300→150→75) → BiLSTM + attention |
| `cnn` | same conv stack, mean-pooled, **no recurrence** |

All three: `--seq-transform asinh`, `--fusion linear`, `--ensemble-seeds
42,43,44`, 80 epoch cap, patience 10. Splits are station-disjoint —
76,494 train / 18,830 val / 15,812 test, balanced 50/50.

**asinh on all three arms, not just the two that needed it.** `cnn` and
`cnn-lstm` crashed with non-finite loss on the first attempt: 7 windows of
95,324 exceed fp16's 65,504 ceiling (max 3.57e5 in station-sigma units) and
AMP casts the *input* to `inf` before any layer runs. The fix is a signed log
compression, which is monotonic — amplitude ordering, and therefore floor
comparability, is preserved (verified Spearman ρ = 1.000000). Applying it to
only the failing arms would have made the transform a second variable and
confounded the comparison, so the incumbent was re-run under it too.

## Floors

Reported on the test set, all single scalars with no learning:

| floor | AUC |
|---|---|
| majority class | 0.5000 |
| `img` mean dB | 0.7330 |
| `seq` std | 0.8678 |
| **`seq` abs-max** | **0.9049** ← strongest |

The majority-class floor is vacuous here. **0.9049 is the number every result
below is measured against**; "headroom" is the fraction of the gap between it
and 1.0 that a model captures.

## Results

3-seed probability-averaged ensembles, test set:

| arm | per-seed AUC | mean ± std | ensemble | headroom | params |
|---|---|---|---|---|---|
| **cnn-lstm** | 0.9891 0.9882 0.9892 | 0.9888 ± 0.0005 | **0.9896** | **89.1%** | 142,059 |
| lstm | 0.9869 0.9876 0.9874 | 0.9873 ± 0.0003 | 0.9883 | 87.7% | 76,707 |
| cnn | 0.9818 0.9826 0.9836 | 0.9827 ± 0.0007 | 0.9843 | 83.5% | 48,555 |
| *2d reference* | *0.9876 0.9874 0.9864* | *0.9871 ± 0.0005* | *0.9882* | *87.6%* | *115,459* |

The 2d row is the existing spectrogram-only detector, included because it is
directly comparable: same seeds, same split sizes, and bit-identical floors
(0.9049 / 0.8678 / 0.7330), which fingerprints the same test set.

Error composition at threshold 0.5 (out of 7,906 events and 7,906 noise):

| arm | missed events | false alarms | precision | recall |
|---|---|---|---|---|
| cnn-lstm | **410** | 141 | 0.9815 | 0.9481 |
| lstm | 546 | 144 | 0.9808 | 0.9309 |
| cnn | 664 | 145 | 0.9804 | 0.9160 |
| *2d* | *489* | *90* | *0.9880* | *0.9381* |

## Findings

**1. Recurrence is the load-bearing component; convolution is a front end for
it, not a replacement.** `cnn` is the clear loser at 0.9843, roughly 0.005
below both recurrent arms — about 10× the seed noise, with per-seed ranges that
do not come close to overlapping. Removing the LSTM saves 66% of parameters and
costs 5.6 points of headroom.

The mechanism is visible in the error split. All three arms sit at essentially
identical precision (0.9804–0.9815), so the entire difference is **recall**:
`cnn` misses 664 events, `cnn-lstm` misses 410. Convolution alone finds the
loud, easy events; recurrence is what recovers the marginal ones.

**2. The conv front end helps, but modestly — and this is the weakest claim
here.** `cnn-lstm` (0.9888 ± 0.0005) over `lstm` (0.9873 ± 0.0003) is a gap of
0.0015. The per-seed ranges do not overlap (0.9882–0.9892 vs 0.9869–0.9876),
which is the strongest form of the claim available, but they are separated by
only 0.0006 and n=3 per arm. **Treat this as suggestive, not established.** It
would not survive a demand for a confidence interval. What is solid is the
direction and that it costs 65k parameters.

**3. A waveform-only model now matches the spectrogram model.** `cnn-lstm` at
0.9896 vs the 2d detector's 0.9882 — a difference of 0.0014 against seed stds
of 0.0005, marginal at n=3. **The defensible claim is equivalence, not
superiority.** The change worth noting is that the waveform branch used to be
the weaker of the two, and the cause now looks architectural rather than
informational: the signal was reachable, the old branch just could not reach it.

**4. They fail differently, and that is the interesting part.** `cnn-lstm`
trades precision for recall against the 2d model — 79 fewer missed events for
51 more false alarms. For a detector feeding the magnitude cascade this is the
better trade: a missed event can never receive a magnitude, while a false alarm
is merely expensive. More importantly, two comparably strong models making
*different* mistakes is the precise condition under which fusing them should
beat either alone.

**5. Convergence tracks the transform, not the architecture.** All arms
early-stopped in 22–45 epochs under asinh, including plain `lstm`. An earlier
`lstm` run was still improving at epoch 45. This is incidental evidence that
the earlier run was not using asinh.

## Operating envelope (added after the grid, `cnn-lstm` ensemble)

Recall stratified by source parameters, threshold 0.5. Produced by
`src/detection/operating_envelope.py`, which re-scores the test set from the
saved checkpoints and joins to the magnitude manifest by filename — a 7,906 /
7,906 join. It reproduces the ensemble ROC-AUC of 0.9896 exactly, which is the
check that the scoring path matches training.

Overall recall **0.9484** (7,498 / 7,906; 408 missed).

| magnitude | n | recall |
|---|---|---|
| 1.5–2.0 | 1,611 | 0.9100 |
| 2.0–2.5 | 4,152 | 0.9458 |
| 2.5–3.0 | 1,445 | 0.9792 |
| 3.0–3.5 | 498 | 0.9900 |
| >3.5 | 200 | 0.9850 |

| log SNR | n | recall |
|---|---|---|
| < −2.0 | 35 | 0.8857 |
| −2.0 – 0.72 | 2,630 | 0.8696 |
| 0.72 – 3.42 | 4,261 | 0.9859 |
| > 3.42 | 976 | **1.0000** |

| distance (km) | n | recall |
|---|---|---|
| 0–25 | 1,588 | 0.9773 |
| 25–50 | 5,044 | 0.9417 |
| 50–100 | 1,274 | 0.9388 |

**The envelope is an SNR envelope, not a magnitude envelope.** Comparing the
408 misses against the 7,498 detections, the medians separate on SNR and
barely on anything else:

| | missed | detected |
|---|---|---|
| log SNR | **−0.006** | **1.431** |
| magnitude | 2.100 | 2.300 |
| distance (km) | 43.06 | 38.47 |

Magnitude and distance differ by 0.2 units and 4.6 km respectively; log SNR
differs by 1.44. Recall is a clean monotonic function of SNR — 0.87 below
log SNR 0.72, 0.986 above it, and **1.0000 above 3.42** (976 events, no
misses). The apparent magnitude trend is largely SNR in disguise, since larger
events are louder at the same station.

This is consistent with the amplitude floor being 0.9049: the detector's
operational limit is set by how far above the local noise the signal sits, not
by earthquake size as such. It also means the honest deployment statement is
conditioned on SNR rather than magnitude:

> ~87% recall for events near the station noise level (log SNR < 0.72), rising
> to ~99% once log SNR exceeds 0.72 and 100% above 3.42.

**Caveats.** Distance coverage stops at 100 km, so the weak distance effect is
a statement about a local network, not about attenuation generally. The count
of misses here is 408 against the training log's 410 — a two-window
threshold-boundary convention difference, not a scoring discrepancy (the AUC
matches to four decimals). And this is the balanced benchmark: recall
transfers to continuous data, precision does not.

## Scope and limits

- This establishes **which 1D architecture is best**. It does **not** establish
  whether the waveform branch contributes anything beyond amplitude — that is
  the fusion question, running now.
- One dataset, one window length (6s), one corpus, station-disjoint. No
  cross-corpus (STEAD) check.
- All numbers are balanced-benchmark AUC. Under continuous-data base rates the
  1.78% false-positive rate implies ~1,540 false alarms/station/day. See
  `docs/TODO.md`.
- n=3 seeds. Adequate to separate `cnn` from the rest; thin for the two
  recurrent arms.

## Follow-up running

`--channels all` fusion, two arms (`cnn-lstm` and `lstm` 1D branches, linear
fusion, same seeds and transform). Two arms because with only the first, a good
score cannot be distinguished from re-measuring the stronger branch with extra
parameters attached; the `lstm` arm is the like-for-like against the old fusion
numbers under a correct asinh path.

## Reproduce

```
python3 src/detection/cnn_lstm_classify.py \
  --dataset-dir .../dataset_specdual_catalog_6s_matched_hard \
  --save-dir trained_model_branch1d_asinh \
  --channels 1d --branch-1d {lstm,cnn-lstm,cnn} \
  --seq-transform asinh --fusion linear \
  --ensemble-seeds 42,43,44 --num-workers 2
```

Logs: `logs/branch1d_asinh_{lstm,cnn-lstm,cnn}.log` (each records its own
command line). Checkpoints: `trained_model_branch1d_asinh/`. Pre-fix
checkpoints are quarantined in `trained_model_branch1d_stale_prefix/` — they
were produced before the asinh fix and `seq_transform` is not encoded in the
checkpoint filename, so they are not distinguishable from valid ones by name.
