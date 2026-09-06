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
nearly the same statement. `src/sismokaos/detection/s_arrival_ablation.py` intervenes
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
| P-only matched fusion | **0.8762** | 0.6679 | +0.2083 | **62.7%** |

Per-seed: matched 1D 0.8673/0.8709/0.8671, 2D 0.8605/0.8610/0.8544, fusion
0.8730/0.8746/0.8737. Spreads 0.0038 / 0.0066 / 0.0016 — stable, not noise.
Fusion beats both arms, matching the 6 s hard-negative result: fusion helps
once the negatives are properly selected.

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

## Negative-regime transfer

Four negative regimes were built over the **same** P-only event windows —
verified identical: same 35 test stations, same 7,908 event windows — so
negative selection is the only variable. `src/sismokaos/detection/negative_regime_transfer.py`.

| regime | monotone floor | non-monotone | gap |
|---|---|---|---|
| matched (amplitude-mirrored) | 0.6679 | 0.6658 | −0.0021 |
| band 0.75–0.99 (loud only) | 0.6447 | 0.7461 | **+0.1015** ← artifact |
| wideband 0.0–0.99 (uniform over quantile) | 0.7927 | 0.7845 | −0.0082 |
| natural (no mining) | 0.7878 | 0.7795 | −0.0082 |

**`natural` and `wideband` are the same regime in practice** — floors within
0.005, AUCs within 0.0004. Spreading evenly across quantiles reproduces the
pool's own density closely enough that the distinction does not matter.

Models trained on **matched**, scored on all four:

| arm | matched | band | wideband | natural |
|---|---|---|---|---|
| **AUC** | | | | |
| 1D | 0.8709 | 0.9058 | 0.8169 | 0.8167 |
| 2D | 0.8602 | 0.8818 | 0.8225 | 0.8221 |
| fusion | **0.8763** | **0.9121** | 0.8225 | 0.8217 |
| **headroom captured** | | | | |
| 1D | 61.1% | 62.9% | **11.7%** | **13.6%** |
| 2D | 57.9% | 53.4% | **14.4%** | **16.2%** |
| fusion | 62.7% | 65.4% | **14.4%** | **16.0%** |

**On the deployment-realistic regime the detector barely beats loudness.**
Natural floor 0.7878, model 0.8167–0.8225: roughly +0.03 AUC over a single
amplitude scalar, against +0.20 on matched.

Two contributions to that, which should not be conflated:

1. **Train/test mismatch.** These models saw only amplitude-matched negatives
   and never the quiet noise that dominates natural. Part of the collapse is a
   generalisation gap, not a capability ceiling. The complementary run — train
   on natural, evaluate on all four — is what separates the two, and has not
   been done.
2. **Floor saturation.** Natural leaves 0.212 of headroom against matched's
   0.332, so some of the drop is arithmetic. This is why captured is quoted
   rather than raw gain.

False alarms barely move across regimes (301–377) and precision stays
0.93–0.96: changing the negatives changes their composition, not how often the
model fires.

**Recall cannot vary across these columns** and is deliberately absent from the
table. The positives are the identical 7,908 windows in every build, so recall
is fixed at ~0.638 by construction; only false alarms respond to the negative
regime. An earlier version of this table reported per-regime recall, which was
meaningless.

Fusion is best on matched and band. On natural, 2D edges it by 0.0004, which is
within seed noise.

## Does it read waveform shape, or loudness with extra steps?

### Why the floor comparison cannot answer this

Clearing a conditional amplitude floor does **not** establish that a model uses
anything but amplitude. The floor is the ROC-AUC of a single scalar, and
ROC-AUC measures only how well that scalar *ranks*. A model that learned a
better-shaped function of the same scalar — a threshold in the right place, or
a non-monotone response — would clear the floor while remaining, in substance,
an amplitude detector.

That is not hypothetical. On the band-mined P-only set the gap between the
monotone floor (0.6447) and a depth-4 tree on that *same single scalar*
(0.7461) was 0.10 AUC. "A better-shaped function of the same number" is worth
about that much here, and 0.10 is a large fraction of the model's apparent
margin. So the question needs a different instrument.

### The test

Hold amplitude nearly constant and see whether discrimination survives. Inside
a narrow amplitude bin the scalar carries almost no information by
construction, so any AUC meaningfully above 0.5 must come from something else —
and the only other thing in the window is the shape of the waveform.

Two design points matter:

1. **Bin width, not bin count.** Amplitude is heavy-tailed, so equal-count
   deciles are wildly unequal in amplitude *range*. On this corpus the top
   decile spans ~530× and the bottom decile ~500×, while the middle ones span
   1.3–2.4×. A high AUC in a 530×-wide bin proves nothing — amplitude is still
   free to vary enormously inside it. `within_amplitude_auc.py` prints the
   width ratio beside every bin and only counts bins ≤ 2.5× as evidence.
   *(An earlier ad-hoc version of this analysis did not, and its reading of the
   bottom decile as "near chance, therefore SNR-gated" was not supportable —
   that bin is too wide to license any conclusion.)*
2. **Class balance varies across bins** (0.23–0.99 here). That is harmless:
   AUC is invariant to class balance, which is exactly why it is the right
   statistic and why accuracy would not be.

### Result

`src/sismokaos/detection/within_amplitude_auc.py`

**matched test set, matched-trained fusion** — pooled 0.8763, floor 0.6679:

| bin | n | P(event) | amplitude range | width | AUC within | evidence? |
|---|---|---|---|---|---|---|
| 1 | 1582 | 0.40 | 0.00 – 0.98 | 508.7× | 0.5664 | no — too wide |
| 2 | 1581 | 0.38 | 0.98 – 1.51 | 1.5× | **0.6298** | yes |
| 3 | 1582 | 0.39 | 1.51 – 2.15 | 1.4× | **0.7090** | yes |
| 4 | 1581 | 0.35 | 2.15 – 3.01 | 1.4× | **0.7781** | yes |
| 5 | 1582 | 0.41 | 3.01 – 4.34 | 1.4× | **0.8013** | yes |
| 6 | 1581 | 0.42 | 4.34 – 6.43 | 1.5× | **0.8578** | yes |
| 7 | 1582 | 0.45 | 6.43 – 10.72 | 1.7× | **0.9274** | yes |
| 8 | 1581 | 0.53 | 10.72 – 21.85 | 2.0× | **0.9689** | yes |
| 9 | 1582 | 0.70 | 21.85 – 65.78 | 3.0× | 0.9902 | no — too wide |
| 10 | 1582 | 0.96 | 65.78 – 34794 | 528.9× | 0.9793 | no — too wide |

**Median across the 7 narrow bins: 0.8013.**

**natural test set, natural-trained fusion** — pooled 0.8410, floor 0.7878:
narrow-bin AUCs 0.5604 / 0.6423 / 0.6792 / 0.7167 / 0.7492 / 0.7922 / 0.9128,
**median 0.7167**.

### What this licenses, and what it does not

**Licensed.** In bins where amplitude varies by at most 1.4–2.4×, the model
reaches AUC 0.63–0.97. No function of loudness can produce that. **The detector
reads waveform morphology.** This is demonstrated rather than asserted, and it
is the strongest positive result in the P-only work.

**Also licensed: the ability scales with signal strength.** Within-bin AUC
rises monotonically with amplitude across every narrow bin — 0.63 at bin 2 to
0.97 at bin 8 on matched, 0.56 to 0.91 on natural. Morphology becomes readable
as the P wave rises out of the noise. This is an independent confirmation of
the operating envelope, which found recall rising 0.469 → 0.945 across SNR
bands from an entirely separate calculation.

**Not licensed.** Nothing about the quietest ~10% of windows: bin 1 is 400–500×
wide, so neither its low AUC nor a high one would mean anything. The claim
"near-chance at the lowest amplitudes, therefore SNR-gated" is *suggested* by
the trend across bins 2–8 but is not established by bin 1.

**Not licensed either:** that the natural-trained model reads shape better.
Median narrow-bin AUC on natural is 0.7167 (natural-trained) against 0.7175 for
the matched-trained model on the same test set — indistinguishable. The
natural-trained model's real advantage is elsewhere: **157 false alarms against
317 at equal recall (~0.633)**. It is not better at recognising P; it is better
at not crying wolf.

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
2. **[DONE 2026-08-27] Report §3.2 discrepancy.** It stated `n_fft = 256,
   hop = 64`, `img (3, 129, 10)` as the method, while the headline results came
   from `dataset_specdual_catalog_6s_matched_hard`, which is `(3, 33, 38)` —
   n_fft 64, hop 16. Fixed in the merged rewrite: the "İki STFT geometrisi"
   paragraph now sits inside §3.2, where a reader meets it before the results,
   instead of only in the window-length section. The superseded
   `tubitak_rapor_bolum_2_5.md` still has it.
3. **[DONE 2026-08-27] `cascade_eval.py`** did `glob("*.pth")` with no
   filtering. Now `find_checkpoints` selects on `_{channels}_{fusion}_` (plus
   an optional `--detector-branch-1d`), then groups the survivors by run
   identity — the tag minus `_pid…_seed…` — and requires exactly one, raising
   with the candidates listed otherwise.

   The anchored regex the other scripts use could not simply be copied: the
   run tag has grown over time, and the 6 s checkpoints this script's own usage
   example points at predate both `--branch-1d` and `--seq-transform`, so they
   are named `2d_linear_dataset_…` and a fully anchored pattern rejects them.
   Matching loosely and then checking the result accepts the old names while
   still turning ambiguity into an error. Verified: the 6 s dir resolves to its
   3 checkpoints, `trained_model_ponly_natural/` narrows from 9 to the 3 `2d`
   ones, and a directory seeded with `cnn` / `cnn-lstm` / `none` / `asinh`
   variants refuses until narrowed.
4. **Wideband diagnostic** (`dataset_specdual_ponly_3p4s_wideband`, band
   0.0–0.99) is built but untrained. The matched result has largely superseded
   the question it was meant to answer.

### Regression tests added 2026-08-27

`tests/` now pins the failure modes this experiment ran into, all of which
produced a plausible number rather than an error. `pytest` runs in ~4 s on
synthetic arrays — no GPU, no dataset, safe to run mid-experiment.

- `test_checkpoints.py` — arm selection, including the `cnn` / `cnn-lstm`
  prefix collision and the pre-`--branch-1d` tag layout.
- `test_metrics.py` — `safe_auc(oriented=True)`, the correction five
  forecasting scripts were missing.
- `test_catalog.py` — the `(t, t+w]` / `(t-w, t]` boundary that keeps a forward
  label from reading its own input.
- `test_amplitude_bins.py` — the bin-width gate that stopped the bottom-decile
  misreading recorded above.
- `test_label_sweep.py`, `test_splits.py`, `test_imports.py`.

Three latent defects surfaced while writing them, each fixed in the same pass:

1. `sismokaos/catalog.py` used `re.match` without importing `re`, so
   `parse_hour_start` raised `NameError` on every call. Its last caller had
   moved to `Zaman_Dk`, so nothing noticed.
2. `label_sweep.sweep_cell` raised `IndexError` on a cell with zero qualifying
   events. Invisible on the full 2010–2026 catalogue, where every cell has one;
   a narrower `--start/--end` would have hit it.
3. `label_sweep.py` was a flat top-to-bottom script, so importing it parsed
   argv and ran the whole 140-cell sweep. It now has `main()` and a guard, and
   `test_imports.py` keeps the next one from appearing.
