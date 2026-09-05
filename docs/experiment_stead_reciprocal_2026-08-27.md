# The reciprocal: our detector on STEAD

**Date:** 2026-08-27
**Scripts:** `src/detection/stead_to_ponly_mseed.py`, `experiments/reproduce/build_stead_ponly.sh`,
`experiments/reproduce/run_stead_reciprocal.sh`
**Tests:** `tests/test_stead_conversion.py`

## Why

`experiment_gpd_baseline_2026-08-27.md` ran GPD on our windows and found our
detector ahead by +0.081 AUC. That result is one-sided, and the obvious rebuttal
writes itself: a locally-trained model beating a foreign model on local data is
the expected outcome, not evidence. The claim only becomes symmetric once the
same model is measured on someone else's corpus, against a floor computed there.

This is that run.

## Building it without reimplementing anything

STEAD traces are 60 s at 100 Hz with a labelled `p_arrival_sample`, so the
project's own window geometry can be cut straight out of them: 340 samples
starting 200 before P, which is the same 2.0 s pre-arrival / 1.4 s post-arrival
cut `arrival_from_catalog.py` makes.

The windows are written back out as **miniSEED in the layout
`generate-spec-dual-dataset` already consumes**, and that command then builds
the dataset with the same flags used for the Aegean corpus — only `--eq-dir`,
`--noise-dir` and `--output-dir` differ. Filtering, per-(station, component)
baseline normalisation, hard-negative mining and STFT geometry are therefore
identical *by construction* rather than by careful copying. Re-deriving that
preprocessing in a separate script is exactly how a cross-corpus number quietly
stops measuring what it claims to.

### Component order was verified, not assumed

STEAD stores `(6000, 3)` as **E, N, Z**; this project's `_COMPONENT_ROLES` is
**Z, N, E**. Reversing them would feed the model horizontals where it expects
the vertical, producing a low score that looks exactly like a generalisation
failure and is not one.

Checked on 400 event traces: column 2 carries the largest P onset jump on 66% of
them, with a median post/pre amplitude ratio of **45.0** against 24.1 and 31.2
for columns 0 and 1. Column 2 is the vertical, as documented. Channel names are
assigned from that, and `tests/test_stead_conversion.py` pins the mapping.

### Only stations present in both STEAD chunks

The baseline normalisation needs long-term noise per station. STEAD's noise
chunk covers 1,155 stations and the event chunk 203, overlapping on **95**.
Events at stations with no noise record cannot be normalised the way the model
expects, so they are dropped — 92.6% of the event chunk survives that cut.

## The two corpora are closely matched on events, and not on noise

Selection was M ≥ 2.0 and ≤ 56 km, mirroring the Aegean corpus's own range.

| | Aegean P-only | STEAD subset |
|---|---|---|
| windows | 55,595 | 12,000 |
| stations | 183 | 81 |
| magnitude mean / median | 2.43 / 2.30 | 2.35 / 2.25 |
| magnitude max | 7.7 | 5.9 |
| distance mean / median (km) | 37.5 / 39.6 | 27.0 / 27.4 |

**The event populations are close, and STEAD's are nearer the station** — median
27 km against 40 km. Whatever happens below, it is not because STEAD's events
are harder in magnitude or distance terms; at equal magnitude they arrive with
*more* signal.

The noise is where the two corpora part company, and it matters:

> `-> test: amplitude-matched to 1535 events; **65.0% of events are louder than
> any candidate noise window and cannot be matched**`

On the Aegean corpus the same figure is **0.0 / 0.0 / 2.4%** across the three
splits. STEAD's noise traces are curated quiet noise; ours are windows taken
3 h before real events at the same stations, which is a far broader and louder
pool. So **the STEAD "matched" regime is not actually matched** — two thirds of
its events sit above anything the noise pool can offer, and its amplitude floor
stays correspondingly high. The natural regime is the meaningful comparison on
STEAD, and the matched column has to be read with that caveat attached.

Splits are station-disjoint on both builds (train/test station overlap: 0),
19 test stations, 1,535 event + 1,535 noise test windows.

## Result 1: each model wins at home

The 2×2 the GPD experiment was missing. Both models scored on the same rows of
each corpus, against a floor computed on those rows.

| | **Aegean matched** (14,821 rows) | **STEAD natural** (3,070 rows) |
|---|---|---|
| amplitude floor | 0.5860 | 0.7728 |
| GPD `original` | 0.7710 (44.7%) | 0.9693 (86.5%) |
| GPD `geofon` | 0.7987 (51.4%) | **0.9796 (91.0%)** |
| **this work (fusion)** | **0.8796 (70.9%)** | 0.9207 (65.1%) |

**We win on our corpus by +0.081; GPD wins on theirs by +0.059.** That is the
symmetric result, and it is the one to report. The one-sided version in
`experiment_gpd_baseline_2026-08-27.md` was a locally-trained model beating a
foreign one on local data, which is what anyone would predict; this says
something more useful, and less flattering.

**A caveat on STEAD's floor that makes our number worse, not better.** The
figures above use the raw-count amplitude scalar, so that both models share one
denominator. But the *station-sigma* scalar — the one the transfer matrix uses,
computed on the normalised tensors — reaches **0.9041** on the same split, and
this project's convention is to quote the higher of the candidate floors as the
honest bar. Against 0.9041, GPD `geofon` captures 78.7% and our fusion captures
**17.3%**. Either way GPD is ahead here; the stricter floor simply makes the gap
wider.

The deeper point is that **STEAD-natural is close to solvable by loudness
alone**. A floor of 0.90 leaves under a tenth of the range for a model to add
anything, which is a property of how the corpus is assembled — curated quiet
noise against events inside 56 km — not of either model.

## Result 2: the 1D branch transfers, the 2D branch does not

Aegean-trained arms, scored everywhere. Headroom captured, against the honest
floor per column:

| trained on | arm | aegean_matched | aegean_natural | stead_matched | stead_natural |
|---|---|---|---|---|---|
| natural | 1D only | 46.2% | 20.7% | **96.8%** | 86.4% |
| natural | 2D only | 33.9% | 23.1% | **−51.7%** | −254.8% |
| natural | fusion | 43.0% | 25.1% | 75.5% | 17.0% |
| matched | 1D only | 61.1% | 13.6% | **96.4%** | 75.1% |
| matched | 2D only | 57.9% | 16.2% | 34.5% | −294.2% |
| matched | fusion | 62.7% | 16.0% | 88.9% | −9.0% |

**The 1D branch does better on STEAD than at home** — 96.8% and 96.4% of
headroom on `stead_matched` against 46.2% and 61.1% in-domain, from both
training regimes. A branch that generalises *up* on a corpus it has never seen,
recorded by different instruments on another continent, is not fitting station
character. Combined with the within-amplitude-bin result in
`experiment_ponly_2026-08-22.md`, this is the strongest evidence the project has
that the 1D branch learned P-wave morphology.

**The 2D branch scores below a loudness scalar on STEAD.** Negative captured
means exactly that: 0.5358 AUC against a 0.6940 floor. There is a plausible
mechanism and it is not mysterious — the spectrogram channel is
station-normalised against spectral baselines computed from the *noise corpus*,
so a STEAD-built dataset hands it a normalisation it never trained under. That
makes it a statement about the 2D channel's normalisation scheme rather than
about spectrograms, and it is testable: rebuilding STEAD with `--normalize`
set to a corpus-independent mode would separate the two.

**Fusion is dragged down by it.** On `stead_matched` the fusion arm reaches
0.9250 where the 1D branch alone reaches 0.9901. The linear fusion has no way to
disown a branch that has gone out of distribution.

## What this licenses, and what it does not

**Licensed.**

- Neither model dominates. Each wins on its own corpus, by a comparable margin,
  against floors computed locally.
- The 1D branch's advantage survives a corpus change; the 2D branch's does not.
- STEAD, assembled this way, is a much easier detection task than the Aegean
  corpus: floors of 0.77–0.90 against 0.59–0.67.

**Not licensed.**

- **Nothing about STEAD as a whole.** This is 12,000 windows from chunk 2 at 81
  stations, restricted to M ≥ 2.0 within 56 km, with negatives drawn from 4,000
  noise records. A different subset would give different numbers.
- **No claim that the 2D branch is useless.** It contributes in-domain (33.9%
  and 57.9% captured on `aegean_matched`). What fails is its transfer, and the
  normalisation mechanism above is untested.
- **No pick-timing comparison.** As in the forward experiment, GPD is being run
  as a detector, which is not the job it was built for.

## Open

1. **Rebuild STEAD with a corpus-independent 2D normalisation** and re-score.
   That is the one experiment that would turn the 2D transfer failure from an
   observation into an explanation.
2. **The `stead_matched` regime is not actually matched** — 65% of its events
   are louder than any candidate noise window, against 0.0–2.4% on the Aegean
   builds. Deepening the noise pool beyond 4,000 records may not fix it if
   STEAD's noise is uniformly quieter than its events; that is worth measuring
   before rebuilding.
3. **1,535 event windows at 19 test stations** is a tenth of the Aegean test
   set. Confidence intervals have not been computed and the STEAD column
   deserves them before it goes in a paper.
