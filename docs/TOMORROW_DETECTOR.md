# Where we left off — 2026-08-20, 00:25

Detector work. `docs/TOMORROW.md` is the *forecasting* family's plan and is
unrelated. Deferred ideas live in `docs/IDEAS.md`.

## Everything is stopped

GPU idle (13 MiB), no training processes, all chain drivers gone, monitors
stopped. Nothing is running.

## One decision waiting for you

**The normalised-dataset result overturns the strong form of §4.2 finding (i)
in the TÜBİTAK report.** I drafted nothing into the report for this — it
changes a stated conclusion in a document going to reviewers, so it is yours
to approve.

The claim is that the 1D branch "only re-learns the scalar it was denied."
On `dataset_specdual_6s` (per-window normalised) the `seq` std floor is
**exactly 0.5000** — absolute amplitude is gone, there is no scalar left to
re-learn — and yet:

| arm | ensemble | vs seq floor 0,7088 | vs strongest floor 0,9205 |
|---|---|---|---|
| `cnn-lstm` | **0,9309** | +0,2221 | **+0,0104** |
| `cnn` | 0,9146 | +0,2058 | −0,0059 |
| `lstm` (report's row) | ~0,9144 | +0,2056 | −0,0061 |

`cnn-lstm` clears **both** floors. Plain recurrence and plain convolution each
sit at ~0,914; the combination reaches 0,9309. Same data, same channel, same
seeds — the difference is architecture (+0,0165). The original result measured
the branch's **reach**, not the data's information content.

A second, separable point: Çizelge 7 judges a waveform-only model against
`img` mean dB (0,9205), a **spectrogram** scalar the model never sees. Both
floors are defensible but answer different questions — "is this branch
deployable alone?" (strongest floor) vs "does it learn more than amplitude?"
(its own channel's floor, 0,7088). Finding (i) asks the second and uses the
first. Suggested fix: print both floors in that row and split the claim.

## Killed mid-run — resume or discard

`logs/norm_1d_lstm.log` — the reproduction control. Seed 42 finished at
**0,9139**, matching the report's 0,9144, so the row *does* reproduce and the
finding above is not a measurement artefact. Seeds 43–44 were killed. The
partial result is sufficient to state reproduction; rerun only if you want a
3-seed ensemble in the table.

`logs/norm_all_cnn-lstm.log` — **never started**. Fusion on the normalised
dataset. Would answer whether fusion helps when amplitude is deleted.

To resume both, rerun the chain script (it is idempotent; use a fresh
`--save-dir` or delete `trained_model_norm_branch1d/` first, since
`seq_transform` is still absent from `run_tag`):

    /tmp/claude-1000/.../scratchpad/run_norm.sh

Scratchpad scripts do not survive a reboot — they were cleared once already
today. Copy anything you want to keep into `scripts/`.

## Completed today (all final, all under --seq-transform asinh)

**Branch-1d grid**, `dataset_specdual_catalog_6s_matched_hard`, floor 0,9049:
`cnn-lstm` 0,9896 · `lstm` 0,9883 · `cnn` 0,9843 · (2B reference 0,9882).
Recurrence is load-bearing; the whole `cnn-lstm`/`cnn` gap is recall
(410 vs 664 misses) at flat precision.

**Fusion**, same dataset: `all/cnn-lstm` **0,9908**, `all/lstm` **0,9907** —
means identical at 0,9901. Best results on the benchmark. Fusion beats *both*
branches on false alarms (64–75 vs 141 and 90). The conv front end's 0,0015
gain vanishes under fusion, so use plain `lstm` in the dual-channel model:
65k fewer parameters, no loss, fewer false alarms.

**Operating envelope**: recall is governed by **SNR, not magnitude**. Missed
events median log SNR −0,006 vs detected 1,431; magnitude differs by only 0,2.
Recall 1,0000 above log SNR 3,42 (976 events, zero misses).

**Özgün re-check** (fp16 suspicion): all rows reproduced inside their original
seed spreads — `1d` 0,9443 · gated fusion 0,9745 · linear fusion 0,9730 (new).
**Limitation 7 is closed**, not open. Fusion underperforms the 2B branch
(0,9779) on this dataset with *either* mechanism, so it is the dataset, not
the gate. My "the gate is the problem" hypothesis was wrong.

**The cross-dataset finding worth writing up**: linear fusion wins on hard
negatives (0,9908, beating both branches) and loses on Özgün (0,9730 vs
0,9779). Same architecture, seeds, code — what flips the sign is how the
negatives were constructed.

## Report state

`~/Desktop/tubitak_rapor_bolum_2_5.{md,docx}`, 835 lines, regenerated and
current. Added §3.9 (fp16/asinh), §4.6 (1D architecture), §4.7 (operating
envelope), §4.8 (fusion); rewrote §5.2; fixed Çizelge 7 including a katkı sign
error (+0,0003 → −0,0018, computed against the wrong floor); `ve diğerleri` →
`et al.`; metric names in English (Recall/Precision/Accuracy).

Not yet in the report: the normalised-dataset results above (pending your
decision), and Çizelge 7's three † rows remain un-re-measured by design.

## Known loose ends

- `seq_transform` is still not in `run_tag`, so asinh and non-asinh
  checkpoints are indistinguishable by filename. `cascade_eval.py` globs
  `*.pth` across a directory. Pre-fix checkpoints are quarantined in
  `trained_model_branch1d_stale_prefix/`.
- The 3s dataset overflows fp16 (max 1,21e6). Its published results are
  2B-only so they are unaffected, but any future 1d/all run there needs asinh.
- STEAD is on disk (`raw/data/dataset_stead_matched_6s`, 27.378 rows) —
  the cross-corpus check listed as unmeasured is actually runnable.
- Çizelge 7's 2B hard-negative figure was 0,9892 in the report but no log
  produces it; replaced with the measured 0,9882.
