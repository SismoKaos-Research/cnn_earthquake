# Handoff — chaotic features → forecasting

> **Historical, superseded 2026-08-31.** Written for a machine move that has
> since happened, and the work it hands off is finished: the chaotic-feature
> forecasting question was answered (negatively) in
> [`experiment_chaos_forecast_2026-08-27.md`](experiment_chaos_forecast_2026-08-27.md),
> and re-run on the corrected catalogue. Kept because its data-location and
> file-format tables are still accurate and still useful. For what is actually
> open, see [`TODO.md`](TODO.md).

Written 2026-08-21, for moving this work to the main PC (more VRAM, ~2× RAM,
faster CPU). Everything below is committed; nothing needed lives only in a
scratch directory.

---

## 1. What state things are in

### Repositories

| Repo | Branch | State |
|---|---|---|
| `model_cnn_lstm` | `main` | forecasting scripts, detector, TÜBİTAK report tooling |
| `sismokaos-cli` | **`testing`** | chaotic features live here, **not on `main`** |
| `Sismokaos/feature-extract` | `main` | Python sibling implementation |

`sismokaos-cli`'s chaos work is on `testing` (`4bcc3ca`, plus `9b48897` for the
speed pass). Merging to `main` is an open decision.

### Data

| What | Where | Size |
|---|---|---|
| Raw BODT miniSEED | `Sismokaos/feature-extract/raw/BODT/aegean_bodt_2024_2026` | 34 GB, 723 daily files, 2024-05-01 → 2026-08-10 |
| ~~KOERI catalogue~~ **AFAD catalogue** (superseded) | `catalogs/archive_superseded_2026-08-30/deprem_katalog_utc.csv` | 42 MB, 482,898 events, 2010 → 2026. **Archived 2026-08-30** — missing ~29% of AFAD's events for this region, incl. almost all of the Feb-2025 Aegean swarm. Use `catalogs/catalog_current.csv`. |
| Pre-chaos features (5 Hz, full archive) | `sismokaos-cli/dataset_features_5hz/` | 819 MB, 1,237,218 windows |
| Pre-chaos features (10 Hz, full archive) | `sismokaos-cli/dataset_features_10hz/` | 819 MB |
| **Chaos features, quarter batch** | `sismokaos-cli/dataset_features_chaos_q1_5hz/` | **does not exist yet** — run was killed for the move, see §2 |

The 34 GB raw archive is the only thing worth thinking about moving. The
feature parquets regenerate from it in hours; the raw data does not regenerate
without re-downloading from FDSN.

BODT station coordinates: **37.0622 N, 27.3103 E**.

---

## 2. The extraction run — killed, needs redoing on the new machine

A quarter batch (the first 181 of 723 days) was **stopped at 38% for the
machine move**. It produced nothing — see the warning below. Re-run it there:

```bash
cd ~/Projects/sismokaos-cli
./target/release/sismokaos-cli --config config_chaos_5hz.json \
    run --data-dir ./bodt_q1_chaos_5hz --out-dir ./dataset_features_chaos_q1_5hz
```

~312,768 windows at ~85 windows/s on 12 threads ≈ **1.1 h**; faster in
proportion to core count. Output is one parquet,
`bodt_q1_chaos_5hz_features.parquet`, 134 columns.

`bodt_q1_chaos_5hz/` is a symlink farm into the raw archive, gitignored, and
its links point at *this* machine's paths — **rebuild it on the new box**:

```bash
A=~/Projects/Sismokaos/feature-extract/raw/BODT/aegean_bodt_2024_2026
mkdir -p ~/Projects/sismokaos-cli/bodt_q1_chaos_5hz
ls $A | head -181 | while read f; do
    ln -s $A/$f ~/Projects/sismokaos-cli/bodt_q1_chaos_5hz/$f
done
```

**The output filename comes from the data-dir basename**, so two runs over
same-named directories silently overwrite each other.

### ⚠ The run is all-or-nothing, and it is a RAM hog

`FeatureWriter` (`src/export.rs`) is documented as "accumulates feature rows in
memory and writes to Parquet on finish". There are no incremental writes:

- **An interrupted run produces zero output.** 24 minutes and 119,640 windows
  of work left an empty directory. Do not assume a partial parquet is
  recoverable — there isn't one.
- **Peak RAM scales with row count.** Rows are held as `Option<f64>` (16 bytes
  each) across 66 features plus 66 `_DEV` columns, and Polars then builds the
  DataFrame from those buffers. The full 723-day archive is ~1.24 M rows ≈
  **2.7 GB of buffers, ~5–6 GB peak**. Fine on the new machine, was tight here.
- Its `Vec::with_capacity(100_000)` pre-allocation is sized by a comment
  reading "100,000 hours ≈ 11.4 years", but rows are *windows* at a 50 s step,
  not hours — so the real count is ~12× that and the buffers repeatedly
  reallocate. Harmless, but the reservation is wrong by design intent.

**Therefore: run the full archive in quarters, not in one go.** Four
181-day directories cap peak RAM around 1.5 GB and mean an interruption costs
at most a quarter of the work. Concatenating four parquets afterwards is
trivial; losing 4.4 hours to a crash is not. Making the writer flush
incrementally would be the real fix if full-archive runs become routine.

---

## 3. The decision that has been made

**Primary label: M≥2.5, within 400 km of BODT, 6 hour horizon.**

232 distinct events over the quarter batch, 25% positive rate, oriented
persistence floor 0.543, minimum detectable edge ±0.064 at 95%.

Chosen by `src/forecasting/label_sweep.py` (catalogue-only, runs in seconds),
which grids 140 (magnitude × radius × horizon) cells and reports distinct
events, positive rate and the oriented floor for each.

Why this cell and not a physically tighter one: **radius trades statistical
power against physical plausibility, and at 181 days you cannot have both.**
The physically preferable M≥3.0 / 100 km cell has 19 events and would need a
+0.229 edge to register. For scale, the best trustworthy result in this
project is the catalog MLP at **+0.017** over its floor.

Catalogue completeness is not a constraint here — modal in-region magnitude is
~1.3, so Mc sits well below 2.5.

**Declare secondaries before looking at them.** 24 of 140 cells pass the
viability filter; sweeping all of them and reporting the best would manufacture
a +0.10 out of noise.

---

## 4. Things that will bite you

### 4.1 The persistence floor was un-oriented in five scripts

Fixed in this handoff's commit. `floor = max(0.5, base_auc, pers_auc)` used the
**raw** persistence AUC. A persistence rule scoring below 0.5 ranks *inversely*
and is exactly as exploitable as a correct one, so the bar it sets is
`max(a, 1-a)`. Without the correction the floor collapsed to a vacuous 0.5
whenever persistence fell under chance.

This is bug #3 from the project ledger. It had been fixed in
`cnn_lstm_catalog_waveform_fusion.py` — with a comment describing exactly this
failure — and never propagated. It was still live in:

- `feature_lstm_forecast.py`
- `feature_gru_tcn.py`
- `raw_cnn_lstm_forecast.py`
- `catalog_lgbm_forecast.py`
- `waveform_lgbm_forecast.py` (two call sites)

**Consequence for an existing result:** `FEATURE_LSTM_CHEATSHEET.md` reported a
hand-feature LSTM at 0.558 against a persistence floor of 0.343, claiming
+0.058. Oriented, that floor is **0.657**, so the ensemble was **0.099 below**
it — the headline was an artifact, and the result was not evidence of
forecasting skill. **That cheat sheet has been deleted** (2026-08-22) rather
than rewritten, since the pipeline it documents — hourly hand features from
KO.GEDZ — is not what this phase does.

`LSTM_FEATURES_VS_RAW_CHEATSHEET.md` quotes the same un-oriented 0.343 floor
and still needs that one row corrected to 0.657. Its headline survives the
correction: the raw CNN-LSTM's 0.787 clears the oriented floor by +0.130,
and the hand-feature model's 0.558 does not clear it at all — which sharpens
its conclusion rather than undoing it.

### 4.2 Interpolated gaps are invisible and poison the chaos columns

`preprocess.rs` fills interior gaps by interpolation and leaves only *edge*
gaps as NaN. Nothing marks a window as reconstructed, so **you cannot filter
these out of the feature parquet.**

Measured on four heavily gapped days against two clean ones:

| | clean | gappy |
|---|---|---|
| windows with `Z_CORR_DIM < 3.0` | 0.43% | **2.87%** |
| NaN fraction | 0.00% | 0.25% |

A flat interpolated segment genuinely is low-dimensional, so the estimators are
correct and the *input* is fake: `CORR_DIM` collapses to 0.26–0.75 and
`SAMP_ENT` to ~0.002.

Fixing it means plumbing an interpolation mask from `interpolate_gaps` through
`ChannelChunk` into `compute_window` and emitting a per-window interpolated
fraction. **Worth doing before any full-archive run**, not after.

### 4.3 `Zaman_Dk` is the window END, not the start

`engine.rs`: `window_epoch = current_buffer_epoch + end/fs`, then `/60`. With
`win_sec 200` that is a systematic 200 s offset from window start. Matters the
moment labels get attached.

### 4.4 Effective sample size is distinct events, not rows

Consecutive 50 s windows under a multi-hour horizon carry near-identical
labels. 311,041 rows carry roughly 232 independent observations for the chosen
label. Per-seed AUC already swung 0.20 on a 1,070-hour test set. **Report
distinct events per fold beside every metric.**

### 4.5 The deepest risk is physical and not fixable by modelling

Median distance of M≥3.0 events from BODT is 249 km; there are **zero** M≥4.5
events within 50 km. If the waveform simply contains nothing about regional
events, no architecture recovers it. This is a real possibility, not a
pessimistic aside.

---

## 5. RESOLVED 2026-08-27 — see `experiment_chaos_forecast_2026-08-27.md`

Steps 0-4 below were all run. The extraction completed (312,626 windows), the
variance gate passed decisively (56.1% hourly, against the 1.2-9.3% band that
degenerated the earlier GRU), the univariate screen found marginal association
(best 0.5726 against a 0.5503 floor, just past a permutation null's 95th
percentile) -- **and none of it survived walk-forward evaluation.** Mean gain
+0.010 AUC, sign flipping across folds, t = 0.37 on 3 df.

**Step 5 is answered too: do not process the full 723 days.** The detectable
edge on this cell is +/-0.064 at 181 days and roughly +/-0.032 at 723; the
observed effect is +0.010. Four times the data would not make it detectable.

The steps below are kept as the record of what was planned and executed.

---

## 5b. Next steps, in order (as written 2026-08-21)

0. **Re-run the extraction** (§2). It was killed at 38% for the machine move
   and left nothing behind.

1. **Within-sequence variance check** — the gate on the whole idea. Run
   `src/forecasting/sequence_variance_check.py` against the new parquet.

   The ruled-out GRU/CNN degenerated because its catalog inputs had within-24 h
   std of only **1.2–9.3%** of overall std, making a 24-step sequence ~24
   identical vectors and the last hidden state an MLP on the last step. Land in
   that band and the verdict transfers regardless of the features being new;
   land at 40–80% and there is real structure for a sequence model.

   **Preliminary, from a two-day smoke test only:** chaos columns sat at
   **38–80% native, median 44.9%** — an order of magnitude clear of the
   degenerate band, Wolf LyE highest at ~79%. Hourly means held at 91–95% over
   24 h of context, but across two days that is only two sequences and means
   little. **Confirm on the full quarter batch before treating it as a
   result.**

   Note the aggregation trap the script measures directly:
   `feature_lstm_forecast.py` eats **hourly** vectors, but extraction emits 72
   windows per hour. Collapsing to an hourly mean may destroy exactly the
   variation the idea depends on.

2. **Re-run the existing hand-feature LSTM** against the now-oriented floor, so
   there is a correct baseline to compare chaos features against.

3. **Univariate screen** before training anything: per-chaos-feature AUC
   against the chosen label versus the oriented floor. Minutes of work. If no
   single feature carries marginal association, no sequence model will find
   one.

4. **Train**, with `log1p_dsp` included. Days-since-previous-event is what the
   persistence floor ranks by; withholding it asks the model to beat a bar
   built from a number it cannot see. It was the single largest lever in the
   GRU experiment (+0.069 / +0.047).

5. **If anything survives, process the full 723 days.** 4× the events halves
   the detectable edge, +0.064 → +0.032.

LSTM vs GRU is not worth a decision at this effective sample size — the GRU
degenerated for input reasons, not gating reasons.

---

## 6. What the better hardware actually buys

- **Feature extraction is CPU-only** and rayon-parallel over windows. It scales
  with cores, not VRAM. The full 723-day archive at 5 Hz is ~4.4 h on 12
  threads of a Ryzen 5 5600H; more cores cuts that proportionally.
- **10 Hz costs ~4× the chaos time**, because the chaotic estimators are O(N²)
  in window samples and 10 Hz doubles samples per window. With `freqmax 2.0`
  the extra samples carry no new band content, so 10 Hz is only worth it to
  test embedding-length sensitivity.
- **More RAM** helps the full-archive feature table (~1.1 GB parquet, but
  pandas will want several × that) and lets more of the 34 GB raw archive stay
  in page cache across repeated runs.
- **More VRAM does almost nothing here.** The models are tiny — the
  hand-feature LSTM is ~20K parameters. The bottleneck in this project has
  always been effective sample size, never GPU memory. The one exception is if
  you go back to raw-waveform CNNs at 100 Hz.

Further speed work on the Rust side, if wanted: `correlation_dimension` is
still 9.6 ms/call of the 19.2 ms total (see `examples/chaos_bench.rs` for
per-function timing). Avoiding the `sqrt` by comparing squared distances
against squared bin edges is worth ~25% of it, at the cost of exact bit-parity
with current output. `rosenstein_lye` computes divergence for all `m` steps but
the fit only uses the first ~100 at 5 Hz — truncating is a larger win, but
changes `nz` and so needs care to stay exact.

---

## 7. Verifying the move

```bash
# Rust side: chaos features present in the binary
cd ~/Projects/sismokaos-cli && git branch --show-current      # -> testing
cargo build --release
strings target/release/sismokaos-cli | grep -c Z_CORR_DIM     # -> >= 1
cargo run --release --example chaos_bench                     # per-function timing

# Label design reproduces
cd ~/Projects/Sismokaos/cnn_earthquake
python3 src/forecasting/label_sweep.py --catalog <path-to-catalogue>
# -> M>=2.5/400km/6h should show 232 events, pos_rate 0.253, floor 0.5426

# Floors are oriented everywhere
grep -c "pers_auc = max(pers_raw" src/forecasting/*.py       # -> 6 sites
```

**Unvalidated dependency:** `chaos.rs` says it was ported from
`sismokaos/chaos_algorithms.py` "so results stay comparable", but no test
asserts that. A pipeline-level comparison cannot settle it — obspy and the Rust
filter/decimate will not agree bit-for-bit. The only clean check is feeding
both implementations an identical synthetic series (fixed-seed random walk, or
a Lorenz trajectory) and comparing the five scalars.
