# Tomorrow's plan (as of 2026-08-13 session, part 2)

## 1. Still open: fused vs catalog-only comparison (top priority, carried over)

Never actually resolved from the first pass. The catalog-only MLP's best
number remains the old 4-feature run's fold-1 ensemble AUC **0.5756** (first
clean floor-beat of the whole project). Nothing since has beaten it.

- Get fold 2's result for `--channels catalog` (4-feature version)
- Rerun the fused model (`--channels all`) with the fixed `CatalogMLPBranch`
  -- still never got a clean fused-vs-catalog-only comparison after the
  false-alarm collapse (val 0.83 -> test 0.33) earlier this session
- If fused doesn't beat catalog-alone, that's a clean, trustworthy answer
  either way to "does waveform add anything on top of a working catalog
  model"

Command reference:
```
python cnn_lstm_catalog_waveform_fusion.py \
  --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \
  --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \
  --consolidated --channels all --horizon-days 14 --cv-folds 2
```

## 2. Tonight's richer-feature + RFE experiment -- inconclusive, needs a clean rerun

- Built 11 catalog features (b-value, energy, magnitude deficit, inter-event
  regularity -- Panakkat-Adeli/Nurtas et al. 2025 style) into
  `build_catalog_features` in `cnn_lstm_catalog_waveform_fusion.py`. Verified
  correct via hand-checked synthetic data AND a real-data timing/NaN check --
  the math is right.
- Result was a step BACKWARD, not forward: all-11-feature catalog-only
  ensemble AUC **0.4916** (below floor), vs the old 4-feature run's 0.5756.
  High per-seed variance (0.4984 to 0.6679, spread 0.17) -- noisier, not just
  worse on average.
- Built `src/catalog_feature_rfe.py`: recursive feature elimination using
  LightGBM gain-importance, 3-fold walk-forward CV, ~10s to run. Findings:
  - `count_7d`, `count_30d`, `b_value_90d` have **zero** importance in every
    fold tried tonight -- safe to drop, don't reintroduce without a reason.
  - Best LightGBM subset by mean CV AUC: just `log1p_dsp` + `cv_interevent_90d`
    (0.5659, vs 0.5310 for all 11).
  - The AUC-vs-feature-count curve is NOT monotonic -- dips in the middle
    (6-8 features: 0.494-0.508), recovers at both extremes. Worth treating
    "n=2 is best" with some skepticism: it's a single-seed LightGBM ranking
    on 3 folds, small enough to be noise rather than a real optimum.
- Tried both the RFE-picked n=2 subset and an n=4 subset
  (`log1p_dsp`+`mean_mag_30d`+`cv_interevent_90d`+`mag_deficit_90d`, the last
  stop point in RFE's trace before the mid-range dip) in the actual MLP.
  First attempt (default hyperparams) collapsed hard: val AUC 0.83 -> 0.25,
  test AUC 0.4612. Added channel-aware lighter hyperparams for
  `--channels catalog` (dropout 0.2, lr 3e-4 vs 0.4/1e-3 for the fused
  model) and reran -- **did not clearly fix it**, both runs kept swinging
  hard in val AUC. Killed both before either produced a test AUC --
  genuinely unresolved, not a negative result, just incomplete. Rerun to
  completion tomorrow without interrupting.
- New lead worth trying: val LOSS looked much smoother epoch-to-epoch than
  val AUC in every one of tonight's logs. Plausible cause: val's positive
  rate is thin (0.096, ~335 positives out of 3493), so epoch-to-epoch val
  AUC is itself a noisy estimate on top of any real train(0.283)->val(0.096)
  regime shift. Try selecting the best checkpoint by val loss instead of val
  AUC and see if that's a more stable, better-behaved signal.

`--keep-features` flag now exists on `cnn_lstm_catalog_waveform_fusion.py`
-- pass any subset of `FEATURE_NAMES` (now defined once in that file,
imported by `catalog_lgbm_forecast.py` and `catalog_feature_rfe.py`, no
more duplicated lists across scripts).

## 3. GBM update

- Reran `catalog_lgbm_forecast.py` with the new 11 features (fixed a real
  gap -- it wasn't loading `bg_times`/`bg_mags` before, so it would've
  silently gotten near-default values for 7 of the 11 features). Fold 1 AUC
  improved to **0.5519** (up from the old best-tuned 4-feature result of
  0.4391) -- first time GBM actually beat the floor. Fold 2 still collapsed
  to exactly 0.5000 (`best_iteration=1`, gave up immediately) on a badly
  skewed test split (positive rate 0.640 vs train's 0.191).
- "MLP beats GBM on this catalog data" still holds directionally, but the
  gap narrowed with richer features. Worth rerunning GBM on the RFE n=2/n=4
  subsets specifically tomorrow too (cheap, seconds) to see if pruning helps
  GBM as much as it might help the MLP.

## 4. 10Hz preprocessing status

- DAT-10Hz preprocessing finished overnight (801/801 files, clean exit) --
  hourly `.npy` tree ready under `Sismokaos/feature-extract/data/aegean_dat_2024_2026_10hz/`,
  feature extraction intentionally skipped.
- Next: consolidate (`consolidate_hourly_raw.py --hour-samples 36000
  --delete-source`, data-root `aegean_dat_2024_2026_10hz`), rerun
  `raw_cnn_lstm_forecast.py` on 10Hz BODT to completion (was cut off at
  epoch 11/40 last time, no conclusive result yet), try 10Hz on the fusion
  model too.

## 5. Waiting on your team

- Declustering (NND method, Zaliapin & Ben-Zion) -- still the most
  theoretically direct fix for the swarm non-stationarity that's caused
  problems in nearly every experiment so far. Nothing to do here until you
  hear back.

## Deprioritize / don't repeat

- `count_7d`, `count_30d`, `b_value_90d` as catalog features -- confirmed
  zero LightGBM importance across every fold tried. Don't reintroduce
  without a specific reason.
- Chunk-based sampling (proximity classifier, week/month chunks, pre-event
  classifier, daily 3-class, hierarchical CNN-LSTM-LSTM) -- every variant
  hit the same sample-size wall. Not worth more time without a fundamentally
  bigger data source.
- Standalone waveform-only forecasting -- cross-station test showed it
  collapses on unseen stations (AUC 0.4736). Only worth pursuing as an
  auxiliary signal to the catalog branch, not standalone.
- ANFIS/fuzzy paper and the two finance CNN-LSTM papers -- confirmed not
  useful, solve different problems.

## Data/infra state to remember

- `FEATURE_NAMES` / `CATALOG_DIM=11` / `--keep-features` flag all live in
  `cnn_lstm_catalog_waveform_fusion.py`, shared by `catalog_lgbm_forecast.py`
  and `catalog_feature_rfe.py` -- one source of truth now.
- `catalog_feature_rfe.py` is new, standalone, ~10s to run -- cheap to rerun
  any time the candidate feature set changes.
- Channel-aware dropout/lr defaults exist for `--channels catalog` (0.2,
  3e-4) vs other channel settings (0.4, 1e-3) in the fusion script -- but
  per section 2, this alone didn't resolve the instability seen tonight, so
  don't assume the branch is "fixed" going in.
- Two runs killed tonight (n=2-lighttuned, n=4 subset) never reached a test
  AUC -- unresolved, not a negative result. Don't count them as failures,
  just finish them.
- BODT and DAT both have working 5Hz consolidated archives (3.5GB, 3.9GB)
  and 10Hz archives (BODT done, 7.4GB; DAT pending, see section 4).
- `truncate_to_reliable_catalog_end()` and the `--embargo` walk-forward-CV
  fix are both live everywhere that needs them.
