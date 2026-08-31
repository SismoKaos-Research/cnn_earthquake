# Catalog MLP Branch — Architecture, Data Pairing & Sources

**Status: second correction, 2026-08-14.** Two previous headline numbers have now been
struck as invalid rather than superseded — each was produced by code containing a bug that
was fixed afterwards. The current headline is **0.5886**, and unlike its predecessors it is
stable across seeds, folds and two independent time spans. §6 documents both retractions.

## 1. What this is

`model_cnn_lstm` (paired with data-pipeline repo `Sismokaos`) forecasts whether an M≥4.5
earthquake will occur in the Aegean region within a 14-day horizon, from hourly catalog
history, optionally paired with raw seismometer waveform (BODT/DAT, 5/10/100 Hz).

The leading architecture is the **catalog-only branch**: a small MLP over a per-hour,
hand-engineered catalog feature vector. It outperforms both the raw-waveform CNN and the
fused model in every clean comparison run to date.

## 2. Architecture

`CatalogMLPBranch`, part of `CatalogWaveformFusionNet`
(`src/cnn_lstm_catalog_waveform_fusion.py`), used standalone via `--channels catalog`.

- **Input:** one hour's catalog-feature vector — not a sequence. An LSTM was tried first and
  sat at chance, because catalog features barely change within a 24-hour window; a plain MLP
  matches the shape of the data.
- Small feedforward net, dropout, weighted BCE, AdamW, cosine LR, early stopping, multi-seed
  ensemble at inference.
- Catalog-only branch uses lighter regularization than the fused model (dropout 0.2 / lr
  3e-4 vs 0.4 / 1e-3).

## 3. Headline result — full-span catalog

The single largest improvement this project has had came from removing an **artificial data
limit**, not from any modelling change. The catalog branch never reads the waveform, yet it
had always been run over the seismometer archive's 2-year window, which holds only 34 M≥4.5
events. The catalog itself runs 2000–2026 and holds 261. `--catalog-span START END`
(added 2026-08-14) builds the hourly index from the catalog alone and feeds the model a
length-1 dummy waveform channel it never consumes.

| run | span | per-fold AUC | mean | fold std | floor | beats floor |
|---|---|---|---|---|---|---|
| `fullspan_2000` | 2000–2026 | 0.5872 / 0.5901 | **0.5886** | **0.0015** | 0.5714 | 1/2 |
| `span_2014` | 2014–2026 | 0.4878 / 0.6861 | 0.5870 | 0.0992 | 0.5423 | 1/2 |

Settings: `--channels catalog --horizon-days 14 --cv-folds 2 --bg-min-mag 3.0
--batch-size 128`, 5 shared random seeds, 4 features (§4), n=57,138 test rows/fold
(full span).

**What this says.** More data did *not* improve performance — it made the number
trustworthy, and the trustworthy number is mediocre. The model is worth roughly **+0.017
AUC over persistence**, beating its floor in 1 of 2 folds. It sits flat at ~0.589 while the
persistence floor swings between folds (0.5975 vs 0.5453), so which fold "wins" reflects how
strong persistence happens to be in that period more than anything the model is doing.

**Why it is nonetheless the first trustworthy result.** Fold-to-fold std is 0.0015, against
0.16 for variant B and a 0.69 → 0.26 inversion for the densification run. Per-seed spread
fell to 0.033 from 0.17. Five seeds, two folds and two spans converge on the same value.

**Completeness control passed.** Catalog detection capability varies enormously across the
record — M≥2.0 counts swing ~60× (194 in 2000, 11,387 in 2011, 1,457 in 2024) as the
monitoring network grew. Three of the four features derive from the M≥3.0 background set and
are exposed to that drift, so a model could have learned "the catalog looks dense → it is
2005–2012," a fact about data collection rather than seismology. `span_2014` restricts to
the stable-completeness era and agrees with the full span to within 0.002, ruling the
confound out.

## 4. Data pairing

- **Labels:** hourly; `1` iff an AEGEAN-region catalog M≥4.5 event occurs in
  `(hour, hour + 14 days]`.
- **Evaluation:** expanding-window walk-forward chronological CV with embargo.
- **Embargo:** must span both the input-window overlap and the full forward-looking label
  horizon — `seq_hours - 1 + horizon_days × 24` = 359 h at defaults, not the 24 h originally
  used (López de Prado Ch. 7). See §6.
- **Persistence floor:** `max(0.5, base_rate_auc, max(pers_auc, 1 - pers_auc))`. The
  orientation term matters — an anti-predictive rule is as exploitable as a predictive one.
  See §6.
- **Features** — best subset via LightGBM RFE (`src/catalog_feature_rfe.py`), 4 of 13:
  1. `log1p(days since previous M≥4.5 event)`
  2. 30-day mean background-catalog magnitude
  3. coefficient of variation of inter-event times (90-day window)
  4. Gutenberg-Richter magnitude deficit (90-day window)

## 5. Feature-engineering sources (citations)

- **Nurtas et al. 2025** — magnitude/energy/b-value/regularity feature family.

  > Nurtas, M., Altaibek, A., Ydyrys, A., Vilayev, A., & Nessipbay, T. (2025). Development of
  > a Long Short-Term Memory (LSTM)-Based Statistical Model for Earthquake Forecasting in
  > Central Asia. *IEEE Access*, 13, 162304–. https://doi.org/10.1109/ACCESS.2025.3610168

- **Aki 1965** — MLE estimator for the Gutenberg-Richter b-value.

  > Aki, K. (1965). Maximum likelihood estimate of b in the formula log n = a − bm and its
  > confidence limits. *Bull. Earthquake Res. Inst., Univ. Tokyo*, 43, 237–250.

- **López de Prado 2018** — purging/embargo for overlapping-label series; basis of the
  embargo correction.

  > López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 7. Wiley.

- **Woessner & Wiemer 2005** — maximum-curvature Mc estimator (+0.2), used to establish that
  the catalog's Mc is already ≈1.5.

  > Woessner, J., & Wiemer, S. (2005). Assessing the quality of earthquake catalogues.
  > *BSSA*, 95(2), 684–698. https://doi.org/10.1785/0120040007

## 6. Retractions and methodology fixes

Four bugs found, each invalidating results produced before it:

1. **Label leakage via short embargo** (fixed 2026-08-13). The embargo covered only the
   input-window overlap (24 h), not the 14-day forward label horizon. Adjacent train and test
   rows shared the same future events. **Invalidates AUC 0.6827.** Post-fix runs are
   identifiable by a smaller test n (3157 vs 3493).

2. **Normalization computed from the first 50 windows** (fixed 2026-08-13T17:12Z). Feature
   scale was estimated from `indices[:50]` — the archive's opening hours, where every
   trailing-window feature is still filling up. Measured on the rate features, sd was
   understated up to 52× (0.0157 vs 0.827), producing z-scores to 156 and saturating the
   GELU MLP. **Invalidates AUC 0.6670**, whose run finished at 16:30Z, 42 minutes before the
   fix landed — the process had already loaded the buggy module. Re-running the identical
   command with the fix gave **0.5582** on fold 1 against 0.6676 before: same data, same
   splits, same seeds, identical floors (0.4607), 0.11 AUC of the result was the bug.

3. **Persistence floor not orientation-corrected** (fixed 2026-08-14). Event mode used
   `max(0.5, base, pers)`, so whenever persistence fell below chance the floor silently
   collapsed to 0.5 — making a vacuous bar look like a real one. Rate mode already applied
   `max(auc, 1-auc)`. The n=4 event-mode result's "0.5000 floor in 2/2 folds" was this; its
   properly-oriented bar was 0.5801.

4. **Constant baseline scored on the wrong statistic** (fixed 2026-08-14, next-event
   regression). MAE is minimised by the median, not the mean; scoring a mean-predictor on MAE
   understated the baseline and inflated the model's apparent edge.

**Standing lesson:** editing a source file while a run using it is in flight produces results
attributable to neither version. Record the code state with each result.

## 7. Tried and rejected (honest ledger)

- **Nearest-neighbour distance (Zaliapin–Ben-Zion η) + spatial Shannon entropy** — the
  13-feature set scored worse than the existing 11.

  > Zaliapin, I., & Ben-Zion, Y. (2020). *JGR: Solid Earth*, 125, e2018JB017120.
  > https://doi.org/10.1029/2018JB017120

- **β-statistic precursor labeling** — made the persistence floor stronger (0.58 vs 0.50)
  without a corresponding model edge.

  > Convertito, V., Giampaolo, F., Amoroso, O., & Piccialli, F. (2024). *Scientific Reports*,
  > 14, 2964. https://doi.org/10.1038/s41598-024-52935-2

- **Hand-engineered DWT/spectral waveform features** (102, time/freq/time-freq) — RFE trace
  hovered at the 0.50 floor across nearly the entire 102→10 range.

  > Bhatia, M., Ahanger, T. A., & Manocha, A. (2023). *Engineering Applications of Artificial
  > Intelligence*, 120, 105856. https://doi.org/10.1016/j.engappai.2023.105856

  (Solves a different problem — discriminating a recorded waveform as earthquake-vs-not, not
  forecasting. Only its feature formulas were reused.)

- **Catalog densification to Mc 2.0** — fold 1 0.6894, fold 2 0.2623. Mc was separately
  measured at ≈1.5, so the phase-picking path this was meant to justify is unnecessary.

- **"Rate-change" relabeling (variant B)** — target reframed as "more seismically active next
  window than last." With trailing-rate features 0.7859/0.4626; without 0.5361/0.7043; floor
  0.7937. **0/2 folds in both arms** — the catalog MLP does not beat Omori decay at rate
  forecasting.

- **Raw 10 Hz CNN** — 0.3068, all seeds inverted.

- **Next-event regression** (`src/next_event_regression.py`) — predict days until the next
  M≥threshold event. In the well-powered arm (M≥3.0, 75–968 events/fold, no censoring) the
  model was **worse than a constant predictor in 3/3 folds**: −16.9% MAE with a log1p target,
  −46.9% raw, negative R² throughout. The M≥4.5 arm looked better (2/3) but its test folds
  held 1, 2 and 22 distinct events and one validation block was 100% censored with zero
  events, so early stopping had no signal.

- **Station split for the catalog branch** — *not possible, by construction.* Catalog features
  and labels are region-wide; BODT and DAT share 95.9% of their hours, so "train BODT / test
  DAT" tests on the very rows it trained on. The meaningful analogue is a spatial split of the
  catalog (`--region-split lat|lon`, added 2026-08-14), which splits space *and* time so a
  shared regional swarm cannot raise features on one side and labels on the other.

## 8. Caveats and standing constraints

- **Effective sample size is the recurring trap.** Hourly rows are not independent
  observations; the label is driven by a handful of distinct earthquakes per fold. Fold 1 of
  the 2-year runs had **one**. Always report distinct-event counts per fold beside any metric.
- **Never compare against a constant baseline.** Use a conditional floor — persistence/Omori
  for classification, `E[target | days_since_prev]` for regression. A constant baseline has
  produced false positives twice here.
- **Sensor–target mismatch is the leading explanation for every waveform failure.** Measured
  from BODT over the waveform window:

  | max distance | M≥3.0 | M≥4.0 | M≥4.5 |
  |---|---|---|---|
  | 50 km | 17 | 1 | **0** |
  | 100 km | 87 | 6 | 1 |
  | 400 km | 1,362 | 149 | 35 |

  Median distance of M≥3.0 events from BODT is **249 km**. The waveform branch has been asked
  to sense short-term precursors of earthquakes far outside its useful range — which is a
  physical explanation for its failures, not a modelling one. Note also that BODT and DAT are
  only 44 km apart, so "cross-station" tests transfer to a nearby sensor, not a new setting.
- **Fixed seeds hide run-to-run variance.** `--random-seeds N` (added 2026-08-14) draws fresh
  seeds and prints them as a paste-ready `--ensemble-seeds` value.
- The model is uncalibrated; temperature scaling needs validation-logit persistence, not yet
  implemented.

---
*`model_cnn_lstm` working sessions, 2026-08-13 / 2026-08-14.*
