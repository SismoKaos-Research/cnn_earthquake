# Catalog Forecasting: Time to the Next Mainshock

**A technical report on the project's original objective**

> **Status.** The scalar baseline is complete and decisive. The dual-channel
> CNN+LSTM run (264 folds, ~85 min) is in progress; §4 is marked accordingly
> and will be filled from that run, not estimated.

---

## Abstract

This is the project's stated objective — forecasting event onset from catalog
data — as distinct from the waveform-classification work in `report.md`, which
was a detour. The task: given a sliding window of 64 consecutive catalog events,
classify how long until the next independent mainshock (M ≥ 4).

The result so far is negative and clean. A gradient-boosted model over the nine
standard seismicity indicators — b-value, Lyapunov exponent, energy release
rate, and six others — scores **31.49 % accuracy against a 33.33 % chance floor,
with kappa −0.028**, under leave-one-event-out cross-validation across 264
target events. It is not merely weak; it is at chance, with a confusion matrix
that is essentially uniform.

Getting to that statement required fixing two defects that would each have
produced a *published wrong number* rather than a crash: class labels wrong by
more than an order of magnitude, and a built-in baseline that is
anti-predictive by construction and made a chance-level model appear to win by
23 points.

---

## 1. The data

`data_downloader/seismic_cli/catalog.py` builds the dataset and is
methodologically careful in ways worth noting, since the negative result below
cannot be blamed on naive construction:

- **Gardner–Knopoff declustering** separates independent mainshocks from
  aftershocks *for target selection*, so one M6.2's aftershock sequence cannot
  masquerade as many independent targets. Dependent events remain in the window
  features, since that seismicity is real.
- **Windows containing an M ≥ 4 event are dropped outright** — they describe
  aftermath, not a precursor state, and would let the model read the answer off
  its own input. Verified: maximum in-window magnitude across all 8,393 windows
  is 3.90.
- **Label-aware embargo** in the chronological split drops exactly those windows
  whose target event falls beyond their own split's boundary.

**Pooled dataset:** 8,393 windows across 4 regions, **264 distinct target
events**, spanning 1999–2026. Per window: `seq (64, 6)` — magnitude, log Δt,
depth, log energy, cumulative energy fraction, distance from window centroid;
`img (3, 32, 32)` — RAM images of three of those series; `aux (9,)` — the
seismicity indicators.

**Verified before use** (all pass): zero windows whose target precedes their own
end time; `days_to_major` matches `target_time − end_time` to 0.000000 days;
264 folds, all non-empty.

## 2. Defect: the class labels were wrong by an order of magnitude

`RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]` was hardcoded in two files, but
`assign_risk_classes` derives **tercile** boundaries by default. On this
dataset the labels actually mean:

| Label says | Actually means |
|---|---|
| `lt_1y` | 0 – 25.8 days |
| `1_5y` | 25.8 – 71.3 days |
| `gt_5y` | 71.3 – 816.9 days |

Median time-to-next-mainshock is **46 days**; the maximum is 817 days. **This is
short-term forecasting on a scale of weeks, not multi-year recurrence.** A
results table reporting "`gt_5y` precision 0.81" would be read as a five-year
claim and would be wrong by more than an order of magnitude.

Nothing failed. The numbers were correct; only their names were wrong. Fixed by
deriving class names from the boundaries actually in force
(`lt_26d / 26d_71d / gt_71d`), and by having both trainers read class names and
their ordinal direction from the manifest's own `days_to_major` rather than a
hardcoded list.

## 3. Defect: the built-in baseline is anti-predictive

Both LOEO scripts compared the model against **per-fold majority-class
prediction**, which scores 8.53 % — far *below* the 33.33 % chance rate for
three balanced classes. That is not a hard baseline; it is a broken one.

The mechanism: classes are globally balanced (2798/2797/2798), and folds are
highly concentrated (mean purity 0.748 — a fold's largest class holds three
quarters of its windows). Removing one fold therefore tips the remaining counts
*away* from that fold's own dominant class, so the resulting "majority" is
systematically the class the fold has **least** of. Measured:

- train-mode matches the fold's true mode: **2 of 264 folds**
- train-mode matches the fold's *rarest* class: **175 of 264 folds**

Comparing against it turns a chance-level model into an apparent +23-point win.
Both scripts now print the chance floor first, flag the majority figure when it
falls below chance, and state plainly when a model fails to beat chance.

## 4. Results

**Leave-one-event-out, 264 folds, 8,393 held-out windows.**

| Model | Accuracy | Balanced acc. | Kappa |
|---|---|---|---|
| *floor:* chance (3 balanced classes) | *33.33 %* | *33.33 %* | *0.000* |
| Gradient boosting on 9 seismicity indicators | **31.49 %** | **31.49 %** | **−0.028** |
| Dual-channel CNN+LSTM (`seq` + `img` + `aux`) | *run in progress* | — | — |
| *(discredited floor:* per-fold majority — see §3*)* | *(8.53 %)* | *(8.53 %)* | — |

Pooled confusion matrix for the scalar model — essentially uniform, which is
what "no signal" looks like:

|  | pred lt_26d | pred 26–71d | pred gt_71d |
|---|---|---|---|
| **true lt_26d** | 974 | 894 | 930 |
| **true 26–71d** | 1031 | 798 | 968 |
| **true gt_71d** | 1015 | 912 | 871 |

Per-class F1 ranges 0.296–0.335 — no class is recovered better than chance.
Per-region mean accuracy: region1 31.6 %, region2 25.1 %, region3 37.6 %,
region4 31.5 % — no region carries signal the others lack.

**Two pooled numbers, because fold sizes are extremely uneven** (1 to 398
windows, median 14). Window-weighted pooling lets one large aftershock-rich
episode outweigh a small one ~400×; the per-fold mean (31.81 %, sd 25.55 %) is
event-weighted and is the one matching the LOEO design. Both agree here.

## 5. What this does and does not show

**Shows:** the nine seismicity indicators, as computed over 64-event sliding
windows, do not predict time-to-next-mainshock on this catalog under
event-disjoint evaluation. This is a direct, well-powered (264 events) negative
result on features that comparable published CNN-LSTM forecasting work builds
its entire model from.

**Does not show:** that catalog forecasting is impossible. Specifically not
tested: different window lengths or strides; a regression target
(`log days_to_major`) instead of terciles; fixed physically-meaningful
boundaries; spatial features beyond within-window centroid distance; or any
external covariate (GNSS strain, groundwater, electromagnetic).

**A caveat on LOEO itself**, from the script's own docstring: it trains on
windows from events occurring *after* the held-out event, so it measures whether
the representation generalizes across mainshocks — not whether a deployed model
could have predicted one in real time. It is a feature-quality check. The
chronological backtest, which is the honest deployment question, has not yet
been run; given a scalar model at chance under the *easier* evaluation, it is
unlikely to be more favourable.

## 6. Reproduction

```bash
seismic-cli generate-catalog-dataset \
    --catalog-path catalogs/deprem_katalog_utc.csv \
    --output-dir data/catalog_dataset_pooled \
    --window-events 64 --stride-events 8 --major-magnitude 4.0 \
    --split-mode loeo --region <lat0> <lat1> <lon0> <lon1>   # repeatable

cd ../cnn_earthquake/src
python catalog_scalar_loeo.py --dataset-dir ../../data_downloader/data/catalog_dataset_pooled
python cnn_lstm_loeo.py       --dataset-dir ../../data_downloader/data/catalog_dataset_pooled
```
