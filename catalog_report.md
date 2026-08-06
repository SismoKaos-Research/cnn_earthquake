# Catalog Forecasting

**A technical report on the project's original objective**

---

## Abstract

This is the project's stated objective — forecasting earthquake occurrence from
catalog data — as distinct from the waveform work in `report.md`, which was a
detour.

The result is a **narrow, defensible forecast and a well-explained failure**.
Reformulated as "will a M ≥ 4.5 event occur in this fault zone within the next
30 days?", the forecaster reaches **test AUC 0.78 in the Aegean extensional
province against a 0.64 persistence floor** — a genuine forecast. In the two
near-Poisson zones it does not work at all, scoring **below chance even with
zone-specific models**.

The reason is physical rather than architectural, and it is the same reason
throughout. **Forecastability tracks clustering.** Across four zones, the
predictive value of recency correlates almost monotonically with the
coefficient of variation of inter-event gaps, and where CV ≈ 1 — exactly
memoryless — no model of this kind can work.

That argument also explains the phase this replaced. The original target
(time to the next independent mainshock, in terciles) measured **at chance**:
31.49 % accuracy against a 33.33 % floor, kappa −0.028, over 264 target events.
Gardner–Knopoff declustering removes aftershocks — the *most predictable* part
of seismicity — leaving mainshock timing that is close to Poisson. The target
was near-unlearnable by construction.

Two defects were fixed along the way, each of which would have published a
wrong number rather than crashed.

---

## 1. The abandoned formulation, and why it failed

`catalog.py` labels each 64-event sliding window with the time until the next
*independent* mainshock, binned into terciles, and evaluates by
leave-one-event-out CV over 264 target events.

**Result: at chance.** Gradient boosting over the nine seismicity indicators
(b-value, Lyapunov exponent, energy release rate, …) scored 31.49 % accuracy
against a 33.33 % floor with **kappa −0.028**, and a pooled confusion matrix
that is essentially uniform:

|  | pred lt_26d | pred 26–71d | pred gt_71d |
|---|---|---|---|
| **true lt_26d** | 974 | 894 | 930 |
| **true 26–71d** | 1031 | 798 | 968 |
| **true gt_71d** | 1015 | 912 | 871 |

Per-class F1 ranged 0.296–0.335 — no class recovered above chance, and no
region carrying signal the others lacked. A 264-fold dual-channel CNN+LSTM run
was started and stopped at fold 69 once the diagnosis below made its outcome
uninteresting; per-fold accuracies to that point were scattered around the same
level.

**The structural diagnosis.** Declustering exists to stop one mainshock's
aftershock sequence masquerading as many independent targets — necessary for
target *selection*, but it removes exactly the predictable structure.
Aftershocks follow Omori decay and are the most forecastable part of a catalog.
What remains is mainshock timing, and measured inter-mainshock gaps have
CV 0.67–1.17 across regions. **CV = 1 is exactly exponential**, i.e.
memoryless, so P(wait | history) = P(wait) and no model can beat chance.

This is not a claim that catalog forecasting is impossible. It is a claim that
*this target*, on *this catalog*, was defined so as to be nearly unlearnable.

## 2. Two defects that would have published wrong numbers

**(a) Class labels wrong by an order of magnitude.**
`RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]` was hardcoded in two files while
`assign_risk_classes` derives **tercile** boundaries. On this catalog those
terciles land at 26 d and 71 d, so:

| Label says | Actually means |
|---|---|
| `lt_1y` | 0 – 25.8 days |
| `1_5y` | 25.8 – 71.3 days |
| `gt_5y` | 71.3 – 816.9 days |

Median time-to-next-mainshock is 46 days. A table reporting "`gt_5y` precision
0.81" would be read as a five-year claim and be wrong by more than an order of
magnitude. Nothing failed; only the names were wrong. Fixed by generating class
names from the boundaries in force and having both trainers read the ordinal
direction from the manifest's own `days_to_major`.

**(b) The built-in LOEO baseline is anti-predictive.**
Both scripts compared against per-fold majority-class prediction, which scores
**8.53 %** — far *below* the 33.33 % chance rate. With globally balanced classes
and highly concentrated folds (mean purity 0.748), removing a fold tips the
training pool *away* from that fold's own dominant class, so the "majority" is
systematically the class the fold has **least** of. Measured: it matched the
fold's true mode in **2 of 264** folds and its rarest class in **175**.
Comparing against it turned a chance-level model into an apparent +23-point
win. Both scripts now lead with the chance floor and flag the artifact.

## 3. The reformulation

The change is to the **target**, not the model — which is what
`catalog.report_major_events`'s own remediation advice recommends
("switch target definition: 'max magnitude in the next N days' is a dense
regression problem rather than a rare-event one").

**Target:** will a M ≥ 4.5 event occur in this fault zone within 30 days of the
window's last event? Dense (every window has one), undeclustered, so clustered
seismicity now *helps* instead of being defined away. Base rates run 0.24–0.61
by zone — neither saturated nor rare. At national scale the same target is
vacuous (M ≥ 4 occurs in essentially every 30-day window), which is why
regionalisation is load-bearing.

**Zones.** The previous pooled dataset's four regions were never recorded
anywhere and could not be reconstructed. `FAULT_ZONES` is now a committed
constant covering 92.8 % of the catalog:

| Zone | Extent (lat, lon) | Events | M≥4.5 |
|---|---|---|---|
| NAFZ — North Anatolian | 39.5–42.0, 26.0–42.0 | 30,744 | 82 |
| EAFZ — East Anatolian | 36.5–39.5, 35.0–42.0 | 50,066 | 299 |
| AEGEAN — western extension | 36.0–40.0, 25.0–30.0 | 73,703 | 200 |
| CENTRAL — Cyprus arc | 34.0–37.5, 28.0–36.0 | 16,841 | 54 |

**Two measured feature fixes.** `n_events` was in the old aux vector and is
*constant* by construction (it equals `window_events`) — one of nine features
was dead weight, now dropped. `days_since_prev_major` was absent despite being
the natural feature for a renewal process and the first indicator in comparable
published work; measured Spearman +0.129 (p = 2.6×10⁻³²) against the old
target, on par with the best feature that *was* included. Added, along with
rate- and energy-acceleration terms.

**Splitting.** Chronological with a 30-day horizon embargo. The label looks
forward exactly one horizon, so dropping one horizon of windows at each
boundary is precisely enough to stop a window being labelled by events on the
far side — and costs only 142 of 21,339 windows.

## 4. Results

### 4.1 Pooled

Test set, 2,414 windows, positive rate 0.589:

| Model | AUC | Accuracy | MCC |
|---|---|---|---|
| *floor:* base rate (majority) | 0.5000 | 41.05 % | +0.000 |
| *floor:* persistence (recency) | 0.6344 | 61.72 % | +0.194 |
| **Logistic (all features)** | **0.7228** | **63.34 %** | **+0.339** |
| Gradient boosting | 0.6927 | 60.98 % | +0.261 |

The **persistence floor** matters more than the base rate: earthquakes cluster,
so "a qualifying event happened recently, predict another" is free. A model
that cannot beat it has learned that seismicity is bursty, not how to forecast
it. AUC is the headline because positive rates run 0.24–0.61 by zone, so
accuracy is dominated by the base rate.

Logistic beating gradient boosting repeats the pattern of `report.md` §8.5 —
the simpler model wins again.

### 4.2 Per zone — where the pooled number comes from

| zone | n_test | pos | persist | pooled | pool+zone | per-zone |
|---|---|---|---|---|---|---|
| NAFZ | 320 | 0.356 | 0.3799 | 0.3711 | 0.3755 | **0.4091** |
| EAFZ | 823 | 0.487 | 0.4990 | 0.5703 | 0.5689 | 0.5648 |
| AEGEAN | 1129 | 0.782 | 0.6398 | **0.7984** | 0.7929 | 0.7773 |
| CENTRAL | 142 | 0.176 | 0.2701 | 0.3491 | 0.3610 | **0.4239** |
| **MACRO** (each zone once) | | | 0.4472 | 0.5222 | 0.5246 | **0.5438** |
| **MICRO** (window-weighted) | | | 0.5356 | 0.6376 | 0.6358 | 0.6353 |

**The pooled 0.72 headline is AEGEAN's number in disguise.** Micro averaging
reproduces it because AEGEAN is the largest zone; macro averaging, where each
zone counts once, collapses it to 0.52–0.54.

Per-zone modelling is the better structure — best macro AUC, and the only
strategy beating persistence in **4 of 4** zones rather than 3 of 4 — but it
does not rescue the failing zones. NAFZ improves 0.371 → 0.409 and CENTRAL
0.349 → 0.424, both still below chance.

### 4.3 Forecastability tracks clustering

Persistence AUC differs in **sign** across zones, and tracks the coefficient of
variation of M ≥ 4.5 inter-event gaps almost monotonically:

| Zone | CV | Persistence AUC | Interpretation |
|---|---|---|---|
| AEGEAN | 1.56 | 0.640 | clustered — recent event ⇒ **more** likely |
| EAFZ | 1.46 | 0.499 | clustered, but no usable recency signal |
| NAFZ | 1.04 | 0.380 | ~Poisson — recent event ⇒ **fewer** |
| CENTRAL | 1.02 | 0.270 | ~Poisson, strongly inverted |

CV = 1 is exactly memoryless. NAFZ and CENTRAL sit there, so
P(event | history) ≈ P(event) and no model of this kind can forecast them —
the same argument that explained §1's chance-level result, now explaining
*which zones work*.

This is why a single pooled model cannot serve all four: it must represent two
opposite temporal dependences at once, so it learns the average and fails on
the minority behaviour. Per-zone models can represent both and do improve the
Poisson-like zones — but improving toward chance is not forecasting.

**The defensible claim** is therefore narrow and specific: *M ≥ 4.5 within 30
days is forecastable in the Aegean extensional province (AUC 0.78 against a
0.64 persistence floor) and is not forecastable in the near-Poisson NAFZ and
Cyprus-arc zones.*

## 5. Limitations

- **Below-chance AUC in two zones is not just "no signal"** — it indicates the
  relationship *inverts* between the 2010–2021 training era and the 2023–2026
  test era. Non-stationarity, not merely absence of signal.
- **The test era contains the 2023 Kahramanmaraş sequence**, which is why the
  validation window has a 0.782 positive rate against training's 0.414. A
  chronological split cannot avoid this, and it disproportionately affects EAFZ.
- **CENTRAL and NAFZ test sets are small** (142 and 320 windows), so their
  figures carry wide uncertainty.
- **Single split, single seed, single threshold/horizon.** No sensitivity
  analysis over M ≥ 4.5 or 30 days, and `report.md` §6.6 showed concretely that
  single-seed margins on this project can reverse.
- **No neural model has been run on the reformulated target.** Given that
  logistic beats gradient boosting here, and the §8.5 precedent, the priors are
  not favourable — but it is untested.

## 6. Reproduction

```bash
cd cnn_earthquake/src
# pooled, with base-rate and persistence floors
python forecast_eval.py    --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
# per-zone vs pooled vs pooled+zone-feature, shared chronological cuts
python forecast_perzone.py --catalog ../../data_downloader/catalogs/deprem_katalog_utc.csv
```

Zone definitions, window construction and the split rule live in
`data_downloader/seismic_cli/forecast.py`; the superseded time-to-mainshock
pipeline remains in `seismic_cli/catalog.py` with its LOEO trainers
(`cnn_lstm_loeo.py`, `catalog_scalar_loeo.py`).
