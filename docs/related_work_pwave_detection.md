# Related work: deep-learning P-wave detection, and why the numbers don't compare

2026-08-22. Sources in `~/Downloads/seismic_network_research/eew_deep_learning/`.

The purpose of this file is **not** to collect headline scores. Published
detection accuracies in this literature range from 0.69 to 0.98, and the spread
is driven more by evaluation protocol than by architecture. What follows is a
protocol table first, scores second, so it is visible which papers this work can
honestly stand beside.

Extend it by adding a row, not by adding a number.

---

## 1. Protocol comparison

| | **TransQuake** (2021) | **NZ edge CNN** (2026) | **CWT + YOLO** (2025) | **This work (P-only)** |
|---|---|---|---|---|
| venue | Earthquake Research Advances | Scientific Reports | IEEE Access | — |
| task | binary: window contains P? | **3-class**: P / S / noise | binary: spectrogram is P? | binary: window contains P? |
| input | 3-comp, 100 Hz | 3-comp, 50 Hz | Morlet CWT image | 3-comp, 100 Hz + log-power STFT |
| **window** | **50 s** | 2 s | per-image | **3.4 s** (2.0 pre-P + **1.4 post-P**) |
| **negatives** | FilterPicker false picks | **same 90 s record**, ±2 s exclusion around P and S | from same station record | screened quiet windows **3 h** before origin, cross-checked ±300 s against a 482,898-event catalogue |
| **split** | chronological (first 5/6 → train) | **random 70/15/15** | **cross-station** (train TLG, test TARG/JNKS/SHLS) | **station-disjoint** |
| test balance | **~11:1 noise:event** (only train balanced) | balanced | balanced | balanced |
| events | Wenchuan aftershocks | **M ≥ 3.0** | within 3° (~333 km) | **M ≥ 2.0** (median 2.3), ≤ 56 km |
| n (test) | ~20 k | ~41 k segments | **~640 windows** | 15,816 windows |
| **conditional floor** | **not reported** | **not reported** | **not reported** | **0.6679** (amplitude-matched) |
| headline | P 0.740 / **R 0.685** / F1 0.712 | acc **97.12%**, P-recall 98% | P 93.4 / **R 94.2** / F1 93.8 | **AUC 0.8712**, R 0.638 @0.5 |

Magnitude/PGA papers in the same directory solve a **different problem**
(regression from early P, not detection) and are not comparable on any of the
above. Listed for completeness: Wang et al. 2023 (*Gondwana Res.*, magnitude
from first seconds of P, single station); Fu et al. 2026 (*J. Asian Earth Sci.*,
on-site PGA, variable windows); Chen et al. 2026 (*JGR ML & Comp.*, multi-station
PGV, window from 5 s before P). Abdullin et al. 2026 (*AI in Geosciences*,
SHAP for microseismic detection, 30 s windows) is a methodology reference for
attribution, not a benchmark.

---

## 2. Three findings that matter more than the scores

### 2.1 TransQuake independently confirms the window-length tradeoff

Their ablation over 20–50 s windows concludes:

> "metrics, especially F1, become better with the increasing of the time window
> length, indicating that the information besides the P wave also contributes to
> detection."

and they hit our exact problem first:

> "Considering the different epicentral distances, we are unable to set a fixed
> time window that only contains a full P-wave."

So a published group identified the same tension — a fixed window cannot be
P-only across a range of epicentral distances — and resolved it the **opposite**
way: lengthen to 50 s, keep S and coda, accept that early-warning value is gone.
This work goes the other way and quantifies the cost (headroom captured 89% at
6 s → 61% P-only). Their result is independent evidence that the drop is a
property of the task, not a deficiency of this model.

### 2.2 The NZ paper is a live specimen of the failure mode Bölüm 5.1 describes

It reports **97.12% accuracy** on 272,424 segments, and discloses:

> "An initial attempt to split them chronologically, intended to better simulate
> real-world deployment, was unsuccessful due to biases observed across
> different periods (e.g., significant deviations during the COVID-19 lockdown
> period due to changes in anthropogenic noise). Random splitting was therefore
> adopted."

Three compounding issues:

1. **Chronological validation failed, so it was abandoned** — the same pattern
   Jover-Alfaro et al. (2026) document, where >97% accuracy fell to 24% under
   time-based validation.
2. **Negatives are not independent of positives.** Noise segments are drawn from
   the *same 90 s record* as the P and S segments, with only ±2 s exclusion
   around each arrival. Under random splitting, the P segment of an event and
   the noise segment from that same record land on opposite sides of the split.
   With only ±2 s exclusion, a "noise" segment can also be drawn from coda.
3. **Amplitude sufficiency is asserted, not measured.** They argue the SNR
   distribution (mean 3.28 dB, sd 9.92 dB) means "the model must learn to
   recognize subtle P-wave features rather than relying on simple amplitude
   changes" — the precise claim a conditional floor exists to test, and it is
   never tested.

**Their engineering contribution is real and is not what is being criticised
here**: ~38 K parameters, sub-7 ms inference, running on Raspberry Pi-class
hardware, with a 2 s window they credibly claim is the shortest reported for
this task. The critique is confined to the evaluation protocol.

### 2.3 No paper in this set reports a conditional floor

Not one of the three reports what a single learning-free statistic achieves on
its own data. This makes Bölüm 5.1's contribution novel in this literature, and
it is the strongest defensible claim this project has — stronger than any
detection score it can report.

---

## 3. Where this work's numbers actually sit

The pattern across the table is that **protocol strictness predicts the score
better than architecture does**:

| protocol on negatives + split | recall |
|---|---|
| FilterPicker negatives, chronological split, imbalanced test (TransQuake) | **0.685** |
| **screened + amplitude-matched negatives, station-disjoint (this work)** | **0.638** |
| same-record noise, random split (NZ) | 0.98 |
| single-station train, cross-station test, n≈640 (YOLO) | 0.942 |

The two papers that control what goes into the negative class land near 0.64–0.69.
The two that do not land near 0.94–0.98.

This work's 0.638 is also measured on a **harder** event population than either
high-scoring paper — M ≥ 2.0 with median 2.3, against NZ's M ≥ 3.0 — and under a
stricter split than any of them.

The honest framing is therefore not "we underperform the literature". It is that
recall 0.64 under station-disjoint splitting, with independently screened and
amplitude-matched negatives and a measured floor of 0.6679, is not evidently
worse than 0.98 under random splitting with same-record noise and no floor at
all. **The comparison cannot be settled from the published numbers**, and saying
so is more defensible than picking a side.

---

## 3b. A comparable number now exists (2026-08-27)

`docs/experiment_gpd_baseline_2026-08-27.md` runs **GPD (Ross et al. 2018)** on
this project's own P-only test windows, scored against the same conditional
amplitude floor. Both models on identical rows, identical floor:

| | AUC | headroom captured |
|---|---|---|
| amplitude floor | 0.5860 | 0% |
| GPD `original` (Ross et al.'s weights) | 0.7710 | 44.7% |
| GPD `geofon` (best of five weight sets) | 0.7987 | 51.4% |
| **this work (fusion)** | **0.8796** | **70.9%** |

Two findings that belong in this file rather than that one:

- **Which pretrained weights you download moves GPD by 0.083 AUC on identical
  data** (0.7154 `scedc` → 0.7987 `geofon`) — a larger spread than most of the
  architecture differences this literature reports. A paper quoting one
  pretrained model is quoting one draw from that spread.
- **GPD scores 0.7710 on the matched negatives and 0.7748 on the natural ones**,
  having seen neither. Our own model moves 0.8796 → 0.8414 across the same two
  sets. A foreign model being nearly invariant where ours is not is independent
  support for the negative-regime finding in
  `experiment_ponly_2026-08-22.md` — the effect is in the negative distribution,
  not the positives.

The reciprocal run (this detector on STEAD, against a floor computed there) is
still missing, and the claim is one-sided until it exists.

---

## 4. What would make a real comparison possible

The metrics this literature actually shares are **detection latency** and
**false alarms on continuous data**. This work is well positioned on the first
and cannot currently report the second.

- **Latency.** 1.4 s after P by construction, verified, with zero S
  contamination under iasp91. Directly comparable to the NZ paper's 2 s window
  and favourable against TransQuake's 50 s.
- **False alarms per station-day.** Every deployed EEW system reports this;
  this project does not have it (Sınırlılık 8). Continuous-data sliding-window
  evaluation is the missing measurement. Note **BODT cannot be used** — it is a
  *training* station in the detection corpus (379 windows, train split). Of the
  183 stations, **35 are held out in test**; continuous data for one or two of
  those would produce the number.
- **ROC-AUC is not reported anywhere in this literature.** It should not be the
  headline figure when comparing outward, even though it is the right primary
  metric internally (threshold-independent, and the floor is expressed in it).

---

## 5. How to extend this file

For each new paper, extract the row before the result: task, input, window,
**how negatives were constructed**, **split protocol**, test class balance,
magnitude range, test n, and whether any learning-free baseline is reported.
A paper that reports a high score without controlling negatives and split
belongs in the table with that noted, not excluded — the spread is the finding.
