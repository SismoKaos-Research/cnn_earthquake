# The Spectrogram Classifier

**A technical report on the best-performing detector in this project**

---

## Abstract

Across every configuration tested in this investigation — including the
dual-channel CNN+LSTM architecture from Wang & Zhao (2025), three fusion
mechanisms, an amplitude auxiliary input, and two hyperparameter sweeps — the
single best earthquake/noise detector is also the simplest: a
Squeeze-and-Excitation ResNet applied to a station-normalized log-power
spectrogram, with no second branch, no auxiliary input, and no fusion at all.
It reaches **test AUC 0.9793, MCC 0.8666, accuracy 93.28 %** on 6-second
arrival-anchored windows, beating a correctly-parameterised STA/LTA trigger by
0.16 AUC.

This report covers that model specifically: what it is, why it outperforms the
RAM-image encoding it was introduced to compete with, and the two ways it can
be made to look better or worse than it is.

---

## 1. Why a spectrogram at all

The project began with the Relative Angle Matrix (RAM) transform, which encodes
a waveform window as an image of angles between normalized sub-vectors. That
normalization has an exact consequence:

$$\max\left|\text{RAM}(x) - \text{RAM}(37.5\,x)\right| = 8.9\times10^{-16}$$

A 37.5× stronger signal produces a byte-identical image. RAM **cannot represent
amplitude**, and amplitude relative to a station's background is precisely the
quantity short-window detection depends on — it is the entire discriminative
content of STA/LTA.

A station-normalized spectrogram has the opposite property: it preserves
amplitude *as a function of both time and frequency*. It was introduced as a
drop-in alternative for the 2D channel, and it won immediately and consistently.

## 2. The model

**Input.** Per window, per channel (Z/N/E): linear + constant detrend, 5 % Hann
taper, 4th-order zero-phase Butterworth bandpass 1–45 Hz, then a log-power
spectrogram (`n_fft=256`, `top_db=80`) normalized against that station's own
long-term noise spectrum (`--normalize station`). The three components stack as
image channels.

**Trunk.** `CNNBranch` — three convolution / batch-norm / GELU stages with
global average pooling. Global pooling makes it resolution-agnostic, which is
why the identical trunk accepts both a square RAM image and a non-square
spectrogram without modification; that is what makes the comparison in §3 a
clean single-variable one.

**Head.** Single logit, `BCEWithLogitsLoss` on label-smoothed targets
($\tilde y = 0.8y + 0.1$), AdamW, gradient-norm clipping at 1.0, mixed
precision, checkpoint selected on validation AUC.

**Size.** 115,459 parameters — the *smallest* model in the comparison. The RAM
classifier it beats is 310k; the full dual-channel model is 183k.

**Data.** 71,672 windows (35,836 per class, balanced), 6 s arrival-anchored,
station-disjoint splits (82/30/40 earthquake and 104/35/38 noise stations across
train/validation/test), 9,548 test windows.

## 3. Results

| Model | Params | Test AUC | MCC | Accuracy |
|---|---|---|---|---|
| **Spectrogram CNN (this report)** | **115k** | **0.9793** | **0.8666** | **93.28 %** |
| Spectrogram + LSTM branch, stacked fusion | 183k | 0.9743 | 0.871 | 93.54 % |
| Spectrogram + LSTM branch, gated fusion | 183k | 0.9761 | 0.850 | 92.51 % |
| Spectrogram + LSTM branch, linear fusion | 183k | 0.9646 | 0.8122 | 90.61 % |
| RAM CNN + amplitude scalars | 310k | 0.9230 | 0.7018 | 84.79 % |
| RAM CNN alone | 310k | 0.8356 | 0.5339 | 76.70 % |
| STA/LTA, correctly parameterised | — | 0.8194 | — | 74.60 % |

Three points follow.

**It beats RAM by 0.144 AUC on an architecture-matched comparison.** Same trunk,
same training procedure, same windows — only the image differs. This is the
cleanest measurement in the investigation of what the encoding itself
contributes.

**Adding the paper's LSTM branch makes it worse, not better.** Linear fusion —
the source paper's own mechanism — costs 0.015 AUC. Gated fusion and late-fusion
stacking both recover most of that loss but neither exceeds the plain model's
AUC. A fixed pair of learned scalars cannot suppress a weaker branch on the
specific windows where it is wrong, and joint training lets the weaker branch
degrade the stronger one's representation.

**Adding amplitude scalars also makes it worse.** `2d+aux` scores 0.9749 against
plain `2d`'s 0.9793. This is not a contradiction of the amplitude finding — it
is the same finding from the other side. Those two scalars produce the largest
single improvement measured anywhere in this project when added to RAM
(+0.087 AUC), because RAM discards amplitude entirely. Added to a spectrogram
that already encodes amplitude across time and frequency, they are redundant and
contribute only estimation noise.

## 4. Two ways to misread this model

**(a) The baseline it beats was broken.** `eval-sta-lta` had never actually run
against this dataset: its filename regex matched only `.png`, so it silently
scored zero windows against the `.pt` tensors the dual-channel encoders write.
Once fixed, its auto-derived parameters gave **AUC 0.5093 — indistinguishable
from random**, because the derived long-term-average window (2.0 s) is longer
than the pre-arrival buffer the anchoring scheme provides (1.2 s), so the P-wave
arrival falls inside `classic_sta_lta`'s mandatory forced-zero warm-up region
and is invisible to the characteristic function. With a validation-selected LTA
that respects the buffer, STA/LTA scores **0.8194**. Quoting the 0.51 figure
would overstate this model's margin by 0.31 AUC.

**(b) Single-seed margins here are not trustworthy.** The gap between this model
and its nearest competitors is 0.003–0.015 AUC. Two comparisons elsewhere in
this project were re-run at three seeds; one **reversed sign**, and the other
shrank to a third of its reported size. The 0.144 AUC margin over RAM is an
order of magnitude above that noise band and is safe. The margins over the
fusion variants are not — they should be treated as "no worse than", not
"better than".

## 5. Limitations

Single seed, single train/validation/test split, 6-second windows only — the
60-second case has never been re-run under the current corrected pipeline.
Hyperparameter tuning was tested and found unproductive on the related RAM
classifier (six configurations within a 0.004 validation-AUC band), so the
defaults here are inherited rather than optimized for the spectrogram
specifically. Noise-station diversity is limited throughout this project and is
the constraint most likely to be masking a generalization problem.

## 6. Reproduction

```bash
seismic-cli generate-spec-dual-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms/noise_pre_3h \
    --output-dir dataset_specdual_6s --window-seconds 6 --max

cd ../cnn_earthquake/src
python cnn_lstm_classify.py --dataset-dir ../../data_downloader/dataset_specdual_6s \
    --channels 2d --batch-size 32

# the honest baseline (NOT the auto-derived default -- see section 4a)
seismic-cli eval-sta-lta --manifest-path dataset_specdual_6s/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5 \
    --sta-seconds 0.03 --lta-seconds 0.3
```

Full detail, including the RAM transform's formal properties and the complete
defect changelog, is in `report.md`.
