"""
Hand-engineered DWT/spectral waveform features, per Tables 3-5 of Bhatia,
Ahanger & Manocha (2023) "Artificial intelligence based real-time earthquake
prediction" (Eng. Appl. Artif. Intell. 120, 105856). Their ANFIS/BBM
classifier and IoT-edge-cloud architecture solve a different problem
(labeling an already-recorded 2-minute waveform as earthquake-vs-not on
STEAD) and don't transfer here, but their time/frequency/time-frequency
domain feature formulas are a reusable, cheap alternative to the CNN's
learned waveform embedding in `cnn_lstm_catalog_waveform_fusion.py`'s
wave_branch.

Per channel, per hourly window:
  - Time domain (9): std, skewness, kurtosis, variance, RMS, mean absolute
    value, approximate entropy, energy entropy, waveform length.
  - Frequency domain (5): peak frequency, mean frequency, mean power,
    spectral moment order-2, total power (via rFFT power spectrum).
  - Time-frequency domain (4 x 5 = 20): for each of the 5 coefficient
    arrays from a 4-level db4 DWT (cA4, cD4, cD3, cD2, cD1) -- std,
    average power, mean absolute value, entropy.

34 features/channel x 3 channels = 102 features per hourly window. That's
a lot for a dataset this size (same lesson as the 11-feature catalog
overfit last night) -- prune with `catalog_feature_rfe.py`-style RFE
before trusting any of this in a neural branch.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pywt

DWT_WAVELET = "db4"
DWT_LEVEL = 4
N_TIME_FEATURES = 9
N_FREQ_FEATURES = 5
N_DWT_LEVELS = DWT_LEVEL + 1  # cA4, cD4, cD3, cD2, cD1
N_TF_FEATURES = 4 * N_DWT_LEVELS
N_FEATURES_PER_CHANNEL = N_TIME_FEATURES + N_FREQ_FEATURES + N_TF_FEATURES  # 34

TIME_FEATURE_NAMES = ["std", "skewness", "kurtosis", "variance", "rms", "mean_abs",
                      "approx_entropy", "energy_entropy", "waveform_length"]
FREQ_FEATURE_NAMES = ["peak_freq", "mean_freq", "mean_power", "spectral_moment2",
                      "total_power"]
TF_STAT_NAMES = ["wavelet_std", "wavelet_power", "wavelet_mean_abs", "wavelet_entropy"]
TF_LEVEL_NAMES = ["cA4", "cD4", "cD3", "cD2", "cD1"]


APEN_MAX_SAMPLES = 500


def _approx_entropy(x, m=2, r_frac=0.2, max_samples=APEN_MAX_SAMPLES):
    """Approximate entropy (Pincus 1991), r = r_frac * std(x).

    Exact ApEn needs an O(n^2) pairwise-distance matrix, which is intractable
    at hour-long-window sample counts (18k-36k -> a multi-GB temp array). We
    uniformly stride the window down to `max_samples` points first -- this
    trades fine-timescale complexity for tractability, so ApEn here reflects
    complexity at a coarser (sub-sampled) timescale than the paper's original
    ~7000-sample STEAD windows.

    Args:
        x: 1-D signal.
        m: Embedding dimension.
        r_frac: Tolerance as a fraction of the signal's std.
        max_samples: Uniform-stride cap applied before the O(n^2) step.

    Returns:
        Scalar ApEn value (0.0 if the signal is constant, since r would be 0).
    """
    if len(x) > max_samples:
        stride = len(x) // max_samples
        x = x[::stride]
    n = len(x)
    r = r_frac * np.std(x)
    if r == 0 or n <= m + 1:
        return 0.0

    def _phi(m_):
        templates = np.array([x[i:i + m_] for i in range(n - m_ + 1)])
        dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        counts = np.sum(dists <= r, axis=1)
        return np.mean(np.log(counts / (n - m_ + 1)))

    return float(_phi(m) - _phi(m + 1))


def _energy_entropy(x):
    """Shannon entropy of the signal's normalized per-sample energy distribution."""
    energy = x.astype(np.float64) ** 2
    total = energy.sum()
    if total <= 0:
        return 0.0
    p = energy / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _time_domain_features(x):
    """Computes the 9 time-domain features (Table 3) for one 1-D channel window."""
    x = x.astype(np.float64)
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        skew, kurt = 0.0, 0.0
    else:
        skew = float(np.mean(((x - mu) / sigma) ** 3))
        kurt = float(np.mean(((x - mu) / sigma) ** 4) - 3.0)
    variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    mean_abs = float(np.mean(np.abs(x)))
    apen = _approx_entropy(x)
    een = _energy_entropy(x)
    wl = float(np.sum(np.abs(np.diff(x))))
    return [sigma, skew, kurt, variance, rms, mean_abs, apen, een, wl]


def _frequency_domain_features(x, fs):
    """Computes the 5 frequency-domain features (Table 4) for one 1-D channel window."""
    x = x.astype(np.float64)
    n = len(x)
    spectrum = np.fft.rfft(x - x.mean())
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total_power = float(power.sum())
    if total_power <= 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    peak_freq = float(freqs[np.argmax(power)])
    mean_freq = float(np.sum(freqs * power) / total_power)
    mean_power = float(power.mean())
    spectral_moment2 = float(np.sum(power * freqs ** 2))
    return [peak_freq, mean_freq, mean_power, spectral_moment2, total_power]


def _time_frequency_features(x):
    """Computes the 4x5=20 time-frequency features (Table 5): a 4-level db4 DWT,
    then std/power/mean-abs/entropy of each of the 5 resulting coefficient arrays.
    """
    coeffs = pywt.wavedec(x.astype(np.float64), DWT_WAVELET, level=DWT_LEVEL)
    feats = []
    for c in coeffs:
        n = len(c)
        mu = c.mean()
        std = float(np.sqrt(np.mean((c - mu) ** 2)))
        power = float(np.mean(c ** 2))
        mean_abs = float(np.mean(np.abs(c)))
        c2 = c ** 2
        c2 = c2[c2 > 0]
        entropy = float(np.sum(c2 * np.log(c2))) if len(c2) else 0.0
        feats.extend([std, power, mean_abs, entropy])
    return feats


def build_dwt_features_1d(x, fs):
    """Builds the 34-feature DWT/spectral vector for one channel's window.

    Args:
        x: 1-D array, one channel's samples for one hourly window.
        fs: Sampling rate in Hz.

    Returns:
        1-D float64 array, length N_FEATURES_PER_CHANNEL (34).
    """
    return np.array(_time_domain_features(x) + _frequency_domain_features(x, fs)
                    + _time_frequency_features(x), dtype=np.float64)


def _build_chunk(args):
    """Worker: builds features for a contiguous slice of hours. Module-level
    so it's picklable for ProcessPoolExecutor.
    """
    raw_chunk, fs = args
    n_hours, n_channels, _ = raw_chunk.shape
    out = np.empty((n_hours, n_channels * N_FEATURES_PER_CHANNEL), dtype=np.float64)
    for h in range(n_hours):
        for c in range(n_channels):
            start = c * N_FEATURES_PER_CHANNEL
            out[h, start:start + N_FEATURES_PER_CHANNEL] = build_dwt_features_1d(raw_chunk[h, c], fs)
    return out


def build_hourly_waveform_features(raw, fs=None, n_jobs=1):
    """Builds DWT/spectral features for every hourly window and channel.

    ApEn's O(n^2) step dominates cost (~50ms/channel-hour even after the
    max_samples cap), so a full multi-thousand-hour archive is a
    multi-minute job -- use n_jobs > 1 to parallelize across hour chunks.

    Args:
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        fs: Sampling rate in Hz. Default: inferred as hour_samples / 3600.
        n_jobs: Number of worker processes. 1 = serial (default).

    Returns:
        (n_hours, 3 * N_FEATURES_PER_CHANNEL) float64 array -- channels
        concatenated in order, each channel's 34 features contiguous.
    """
    n_hours, n_channels, hour_samples = raw.shape
    if fs is None:
        fs = hour_samples / 3600.0
    if n_jobs <= 1:
        return _build_chunk((raw, fs))

    bounds = np.linspace(0, n_hours, n_jobs + 1, dtype=int)
    chunks = [(np.asarray(raw[bounds[i]:bounds[i + 1]]), fs)
             for i in range(n_jobs) if bounds[i + 1] > bounds[i]]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(_build_chunk, chunks))
    return np.concatenate(results, axis=0)


def feature_names(n_channels=3):
    """Builds the full per-channel-prefixed feature name list, matching
    `build_hourly_waveform_features`'s column order.
    """
    base = [f"time_{n}" for n in TIME_FEATURE_NAMES] + [f"freq_{n}" for n in FREQ_FEATURE_NAMES]
    for level in TF_LEVEL_NAMES:
        base.extend(f"tf_{level}_{stat}" for stat in TF_STAT_NAMES)
    names = []
    for c in range(n_channels):
        names.extend(f"ch{c}_{b}" for b in base)
    return names
