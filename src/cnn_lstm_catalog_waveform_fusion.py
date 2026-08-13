"""
Catalog-LSTM + raw-waveform-CNN fusion forecaster, trained end-to-end.

Motivated by today's finding that the catalog-only gradient-boosted floor
(~0.73 pooled AUC per report.md) already beats every raw-waveform-only model
tried today, while a plain-CNN embedding of raw waveform found real (if
weak) signal that hand-picked statistical features (RMS/kurtosis/PSD ratio,
via this session's MMD test) could not. Rather than asking waveform to
forecast from scratch (small samples, noisy, everything tried today), this
asks a smaller, easier-to-answer question: does adding a raw-waveform CNN
branch improve a model that already has a working catalog-based branch?

Two branches, both `LSTMAttentionBranch` (this project's existing 1D
sequence branch), fused by concatenation + a shared head -- trained
JOINTLY, not pre-extract-then-freeze, so the CNN's waveform embedding is
shaped by gradient signal from this exact forecast target rather than
repeating whatever it learned from an unrelated task (e.g. today's
proximity classifier).

  - Catalog branch: LSTMAttentionBranch directly on a small per-hour
    catalog-feature vector (log1p(days since previous qualifying event),
    event counts in the trailing 7/30/90 days) -- all backward-looking, no
    leakage, no magnitude needed (load_aegean_events only returns times).
  - Waveform branch: RawWaveformEncoder (same CNN as every other raw-
    waveform script here) per hour, then LSTMAttentionBranch over the
    resulting hourly-embedding sequence -- structurally identical to
    RawCNNLSTM's own branch, just stopped before its head so the pooled
    embedding can be fused instead.

`--channels catalog` trains the catalog branch alone (the ablation this
experiment's whole point rests on -- compare against `--channels all` to
see whether waveform adds anything). `--channels all` (default) is the
fused model.

Usage:
    python cnn_lstm_catalog_waveform_fusion.py \\
        --data-root ../../Sismokaos/feature-extract/data/aegean_bodt_2024_2026_consolidated \\
        --catalog-path ../../Sismokaos/data_downloader/catalogs/data_large.csv \\
        --consolidated --channels all --cv-folds 2

Not imported by anything else -- standalone script.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader, Dataset

from feature_lstm_forecast import (count_events_in_window, days_since_prev_major,
                                   label_hours, label_hours_rate_change, load_aegean_events,
                                   load_aegean_events_with_location, print_split_diagnostics,
                                   rate_persistence_auc, safe_auc,
                                   truncate_to_reliable_catalog_end, walk_forward_splits)
from metrics import binary_report, print_report
from model.blocks import LSTMAttentionBranch
from raw_cnn_lstm_forecast import (RawWaveformEncoder, load_hourly_raw,
                                   load_hourly_raw_consolidated)
from training import seed_everything

# log1p(days since prev major event), count_7d/30d/90d (major events, M>=threshold --
# same set label_hours/persistence use), mean_mag_30d, max_mag_90d, b_value_90d,
# energy_sqrt_30d, mean_interevent_90d, cv_interevent_90d, magnitude_deficit_90d
# (all from a lower-completeness-threshold "background" catalog -- more data points
# than the rare M>=threshold events alone give for magnitude-distribution features),
# nnd_log_eta_90d, shannon_entropy_90d (Zaliapin-Ben-Zion nearest-neighbour distance
# and spatial Shannon entropy, per Convertito et al. 2024, Sci. Rep. 14:2964 -- the
# same paper's coefficient-of-variation feature independently matches our
# cv_interevent_90d, which RFE already found to be the strongest of the original 11).
CATALOG_DIM = 13
LN10 = np.log(10.0)
FEATURE_NAMES = ["log1p_dsp", "count_7d", "count_30d", "count_90d", "mean_mag_30d",
                 "max_mag_90d", "b_value_90d", "energy_sqrt_30d", "mean_interevent_90d",
                 "cv_interevent_90d", "mag_deficit_90d", "nnd_log_eta_90d",
                 "shannon_entropy_90d"]

NND_FRACTAL_DIM = 1.6  # standard Zaliapin-Ben-Zion literature default
NND_LOOKBACK = 500  # bound the O(n*lookback) nearest-neighbour search for tractability
ENTROPY_GRID_SIZE = 10  # 10x10 spatial cells over AEGEAN_BBOX for Shannon entropy

# Trailing-rate features, for --label-mode rate. None of the features above encode
# the trailing count of the LOW-magnitude events that define the rate target:
# count_7d/30d/90d count M>=threshold (4.5) events, not M>=rate_min_mag (3.0) ones.
# That left the model trying to beat a persistence floor built from exactly the
# number it was never given -- it lost fold 1 0.6573 vs 0.7991 while scoring a
# POSITIVE Brier skill (+0.129), i.e. learning something real but unable to rank
# without the rate signal. The ratio features are the acceleration term itself
# (short-window rate over long-window rate), which is what the label asks about.
RATE_WINDOWS = (3, 7, 14, 30, 90)
RATE_RATIO_PAIRS = ((3, 14), (7, 30), (14, 90))
RATE_FEATURE_NAMES = ([f"rate_log1p_count_{w}d" for w in RATE_WINDOWS]
                      + [f"rate_logratio_{a}_{b}" for a, b in RATE_RATIO_PAIRS])
ALL_FEATURE_NAMES = FEATURE_NAMES + RATE_FEATURE_NAMES


def build_rate_features(hour_index, rate_times) -> np.ndarray:
    """Backward-looking trailing-rate features for the rate-change target.

    Args:
        hour_index: DatetimeIndex of hour starts.
        rate_times: Sorted array of the event times defining the rate (the
            M>=rate_min_mag set that `label_hours_rate_change` uses).

    Returns:
        float32 array, shape (n_hours, len(RATE_FEATURE_NAMES)). Counts are
        log1p'd (raw counts are heavy-tailed -- median 9, max 297 at 14d) and
        ratios are log'd so acceleration and deceleration are symmetric around 0.
    """
    counts = {w: count_events_in_window(hour_index, rate_times, w, forward=False)
             for w in RATE_WINDOWS}
    cols = [np.log1p(counts[w].astype(np.float64)) for w in RATE_WINDOWS]
    for a, b in RATE_RATIO_PAIRS:
        # per-day rates, so the ratio is a clean acceleration factor rather than a
        # window-length artifact; eps keeps quiet stretches (0 events) finite.
        rate_a = counts[a] / float(a)
        rate_b = counts[b] / float(b)
        cols.append(np.log((rate_a + 1e-3) / (rate_b + 1e-3)))
    return np.stack(cols, axis=1).astype(np.float32)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _nearest_neighbor_log_eta(times, mags, lats, lons, b_value=1.0,
                              df=NND_FRACTAL_DIM, lookback=NND_LOOKBACK):
    """Per-event Zaliapin-Ben-Zion nearest-neighbour distance, log10(eta).

    eta_ij = T_ij * R_ij, rescaled time x rescaled distance to the closest
    preceding candidate "parent" event i:
        T_ij = (t_j - t_i) * 10^(-0.5 b m_i)
        R_ij = (haversine_km(i, j))^df * 10^(-0.5 b m_i)
    Small eta = temporally/spatially close to a prior event relative to its
    magnitude -- the signature of a triggered (foreshock/aftershock-like)
    event; large eta = an independent "background" event. Bounded to the
    `lookback` most recent candidate parents per event (full O(n^2) is
    intractable at ~17k events) -- true nearest neighbors are overwhelmingly
    recent for local clustering, so this is a tractable approximation, not
    the exact full-catalog nearest neighbor.

    Args:
        times: Sorted event times (datetime64).
        mags: Matching magnitudes.
        lats: Matching latitudes.
        lons: Matching longitudes.
        b_value: Gutenberg-Richter b-value for the rescaling.
        df: Fractal dimension of the epicenter distribution.
        lookback: Max number of preceding events considered as candidate parents.

    Returns:
        float64 array, length len(times); NaN for the first event (no candidates).
    """
    n = len(times)
    log_eta = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return log_eta
    t_days = (times - times[0]) / np.timedelta64(1, "D")
    for j in range(1, n):
        lo = max(0, j - lookback)
        dt = t_days[j] - t_days[lo:j]
        dist_km = _haversine_km(lats[lo:j], lons[lo:j], lats[j], lons[j])
        scale = 10.0 ** (-0.5 * b_value * mags[lo:j])
        eta = dt * scale * (np.maximum(dist_km, 1e-6) ** df) * scale
        eta = eta[eta > 0]
        if len(eta):
            log_eta[j] = np.log10(eta.min())
    return log_eta


def build_catalog_features(hour_index, major_times, dsp, bg_times=None, bg_mags=None,
                           bg_min_mag=3.0, bg_lats=None, bg_lons=None) -> np.ndarray:
    """Per-hour backward-looking catalog features -- no leakage.

    Timing features (0-3) come from `major_times` (the M>=threshold set
    labels/persistence use). Magnitude/energy/regularity features (4-10)
    come from a separate, lower-completeness-threshold "background" catalog
    (`bg_times`/`bg_mags`, e.g. M>=3.0) -- the Panakkat-Adeli-style feature
    set used in Nurtas et al. 2025 (IEEE Access, the Central Asia LSTM
    forecasting paper): mean magnitude, sqrt(energy release), Gutenberg-
    Richter b-value, magnitude deficit, mean inter-event time, coefficient
    of variation of inter-event times. `bg_times`/`bg_mags` default to
    `major_times` (with an all-zero-ish magnitude array) if not given, which
    degrades gracefully to just the 4 original timing features plus zeros.

    Features 11-12 (nearest-neighbour log-eta, spatial Shannon entropy) need
    `bg_lats`/`bg_lons`; without them they default to 0.0 for every hour.

    Args:
        hour_index: DatetimeIndex of hour starts.
        major_times: Sorted array of qualifying (M>=threshold) event times.
        dsp: Days since the previous qualifying event, per hour (see
            `days_since_prev_major`); NaN where none exists.
        bg_times: Sorted array of lower-threshold "background" event times
            (see `load_aegean_events_with_magnitude`). Defaults to
            `major_times` if None.
        bg_mags: Matching magnitudes for `bg_times`, same order. Defaults
            to an all-`bg_min_mag` array if None.
        bg_min_mag: Completeness threshold of the background catalog --
            the Gutenberg-Richter reference magnitude for the b-value and
            magnitude-deficit calculations.
        bg_lats: Matching latitudes for `bg_times`, same order (see
            `load_aegean_events_with_location`). Optional.
        bg_lons: Matching longitudes for `bg_times`, same order. Optional.

    Returns:
        float32 array, shape (n_hours, CATALOG_DIM).
    """
    if bg_times is None:
        bg_times = major_times
    if bg_mags is None:
        bg_mags = np.full(len(bg_times), bg_min_mag, dtype=np.float64)

    have_location = bg_lats is not None and bg_lons is not None
    if have_location:
        mean_excess_global = max(bg_mags.mean() - bg_min_mag, 1e-3)
        b_value_global = (1.0 / LN10) / mean_excess_global
        nnd_log_eta = _nearest_neighbor_log_eta(bg_times, bg_mags, bg_lats, bg_lons,
                                                b_value=b_value_global)
        lat0, lat1, lon0, lon1 = 36.0, 40.0, 25.0, 30.0  # AEGEAN_BBOX

    t = hour_index.to_numpy()
    feat = np.zeros((len(t), CATALOG_DIM), dtype=np.float32)
    feat[:, 0] = np.log1p(np.nan_to_num(dsp, nan=3650.0))
    for i, ti in enumerate(t):
        past = major_times[major_times < ti]
        feat[i, 1] = np.sum(past > ti - np.timedelta64(7, "D"))
        feat[i, 2] = np.sum(past > ti - np.timedelta64(30, "D"))
        feat[i, 3] = np.sum(past > ti - np.timedelta64(90, "D"))

        mask_30 = (bg_times < ti) & (bg_times > ti - np.timedelta64(30, "D"))
        mask_90 = (bg_times < ti) & (bg_times > ti - np.timedelta64(90, "D"))
        mags_30, mags_90 = bg_mags[mask_30], bg_mags[mask_90]
        times_90 = bg_times[mask_90]
        n90 = len(mags_90)

        feat[i, 4] = mags_30.mean() if len(mags_30) else 0.0
        feat[i, 5] = mags_90.max() if n90 else bg_min_mag
        feat[i, 7] = np.sqrt(np.sum(10.0 ** (1.5 * mags_30))) if len(mags_30) else 0.0

        if n90 >= 5:
            mean_excess = max(mags_90.mean() - bg_min_mag, 1e-3)
            b_value = (1.0 / LN10) / mean_excess
        else:
            b_value = 1.0  # global-average default (standard tectonic seismicity value)
        feat[i, 6] = b_value

        if n90 >= 3:
            intervals = np.diff(np.sort(times_90)) / np.timedelta64(1, "D")
            mean_iv = intervals.mean()
            feat[i, 8] = mean_iv
            feat[i, 9] = (intervals.std() / mean_iv) if mean_iv > 0 else 1.0
        else:
            feat[i, 8] = 90.0  # default: one event per window, i.e. quiet
            feat[i, 9] = 1.0   # default: Poisson-like regularity

        # magnitude deficit: Gutenberg-Richter-extrapolated expected max magnitude
        # (from local b-value + event count) minus what's actually been observed --
        # a proxy for "overdue" stress release.
        expected_max = bg_min_mag + (np.log10(max(n90, 1)) / max(b_value, 1e-3))
        feat[i, 10] = expected_max - feat[i, 5]

        if have_location:
            eta_90 = nnd_log_eta[mask_90]
            eta_90 = eta_90[~np.isnan(eta_90)]
            feat[i, 11] = eta_90.mean() if len(eta_90) else 0.0

            if n90 >= 2:
                lats_90, lons_90 = bg_lats[mask_90], bg_lons[mask_90]
                energies = 10.0 ** (1.5 * mags_90)
                lat_bin = np.clip(((lats_90 - lat0) / (lat1 - lat0) * ENTROPY_GRID_SIZE)
                                  .astype(int), 0, ENTROPY_GRID_SIZE - 1)
                lon_bin = np.clip(((lons_90 - lon0) / (lon1 - lon0) * ENTROPY_GRID_SIZE)
                                  .astype(int), 0, ENTROPY_GRID_SIZE - 1)
                cell_id = lat_bin * ENTROPY_GRID_SIZE + lon_bin
                cell_energy = np.bincount(cell_id, weights=energies,
                                          minlength=ENTROPY_GRID_SIZE ** 2)
                total = cell_energy.sum()
                p = cell_energy[cell_energy > 0] / total if total > 0 else np.array([])
                feat[i, 12] = float(-np.sum(p * np.log(p))) if len(p) else 0.0
    return feat


class FusionSeqDataset(Dataset):
    """One sample per window -- both the raw waveform sequence and the
    aligned catalog-feature sequence, same window, same label. Both
    normalized per-channel/per-feature using the training split's stats."""

    def __init__(self, raw, cat_features, labels, seq_hours, indices, stats=None):
        """Builds the dataset.

        Args:
            raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
            cat_features: Per-hour catalog feature array, shape
                (n_hours, CATALOG_DIM).
            labels: Per-hour binary labels, shape (n_hours,).
            seq_hours: Number of consecutive hours per window.
            indices: Window end-indices.
            stats: Optional (wave_mu, wave_sd, cat_mu, cat_sd) tuple; if
                None, computed from this split's first 50 windows.
        """
        self.raw = raw
        self.cat_features = cat_features
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
        if stats is None:
            # Sample windows spread across the WHOLE training split, not the first 50.
            # The archive's opening hours are unrepresentative for any trailing-window
            # feature, whose lookback is still filling up there: measured on the rate
            # features, sd over the first 50 windows understated the true training sd by
            # up to 52x (rate_log1p_count_90d: 0.0157 vs 0.827), so dividing by it
            # produced z-scores up to 156 and saturated the GELU MLP -- val AUC fell from
            # epoch 1 and the "best" checkpoint was the untrained one.
            stat_idx = indices[np.linspace(0, len(indices) - 1, min(500, len(indices))).astype(int)]
            wsub = np.concatenate([raw[max(0, i - seq_hours + 1):i + 1] for i in stat_idx], axis=0)
            wave_mu = wsub.mean(axis=(0, 2), keepdims=True)[0]
            wave_sd = wsub.std(axis=(0, 2), keepdims=True)[0] + 1e-6
            csub = np.concatenate([cat_features[max(0, i - seq_hours + 1):i + 1] for i in stat_idx], axis=0)
            cat_mu = csub.mean(axis=0, keepdims=True)
            cat_sd = csub.std(axis=0, keepdims=True) + 1e-6
            stats = (wave_mu, wave_sd, cat_mu, cat_sd)
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (wave_seq, cat_seq, label) for one window.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (seq_hours, 3, hour_samples),
            float32 tensor shape (seq_hours, CATALOG_DIM), float32 scalar
            tensor label).
        """
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        wave_mu, wave_sd, cat_mu, cat_sd = self.stats
        wave_seq = (self.raw[start:end + 1] - wave_mu[None]) / wave_sd[None]
        cat_seq = (self.cat_features[start:end + 1] - cat_mu) / cat_sd
        return (torch.from_numpy(wave_seq).float(), torch.from_numpy(cat_seq).float(),
               torch.tensor(self.labels[end], dtype=torch.float32))


class CatalogMLPBranch(nn.Module):
    """Small MLP on the catalog feature vector at the window's LAST hour.

    Catalog features barely change within a window -- empirically ~1-6% of
    their across-dataset variation (measured directly: within-window std /
    overall std for log1p(dsp)/count_7d/count_30d/count_90d was
    0.055/0.056/0.025/0.011). An LSTM's whole point is tracking change
    across a sequence; fed a near-constant-repeated-24-times input, it has
    almost nothing to track and (empirically, in this project) got stuck at
    chance for the entire catalog-only ablation run. A direct MLP on the
    most recent reading is a better-matched, simpler tool for what is
    functionally a single point-in-time tabular reading, not a time series.
    """

    def __init__(self, catalog_dim, hidden=16, dropout=0.4):
        """Initializes the MLP.

        Args:
            catalog_dim: Width of the per-hour catalog feature vector.
            hidden: Hidden width (also the output embedding width).
            dropout: Dropout used between layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(catalog_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.out_dim = hidden

    def forward(self, cat_seq):
        """Embeds the last hour of one batch of catalog-feature sequences.

        Args:
            cat_seq: Shape (batch, seq_hours, catalog_dim).

        Returns:
            Tensor of shape (batch, hidden).
        """
        return self.net(cat_seq[:, -1, :])


class CatalogWaveformFusionNet(nn.Module):
    """Catalog `CatalogMLPBranch` + waveform (`RawWaveformEncoder` ->
    `LSTMAttentionBranch`), fused by concatenation, trained end-to-end.
    `channels` ablates which branch(es) are active."""

    def __init__(self, catalog_dim=CATALOG_DIM, cnn_out=32, cat_hidden=16, wave_hidden=16,
                fusion_hidden=32, dropout=0.4, channels="all"):
        """Initializes the active branch(es) and fusion head.

        Args:
            catalog_dim: Width of the per-hour catalog feature vector.
            cnn_out: Width of the waveform CNN's per-hour embedding.
            cat_hidden: Catalog LSTM hidden size (per direction).
            wave_hidden: Waveform LSTM hidden size (per direction).
            fusion_hidden: Hidden width of the fusion head.
            dropout: Dropout used throughout.
            channels: "all" (both branches), "catalog" (catalog only, the
                ablation), or "waveform" (waveform only).
        """
        super().__init__()
        self.channels = channels
        fused_dim = 0
        if channels in ("all", "catalog"):
            self.cat_branch = CatalogMLPBranch(catalog_dim, hidden=cat_hidden, dropout=dropout)
            fused_dim += self.cat_branch.out_dim
        if channels in ("all", "waveform"):
            self.wave_encoder = RawWaveformEncoder(out_dim=cnn_out, dropout=dropout)
            self.wave_branch = LSTMAttentionBranch(cnn_out, hidden=wave_hidden, dropout=dropout)
            fused_dim += self.wave_branch.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, wave_seq, cat_seq):
        """Forecasts from one batch of (waveform, catalog) window pairs.

        Args:
            wave_seq: Shape (batch, seq_hours, 3, hour_samples).
            cat_seq: Shape (batch, seq_hours, catalog_dim).

        Returns:
            Tensor of shape (batch,), raw logits.
        """
        parts = []
        if self.channels in ("all", "catalog"):
            parts.append(self.cat_branch(cat_seq))
        if self.channels in ("all", "waveform"):
            b, t = wave_seq.shape[:2]
            day_emb = self.wave_encoder(wave_seq.reshape(b * t, *wave_seq.shape[2:])).reshape(b, t, -1)
            parts.append(self.wave_branch(day_emb))
        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.head(fused).squeeze(-1)


def parse_args():
    """Parses CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--bg-min-mag", type=float, default=3.0,
                  help="Completeness threshold for the lower-magnitude 'background' catalog "
                       "used for the magnitude/energy/b-value/regularity features -- separate "
                       "from --threshold (the M>=threshold set that defines the label and "
                       "persistence floor). Lower threshold = more events = a more stable "
                       "b-value estimate.")
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--consolidated", action="store_true")
    p.add_argument("--label-mode", choices=["event", "rate"], default="event",
                  help="'event' (default): does an M>=--threshold event occur within "
                       "--horizon-days. Its positive class is driven by very few distinct "
                       "events (4 in fold 1 at M>=4.5), so the effective sample size is far "
                       "below the hour count. 'rate' (variant B): will the next "
                       "--horizon-days contain MORE M>=--rate-min-mag events than the "
                       "trailing window -- a rate/acceleration forecast in the ETAS/CSEP "
                       "tradition, driven by ~10^3 events instead of ~10^1.")
    p.add_argument("--rate-min-mag", type=float, default=3.0,
                  help="Magnitude threshold defining the rate in --label-mode rate. Distinct "
                       "from --threshold (rare-event label) and --bg-min-mag (features).")
    p.add_argument("--rate-baseline-days", type=float, default=None,
                  help="Trailing comparison window for --label-mode rate. Default: equal to "
                       "--horizon-days, so the comparison is like-for-like.")
    p.add_argument("--keep-features", nargs="+", default=None, metavar="FEATURE",
                  help="Restrict the catalog branch to this subset of features, by name from "
                       "FEATURE_NAMES (or ALL_FEATURE_NAMES when --rate-features is on) "
                       "instead of all of them -- e.g. an RFE-picked subset. Default: keep all.")
    p.add_argument("--rate-features", action="store_true",
                  help="Append trailing-rate count/ratio features over the M>=--rate-min-mag "
                       "catalog (windows 3/7/14/30/90d plus short-over-long ratios). Without "
                       "these the model has no trailing-rate signal at all -- the counts in "
                       "FEATURE_NAMES track M>=--threshold events -- so in --label-mode rate "
                       "it is trying to beat a persistence floor built from a number it was "
                       "never given.")
    p.add_argument("--channels", default="all", choices=["all", "catalog", "waveform"])
    p.add_argument("--seq-hours", type=int, default=24)
    p.add_argument("--cnn-out", type=int, default=32)
    p.add_argument("--cat-hidden", type=int, default=16)
    p.add_argument("--wave-hidden", type=int, default=16)
    p.add_argument("--fusion-hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=None,
                  help="Default: 0.2 for --channels catalog (a tiny 2-11-input MLP -- "
                       "aggressive dropout can knock it off a good basin instead of "
                       "regularizing it), 0.4 otherwise (the bigger CNN+LSTM fused model).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                  help="Default: 3e-4 for --channels catalog, 1e-3 otherwise (same reasoning "
                       "as --dropout -- a near-linear few-scalar-feature problem needs "
                       "gentler steps than the fused model does).")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--checkpoint-metric", choices=["auc", "loss"], default="auc",
                  help="Metric used to pick the best epoch's weights. Val AUC swings hard "
                       "epoch-to-epoch on the catalog branch (thin val positive class, ~335 "
                       "positives) while val loss stays comparatively smooth -- try 'loss' "
                       "there if AUC-based selection looks noisy.")
    p.add_argument("--ensemble-seeds", type=str, default="42,43,44")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--balanced-folds", action="store_true",
                  help="With --cv-folds > 1, place walk-forward block boundaries by equal "
                       "positive-label mass instead of equal hour count, so a single "
                       "sustained swarm cannot fill one block almost entirely. Boundaries "
                       "come from the label series alone, before training, and apply to "
                       "every fold -- distinct from selecting whichever fold scores best, "
                       "which would be test-set selection. Blocks stop being equal-width "
                       "in wall-clock time as a result.")
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="FOLD")
    return p.parse_args()


def train_one_seed(args, seed, raw, cat_features, labels, train_idx, val_idx, test_idx, device):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: Parsed CLI args.
        seed: Random seed for init/shuffling.
        raw: Hourly raw waveform array, shape (n_hours, 3, hour_samples).
        cat_features: Per-hour catalog feature array.
        labels: Per-hour binary labels.
        train_idx: Window end-indices for the training split.
        val_idx: Window end-indices for the validation split.
        test_idx: Window end-indices for the test split.
        device: torch device to train on.

    Returns:
        Tuple of (y_true, y_score) arrays for the test split, from the
        best (by val AUC) epoch's weights.
    """
    seed_everything(seed)
    train_ds = FusionSeqDataset(raw, cat_features, labels, args.seq_hours, train_idx)
    val_ds = FusionSeqDataset(raw, cat_features, labels, args.seq_hours, val_idx, stats=train_ds.stats)
    test_ds = FusionSeqDataset(raw, cat_features, labels, args.seq_hours, test_idx, stats=train_ds.stats)

    model = CatalogWaveformFusionNet(catalog_dim=cat_features.shape[1], cnn_out=args.cnn_out,
                                     cat_hidden=args.cat_hidden, wave_hidden=args.wave_hidden,
                                     fusion_hidden=args.fusion_hidden, dropout=args.dropout,
                                     channels=args.channels).to(device)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=2)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = labels[train_idx].mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for wave_seq, cat_seq, y in loader:
                wave_seq, cat_seq, y = wave_seq.to(device), cat_seq.to(device), y.to(device)
                logit = model(wave_seq, cat_seq)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    best = float("inf") if args.checkpoint_metric == "loss" else -1.0
    no_improve, best_state = 0, None
    for epoch in range(args.epochs):
        model.train()
        for wave_seq, cat_seq, y in train_loader:
            wave_seq, cat_seq, y = wave_seq.to(device), cat_seq.to(device), y.to(device)
            loss = criterion(model(wave_seq, cat_seq), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()

        yv, sv, val_loss = evaluate(val_loader)
        val_auc = safe_auc(yv, sv)
        print(f"  [seed {seed}] epoch {epoch+1}/{args.epochs} val AUC {val_auc:.4f} val loss {val_loss:.4f}")
        metric = val_loss if args.checkpoint_metric == "loss" else val_auc
        improved = metric < best if args.checkpoint_metric == "loss" else metric > best
        if improved:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = evaluate(test_loader)
    print(f"  [seed {seed}] test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def run_fold(fold_label, args, raw, cat_features, labels, dsp, hour_index, train_idx, val_idx, test_idx,
            seeds, device, rate_trailing=None):
    """Trains the seed ensemble on one split and reports it.

    Args:
        fold_label: Header string printed above this fold's report.
        args: Parsed CLI args, forwarded to `train_one_seed`.
        raw: Hourly raw waveform array.
        cat_features: Per-hour catalog feature array.
        labels: Per-hour binary labels.
        dsp: Days-since-previous-major-event array, for the persistence floor
            in `--label-mode event`. Ignored when `rate_trailing` is given.
        hour_index: DatetimeIndex of hour starts, for split diagnostics.
        train_idx: Window end-indices for the training split.
        val_idx: Window end-indices for the validation split.
        test_idx: Window end-indices for the test split.
        rate_trailing: Trailing-window event counts from
            `label_hours_rate_change`, present only in `--label-mode rate`.
            When given, the persistence floor is computed from the trailing
            rate instead of days-since-previous-event.
        seeds: List of random seeds to train and ensemble.
        device: torch device to train on.

    Returns:
        Tuple of (ensemble_auc, floor_auc, report_dict), or None if the
        split is too thin (fewer than 10 train or 5 test windows).
    """
    print(f"\n{'=' * 64}\n{fold_label} [channels={args.channels}]\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores = []
    yt_ref = None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, raw, cat_features, labels, train_idx, val_idx, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    ensemble_score = np.mean(per_seed_scores, axis=0)

    print("\n--- Floors (test set) ---")
    pos_tr = labels[train_idx].mean()
    base_pred = np.full_like(yt_ref, int(round(pos_tr)), dtype=np.float64)
    base_auc = safe_auc(yt_ref, base_pred)
    print(f"  base-rate (majority)   AUC {base_auc:.4f}   n={len(yt_ref)}")
    single_class = len(np.unique(yt_ref)) < 2
    if rate_trailing is not None:
        # Rate-change target: the trailing event count is the natural backward-looking
        # predictor, but it is ANTI-correlated with a rate-INCREASE label (Omori decay
        # -- busy now implies calmer next), so the achievable baseline is max(auc,
        # 1-auc), not auc. Predicting "increase iff currently quieter than the training
        # median" is the corresponding hard rule, used for the Brier term.
        pers_auc = rate_persistence_auc(yt_ref, rate_trailing[test_idx])
        thresh = np.median(rate_trailing[train_idx])
        pers_pred = (rate_trailing[test_idx] < thresh).astype(np.float64)
        if safe_auc(yt_ref, rate_trailing[test_idx].astype(np.float64)) > 0.5:
            pers_pred = 1.0 - pers_pred  # trailing rate positively correlated here
    else:
        pers_dsp = dsp[test_idx]
        pers_pred = np.where(np.isnan(pers_dsp), 0,
                             (pers_dsp <= args.horizon_days).astype(int)).astype(np.float64)
        pers_auc = safe_auc(yt_ref, pers_pred)
    pers_brier = float("nan") if single_class else float(brier_score_loss(yt_ref, pers_pred))
    print(f"  persistence             AUC {pers_auc:.4f}   Brier {pers_brier:.4f}   n={len(yt_ref)}")

    print(f"\n--- Catalog+Waveform Fusion [channels={args.channels}] ---")
    per_seed_aucs = [safe_auc(yt_ref, s) for s in per_seed_scores]
    print(f"  per-seed AUC: {[f'{a:.4f}' for a in per_seed_aucs]}  "
         f"mean {np.mean(per_seed_aucs):.4f}  spread {max(per_seed_aucs)-min(per_seed_aucs):.4f}")
    ensemble_auc = safe_auc(yt_ref, ensemble_score)
    print(f"  ENSEMBLE (mean of {len(seeds)} seeds' probabilities)   AUC {ensemble_auc:.4f}   n={len(yt_ref)}")

    floor = max(0.5, base_auc, pers_auc)
    report = binary_report(yt_ref, ensemble_score)
    bss = (float("nan") if (single_class or not np.isfinite(pers_brier) or pers_brier == 0)
          else 1.0 - report["brier"] / pers_brier)
    report["brier_skill_score_vs_persistence"] = bss
    print_report(f"Catalog+Waveform Fusion ensemble [{args.channels}] ({fold_label}, test set)", report)
    return ensemble_auc, floor, report


def main():
    """Loads the raw waveform archive/catalog, builds catalog+waveform
    features, and runs the fold sweep."""
    args = parse_args()
    if args.dropout is None:
        args.dropout = 0.2 if args.channels == "catalog" else 0.4
    if args.lr is None:
        args.lr = 3e-4 if args.channels == "catalog" else 1e-3

    print("Loading raw preprocessed waveform and building catalog+hourly labels...")
    if args.consolidated:
        hour_index, raw = load_hourly_raw_consolidated(args.data_root)
    else:
        hour_index, raw = load_hourly_raw(args.data_root, max_days=args.max_days)
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    bg_times, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag)
    # The NND precompute inside build_catalog_features is O(n_bg * NND_LOOKBACK)
    # haversine in a Python loop and fires whenever coordinates are supplied. Skip it
    # when this run's feature subset doesn't include a location-derived feature --
    # otherwise a 2-feature run pays the full cost for columns it then discards.
    if args.keep_features is not None and not any(
            f in args.keep_features for f in ("nnd_log_eta_90d", "shannon_entropy_90d")):
        bg_lats = bg_lons = None
    print(f"  {len(hour_index)} hourly raw vectors {raw.shape}, {len(major_times)} M>={args.threshold} "
         f"AEGEAN events in the full catalog, {len(bg_times)} M>={args.bg_min_mag} background events")

    hour_index, raw = truncate_to_reliable_catalog_end(hour_index, raw, major_times,
                                                        buffer_days=args.horizon_days)

    dsp = days_since_prev_major(hour_index, major_times)
    cat_features = build_catalog_features(hour_index, major_times, dsp, bg_times, bg_mags,
                                          args.bg_min_mag, bg_lats, bg_lons)

    # Rate features must be appended BEFORE --keep-features subsets, so the flag can
    # select them by name alongside the originals.
    rate_times = None
    active_names = FEATURE_NAMES
    if args.label_mode == "rate" or args.rate_features:
        rate_times, _, _, _ = load_aegean_events_with_location(args.catalog_path,
                                                               args.rate_min_mag)
    if args.rate_features:
        cat_features = np.hstack([cat_features, build_rate_features(hour_index, rate_times)])
        active_names = ALL_FEATURE_NAMES
        print(f"  + {len(RATE_FEATURE_NAMES)} trailing-rate features "
             f"(M>={args.rate_min_mag}, windows {RATE_WINDOWS}d)")

    if args.keep_features is not None:
        unknown = [f for f in args.keep_features if f not in active_names]
        if unknown:
            raise SystemExit(f"[ERROR] --keep-features got {unknown}, which are rate features; "
                             f"pass --rate-features to enable them.")
        keep_idx = [active_names.index(f) for f in args.keep_features]
        cat_features = cat_features[:, keep_idx]
        print(f"  restricting catalog branch to {len(keep_idx)} feature(s): {args.keep_features}")

    rate_trailing = None
    if args.label_mode == "rate":
        labels, fwd_counts, rate_trailing = label_hours_rate_change(
            hour_index, rate_times, args.horizon_days, args.rate_baseline_days)
        base = args.rate_baseline_days or args.horizon_days
        print(f"  [label-mode=rate] target: more M>={args.rate_min_mag} events in the next "
             f"{args.horizon_days:.0f}d than in the trailing {base:.0f}d")
        print(f"    {len(rate_times)} M>={args.rate_min_mag} events define the rate; forward "
             f"counts: median {int(np.median(fwd_counts))} mean {fwd_counts.mean():.1f} "
             f"max {fwd_counts.max()}")
        print(f"    trailing-rate persistence floor (Omori-inverted): "
             f"{rate_persistence_auc(labels, rate_trailing):.4f} -- this, not 0.5, is the bar")
    else:
        labels = label_hours(hour_index, major_times, args.horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    n = len(hour_index)
    valid_end_indices = np.arange(args.seq_hours - 1, n)
    # seq_hours-1 removes *input*-window overlap across a block boundary; the label
    # additionally looks horizon_days forward, so without the extra horizon term the
    # last ~horizon_days of each block carry labels determined by events inside the
    # NEXT block -- i.e. train labels encoding what happens in val, and val labels
    # encoding what happens in test (~9% of samples at horizon=14d, seq=24h). That's
    # the overlapping-label leakage that purging/embargo exists to prevent
    # (Lopez de Prado, Advances in Financial Machine Learning, Ch. 7).
    embargo = args.seq_hours - 1 + int(round(args.horizon_days * 24))

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train], valid_end_indices[i_train + embargo:i_val],
                 valid_end_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    elif args.balanced_folds:
        # Place block boundaries by equal positive-label MASS rather than equal hour
        # count, so one sustained swarm can't fill a block almost entirely. Decided
        # from the label series before any model runs and applied uniformly to every
        # fold -- this is not the same as picking whichever fold scores best, which
        # would be test-set selection.
        print("  [balanced-folds] block boundaries placed by positive-label mass, "
             "not equal hour count")
        folds = walk_forward_splits(valid_end_indices, args.cv_folds,
                                    labels=labels[valid_end_indices], embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]
        # Equal positive-MASS boundaries degenerate when the positive class is rare and
        # clustered: the blocks holding the clusters come out nearly all-positive. At
        # --label-mode event this produced a 0.999-positive test block (AUC there is
        # meaningless). Measured, not hypothetical -- so fail loudly rather than
        # reporting a number computed on a single-class split.
        for k, (_, va, te) in enumerate(folds, 1):
            for split_name, idx in (("val", va), ("test", te)):
                if len(idx) and not 0.02 <= labels[idx].mean() <= 0.98:
                    print(f"  [!] --balanced-folds made fold {k}'s {split_name} block "
                         f"{labels[idx].mean():.3f}-positive -- near-degenerate, so its AUC "
                         f"would be uninformative. Re-run without --balanced-folds (this "
                         f"flag suits balanced targets like --label-mode rate, not rare "
                         f"clustered ones).")
    else:
        folds = walk_forward_splits(valid_end_indices, args.cv_folds, embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(zip(fold_labels, folds), 1):
        if k in skip:
            continue
        result = run_fold(fold_label, args, raw, cat_features, labels, dsp, hour_index,
                          train_idx, val_idx, test_idx, seeds, device,
                          rate_trailing=rate_trailing)
        if result is not None:
            results.append(result)

    if args.cv_folds > 1 and results:
        aucs = np.array([r[0] for r in results])
        floors = np.array([r[1] for r in results])
        print(f"\n{'=' * 64}\nWalk-forward CV summary [{args.channels}] ({len(results)}/{args.cv_folds} "
             f"folds)\n{'=' * 64}")
        print(f"  ensemble AUC per fold: {[f'{a:.4f}' for a in aucs]}")
        print(f"  ensemble AUC:  mean {np.nanmean(aucs):.4f}  std {np.nanstd(aucs):.4f}")
        print(f"  floor AUC:     mean {floors.mean():.4f}  std {floors.std():.4f}")
        print(f"  beats its own fold's floor in {int((aucs > floors).sum())}/{len(results)} folds")


if __name__ == "__main__":
    main()
