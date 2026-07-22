import numpy as np
from obspy import read
from PIL import Image


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """z-score standardization."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < eps:
        sigma = eps
    return (x - mu) / sigma


def reshape_to_d_by_n(x: np.ndarray, d: int) -> np.ndarray:
    """
    Rearrange 1D standardized series into matrix M of shape (d, n), n=ceil(m/d),
    filling column-wise as in RAM paper.
    If m is not divisible by d, wrap-around fill.
    """
    m = len(x)
    n = int(np.ceil(m / d))
    M = np.empty((d, n), dtype=np.float64)

    # Fill M[row, col] with x[(col*d + row) % m]
    for col in range(n):
        base = col * d
        for row in range(d):
            idx = (base + row) % m
            M[row, col] = x[idx]

    return M


def ram_matrix(x: np.ndarray, d: int, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Relative Angle Matrix (RAM) from 1D signal.
    Returns R of shape (n, n), where n=ceil(m/d).
    """
    x_std = standardize(x, eps=eps)
    M = reshape_to_d_by_n(x_std, d=d)  # M = [X1, X2, ..., Xn], Xi is each column
    X_cols = M  # shape (d, n)

    # Central feature vector Xbar = mean of columns
    Xbar = np.mean(X_cols, axis=1)  # shape (d,)
    norm_Xbar = np.linalg.norm(Xbar)
    if norm_Xbar < eps:
        norm_Xbar = eps

    # beta_i = arccos( (Xi . Xbar) / (||Xi|| ||Xbar||) )
    n = X_cols.shape[1]
    betas = np.empty(n, dtype=np.float64)

    for i in range(n):
        Xi = X_cols[:, i]
        norm_Xi = np.linalg.norm(Xi)
        if norm_Xi < eps:
            norm_Xi = eps
        cos_val = np.dot(Xi, Xbar) / (norm_Xi * norm_Xbar)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        betas[i] = np.arccos(cos_val)

    # R_ij = beta_j - beta_i
    R = betas[None, :] - betas[:, None]  # shape (n, n)
    return R


def to_uint8(mat: np.ndarray) -> np.ndarray:
    """
    Min-max normalize matrix to [0,255] uint8.
    """
    mat = np.asarray(mat, dtype=np.float64)
    mn, mx = np.min(mat), np.max(mat)
    if np.isclose(mx, mn):
        return np.zeros(mat.shape, dtype=np.uint8)
    out = (mat - mn) / (mx - mn)
    out = (out * 255.0).round().astype(np.uint8)
    return out


def select_three_traces(stream):
    """
    Pick exactly 3 traces from stream.
    Strategy:
      1) If exactly 3 traces, use them.
      2) Else choose first 3 after sorting by id.
    """
    if len(stream) < 3:
        raise ValueError(f"Need at least 3 traces, got {len(stream)}")
    st = stream.copy().sort(keys=["network", "station", "location", "channel"])
    return st[:3]


def align_trim_resample(traces, target_fs=None):
    """
    Align 3 traces to common time window and sampling rate.
    - Trim to overlapping interval.
    - Resample to target_fs if provided; else to minimum fs among traces.
    """
    # Common overlap
    start = max(tr.stats.starttime for tr in traces)
    end = min(tr.stats.endtime for tr in traces)
    if end <= start:
        raise ValueError("No overlapping time window among 3 traces.")

    aligned = []
    for tr in traces:
        t = tr.copy().trim(starttime=start, endtime=end, pad=False)
        aligned.append(t)

    # Choose sampling rate
    if target_fs is None:
        target_fs = min(tr.stats.sampling_rate for tr in aligned)

    # Resample all to target_fs
    out = []
    for tr in aligned:
        tc = tr.copy()
        if abs(tc.stats.sampling_rate - target_fs) > 1e-6:
            tc.resample(target_fs)
        out.append(tc)

    # Ensure equal length after resample
    min_len = min(len(tr.data) for tr in out)
    for tr in out:
        tr.data = tr.data[:min_len]

    return out


def mseed_3ch_to_ram_rgb(
    mseed_path: str,
    out_png: str,
    d: int = 64,
    target_fs: float = None
):
    """
    Convert 3-channel MiniSEED to 3-band (RGB) 8-bit image using RAM transform.
    """
    st = read(mseed_path)
    traces = select_three_traces(st)
    traces = align_trim_resample(traces, target_fs=target_fs)

    # RAM per channel
    ram_u8_channels = []
    for tr in traces:
        sig = tr.data.astype(np.float64)
        R = ram_matrix(sig, d=d)
        R_u8 = to_uint8(R)
        ram_u8_channels.append(R_u8)

    # Ensure same shape
    h, w = ram_u8_channels[0].shape
    for c in ram_u8_channels[1:]:
        if c.shape != (h, w):
            raise RuntimeError("RAM channel shapes differ; check preprocessing.")

    # Stack as RGB: channel order follows selected trace order
    rgb = np.stack(ram_u8_channels, axis=-1)  # (H, W, 3), uint8
    img = Image.fromarray(rgb, mode="RGB")
    img.save(out_png)

    print(f"Saved RGB RAM image: {out_png}")
    for i, tr in enumerate(traces):
        print(f"Band {i} <- {tr.id}, fs={tr.stats.sampling_rate}, n={len(tr.data)}")
    print(f"Image shape: {rgb.shape}")


if __name__ == "__main__":
    # Example usage:
    # python ram_mseed_to_rgb.py
    # (edit paths below)
    mseed_path = "input.mseed"
    out_png = "ram_rgb.png"

    # d controls local vector size and resulting image size n x n where n=ceil(m/d)
    # Larger d -> smaller n (smaller image), smaller d -> larger image.
    mseed_3ch_to_ram_rgb(
        mseed_path=mseed_path,
        out_png=out_png,
        d=64,
        target_fs=None,  # e.g., 100.0 to force 100 Hz
    )
