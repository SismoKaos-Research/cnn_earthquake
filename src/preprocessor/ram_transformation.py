import math
import os

import numpy as np
from obspy import read
from PIL import Image


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Applies z-score standardization to a 1D numerical array.
    
    Centers the data around a mean of 0 and a standard deviation of 1. 
    Prevents division by zero in cases of flatlined signals.

    Args:
        x (np.ndarray): The 1D input array (e.g., raw trace data).
        eps (float, optional): A small epsilon value to prevent division by zero. Defaults to 1e-12.

    Returns:
        np.ndarray: The standardized 1D array of type np.float64.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < eps:
        sigma = eps
    return (x - mu) / sigma


def reshape_to_d_by_n(x: np.ndarray, d: int) -> np.ndarray:
    """
    Rearranges a 1D standardized series into a 2D matrix filled column-wise.
    
    If the length of the array is not perfectly divisible by `d`, the function 
    wraps around to the beginning of the array to fill the remaining slots.

    Args:
        x (np.ndarray): The 1D standardized time-series array.
        d (int): The number of rows for the target matrix. Controls the local 
                 feature vector size.

    Returns:
        np.ndarray: A 2D matrix of shape (d, n) where n = ceil(len(x) / d).
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
    Computes the Relative Angle Matrix (RAM) from a 1D signal.
    
    Transforms a 1D time-series into a 2D spatial representation by calculating 
    the cosine angle differences between local feature vectors (columns) and the 
    central feature vector (mean of all columns).

    Args:
        x (np.ndarray): The raw 1D input signal.
        d (int): The row dimension used for intermediate reshaping. 
                 Larger d results in a smaller output image, and vice versa.
        eps (float, optional): Epsilon for numerical stability. Defaults to 1e-12.

    Returns:
        np.ndarray: A square 2D Relative Angle Matrix of shape (n, n), 
                    where n = ceil(len(x) / d).
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
    Min-max normalizes a 2D float matrix into an 8-bit unsigned integer matrix.
    
    Converts mathematical matrices into a format suitable for standard RGB image 
    generation (pixel values between 0 and 255).

    Args:
        mat (np.ndarray): The 2D float matrix (e.g., the output of ram_matrix).

    Returns:
        np.ndarray: The normalized matrix with dtype np.uint8.
    """
    mat = np.asarray(mat, dtype=np.float64)
    mn, mx = np.min(mat), np.max(mat)
    if np.isclose(mx, mn):
        return np.zeros(mat.shape, dtype=np.uint8)
    out = (mat - mn) / (mx - mn)
    out = (out * 255.0).round().astype(np.uint8)
    return out


def select_traces(stream):
    """
    Safely selects between 1 and 3 orthogonal traces from an ObsPy Stream.
    
    Attempts to explicitly grab components Z, N (or 1), and E (or 2) based on 
    the last character of the channel name. Avoids IndexErrors if a specific 
    component is offline or missing.

    Args:
        stream (obspy.core.Stream): The raw input stream containing seismic traces.

    Raises:
        ValueError: If the stream contains zero traces.

    Returns:
        list: A list of obspy.core.Trace objects (length 1 to 3).
    """ 
    if len(stream) < 1:
        raise ValueError(f"Need at least 1 trace, got {len(stream)}")
    
    selected_traces = []
    
    st_z = stream.select(component="Z")
    if st_z:
        selected_traces.append(st_z[0])
        
    st_n = stream.select(component="N") or stream.select(component="1")
    if st_n:
        selected_traces.append(st_n[0])
        
    st_e = stream.select(component="E") or stream.select(component="2")
    if st_e:
        selected_traces.append(st_e[0])
        
    if not selected_traces:
        print("Warning: Standard Z, N, E (or 1, 2) components not found. Falling back to sort.")
        st = stream.copy().sort(keys=["network", "station", "location", "channel"])
        return st[:3]
        
    return selected_traces



def process_and_window_traces(traces, target_fs=100.0, window_seconds=60.0, overlap=0.5, freqmin=1.0, freqmax=45.0):
    """
    Cleans, resamples, and chunks traces into sliding windows of strict lengths.
    
    Traces are processed independently without time-alignment constraints. The 
    function ensures that no signal is left unprocessed by zero-padding the 
    final chunk if it falls short of the target sample length.

    Args:
        traces (list): List of obspy.core.Trace objects.
        target_fs (float, optional): The target sampling rate in Hz. Defaults to 100.0.
        window_seconds (float, optional): The duration of each windowed chunk. Defaults to 60.0.
        overlap (float, optional): Fraction of the window to overlap (0.0 to <1.0). Defaults to 0.5 (50%).
        freqmin (float, optional): Lower bound for bandpass filter. Defaults to 1.0.
        freqmax (float, optional): Upper bound for bandpass filter. Defaults to 45.0.

    Raises:
        ValueError: If the overlap parameter requires a step size smaller than 1 sample.

    Returns:
        list of lists: A nested list where each inner list contains the windowed 
                       sub-traces for one of the original input traces.
                       Example: [[Z_win1, Z_win2], [N_win1, N_win2]]
    """
    windowed_traces_by_component = []
    
    for tr in traces:
        # Clean the signal before trimming
        tc = tr.copy()
        tc.detrend("linear")
        tc.detrend("demean")
        tc.taper(max_percentage=0.05, type="hann")
        
        if tc.stats.sampling_rate / 2 > freqmax:
            tc.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

        # Resample to target_fs
        if abs(tc.stats.sampling_rate - target_fs) > 1e-6:
            tc.interpolate(sampling_rate=target_fs, method="lanczos", a=20)

        # Windowing constraints
        target_samples = int(target_fs * window_seconds)
        step_samples = int(target_samples * (1.0 - overlap))
        
        if step_samples < 1:
            raise ValueError("Overlap fraction too high; step size must be at least 1 sample.")
            
        data = tc.data
        n_samples = len(data)
        
        # Calculate how many windows are needed to cover the entire trace
        if n_samples <= target_samples:
            n_windows = 1
        else:
            n_windows = math.ceil((n_samples - target_samples) / step_samples) + 1

        # Slide window across the individual trace
        trace_windows = []
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + target_samples
            
            window_data = data[start_idx:end_idx]
            
            # Enforce STRICT sample length (Zero-pad the final window if it falls short)
            if len(window_data) < target_samples:
                pad_length = target_samples - len(window_data)
                window_data = np.pad(window_data, (0, pad_length), mode='constant')
            
            # Create the sub-trace
            win_tr = tc.copy()
            win_tr.data = window_data
            # Accurately update the start time for this specific window
            win_tr.stats.starttime = tc.stats.starttime + (start_idx / target_fs)
            
            trace_windows.append(win_tr)
            
        windowed_traces_by_component.append(trace_windows)

    return windowed_traces_by_component


def mseed_to_ram_rgb(
    mseed_path: str,
    out_png: str,
    d: int = 64,
    target_fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5
):
    """
    Converts an up-to-3-channel MiniSEED file into multiple RGB RAM images.
    
    Generates an image for each sliding window step. Ensures consistent channel-to-color 
    mapping (Red=Z, Green=N/1, Blue=E/2). Missing channels or exhausted windows 
    are dynamically zero-padded (rendered as black pixels) to strictly maintain 
    an (H, W, 3) image shape.

    Args:
        mseed_path (str): Filepath to the input .mseed file.
        out_png (str): Base filepath for the output image. Will be appended with 
                       sequential window IDs (e.g., base_win000.png).
        d (int, optional): RAM matrix dimension controller. Defaults to 64.
        target_fs (float, optional): Target sampling rate. Defaults to 100.0.
        window_seconds (float, optional): Length of each data chunk. Defaults to 60.0.
        overlap (float, optional): Sliding window overlap fraction. Defaults to 0.5.

    Raises:
        RuntimeError: If no valid traces exist to calculate a base matrix shape.
    """
    st = read(mseed_path)
    
    traces = select_traces(st)
    # traces_windowed is a list of lists: e.g., [ [Z_win1, Z_win2], [N_win1, N_win2] ]
    traces_windowed = process_and_window_traces(
        traces, target_fs=target_fs, window_seconds=window_seconds, overlap=overlap
    )

    # Initialize 3 slots for R, G, B. Each slot will hold a LIST of windowed traces.
    rgb_window_slots = [None, None, None]
    
    for tr_list in traces_windowed:
        if not tr_list:
            continue
            
        # Determine mapping based on the first window's channel code
        comp = tr_list[0].stats.channel[-1].upper()
        idx = -1
        
        if comp == 'Z':
            idx = 0  # Red
        elif comp in ['N', '1']:
            idx = 1  # Green
        elif comp in ['E', '2']:
            idx = 2  # Blue
            
        # Place the entire list of windows into the correct slot
        if idx != -1 and rgb_window_slots[idx] is None:
            rgb_window_slots[idx] = tr_list
        else:
            for i in range(3):
                if rgb_window_slots[i] is None:
                    rgb_window_slots[i] = tr_list
                    break

    # Find the maximum number of windows generated across any available channel
    max_windows = 0
    for slot in rgb_window_slots:
        if slot is not None:
            max_windows = max(max_windows, len(slot))
            
    if max_windows == 0:
        raise RuntimeError("No valid traces were processed to generate an image.")

    base_name, ext = os.path.splitext(out_png)
    band_names = ["Red (Z)", "Green (N/1)", "Blue (E/2)"]

    # Generate an image for each window index
    for w_idx in range(max_windows):
        ram_u8_channels = []
        valid_shape = None
        current_traces = [None, None, None]
        
        # First pass: Process available traces to find the RAM matrix shape
        for i in range(3):
            if rgb_window_slots[i] is not None and w_idx < len(rgb_window_slots[i]):
                tr = rgb_window_slots[i][w_idx]
                current_traces[i] = tr
                sig = tr.data.astype(np.float64)
                R = ram_matrix(sig, d=d)
                R_u8 = to_uint8(R)
                ram_u8_channels.append(R_u8)
                
                if valid_shape is None:
                    valid_shape = R_u8.shape
            else:
                # Placeholder, will be replaced with zeros once we know the shape
                ram_u8_channels.append(None)
                
        # Second pass: Pad missing or exhausted channels with zeros
        for i in range(3):
            if ram_u8_channels[i] is None:
                ram_u8_channels[i] = np.zeros(valid_shape, dtype=np.uint8)
                
        # Stack and save
        rgb = np.stack(ram_u8_channels, axis=-1)  # (H, W, 3)
        img = Image.fromarray(rgb, mode="RGB")
        
        # Append window index to the filename (e.g., ram_rgb_win000.png)
        out_filename = f"{base_name}_win{w_idx:03d}{ext}"
        img.save(out_filename)
        
        # Print stats for this specific image
        print(f"\n--- Saved {out_filename} ---")
        for i in range(3):
            tr = current_traces[i]
            if tr is not None:
                print(f"{band_names[i]} <- {tr.id} (Start: {tr.stats.starttime})")
            else:
                print(f"{band_names[i]} <- None (Zero-padded)")


if __name__ == "__main__":
    mseed_path = "input.mseed"
    out_png = "ram_rgb.png"

    # d controls local vector size and resulting image size n x n where n=ceil(m/d)
    # Larger d -> smaller n (smaller image), smaller d -> larger image.
    mseed_to_ram_rgb(
        mseed_path=mseed_path,
        out_png=out_png,
        d=64,
        target_fs=100.0,  # e.g., 100.0 to force 100 Hz
    )
