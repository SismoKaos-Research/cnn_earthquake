"""
Extracts rolling physical features from a seismic catalog.
Designed to run alongside continuous waveform features (DWT/CNN).

Features per hourly window (trailing 7, 30, and 90 days):
  - Seismicity Rate (Count)
  - Seismic Moment Release (Joules / N·m)
  - Maximum Magnitude
  - Gutenberg-Richter b-value (Aki-Utsu MLE)
"""

import numpy as np
import pandas as pd


def build_catalog_features(catalog_df: pd.DataFrame, 
                           hourly_index: pd.DatetimeIndex, 
                           windows_days: list = [7, 30, 90], 
                           Mc: float = 2.5,
                           min_b_events: int = 25) -> pd.DataFrame:
    """
    Computes rolling seismological features mapped to a specific hourly timeline.
    
    Args:
        catalog_df: DataFrame containing at least 'dt' (datetime) and 'Magnitude'.
        hourly_index: The DatetimeIndex (your continuous feature timeline) to align to.
        windows_days: Trailing windows in days to compute features over.
        Mc: Magnitude of completeness. Events below this are excluded from stats.
        min_b_events: Minimum events required in a window to compute a stable b-value.
        
    Returns:
        DataFrame of catalog features, indexed identically to `hourly_index`.
    """
    # Isolate complete events and sort chronologically
    cat = catalog_df.dropna(subset=['dt', 'Magnitude']).sort_values("dt")
    cat = cat[cat["Magnitude"] >= Mc].copy()
    
    times = cat["dt"].to_numpy()
    mags = cat["Magnitude"].to_numpy()
    
    # Calculate Seismic Moment (M0) in N·m using standard Hanks & Kanamori formula
    moments = np.power(10, 1.5 * mags + 9.1)
    
    hours = hourly_index.to_numpy()
    out_features = {}
    
    for w_days in windows_days:
        w_sec = np.timedelta64(w_days, 'D')
        
        rates = np.zeros(len(hours), dtype=np.float32)
        moment_sums = np.zeros(len(hours), dtype=np.float32)
        max_mags = np.zeros(len(hours), dtype=np.float32)
        b_values = np.full(len(hours), np.nan, dtype=np.float32)
        
        # O(N log M) vectorized boundary search
        right_idx = np.searchsorted(times, hours, side='right')
        left_idx = np.searchsorted(times, hours - w_sec, side='right')
        
        for i in range(len(hours)):
            l, r = left_idx[i], right_idx[i]
            rates[i] = r - l
            
            if r > l:
                window_mags = mags[l:r]
                moment_sums[i] = np.sum(moments[l:r])
                max_mags[i] = np.max(window_mags)
                
                # Aki-Utsu Maximum Likelihood Estimate for b-value
                # Includes the 0.05 correction for standard 0.1 magnitude binning
                if len(window_mags) >= min_b_events:
                    mean_mag = np.mean(window_mags)
                    b = np.log10(np.e) / (mean_mag - (Mc - 0.05))
                    b_values[i] = b
                    
        out_features[f"cat_rate_{w_days}d"] = rates
        out_features[f"cat_moment_{w_days}d"] = np.log10(moment_sums + 1) # Log-scaled for neural net stability
        out_features[f"cat_max_mag_{w_days}d"] = max_mags
        out_features[f"cat_b_value_{w_days}d"] = b_values
        
    df_out = pd.DataFrame(out_features, index=hourly_index)
    
    # Fill NaN b-values (windows with too few events) with the regional global average ~1.0
    # Alternatively, you can ffill() to carry forward the last stable b-value
    b_cols = [c for c in df_out.columns if "b_value" in c]
    df_out[b_cols] = df_out[b_cols].fillna(1.0)
    
    return df_out
