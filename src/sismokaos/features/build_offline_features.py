"""
Offline feature builder for preprocessed Parquet waveforms.
Reads compressed Zstd Parquet row-groups sequentially to prevent OOM crashes,
computes DWT features, and merges catalog physics.
"""

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sismokaos.features.catalog_feature_processor import build_catalog_features
from sismokaos.features.waveform_dwt_features import (build_hourly_waveform_features,
                                            feature_names)
from sismokaos.catalog import (load_aegean_events,
                               truncate_to_reliable_catalog_end)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-parquet", required=True, help="Output from Rust --preprocess")
    p.add_argument("--catalog-path", required=True)
    p.add_argument("--out-path", required=True, help="Path to save the combined 114d parquet")
    p.add_argument("--hour-samples", type=int, default=18000, help="18000 for 5Hz, 36000 for 10Hz")
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--bg-min-mag", type=float, default=3.0)
    p.add_argument("--n-jobs", type=int, default=4, help="Cores for DWT extraction")
    return p.parse_args()

def stream_parquet_and_extract_dwt(parquet_path, hour_samples, n_jobs):
    print(f"Streaming Parquet row-groups from {parquet_path}...")
    
    pf = pq.ParquetFile(parquet_path)
    overflow_df = pd.DataFrame()
    dwt_rows = []
    hour_index_list = []
    
    for batch_idx, batch in enumerate(pf.iter_batches()):
        df = pd.concat([overflow_df, batch.to_pandas()], ignore_index=True)
        
        # Zaman_Dk is epoch minutes from Rust. Convert to absolute Datetime and floor to hour.
        df["hour_start"] = pd.to_datetime(df["Zaman_Dk"], unit="m").dt.floor("h")
        unique_hours = df["hour_start"].unique()
        
        if len(unique_hours) <= 1:
            overflow_df = df
            continue
            
        completed_hours = unique_hours[:-1]
        raw_hours = np.zeros((len(completed_hours), 3, hour_samples), dtype=np.float32)
        
        for i, h in enumerate(completed_hours):
            hour_data = df[df["hour_start"] == h]
            
            for c, comp in enumerate(["E", "N", "Z"]):
                if comp in hour_data.columns:
                    vals = hour_data[comp].to_numpy(dtype=np.float32)
                    
                    if len(vals) < hour_samples:
                        fixed = np.full(hour_samples, np.nan, dtype=np.float32)
                        fixed[:len(vals)] = vals
                        vals = fixed
                    else:
                        vals = vals[:hour_samples]
                        
                    raw_hours[i, c, :] = np.nan_to_num(vals, nan=0.0)
                    
            hour_index_list.append(h)
            
        # Execute parallel DWT on this batch of hours
        dwt_matrix = build_hourly_waveform_features(raw_hours, n_jobs=n_jobs)
        dwt_rows.append(dwt_matrix)
        
        overflow_df = df[df["hour_start"] == unique_hours[-1]].copy()
        print(f"  Processed through batch {batch_idx+1}/{pf.num_row_groups}...")

    # Flush final hour
    if not overflow_df.empty:
        h = overflow_df["hour_start"].iloc[0]
        raw_hours = np.zeros((1, 3, hour_samples), dtype=np.float32)
        for c, comp in enumerate(["E", "N", "Z"]):
            if comp in overflow_df.columns:
                vals = overflow_df[comp].to_numpy(dtype=np.float32)
                raw_hours[0, c, :len(vals)] = np.nan_to_num(vals, nan=0.0)
                
        hour_index_list.append(h)
        dwt_rows.append(build_hourly_waveform_features(raw_hours, n_jobs=1))

    return pd.DatetimeIndex(hour_index_list), np.concatenate(dwt_rows, axis=0)

def main():
    args = parse_args()
    
    hour_index, dwt_matrix = stream_parquet_and_extract_dwt(args.raw_parquet, args.hour_samples, args.n_jobs)
    
    dwt_cols = feature_names(n_channels=3)
    df_dwt = pd.DataFrame(dwt_matrix, index=hour_index, columns=dwt_cols).astype(np.float32)
    
    print("\nExtracting physics-based catalog features...")
    catalog_df = pd.read_csv(args.catalog_path)
    catalog_df["dt"] = pd.to_datetime(catalog_df["Date"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    
    df_cat = build_catalog_features(catalog_df, hour_index, Mc=args.bg_min_mag)
    
    print("Concatenating and saving final feature matrix...")
    df_combined = pd.concat([df_dwt, df_cat], axis=1)
    
    major_times = load_aegean_events(args.catalog_path, args.threshold)
    cutoff = major_times[-1] - pd.Timedelta(days=30)
    df_combined = df_combined[df_combined.index <= cutoff]
    
    df_combined.to_parquet(args.out_path, engine="pyarrow")
    print(f"Successfully saved {df_combined.shape[1]} features to {args.out_path}")

if __name__ == "__main__":
    main()
