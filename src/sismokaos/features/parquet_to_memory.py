"""
Converts a preprocessed Parquet time-series into a flat binary memmap.
Structures data into an (N_hours, 3_channels, hour_samples) tensor for 
instantaneous, O(1) disk-read access in PyTorch DataLoaders.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-parquet", required=True, help="Input from Rust --preprocess")
    p.add_argument("--out-bin", required=True, help="Path for the output .dat binary file")
    p.add_argument("--hour-samples", type=int, default=18000, help="18000 for 5Hz, 36000 for 10Hz")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Streaming {args.raw_parquet} to binary memmap format...")

    pf = pq.ParquetFile(args.raw_parquet)
    overflow_df = pd.DataFrame()
    
    n_hours = 0
    
    # Open a raw binary file for continuous byte-appending
    with open(args.out_bin, 'wb') as f_out:
        for batch_idx, batch in enumerate(pf.iter_batches()):
            df = pd.concat([overflow_df, batch.to_pandas()], ignore_index=True)
            
            # Reconstruct the hourly boundaries
            df["hour_start"] = pd.to_datetime(df["Zaman_Dk"], unit="m").dt.floor("h")
            unique_hours = df["hour_start"].unique()
            
            if len(unique_hours) <= 1:
                overflow_df = df
                continue
                
            completed_hours = unique_hours[:-1]
            
            # Pre-allocate a contiguous C-order array for this batch of hours
            raw_hours = np.zeros((len(completed_hours), 3, args.hour_samples), dtype=np.float32)
            
            for i, h in enumerate(completed_hours):
                hour_data = df[df["hour_start"] == h]
                
                for c, comp in enumerate(["E", "N", "Z"]):
                    if comp in hour_data.columns:
                        vals = hour_data[comp].to_numpy(dtype=np.float32)
                        if len(vals) < args.hour_samples:
                            raw_hours[i, c, :len(vals)] = vals
                        else:
                            raw_hours[i, c, :] = vals[:args.hour_samples]
                            
                # Sanitize any lingering NaNs to 0.0 to prevent PyTorch gradient explosions
                np.nan_to_num(raw_hours[i], copy=False, nan=0.0)
            
            # Dump the raw memory buffer straight to the disk
            f_out.write(raw_hours.tobytes())
            n_hours += len(completed_hours)
            
            overflow_df = df[df["hour_start"] == unique_hours[-1]].copy()
            print(f"  Processed batch {batch_idx+1}/{pf.num_row_groups}. Hours written: {n_hours}", end="\r")
    
        # Handle the final overflow hour at the end of the file
        if not overflow_df.empty:
            raw_hours = np.zeros((1, 3, args.hour_samples), dtype=np.float32)
            for c, comp in enumerate(["E", "N", "Z"]):
                if comp in overflow_df.columns:
                    vals = overflow_df[comp].to_numpy(dtype=np.float32)
                    if len(vals) < args.hour_samples:
                        raw_hours[0, c, :len(vals)] = vals
                    else:
                        raw_hours[0, c, :] = vals[:args.hour_samples]
            np.nan_to_num(raw_hours, copy=False, nan=0.0)
            
            f_out.write(raw_hours.tobytes())
            n_hours += 1

    print(f"\nDone! Wrote {n_hours} hours to {args.out_bin}")
    
    # Save the shape metadata so np.memmap knows exactly how to read the binary blob
    meta = {
        "dtype": "float32",
        "shape": (n_hours, 3, args.hour_samples)
    }
    meta_path = Path(args.out_bin).with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
        
    print(f"Saved memmap metadata to {meta_path}")

if __name__ == "__main__":
    main()
