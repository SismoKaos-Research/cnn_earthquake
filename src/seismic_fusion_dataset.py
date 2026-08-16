import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

OPTIMIZED_CATALOG_FEATURES = [
    "cat_moment_7d",
    "cat_rate_30d",
    "cat_rate_90d"
]

class SeismicFusionDataset(Dataset):
    def __init__(self, catalog_parquet_path, raw_waveform_memmap, labels, seq_len=24):
        """
        Args:
            catalog_parquet_path: Path to the combined Parquet file.
            raw_waveform_memmap: Memory-mapped numpy array of raw E/N/Z waveforms.
                                 Shape: (N_hours, 3, hour_samples).
            labels: 1D array of binary labels aligned with the hours.
            seq_len: Number of hours to look back for the GRU/CNN sequence.
        """
        df_cat = pd.read_parquet(catalog_parquet_path, columns=OPTIMIZED_CATALOG_FEATURES)
        cat_array = df_cat.to_numpy(dtype=np.float32)
        
        self.cat_mean = cat_array.mean(axis=0)
        self.cat_std = cat_array.std(axis=0) + 1e-8
        normalized_cat = (cat_array - self.cat_mean) / self.cat_std
        
        self.catalog_x = torch.tensor(normalized_cat, dtype=torch.float32)
        
        self.raw_x = raw_waveform_memmap 
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.catalog_x) - self.seq_len
        
    def __getitem__(self, idx):
        end_idx = idx + self.seq_len
        
        cat_seq = self.catalog_x[idx:end_idx]
        
        wave_seq = torch.tensor(np.array(self.raw_x[idx:end_idx]), dtype=torch.float32)
        
        label = self.labels[end_idx - 1]
        
        return cat_seq, wave_seq, label
