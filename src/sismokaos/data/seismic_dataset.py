import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeismicDataset(Dataset):
    """
    Unified dataset for seismic tasks (classification, regression, forecast).
    Handles dual-channel (seq, img) and single-channel inputs, with optional
    auxiliary scalars.
    """

    def __init__(
        self,
        samples: List[Union[Path, str, Tuple[Union[Path, str], int]]],
        root_dir: Optional[Path] = None,
        mode: str = "classification",
        targets: Optional[np.ndarray] = None,
        aux: Optional[np.ndarray] = None,
        transform=None,
        normalization_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Args:
            samples: List of paths to .pt files, or list of tuples (path, label).
            root_dir: Root directory for samples if paths are relative.
            mode: "classification" or "regression".
            targets: Array of targets (labels or continuous values).
            aux: Optional array of auxiliary scalar features.
            transform: Optional transform to apply to images/sequences.
            normalization_stats: Dict of (mean, std) for "seq" and "aux".
        """
        self.samples = samples
        self.root_dir = Path(root_dir) if root_dir else None
        self.mode = mode
        self.targets = targets
        self.aux = aux
        self.transform = transform
        self.normalization_stats = normalization_stats or {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, path: Union[Path, str]) -> Dict[str, torch.Tensor]:
        full_path = self.root_dir / path if self.root_dir else Path(path)
        return torch.load(full_path, weights_only=True)

    def __getitem__(self, i: int) -> Tuple:
        sample = self.samples[i]
        if isinstance(sample, tuple):
            path, label = sample
        else:
            path, label = sample, (self.targets[i] if self.targets is not None else None)

        data = self._load_tensor(path)
        
        # Extract components
        seq = data.get("seq")
        img = data.get("img")
        
        # If the .pt was saved by an older version, it might be the tensor directly
        if seq is None and img is None:
            if isinstance(data, dict) and "img" in data:
                 img = data["img"]
            elif isinstance(data, dict) and "seq" in data:
                 seq = data["seq"]
            else:
                 # Assume it's a single tensor (legacy RAM or Spectrogram)
                 img = data if torch.is_tensor(data) else None

        # Apply normalization to seq if stats exist
        if seq is not None and "seq" in self.normalization_stats:
            mu, sd = self.normalization_stats["seq"]
            seq = (seq.numpy() - mu) / sd
            seq = torch.from_numpy(np.nan_to_num(seq, nan=0.0)).float()
        elif seq is not None:
            seq = seq.float()

        if img is not None:
            img = img.float()

        # Handle aux
        aux_val = None
        if self.aux is not None:
            aux_val = torch.tensor(self.aux[i], dtype=torch.float32)
        elif "aux" in data:
            aux_val = data["aux"]
            if "aux" in self.normalization_stats:
                mu, sd = self.normalization_stats["aux"]
                aux_val = (aux_val.numpy() - mu) / sd
                aux_val = torch.from_numpy(np.nan_to_num(aux_val, nan=0.0)).float()
            else:
                aux_val = aux_val.float()

        # Construct return tuple based on what's available
        # Order: seq, img, aux, target
        out = []
        if seq is not None: out.append(seq)
        if img is not None: out.append(img)
        if aux_val is not None: out.append(aux_val)
        
        target_dtype = torch.long if self.mode == "classification" else torch.float32
        out.append(torch.tensor(label, dtype=target_dtype) if label is not None else None)
        
        return tuple(out)


def resplit_manifest(
    df: pd.DataFrame,
    how: str = "event",
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> pd.DataFrame:
    """
    Centralized splitting logic for manifests.
    'how': 'event' (disjoint events), 'station' (disjoint stations), 'both'.
    """
    d = df.copy()
    if "file_split" not in d:
        d["file_split"] = d["split"] if "split" in d else "all"
        
    if how == "event" or "station_key" not in d.columns:
        return d

    rng = random.Random(seed)
    stations = sorted(set(d.station_key))
    rng.shuffle(stations)
    
    size = d.station_key.value_counts().to_dict()
    total = len(d)
    targets = {s: r * total for s, r in zip(("train", "val", "test"), ratios)}
    running = {s: 0 for s in targets}
    assign = {}
    
    for st in stations:
        # Assign to the split furthest from its target ratio
        best = max(targets, key=lambda s: (targets[s] - running[s]) / max(targets[s], 1.0))
        assign[st] = best
        running[best] += size[st]
        
    d["split"] = d.station_key.map(assign)

    if how == "both" and "event_id" in d.columns:
        train_events = set(d.loc[d.split == "train", "event_id"])
        clash = (d.split != "train") & d.event_id.isin(train_events)
        print(f"[split] doubly-disjoint: dropping {int(clash.sum())} rows")
        d = d[~clash].copy()
        
    return d
