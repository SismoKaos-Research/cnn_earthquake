"""Which horizons give evaluable folds? Label composition only -- no training."""
import numpy as np, pandas as pd
from sismokaos.catalog import load_hourly_features, load_aegean_events, label_hours
C="/home/hogib/Projects/Sismokaos/seismic_cli/catalogs/"
FE="/home/hogib/Projects/Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy"
h=load_hourly_features(FE); idx=pd.DatetimeIndex(h.index)
print(f"{len(idx)} hours, {idx.min():%Y-%m-%d} .. {idx.max():%Y-%m-%d}\n")
print(f"{'horizon':>9s} {'cat':>5s} {'pos rate':>9s}   per-fold TEST positive rate (5 chronological folds)")
for tag,cat in (("old",C+"archive_superseded_2026-08-30/data_large.csv"),
                ("new",C+"catalog_current.csv")):
    ev=load_aegean_events(cat, min_magnitude=4.5)
    for hd in (0.25, 1, 3, 7, 14, 30):
        y=label_hours(idx, ev, hd)
        n=len(y); fold=n//6   # walk-forward-ish: expanding train, fixed test blocks
        rates=[]
        for k in range(5):
            lo=n-(5-k)*fold; hi=lo+fold
            rates.append(y[lo:hi].mean())
        bad=sum(1 for r in rates if r in (0.0,1.0) or r<0.02 or r>0.98)
        print(f"{hd:8.2f}d {tag:>5s} {y.mean():9.3f}   "
              + " ".join(f"{r:.2f}" for r in rates)
              + (f"   <-- {bad} degenerate" if bad else "   all evaluable"))
    print()
