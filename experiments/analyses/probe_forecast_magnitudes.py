"""Magnitude x horizon: which cells give evaluable folds AND enough episodes?"""
import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from seismolib.catalog import load_hourly_features, load_aegean_events, label_hours
C="/home/hogib/Projects/Sismokaos/data_downloader/catalogs/catalog_current.csv"
FE="/home/hogib/Projects/Sismokaos/feature-extract/results/BODT/BODT_2024_05_01-2026_08_10_ENZ_features.npy"
h=load_hourly_features(FE); idx=pd.DatetimeIndex(h.index)
T0,T1=idx.min().to_datetime64(), idx.max().to_datetime64()
n=len(idx); fold=n//6

def decluster(ev, days=3.0):
    """Crude: collapse events within `days` of the previous kept one."""
    keep=[]; last=None
    for t in np.sort(ev):
        if last is None or (t-last)/np.timedelta64(1,'D') > days:
            keep.append(t); last=t
    return np.array(keep)

print(f"{'M>=':>5s} {'events':>7s} {'indep':>6s} | " + " ".join(f"{h:>5.0f}d" for h in (3,7,14,30)))
print(f"{'':>5s} {'in win':>7s} {'(3d)':>6s} | " + " ".join(f"{'ok/5':>6s}" for _ in range(4)))
for mag in (4.5, 4.0, 3.5, 3.0, 2.5):
    ev=np.asarray(load_aegean_events(C,min_magnitude=mag))
    inw=ev[(ev>=T0)&(ev<T1)]
    ind=decluster(inw)
    cells=[]
    for hd in (3,7,14,30):
        y=label_hours(idx, ev, hd)
        rates=[y[n-(5-k)*fold:n-(5-k)*fold+fold].mean() for k in range(5)]
        ok=sum(1 for r in rates if 0.05 <= r <= 0.95)
        cells.append(f"{ok:>4d}/5")
    print(f"{mag:5.1f} {len(inw):7d} {len(ind):6d} | " + " ".join(f"{c:>6s}" for c in cells))
print("\n'ok' = fold test positive rate within [0.05, 0.95] -- an evaluable AUC")
print("\nper-fold detail for the most promising cells:")
for mag,hd in ((3.5,7),(3.0,3),(3.0,7),(2.5,3)):
    ev=np.asarray(load_aegean_events(C,min_magnitude=mag))
    y=label_hours(idx, ev, hd)
    rates=[y[n-(5-k)*fold:n-(5-k)*fold+fold].mean() for k in range(5)]
    print(f"  M>={mag}, {hd:2d}d: " + " ".join(f"{r:.2f}" for r in rates) + f"   overall {y.mean():.3f}")
