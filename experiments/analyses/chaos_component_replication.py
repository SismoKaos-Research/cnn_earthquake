"""Do BODT and DAT agree on WHICH chaos features carry signal, per component?

The station replication asks whether a feature set transfers between stations.
This asks a sharper version: not whether the scores are similar, but whether the
two stations rank the same features in the same order, split by seismometer
component. Agreement would mean the signal is physical; disagreement would mean
each station is fitting its own noise, which is the failure the whole chaos line
has been circling.

Scores are recomputed from the parquets rather than parsed out of a log, so this
does not depend on a log format that has already changed once.

NOTE: `seismolib.catalog.label_hours` was corrected on 2026-09-06 to open its
horizon at the END of the feature window. At the 6-hour chaos horizon that moved
29% of the positive class, so any number produced by this file before that date
is not comparable with one produced after.
"""
import re, sys, numpy as np
sys.path.insert(0,"src")
from scipy.stats import spearmanr

# Recomputed directly rather than parsed out of a log: score both stations,
# split by component. (A `parse()` stub used to sit here whose docstring said it
# read the log's top-15 block and whose body was `return None` -- it never did
# that, and a caller would have got None without a word.)
from forecasting.chaos_dataset import build
import pandas as pd
CAT="/home/hogib/Projects/Sismokaos/seismic_cli/catalogs/catalog_current.csv"
CLI="/home/hogib/Projects/Sismokaos/sismokaos-cli/dataset_features_chaos_q1_5hz/"
from seismolib.metrics import safe_auc
res={}
for st,pq in (("BODT",CLI+"bodt_q1_chaos_5hz_features.parquet"),
              ("DAT",CLI+"dat_q1_chaos_5hz_features.parquet")):
    X,y,_,_=build(pq,CAT,station=st)
    cols=[c for c in X.columns if c not in ("hour","log1p_dsp")]
    a={}
    for c in cols:
        v=X[c].values
        if np.all(np.isnan(v)) or np.nanstd(v)==0: continue
        a[c]=safe_auc(y,np.nan_to_num(v,nan=np.nanmedian(v)),oriented=True)
    res[st]=a
common=sorted(set(res["BODT"])&set(res["DAT"]))
b=np.array([res["BODT"][c] for c in common]); d=np.array([res["DAT"][c] for c in common])
print(f"{len(common)} features scored at both stations")
print(f"overall Spearman rho = {spearmanr(b,d).statistic:+.4f}\n")
print(f"{'comp':6s} {'n':>4s} {'rho':>8s}   {'BODT best':>10s} {'DAT best':>9s}")
for comp in ("Z","N","E"):
    m=np.array([c.startswith(comp+"_") for c in common])
    if m.sum()<5: continue
    r=spearmanr(b[m],d[m]).statistic
    print(f"{comp:6s} {m.sum():4d} {r:+8.4f}   {b[m].max():10.4f} {d[m].max():9.4f}")
