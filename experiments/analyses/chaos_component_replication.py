import re, sys, numpy as np
sys.path.insert(0,"src")
from scipy.stats import spearmanr

def parse(path):
    """Pulls the per-station AUC table out of the replication log's top-15 block."""
    return None

# recompute directly instead of parsing: score both stations, split by component
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
