"""Collate the three branch-1d runs into one table.

Parses the end-of-run block each training log emits. Patterns are anchored with
re.M because the fields sit at line starts inside an indented block -- a missing
re.M silently matched nothing here once before and produced an all-NaN table.
"""
import re
import sys
from pathlib import Path

# Resolved from this file, not from where the repo happens to live: the
# checkout moves (Projects/model_cnn_lstm -> Projects/Sismokaos/cnn_earthquake)
# and an absolute path silently reads an old tree or none at all.
LOGS = Path(__file__).resolve().parents[2] / "logs"
VARIANTS = ["lstm", "cnn-lstm", "cnn"]

PAT = {
    "floor":   re.compile(r"^\s*seq abs-max\s+AUC\s+([\d.]+)", re.M),
    "perseed": re.compile(r"^\s*per-seed test AUC:\s*\[([^\]]*)\]", re.M),
    "spread":  re.compile(r"^\s*mean\s+([\d.]+)\s+std\s+([\d.]+)\s+spread\s+([\d.]+)", re.M),
    "auc":     re.compile(r"^\s*roc_auc\s+([\d.]+)", re.M),
    "mcc":     re.compile(r"^\s*mcc\s+([\d.]+)", re.M),
    "recall":  re.compile(r"^\s*recall\s+([\d.]+)", re.M),
    "prec":    re.compile(r"^\s*precision\s+([\d.]+)", re.M),
    "params":  re.compile(r"^\s*model parameters:\s*([\d,]+)", re.M),
}


def grab(txt, key, cast=float):
    m = PAT[key].search(txt)
    return cast(m.group(1)) if m else None


rows = []
for v in VARIANTS:
    p = LOGS / f"branch1d_asinh_{v}.log"
    if not p.exists():
        print(f"!! missing {p}")
        continue
    t = p.read_text()
    seeds = re.findall(r"\[seed (\d+)\] test AUC ([\d.]+)", t)
    done = "ALL SEEDS" in t or grab(t, "auc") is not None
    rows.append({
        "variant": v,
        "seeds_done": len(seeds),
        "per_seed": [float(s[1]) for s in seeds],
        "ens_auc": grab(t, "auc"),
        "mcc": grab(t, "mcc"),
        "recall": grab(t, "recall"),
        "precision": grab(t, "prec"),
        "params": grab(t, "params", lambda s: int(s.replace(",", ""))),
        "floor": grab(t, "floor"),
        "finished": done,
    })

floor = next((r["floor"] for r in rows if r["floor"]), 0.9049)
print(f"\nseq abs-max floor (test): {floor:.4f}\n")
hdr = f"{'variant':<10}{'seeds':>6}{'per-seed test AUC':>34}{'ens AUC':>9}{'headroom':>10}{'params':>9}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    ps = " ".join(f"{x:.4f}" for x in r["per_seed"]) or "-"
    ens = f"{r['ens_auc']:.4f}" if r["ens_auc"] else "running"
    hd = (f"{(r['ens_auc'] - floor) / (1 - floor) * 100:.1f}%"
          if r["ens_auc"] else "-")
    pr = f"{r['params']:,}" if r["params"] else "-"
    print(f"{r['variant']:<10}{r['seeds_done']:>6}{ps:>34}{ens:>9}{hd:>10}{pr:>9}")

if not all(r["finished"] for r in rows):
    print("\n(runs still in progress -- ensemble columns fill in as each completes)")
