"""
Collate the ground-motion experiment grid into one table.

Reads every `groundmotion_cnn_*.csv` written by `run_groundmotion_experiments.sh`
and prints the comparison in both metric spaces, with the seed spread beside
each mean, so no configuration can be quoted without its uncertainty or without
the floor it is being measured against.
"""

import glob
import re
from pathlib import Path

import pandas as pd

# What each run was designed to answer, written before the grid was run.
QUESTIONS = {
    "A_main":     "does shape beat the peak-amplitude floor at all?",
    "B_nolstm":   "does the BiLSTM+attention earn its parameters?",
    "C_nodist":   "given shape, is the distance scalar still needed?",
    "D_waveonly": "with no scalars, can the network recover amplitude itself?",
    "E_pga":      "the paper's quantity, on the honest non-overlapping window",
    "F_pgafull":  "the DEGENERATE target -- contrast only, not like-for-like",
}


def main():
    src = Path(__file__).parent
    files = sorted(glob.glob(str(src / "groundmotion_cnn_*.csv")))
    if not files:
        print("No result CSVs found -- has the grid been run?")
        return

    rows = []
    for f in files:
        tag = re.sub(r"^groundmotion_cnn_|\.csv$", "", Path(f).name)
        d = pd.read_csv(f)
        rows.append({
            "run": tag,
            "target": d.target.iloc[0],
            "arch": d.arch.iloc[0],
            "norm": d.input_norm.iloc[0],
            "n_aux": int(d.n_aux.iloc[0]),
            "seeds": len(d),
            "MAE_log": d.MAE_log.mean(),
            "MAE_sd": d.MAE_log.std(ddof=0),
            "R2_log": d.R2_log.mean(),
            "R2_lin": d.R2_lin.mean(),
        })
    r = pd.DataFrame(rows).sort_values("run")

    print(f"\n{'='*104}")
    print("GROUND MOTION EXPERIMENT GRID")
    print(f"{'='*104}")
    print(f"{'run':12s} {'target':9s} {'arch':9s} {'norm':5s} {'aux':>4s} "
          f"{'seeds':>5s} {'MAE_log':>16s} {'R2_log':>8s} {'R2_lin':>9s}   question")
    print("-" * 104)
    for _, x in r.iterrows():
        q = QUESTIONS.get(x.run, "")
        print(f"{x.run:12s} {x.target:9s} {x.arch:9s} {x.norm:5s} {x.n_aux:4d} "
              f"{x.seeds:5d} {x.MAE_log:9.4f} +-{x.MAE_sd:5.4f} {x.R2_log:8.4f} "
              f"{x.R2_lin:9.4f}   {q}")
    print("-" * 104)
    print("  Lower MAE_log is better. Floors are printed per-run by "
          "cnn_groundmotion.py; the\n  strongest is 'amplitude + distance + station', "
          "which is the one to quote against.")
    print("  F_pgafull's target window CONTAINS the input window -- its numbers are "
          "not\n  comparable to the _fwd runs and exist only to show the size of that "
          "degeneracy.")

    r.to_csv(src / "groundmotion_grid_summary.csv", index=False)
    print(f"\n[write] {src / 'groundmotion_grid_summary.csv'}")


if __name__ == "__main__":
    main()
