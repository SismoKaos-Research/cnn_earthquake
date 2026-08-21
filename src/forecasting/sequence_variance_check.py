"""Does a sequence model have anything to chew on, or is every step the same?

The ruled-out GRU/CNN catalog forecaster degenerated for a measurable reason:
its inputs were 7/30/90-day rolling aggregates whose within-24 h std was only
**1.2-9.3%** of their overall std. A 24-step sequence was therefore ~24
near-identical vectors, and the last hidden state was an MLP on the last step.
No amount of gating fixes that -- it is a property of the input.

Chaotic features at a 50 s step are a genuinely different input, so the verdict
does not automatically transfer. This script measures whether it does, and it
is the gate on the whole idea: run it before writing any training code.

The number reported per column is

    within-sequence variance ratio = mean_over_sequences( std within sequence )
                                     -------------------------------------------
                                                 std over everything

Read it as: land near the 1.2-9.3% band and the sequence carries almost no
internal variation, so the verdict transfers. Land at 40-80% and there is real
within-sequence structure for an LSTM/GRU to model.

**Granularity is the design decision, not a detail.** Extraction emits 72
windows per hour (200 s windows, 50 s step), but `feature_lstm_forecast.py`
consumes *hourly* vectors. Collapsing 72 windows to an hourly mean may destroy
exactly the variation the idea depends on, so this reports both and the answer
may differ.

Usage:
    python3 src/forecasting/sequence_variance_check.py \\
        --parquet ~/Projects/sismokaos-cli/dataset_features_chaos_q1_5hz/\\
bodt_q1_chaos_5hz_features.parquet
"""
import argparse
import numpy as np
import pandas as pd

CHAOS_KEYS = ("WOLF_LYE", "ROS_SHORT", "ROS_LONG", "SAMP_ENT", "CORR_DIM")
# The catalog aggregates that made the GRU degenerate, for reference.
DEGENERATE_LO, DEGENERATE_HI = 0.012, 0.093


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", required=True, help="feature parquet from sismokaos-cli run")
    p.add_argument("--seq-len", type=int, default=24,
                   help="sequence length in steps, at whichever granularity (default 24)")
    p.add_argument("--step-sec", type=float, default=50.0,
                   help="seconds between windows; must match the extraction config")
    p.add_argument("--top", type=int, default=0,
                   help="also list the N most and least variable non-chaos columns")
    return p.parse_args()


def variance_ratio(values, seq_len):
    """Mean within-sequence std as a fraction of the overall std.

    Sequences are consecutive, non-overlapping and complete; a trailing partial
    sequence is dropped rather than padded, which would understate the spread.
    """
    v = np.asarray(values, dtype=float)
    overall = np.nanstd(v)
    if not np.isfinite(overall) or overall == 0:
        return np.nan
    n = (len(v) // seq_len) * seq_len
    if n == 0:
        return np.nan
    blocks = v[:n].reshape(-1, seq_len)
    with np.errstate(invalid="ignore"):
        within = np.nanstd(blocks, axis=1)
    return float(np.nanmean(within) / overall)


def report(df, cols, seq_len, label, span_note):
    print(f"\n=== {label} — sequence = {seq_len} steps ({span_note}) ===")
    print(f"{'column':<18}{'within/overall':>16}   verdict")
    rows = []
    for c in cols:
        r = variance_ratio(df[c].values, seq_len)
        rows.append((c, r))
    for c, r in sorted(rows, key=lambda t: -(t[1] if t[1] == t[1] else -1)):
        if r != r:
            verdict = "degenerate (no variance)"
        elif r < DEGENERATE_HI:
            verdict = "FLAT — verdict transfers"
        elif r < 0.40:
            verdict = "marginal"
        else:
            verdict = "varies — worth modelling"
        print(f"{c:<18}{r:>15.1%}   {verdict}")
    finite = [r for _, r in rows if r == r]
    if finite:
        print(f"{'MEDIAN':<18}{np.median(finite):>15.1%}")
    return rows


def main():
    args = parse_args()
    df = pd.read_parquet(args.parquet)
    print(f"{args.parquet}\n  {len(df):,} rows x {len(df.columns)} columns")

    chaos = sorted(c for c in df.columns
                   if any(k in c for k in CHAOS_KEYS) and not c.endswith("_DEV"))
    print(f"  {len(chaos)} chaos columns (excluding _DEV first differences)")

    per_hour = int(round(3600.0 / args.step_sec))
    print(f"  {per_hour} windows per hour at a {args.step_sec:g} s step")
    print(f"\n  Reference: the catalog aggregates that made the GRU degenerate sat at "
          f"{DEGENERATE_LO:.1%}–{DEGENERATE_HI:.1%}.")

    native = report(df, chaos, args.seq_len,
                    f"NATIVE {args.step_sec:g} s windows",
                    f"{args.seq_len * args.step_sec / 60:.0f} min of context")

    hourly = df[chaos].groupby(np.arange(len(df)) // per_hour).mean()
    hr = report(hourly, chaos, args.seq_len, "HOURLY means",
                f"{args.seq_len} h of context")

    print("\n=== the aggregation cost ===")
    print(f"{'column':<18}{'native':>10}{'hourly':>10}{'lost':>10}")
    dn, dh = dict(native), dict(hr)
    losses = []
    for c in chaos:
        a, b = dn.get(c), dh.get(c)
        if a == a and b == b and a:
            losses.append(a - b)
            print(f"{c:<18}{a:>9.1%}{b:>9.1%}{a - b:>9.1%}")
    if losses:
        print(f"\n  Averaging to hourly costs {np.mean(losses):.1%} of the "
              f"within-sequence variation, on average.")

    if args.top:
        other = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in chaos and not c.endswith("_DEV") and c != "Zaman_Dk"]
        rows = [(c, variance_ratio(df[c].values, args.seq_len)) for c in other]
        rows = [r for r in rows if r[1] == r[1]]
        rows.sort(key=lambda t: -t[1])
        print(f"\n=== non-chaos columns for comparison (native, top/bottom {args.top}) ===")
        for c, r in rows[:args.top] + [("...", float("nan"))] + rows[-args.top:]:
            print(f"  {c:<26}{'' if r != r else f'{r:.1%}':>8}")


if __name__ == "__main__":
    main()
