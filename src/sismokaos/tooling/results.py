"""What have we measured, on what, with which command.

The question this project has answered worst. Every number in it has had to be
re-derived when questioned -- the continuous-detection figures sat in
`docs/TODO.md` for a day as first-195-day values next to a finished 728-day run,
and nobody could tell from the numbers which they were.

`sismokaos.runlog` writes one JSON per run and `sk train` opens one around every
task, so the records exist. This reads them back.

    sk results                        # every run, newest first
    sk results --task magnitude       # one family
    sk results --metric MAE --best    # the best MAE we have measured, and its command
    sk results --failed               # runs that did not finish
    sk results --show <tag>           # one record in full

**A run from a dirty tree is marked.** The commit alone does not reproduce it,
and a table that showed the SHA without saying so would imply otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

from sismokaos.runlog import load_runs

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
# Metric names seen in this repo's reports, best-first by convention.
LOWER_IS_BETTER = {"mae", "rmse", "loss", "mean_absolute_error", "median_ae",
                   "resid_std", "max_error"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--task", default=None, help="substring match on the task name")
    p.add_argument("--metric", default=None,
                   help="show and sort by this metric (case-insensitive)")
    p.add_argument("--best", action="store_true",
                   help="only the best run for --metric, with its exact command")
    p.add_argument("--failed", action="store_true", help="only runs that did not finish")
    p.add_argument("--show", default=None, help="print one record in full, by tag substring")
    p.add_argument("--limit", type=int, default=25)
    return p.parse_args()


def metric_of(rec, name):
    """One metric from a record, searching nested report blocks.

    `print_report` files each block under its own label, so a run with three
    seeds has three blocks; the best value across them is what the run
    achieved. Returns (value, label) or (None, None).
    """
    best = (None, None)
    for label, block in (rec.get("metrics") or {}).items():
        if not isinstance(block, dict):
            if label.lower() == name.lower() and isinstance(block, (int, float)):
                best = _pick(best, (block, "metrics"), name)
            continue
        for k, v in block.items():
            if k.lower() == name.lower() and isinstance(v, (int, float)):
                best = _pick(best, (v, label), name)
    return best


def _pick(a, b, name):
    if a[0] is None:
        return b
    lower_better = name.lower() in LOWER_IS_BETTER
    return b if ((b[0] < a[0]) == lower_better) else a


def command_of(rec):
    """The invocation, as a user would retype it."""
    argv = rec.get("argv") or []
    return " ".join(str(a) for a in argv) if argv else "(not recorded)"


def main():
    """Prints the run table, one record, or the best run for a metric."""
    args = parse_args()
    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"no run records under {args.runs_dir}/ -- runs appear once a task "
              f"has been launched through `sk train`", file=sys.stderr)
        return 1

    if args.show:
        hit = [r for r in runs if args.show in r.get("_file", "")]
        if not hit:
            print(f"no record matching {args.show!r}", file=sys.stderr)
            return 2
        print(json.dumps(hit[0], indent=2))
        return 0

    if args.task:
        runs = [r for r in runs if args.task in str(r.get("task", ""))]
    if args.failed:
        runs = [r for r in runs if r.get("status") != "ok"]

    if args.metric and args.best:
        scored = [(metric_of(r, args.metric), r) for r in runs]
        scored = [(v, lbl, r) for (v, lbl), r in scored if v is not None]
        if not scored:
            print(f"no run records a metric called {args.metric!r}", file=sys.stderr)
            return 1
        lower = args.metric.lower() in LOWER_IS_BETTER
        v, lbl, r = sorted(scored, key=lambda x: x[0], reverse=not lower)[0]
        print(f"{BOLD}best {args.metric} = {v:.4f}{OFF}   ({'lower' if lower else 'higher'} is better)")
        print(f"  block    {lbl}")
        print(f"  task     {r.get('task')}")
        print(f"  when     {(r.get('started_utc') or '')[:19].replace('T', ' ')} UTC")
        print(f"  commit   {r.get('git_commit', '?')[:12]}"
              + (f"  {BOLD}(tree was DIRTY -- this SHA does not reproduce it){OFF}"
                 if r.get("git_dirty") else ""))
        print(f"  out      {r.get('out_dir')}")
        print(f"  command  {command_of(r)}")
        print(f"  record   {r.get('_file')}")
        return 0

    print(f"  {'when':<17}{'status':<9}{'task':<34}{'metric':>10}  command")
    for r in runs[: args.limit]:
        when = (r.get("started_utc") or "")[:16].replace("T", " ")
        status = r.get("status", "?")
        val = ""
        if args.metric:
            v, lbl = metric_of(r, args.metric)
            val = f"{v:.4f}" if v is not None else "-"
        else:
            for cand in ("roc_auc", "MAE", "mae", "F1", "f1", "R2"):
                v, _ = metric_of(r, cand)
                if v is not None:
                    val = f"{cand} {v:.4f}"
                    break
        dirty = "*" if r.get("git_dirty") else " "
        print(f" {dirty}{when:<17}{status:<9}{str(r.get('task'))[:33]:<34}{val:>10}  "
              f"{command_of(r)[:60]}")
    print(f"\n{DIM}  * = run from a dirty tree; its commit does not reproduce it{OFF}")
    print(f"{DIM}  `sk results --metric MAE --best` for the best run and its command{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
