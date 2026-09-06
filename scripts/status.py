"""What is running, how far along, and did the last thing work?

This is the question asked most often in this project and the one the tooling
answered worst. Jobs run detached on a remote box for hours, `ps` shows several
processes of the same script that differ only in an argument 80 characters in,
and whether a finished run succeeded lived in scrollback.

Three sections, in the order the question is usually asked:

**Running** -- one line per job, with the argument that distinguishes it. Three
`afad_imap` processes are identical in `ps` until you can see which ledger each
holds; that single field is most of the value here.

**Recent runs** -- from `runs/*.json` (`seismolib.runlog`), so a finished
training run reports its own headline metric and whether it succeeded, instead
of requiring a grep through a log whose name encodes the config.

**Disk** -- because every campaign here is bounded by it, and finding out at 90%
through a 200 GB pull is expensive.

    sk status                 # this machine
    sk status --host vegs     # the box the jobs are actually on
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Argument whose value identifies one invocation of a tool from another. Without
# these, `ps` output for three pollers is three identical lines.
DISTINGUISHING = ("--ledger", "--out-dir", "--zips", "--requests", "--dataset-dir",
                  "--scores", "--out", "--station", "--save-dir")

# Processes worth reporting: this project's tools and trainers, not every python.
INTERESTING = ("afad_imap", "afad_campaign", "fdsn_magnitude_pull", "cut_event_windows",
               "cut_window_length", "continuous_false_alarms", "station_detection_range",
               "magnitude_error_profile", "plan_pbefores_pull", "status", "sk ",
               "cnn_lstm", "cnn_", "chaos_", "feature_", "gru_cnn", "raw_cnn",
               "catalog_", "waveform_", "next_event", "label_sweep")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=None,
                   help="ssh host to inspect instead of this machine")
    p.add_argument("--dir", default=None,
                   help="project directory on that host (default: this repo's path)")
    p.add_argument("--runs", type=int, default=6, help="recent runs to show")
    p.add_argument("--all", action="store_true",
                   help="every python process, not just this project's tools")
    return p.parse_args()


def sh(cmd, host=None):
    """Runs a shell command locally or over ssh, returning stdout ('' on error)."""
    full = ["ssh", host, cmd] if host else ["bash", "-lc", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=60)
        return r.stdout
    except Exception:
        return ""


def short_arg(cmdline):
    """Picks the argument that tells this invocation apart from its siblings."""
    try:
        parts = shlex.split(cmdline)
    except ValueError:
        parts = cmdline.split()
    for flag in DISTINGUISHING:
        if flag in parts:
            i = parts.index(flag)
            if i + 1 < len(parts):
                v = parts[i + 1]
                # a glob or long path is only useful at its tail
                return f"{flag} {v if len(v) <= 34 else '…' + v[-33:]}"
    for p in parts:
        if p.endswith((".py", ".jsonl", ".csv")):
            return Path(p).name
    return ""


def tool_of(cmdline):
    """The script name, without path or interpreter."""
    for p in cmdline.split():
        if p.endswith(".py"):
            return Path(p).stem
    # `sk <cmd>` dispatches in-process, so the cmdline carries no .py at all;
    # name it by the subcommand rather than by the interpreter's path.
    parts = cmdline.split()
    for i, p in enumerate(parts):
        if p.endswith("/sk") or p == "sk":
            return f"sk {parts[i+1]}" if i + 1 < len(parts) else "sk"
    return Path(parts[0]).name if parts else "?"


def show_running(host, show_all):
    """Prints one line per interesting process."""
    # Only meaningful locally; over ssh our pids are on the other machine.
    self_pids = set() if host else {str(os.getpid()), str(os.getppid())}
    raw = sh("ps -eo pid,etime,pcpu,rss,args --no-headers -ww", host)
    rows = []
    for line in raw.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, et, cpu, rss, cmd = parts
        # A shell wrapper is not the job. `zsh -c "... foo.py ..."` carries the
        # script name inside its -c string, so any name-based test matches it;
        # the actual process appears separately as its own child. Rejecting
        # argv[0] shells is what distinguishes "running" from "mentioned".
        argv0 = Path(cmd.split()[0]).name if cmd.split() else ""
        if argv0 in ("sh", "bash", "zsh", "dash", "ssh", "nohup", "timeout"):
            continue
        if "python" not in cmd and "/sk " not in cmd:
            continue
        # Do not report the query itself -- by pid, not by matching "status" in
        # the command line. A substring test also hid legitimate jobs such as
        # `afad_campaign.py --ledger x status`, which is the exact thing this
        # tool exists to show.
        if pid in self_pids or "ps -eo" in cmd:
            continue
        # Match on the SCRIPT being executed, not on any substring of the
        # command line. A shell whose line merely mentions a tool -- an editor,
        # a wrapper, a previous command in the same string -- is not that tool
        # running, and reporting it as a live job is worse than saying nothing.
        tool = tool_of(cmd)
        if not show_all and not any(tool.startswith(k) or tool == k for k in INTERESTING):
            continue
        rows.append((pid, et, cpu, int(rss) / 1e6, tool, short_arg(cmd)))

    print(f"\033[1mRUNNING\033[0m ({len(rows)})" if rows else "\033[1mRUNNING\033[0m — nothing")
    if rows:
        print(f"  {'pid':>8}{'elapsed':>12}{'cpu%':>7}{'GB':>6}  {'tool':<24}what")
        for pid, et, cpu, gb, tool, arg in rows:
            print(f"  {pid:>8}{et:>12}{cpu:>7}{gb:>6.1f}  {tool:<24}{arg}")
    return len(rows)


def show_runs(host, cwd, n):
    """Prints the most recent training runs and their headline metric."""
    target = cwd if cwd.startswith("~") else shlex.quote(cwd)
    out = sh(f"cd {target} 2>/dev/null && ls -1t runs/*.json 2>/dev/null | head -{n} "
             f"| while read f; do cat \"$f\"; echo; done", host)
    records = []
    for blob in out.split("\n}"):
        blob = blob.strip()
        if not blob:
            continue
        try:
            records.append(json.loads(blob + "\n}"))
        except Exception:
            continue
    print(f"\n\033[1mRECENT RUNS\033[0m ({len(records)})" if records
          else "\n\033[1mRECENT RUNS\033[0m — none (runs/ is empty; is runlog wired into this trainer?)")
    if not records:
        return
    print(f"  {'when':<18}{'task':<32}{'status':<9}{'time':>8}  headline")
    for r in records:
        m = r.get("metrics") or {}
        head = ""
        for k in ("roc_auc", "MAE", "F1", "f1", "R2"):
            if k in m and isinstance(m[k], (int, float)):
                head = f"{k} {m[k]:.4f}"
                floor = m.get("floor") or m.get("ridge_floor")
                if isinstance(floor, (int, float)):
                    head += f" (floor {floor:.4f})"
                break
        when = (r.get("started_utc") or "")[:16].replace("T", " ")
        dur = r.get("duration_s")
        dur = f"{dur/60:.0f}m" if isinstance(dur, (int, float)) and dur >= 90 else \
              (f"{dur:.0f}s" if isinstance(dur, (int, float)) else "-")
        status = r.get("status", "?")
        mark = "" if status == "ok" else "  <-- did not finish"
        print(f"  {when:<18}{r.get('task','?'):<32}{status:<9}{dur:>8}  {head}{mark}")


def show_disk(host, cwd):
    """Prints free space where the data lands."""
    # A quoted "~" is literal to the remote shell and df then finds nothing, so
    # leave a leading tilde unquoted for it to expand.
    target = cwd if cwd.startswith("~") else shlex.quote(cwd)
    out = sh(f"df -h {target} 2>/dev/null | tail -1", host)
    if out.split():
        f = out.split()
        print(f"\n\033[1mDISK\033[0m  {f[3]} free of {f[1]} ({f[4]} used) on {f[0]}")


def main():
    """Prints running jobs, recent runs and free disk."""
    args = parse_args()
    cwd = args.dir or str(Path(__file__).resolve().parents[1])
    where = args.host or os.uname().nodename
    print(f"\033[1m{where}\033[0m:{cwd}")
    show_running(args.host, args.all)
    show_runs(args.host, cwd, args.runs)
    show_disk(args.host, cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
