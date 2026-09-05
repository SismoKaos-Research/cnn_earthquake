"""`sk` -- one entry point for the tools in `scripts/`.

Fifteen tools with fifteen invocation styles is why nobody remembers which one
does what. This gives them one front door and a listing, without changing any of
them: each tool keeps its own `main()` and its own `--help`, and still runs
standalone as `python3 scripts/<tool>.py`.

**It dispatches by file path rather than by import.** The tools deliberately are
not moved into the package, because `afad_imap.py` spawns `afad_campaign.py` as
a subprocess by path to refill queue slots -- moving them under a live campaign
would break a poller mid-run. Dispatch buys the discoverability now; relocating
the files is a separate change for when nothing is in flight.

    sk                      # list the tools
    sk fdsn plan --help     # a tool's own help, unchanged
    sk falsealarm scan --zips 'afad_raw/MANT/*.zip' ...

Every command is exactly the underlying script, so anything a write-up records
as `sk fdsn fetch ...` can also be run as `python3 scripts/fdsn_magnitude_pull.py
fetch ...` and vice versa. That matters here: results in this repo are expected
to be traceable to a command, and a front end that rewrote arguments would make
the recorded command and the real one diverge.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

# name -> (script, one-line description). Grouped by what you are trying to do,
# which is how the tools get looked for, rather than by which family wrote them.
COMMANDS = {
    # acquisition
    "campaign":   ("afad_campaign.py", "TDVMS ledger: plan, submit, paste, status, mark, reset"),
    "poll":       ("afad_imap.py", "watch the mailbox, fetch links, refill freed queue slots"),
    "fdsn":       ("fdsn_magnitude_pull.py", "plan/fetch event windows from KOERI FDSN (KO network)"),
    "plan-pull":  ("plan_pbefores_pull.py", "plan a TDVMS pull that can report before S"),
    "catalog":    ("fetch_afad_catalog.py", "rebuild the event catalogue from AFAD's API"),
    # stations
    "station-select": ("select_afad_stations.py", "rank stations by catalogue coverage"),
    "station-range":  ("station_detection_range.py", "per-event SNR at one station"),
    "station-loss":   ("station_catalog_loss.py", "what a station's catalogue misses"),
    # windowing
    "cut-events": ("cut_event_windows.py", "cut arrival-anchored windows from continuous record"),
    "cut-length": ("cut_window_length.py", "re-cut existing windows to another length"),
    # evaluation
    "falsealarm": ("continuous_false_alarms.py", "false-alarm rate on continuous data"),
    "magprofile": ("magnitude_error_profile.py", "where a magnitude regressor's error lives"),
    # what is going on
    "status":     ("status.py", "what is running, how far along, and did the last run work"),
    "models":     ("models.py", "the model registry: architectures, branches, flags"),
    # reporting
    "docx":       ("md2docx.py", "Markdown -> .docx (no pandoc on this box)"),
    "figures":    ("make_report_figures.py", "report figures"),
}

GROUPS = [
    ("acquire", ["campaign", "poll", "fdsn", "plan-pull", "catalog"]),
    ("stations", ["station-select", "station-range", "station-loss"]),
    ("windows", ["cut-events", "cut-length"]),
    ("evaluate", ["falsealarm", "magprofile"]),
    ("report", ["docx", "figures"]),
    ("inspect", ["status", "models"]),
]


def usage():
    """Prints the grouped command listing."""
    print("sk -- tooling for the seismic pipeline\n")
    print("usage: sk <command> [args...]        (each command has its own --help)\n")
    for group, names in GROUPS:
        print(f"  {group}")
        for n in names:
            script, desc = COMMANDS[n]
            print(f"    {n:<16} {desc}")
        print()
    print("Reproduction runners for published results live in experiments/reproduce/;")
    print("they are deliberately not commands here -- their value is being the exact")
    print("thing that was run.")


def main():
    """Dispatches to one script's `main()`, leaving its arguments untouched."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        usage()
        return 0
    name = sys.argv[1]
    if name not in COMMANDS:
        near = [c for c in COMMANDS if c.startswith(name[:3])]
        print(f"sk: unknown command {name!r}" +
              (f" -- did you mean {' or '.join(near)}?" if near else ""), file=sys.stderr)
        print("run `sk` for the list", file=sys.stderr)
        return 2

    script = SCRIPTS / COMMANDS[name][0]
    if not script.exists():
        print(f"sk: {script} is missing -- the tool moved without updating "
              f"seismolib/cli.py", file=sys.stderr)
        return 2

    # Rewrite argv so the tool sees exactly what it would standalone, including
    # a prog name that matches what a user would type back.
    sys.argv = [f"sk {name}"] + sys.argv[2:]
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_sk_{name}", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "main", None)
    if fn is None:
        print(f"sk: {script.name} has no main()", file=sys.stderr)
        return 2
    return fn() or 0


if __name__ == "__main__":
    sys.exit(main())
