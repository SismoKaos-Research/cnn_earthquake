"""`sk` -- the front door to this repo's tools.

Twenty tools with twenty invocation styles is why nobody remembers which one
does what. This gives them one entry point and a listing, grouped by what you
are trying to do rather than by which family wrote them.

**It dispatches by import.** It used to dispatch by file path, because the
tools lived in `scripts/` -- a directory that cannot be imported, so the only
way to reach one was to load it from its path. That is also why
`cut_event_windows.py` had to `sys.path.insert` its own directory to borrow six
functions from a sibling, and why the most scientifically important tool in the
repo went untested for as long as it did. The tools are modules now:

    sk                      # the listing
    sk fdsn plan --help     # a tool's own help, unchanged
    sk falsealarm scan --zips 'afad_raw/MANT/*.zip' ...

Every command is exactly the underlying module's `main()`, called with the
arguments you typed. Nothing is rewritten in between. That matters here:
results in this repo are expected to be traceable to a command, and a front end
that reshaped arguments would make the recorded command and the real one
diverge. Each tool also still runs standalone as
`python -m sismokaos.<group>.<tool>`.
"""
import importlib
import sys

from sismokaos.runlog import RunLog

# name -> (module, one-line description). Grouped below by what you are trying
# to do, which is how a tool actually gets looked for.
COMMANDS = {
    # acquisition
    "campaign":   ("sismokaos.acquisition.afad_campaign", "TDVMS ledger: plan, submit, pump, paste, status, mark, reset"),
    "poll":       ("sismokaos.acquisition.afad_imap", "watch the mailbox, fetch links, refill freed queue slots"),
    "fdsn":       ("sismokaos.acquisition.fdsn_magnitude_pull", "plan/fetch event windows from KOERI FDSN (KO network)"),
    "fdsn-noise": ("sismokaos.acquisition.plan_fdsn_noise", "plan noise windows for the stations an FDSN pull returned"),
    "plan-pull":  ("sismokaos.acquisition.plan_pbefores_pull", "plan a TDVMS pull that can report before S"),
    "catalog":    ("sismokaos.acquisition.fetch_afad_catalog", "rebuild the event catalogue from AFAD's API"),
    # stations
    "station-select": ("sismokaos.stations.select_afad_stations", "rank stations by catalogue coverage"),
    "station-range":  ("sismokaos.stations.station_detection_range", "per-event SNR at one station"),
    "station-loss":   ("sismokaos.stations.station_catalog_loss", "what a station's catalogue misses"),
    "station-neighbors": ("sismokaos.stations.station_neighbors", "which stations sit within N km of one station"),
    "distances":      ("sismokaos.stations.manifest_distances", "recompute a manifest's distance_km from station coordinates"),
    # windowing
    "cut-events": ("sismokaos.windows.cut_event_windows", "cut arrival-anchored windows from continuous record"),
    "cut-length": ("sismokaos.windows.cut_window_length", "re-cut existing windows to another length"),
    # evaluation
    "falsealarm": ("sismokaos.continuous.cli", "false-alarm rate on continuous data"),
    "magprofile": ("sismokaos.magnitude.magnitude_error_profile", "where a magnitude regressor's error lives"),
    # training
    "train":      ("sismokaos.tooling.train", "train a model against one of this repo's labels"),
    # what is going on
    "status":     ("sismokaos.tooling.status", "what is running, how far along, and did the last run work"),
    "models":     ("sismokaos.tooling.models", "the model registry: architectures, branches, flags"),
    "results":    ("sismokaos.tooling.results", "what we have measured, on what, with which command"),
    # reporting
    "docx":       ("sismokaos.reporting.md2docx", "Markdown -> .docx (no pandoc on this box)"),
    "pdf":        ("sismokaos.reporting.md2pdf", "Markdown -> .pdf (run through uv; see its docstring)"),
    "figures":    ("sismokaos.reporting.make_report_figures", "report figures"),
}

GROUPS = [
    ("acquire", ["campaign", "poll", "fdsn", "fdsn-noise", "plan-pull", "catalog"]),
    ("stations", ["station-select", "station-range", "station-loss", "station-neighbors", "distances"]),
    ("windows", ["cut-events", "cut-length"]),
    ("train", ["train"]),
    ("evaluate", ["falsealarm", "magprofile"]),
    ("report", ["docx", "pdf", "figures"]),
    ("inspect", ["status", "models", "results"]),
]


def usage():
    """Prints the grouped command listing."""
    print("sk -- tooling for the seismic pipeline\n")
    print("usage: sk <command> [args...]        (each command has its own --help)\n")
    for group, names in GROUPS:
        print(f"  {group}")
        for n in names:
            print(f"    {n:<18} {COMMANDS[n][1]}")
        print()
    print("Reproduction runners for published results live in experiments/reproduce/;")
    print("they are deliberately not commands here -- their value is being the exact")
    print("thing that was run.")


def main():
    """Dispatches to one tool's `main()`, leaving its arguments untouched."""
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

    # Rewrite argv so the tool sees exactly what it would standalone, including
    # a prog name that matches what a user would type back.
    sys.argv = [f"sk {name}"] + sys.argv[2:]
    mod = importlib.import_module(COMMANDS[name][0])
    fn = getattr(mod, "main", None)
    if fn is None:
        print(f"sk: {COMMANDS[name][0]} has no main()", file=sys.stderr)
        return 2
    args = sys.argv[1:]
    if name not in RECORDED or any(a in ("-h", "--help") for a in args):
        return fn() or 0
    # The commands that write data get the same run record as `sk train`
    # (which opens its own). The FDSN pull and the distance repair behind the
    # magnitude corpus left none, so that corpus could only be traced through
    # shell history.
    with RunLog(f"sk {name}", _out_dir(args), {"argv": args}) as log:
        rc = fn() or 0
        log.note(exit_code=rc)
        if rc != 0:
            log.finish(status="failed")
    return rc


# Commands that write data a later step trains or evaluates on.
RECORDED = {"fdsn", "fdsn-noise", "catalog", "plan-pull", "station-range",
            "distances", "cut-events", "cut-length"}


def _out_dir(args):
    """The output path named on a command line, or "" when there is none."""
    for flag in ("--out-dir", "--output-dir", "--out", "--manifest"):
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
    return ""


if __name__ == "__main__":
    sys.exit(main())
