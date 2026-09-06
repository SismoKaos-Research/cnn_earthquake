"""One JSON per training run, so a number can be traced back to what produced it.

Run identity currently lives in checkpoint filenames and in whichever shell
wrapper happened to echo its command: **15 of 110 logs record the invocation that
produced them.** The rest can only be reconstructed by parsing filenames and
grepping for metric lines, which is how every result in this repo has had to be
re-derived when questioned.

This writes `runs/<utc>_<tag>.json` next to the checkpoint dir with everything
needed to reproduce and to compare: the exact argv, the git commit and whether
the tree was dirty, dataset identity and row counts, the resolved split, seeds,
final metrics and checkpoint paths.

Two properties worth stating, because they are what make the record trustworthy
rather than decorative:

**The git SHA is recorded with a dirty flag.** A run from an edited tree is not
reproducible from its SHA alone, and silently recording the SHA anyway would
imply otherwise.

**Metrics are written at the end, in a second call.** A run that crashes leaves a
record with `status: "started"` and no metrics rather than no record at all, so
an interrupted run is visible instead of invisible.

    from seismolib.runlog import RunLog
    log = RunLog("detection/cnn_lstm_classify", save_dir, vars(args))
    log.note(dataset=str(root), n_train=len(tr), n_test=len(te))
    ...
    log.finish(metrics={"roc_auc": auc, "mae": mae}, checkpoints=paths)
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# The run this process is inside, if any. `sk train` opens one around every
# task, so a trainer can record metrics without being told where to put them --
# and without 23 trainers each needing an edit to a file that produces
# published numbers. None when a script is run directly, and every entry point
# below is a no-op then rather than an error.
_CURRENT = None


def current():
    """The ambient RunLog, or None outside one."""
    return _CURRENT


def record(**kw):
    """Adds fields to the ambient run. No-op when there is no run.

    This is what lets `seismolib.metrics.print_report` file a trainer's headline
    numbers automatically: everything that prints a report already calls it, so
    the metrics land in the record without the trainer knowing a record exists.
    """
    if _CURRENT is not None:
        _CURRENT.note(**kw)


def record_metrics(name, report):
    """Files one report block under `metrics`, keyed by its label.

    Several trainers print more than one report (per seed, per fold, per
    ablation arm), so they accumulate rather than overwrite -- a record that
    kept only the last block would silently answer a different question than
    the one asked.
    """
    if _CURRENT is None:
        return
    m = dict(_CURRENT.record.get("metrics") or {})
    keep = {k: v for k, v in report.items()
            if isinstance(v, (int, float, str, bool)) or v is None}
    m[str(name)] = keep
    _CURRENT.note(metrics=m)


def _git(*args, default=None):
    """Runs a git command, returning `default` when git or the repo is absent."""
    try:
        out = subprocess.run(("git",) + args, capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parents[2])
        return out.stdout.strip() if out.returncode == 0 else default
    except Exception:
        return default


class RunLog:
    """Accumulates provenance for one run and writes it as JSON."""

    def __init__(self, task, out_dir, args=None, runs_dir="runs"):
        """Opens a record and writes it immediately with status 'started'.

        Writing at construction rather than at the end is deliberate: a run that
        dies mid-training still leaves evidence that it happened and with what
        arguments, which is exactly the case where the record is most wanted.

        Args:
            task: Dotted or slashed task name, e.g. "detection/cnn_lstm_classify".
            out_dir: Where the run writes checkpoints; recorded, not created.
            args: Parsed arguments as a dict (`vars(args)`), or None.
            runs_dir: Directory for the JSON records.
        """
        self.started = time.time()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Any separator becomes a dot: task names now arrive as "sk train
        # magnitude" from the dispatcher, and a space in the filename breaks
        # every glob that reads these back.
        safe = re.sub(r"[^A-Za-z0-9._-]+", ".", task).strip(".")
        self.tag = f"{stamp}_{safe}_pid{os.getpid()}"
        Path(runs_dir).mkdir(parents=True, exist_ok=True)

        # The stamp is per-second and the pid is per-process, so two runs
        # started in the same second by the same process collided and the
        # second silently overwrote the first. That is not hypothetical: a
        # seed loop or an ensemble runs several trainings in one process, and
        # a lost record makes `sk results --best` answer from whatever
        # survived -- a plausible number, quietly wrong. Suffix until free.
        self.path = Path(runs_dir) / f"{self.tag}.json"
        n = 2
        while self.path.exists():
            self.path = Path(runs_dir) / f"{self.tag}-{n}.json"
            n += 1
        self.tag = self.path.stem

        dirty = _git("status", "--porcelain", default="")
        self.record = {
            "task": task,
            "status": "started",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "args": {k: _plain(v) for k, v in (args or {}).items()},
            "out_dir": str(out_dir),
            "git_commit": _git("rev-parse", "HEAD"),
            # A run from an edited tree cannot be reproduced from its SHA, so
            # say so rather than recording a SHA that implies it can.
            "git_dirty": bool(dirty),
            "host": os.uname().nodename,
            "python": sys.version.split()[0],
        }
        self._flush()

    def note(self, **kw):
        """Adds fields as they become known (dataset sizes, split counts)."""
        self.record.update({k: _plain(v) for k, v in kw.items()})
        self._flush()

    def finish(self, metrics=None, checkpoints=None, status="ok"):
        """Closes the record with metrics and the checkpoints produced."""
        self.record["status"] = status
        self.record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        self.record["duration_s"] = round(time.time() - self.started, 1)
        if metrics:
            self.record["metrics"] = {k: _plain(v) for k, v in metrics.items()}
        if checkpoints:
            self.record["checkpoints"] = [str(c) for c in checkpoints]
        self._flush()
        return self.path

    def _flush(self):
        self.path.write_text(json.dumps(self.record, indent=2, default=str))

    # -- ambient use ------------------------------------------------------
    def __enter__(self):
        """Makes this the ambient run for the duration of the block."""
        global _CURRENT
        self._previous = _CURRENT
        _CURRENT = self
        return self

    def __exit__(self, exc_type, exc, tb):
        """Closes the record, marking it failed if the block raised.

        A crashed run leaves `status: "failed"` and the exception, which is the
        case where the record is most wanted and least likely to be written by
        hand.
        """
        global _CURRENT
        if self.record.get("status") == "started":
            if exc_type is None:
                self.finish()
            else:
                self.note(error=f"{exc_type.__name__}: {exc}")
                self.finish(status="failed")
        _CURRENT = self._previous
        return False


def _plain(v):
    """Coerces numpy scalars, Paths and the like into JSON-safe values."""
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(v)


def load_runs(runs_dir="runs"):
    """Reads every run record, newest first. Returns a list of dicts."""
    out = []
    for f in sorted(Path(runs_dir).glob("*.json"), reverse=True):
        try:
            out.append(json.loads(f.read_text()) | {"_file": str(f)})
        except Exception:
            continue
    return out
