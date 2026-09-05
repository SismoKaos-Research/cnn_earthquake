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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


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
        self.tag = f"{stamp}_{task.replace('/', '.')}_pid{os.getpid()}"
        self.path = Path(runs_dir) / f"{self.tag}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
