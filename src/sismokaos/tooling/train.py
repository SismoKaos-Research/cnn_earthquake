"""One front door for training, indexed by what you are trying to predict.

Twenty-five trainers live under `src/`, named after their architecture rather
than their target, in five packages. Finding the one that learns a given label
meant knowing the naming history. `sismokaos.tasks` is the index; this runs it.

    sk train                          # every task, grouped by what it predicts
    sk train --predicts forecast      # just the forecasting targets
    sk train detect --help            # the trainer's own help, unchanged
    sk train detect --dataset-dir ds --model-branch cnn-lstm
    sk train forecast-features --model tcn --features-csv f.npy --catalog-path c

Arguments after the task name are passed through untouched, and each task is
exactly its module, so anything recorded as `sk train magnitude ...` also runs
as `python3 -m magnitude.cnn_lstm_regression ...`. That matters here: results
are expected to be traceable to a command, and a front end that rewrote
arguments would make the recorded command and the real one diverge.

Tasks marked `--model` accept the registry's architecture flags, so the same
labels and the same splits can be put to a different network without a second
script. `sk models` describes what each name builds.
"""
import argparse
import importlib
import sys

from sismokaos.model.registry import REGISTRY, by_family
from sismokaos.runlog import RunLog
from sismokaos.tasks import PREDICTS, TASKS, by_prediction

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"


def choices_for(task):
    """The `--model` values a task offers, or an empty list."""
    if task.models:
        return list(task.models)
    if task.family:
        return [a.key for a in by_family(task.family)]
    return []


def show_listing(predicts):
    """Prints the tasks, grouped by what their label answers."""
    print("sk train -- train a model against one of this repo's labels\n")
    for group, question in PREDICTS.items():
        tasks = [t for t in by_prediction(group)
                 if predicts is None or predicts == group]
        if not tasks:
            continue
        print(f"  {BOLD}{group}{OFF}  {DIM}{question}{OFF}")
        for t in tasks:
            print(f"    {t.key:<22}{t.summary}")
            models = choices_for(t)
            if models:
                print(f"    {'':<22}{DIM}--model {'|'.join(models)}{OFF}")
        print()
    print(f"{DIM}`sk train <task> --help` for a task's own flags; "
          f"`sk train <task> --label` for what it learns.{OFF}")
    print(f"{DIM}`sk models` describes the architectures.{OFF}")
    return 0


def show_label(task):
    """Prints what one task's training label actually is."""
    print(f"{BOLD}{task.key}{OFF}  {DIM}({task.predicts}: {PREDICTS[task.predicts]}){OFF}")
    print(f"  {task.summary}")
    print(f"\n  {BOLD}label{OFF}   {task.label}")
    print(f"  {BOLD}trainer{OFF} {task.module}")
    models = choices_for(task)
    if models:
        print(f"  {BOLD}--model{OFF} {' '.join(models)}")
        for m in models:
            a = REGISTRY[m]
            branches = "|".join(a.branches) if a.branches else "-"
            print(f"    {m:<24}--model-branch {branches}")
    else:
        print(f"  {DIM}--model does not apply: this trainer is not registry-wired, "
              f"so it takes the flags it always had{OFF}")
    return 0


def _save_dir(rest):
    """Where the task writes, taken from its own `--save-dir` if it has one.

    Recorded, not created. Tasks without the flag still get a record -- the
    field just says so, which is more useful than refusing to keep one.
    """
    for flag in ("--save-dir", "--out-dir", "--out"):
        if flag in rest:
            i = rest.index(flag)
            if i + 1 < len(rest):
                return rest[i + 1]
    return "(task has no --save-dir)"


def main():
    """Lists the tasks, or dispatches to one trainer's `main()`."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        return show_listing(None)
    if argv[0] == "--predicts":
        if len(argv) < 2 or argv[1] not in PREDICTS:
            print(f"--predicts takes one of {', '.join(PREDICTS)}", file=sys.stderr)
            return 2
        return show_listing(argv[1])

    name = argv[0]
    if name not in TASKS:
        near = [k for k in TASKS if k.startswith(name[:4])]
        print(f"sk train: unknown task {name!r}" +
              (f" -- did you mean {' or '.join(near)}?" if near else ""),
              file=sys.stderr)
        print("run `sk train` for the list", file=sys.stderr)
        return 2
    task = TASKS[name]
    if argv[1:2] == ["--label"]:
        return show_label(task)

    # The trainer sees exactly what it would standalone, including a prog name
    # matching what a user would type back.
    sys.argv = [f"sk train {name}"] + argv[1:]
    mod = importlib.import_module(task.module)
    fn = getattr(mod, "main", None)
    if fn is None:
        print(f"sk train: {task.module} has no main()", file=sys.stderr)
        return 2

    # Provenance is opened HERE rather than inside each trainer. Wiring 23
    # trainers by hand is a mechanical edit to files that produce published
    # numbers, which is how a typo gets into one; opening it around the
    # dispatch costs nothing and covers every task at once. The trainer does
    # not need to know: `sismokaos.metrics.print_report` files its numbers into
    # the ambient run, and `RunLog.__exit__` marks a crash as failed.
    out_dir = _save_dir(argv[1:])
    with RunLog(f"sk train {name}", out_dir, {"argv": argv[1:]}) as log:
        log.note(task_key=name, module=task.module, predicts=task.predicts)
        rc = fn() or 0
        log.note(exit_code=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
