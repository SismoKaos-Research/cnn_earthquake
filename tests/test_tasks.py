"""Every task dispatches, and every task that offers `--model` means it.

`sk train` is an index, and an index that points at something that has moved is
worse than no index. Two things go stale on their own: a module rename leaves a
task pointing nowhere, and a task claiming a model family whose trainer never
called `add_model_args` accepts `--model` in the listing and rejects it on the
command line.

Both are checked here against the code rather than against the table, so the
table cannot drift from what actually runs.
"""
import argparse
import importlib
import sys

import pytest

from sismokaos.model.registry import FAMILIES, REGISTRY, by_family
from sismokaos.tasks import PREDICTS, TASKS, Task, by_prediction


@pytest.mark.parametrize("key", sorted(TASKS))
def test_task_module_imports_and_has_main(key):
    """`sk train <task>` dispatches by calling main() on this module."""
    mod = importlib.import_module(TASKS[key].module)
    assert callable(getattr(mod, "main", None)), (
        f"{TASKS[key].module} has no main(); sk train {key} would fail")


@pytest.mark.parametrize("key", sorted(TASKS))
def test_task_group_and_models_are_real(key):
    t = TASKS[key]
    assert t.predicts in PREDICTS, f"{key} is in no known prediction group"
    if t.family:
        assert t.family in FAMILIES, f"{key} names family {t.family!r}"
    for m in t.models:
        assert m in REGISTRY, f"{key} names model {m!r}, which is not registered"


def test_no_two_tasks_share_a_trainer():
    """Two names for one module is an index that lies about how much there is."""
    seen = {}
    for k, t in TASKS.items():
        assert t.module not in seen, f"{k} and {seen[t.module]} are the same trainer"
        seen[t.module] = k


@pytest.mark.parametrize("key", sorted(k for k, t in TASKS.items() if t.choosable))
def test_choosable_tasks_actually_accept_model_flags(key):
    """The listing promises `--model`; the trainer's parser must deliver it.

    A task can name a family in the table while its trainer still builds its
    model by class name. The listing would advertise `--model` and argparse
    would reject it, which is exactly the drift the registry exists to end.
    """
    t = TASKS[key]
    mod = importlib.import_module(t.module)
    parser_fn = getattr(mod, "parse_args", None)
    assert parser_fn is not None, f"{t.module} has no parse_args to inspect"

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **kw):
        captured["actions"] = {s for act in self._actions for s in act.option_strings}
        raise SystemExit(0)          # stop before the trainer needs real files

    monkey = pytest.MonkeyPatch()
    monkey.setattr(argparse.ArgumentParser, "parse_args", spy)
    old_argv = sys.argv
    try:
        sys.argv = [key]
        with pytest.raises(SystemExit):
            parser_fn()
    finally:
        sys.argv = old_argv
        monkey.undo()
        assert argparse.ArgumentParser.parse_args is real_parse

    flags = captured.get("actions", set())
    assert "--model" in flags, (
        f"sk train lists {key} as taking --model, but {t.module}'s parser has "
        f"no such flag -- either wire add_model_args into it or drop the "
        f"family/models entry from sismokaos/tasks.py")
    offers_branch = any(REGISTRY[m].branches for m in
                        (t.models or [a.key for a in by_family(t.family)]))
    if offers_branch:
        assert "--model-branch" in flags, f"{key} offers branchy models but no flag"


def test_by_prediction_covers_every_task():
    assert sum(len(by_prediction(g)) for g in PREDICTS) == len(TASKS)
    with pytest.raises(ValueError, match="unknown group"):
        by_prediction("nonsense")


def test_choosable_is_what_the_listing_reads():
    assert Task("k", "detect", "m", "s", "l").choosable is False
    assert Task("k", "detect", "m", "s", "l", family="dual").choosable is True
    assert Task("k", "detect", "m", "s", "l", models=("gru",)).choosable is True
