"""Layout invariants: what may reach outside its own package, and how.

`sys.path` manipulation is how this repo used to paper over a layout problem,
and every instance of it hid something worth fixing. Three scripts added their
own directory so they could import a sibling script -- which is why
`cut_event_windows.py` and `continuous_false_alarms.py` could not be tested
together. Seven files in `src/sismokaos/detection/` added `src/`, which the editable
install has provided since the package existed. Four in `experiments/` added
the literal string `"src"`, which only resolves when the cwd happens to be the
repo root.

All of those are gone. The two that remain are here by name, each with the
reason it is not the same thing.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = (sorted((ROOT / "src").rglob("*.py"))
         + sorted((ROOT / "experiments").rglob("*.py")))

# path -> why this one is not the anti-pattern.
ALLOWED = {
    "src/sismokaos/forecasting/cnn_lstm_forecast.py":
        "imports `seismic_cli.forecast` from a SIBLING REPO whose location "
        "comes from --seismic-cli-root. That path is not knowable at "
        "install time and the import is guarded, skipping with a message "
        "rather than crashing when the sibling is absent.",
}


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_nothing_manipulates_sys_path(path):
    rel = str(path.relative_to(ROOT))
    # Parsed, not grepped: `continuous/__init__.py` explains the anti-pattern
    # in its docstring, and a text search cannot tell prose from code.
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = [f"line {n.lineno}: sys.path.{n.func.attr}(...)"
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("insert", "append")
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "path"
            and isinstance(n.func.value.value, ast.Name)
            and n.func.value.value.id == "sys"]
    if rel in ALLOWED:
        assert hits, (f"{rel} is on the sys.path allowlist but no longer "
                      f"manipulates sys.path -- drop the entry")
        return
    assert not hits, (
        f"{rel} manipulates sys.path:\n    " + "\n    ".join(hits) +
        f"\n\nShared code belongs in the sismokaos package, which every module can import "
        f"because the package is installed. If this one is genuinely different, "
        f"add it to ALLOWED in {Path(__file__).name} with the reason.")


# --- recorded paths -------------------------------------------------------

DOC_ROOTS = ("docs", "src", "tests", "experiments")
PATH_RE = re.compile(
    r"(?<![\w/.-])(?:" + "|".join(DOC_ROOTS) + r")/[\w./-]+\.(?:py|sh|md)\b")


def _documents():
    out = [ROOT / "README.md"]
    for d in DOC_ROOTS:
        for p in (ROOT / d).rglob("*"):
            if p.suffix not in (".md", ".py", ".sh"):
                continue
            if "__pycache__" in p.parts or "experiment_results" in p.parts:
                continue
            out.append(p)
    return sorted(out)


@pytest.mark.parametrize("path", _documents(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_recorded_path_still_resolves(path):
    """A path this repo writes down must be a path this repo has.

    Results here are expected to be traceable to the command that produced
    them, which is worth nothing if the command names a file that moved. This
    caught 31 at once after the namespace consolidation: sixteen were docstrings
    naming their own old location, and the rest predated the split into family
    packages, still pointing at the flat `src/` era -- including two that had
    moved to the sibling `seismic_cli` repo entirely, which is exactly the
    reference a reader cannot resolve on their own.

    Only paths under this repo's own code directories are checked, so a
    reference to a sibling repo or to a data file is left alone.
    """
    missing = sorted({m for m in PATH_RE.findall(path.read_text())
                      if not (ROOT / m).exists()})
    assert not missing, (
        f"{path.relative_to(ROOT)} names {missing}, which do not exist. "
        f"Repoint them, or write the path so it is clearly not this repo's.")


def test_sk_registers_a_real_module_for_every_command():
    """`sk` is the only front door, so a command that does not resolve is a
    tool nobody can run.

    It used to dispatch by file path, and the check here was that `scripts/`
    and the registry held the same filenames. Dispatch is by import now, so
    the question is the stronger one: does the module exist, does it import,
    and does it expose the `main()` that `sk` is about to call. A command in
    `COMMANDS` that no group prints is registered but invisible, which is its
    own way of being unreachable.
    """
    import importlib

    from sismokaos.cli import COMMANDS, GROUPS

    grouped = {n for _, names in GROUPS for n in names}
    assert not set(COMMANDS) - grouped, \
        f"{sorted(set(COMMANDS) - grouped)} are registered but in no GROUPS section"
    assert not grouped - set(COMMANDS), \
        f"{sorted(grouped - set(COMMANDS))} are listed in a group but not registered"

    broken = []
    for name, (modname, _) in sorted(COMMANDS.items()):
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if modname.endswith("md2docx") and e.name == "docx":
                continue        # run through `uv run --with python-docx`
            broken.append(f"sk {name} -> {modname}: {e}")
            continue
        if not callable(getattr(mod, "main", None)):
            broken.append(f"sk {name} -> {modname} has no main()")
    assert not broken, "\n".join(broken)


def test_no_tool_module_is_left_unregistered():
    """A tool in one of the command groups that `sk` does not list.

    The groups exist to be run from the command line; a module that lands in
    one without a registry entry is reachable only by someone who already
    knows the module path, which is the discoverability problem `sk` exists to
    solve.
    """
    from sismokaos.cli import COMMANDS

    registered = {m for m, _ in COMMANDS.values()}
    on_disk = set()
    for group in ("acquisition", "stations", "windows", "reporting", "tooling"):
        for p in (ROOT / "src" / "sismokaos" / group).glob("*.py"):
            if p.name != "__init__.py":
                on_disk.add(f"sismokaos.{group}.{p.stem}")
    assert not on_disk - registered, \
        f"{sorted(on_disk - registered)} are tools that `sk` does not list"


def test_the_readme_lists_the_commands_sk_actually_has():
    """The README's command block is a copy of `sk`'s listing, so it drifts.

    It had: `fdsn-noise`, `pdf` and `results` were missing, three commands that
    exist and work but that the manual did not mention. A reader's first look
    at this repo is the README, and a front door that under-reports itself is
    worse than no listing at all.
    """
    from sismokaos.cli import GROUPS

    block = (ROOT / "README.md").read_text().split("```\nacquire", 1)[1].split("```", 1)[0]
    documented = {}
    for line in ("acquire" + block).strip().splitlines():
        group, *names = line.split()
        documented[group] = names

    real = {group: names for group, names in GROUPS}
    assert documented == real, (
        "README command block is out of date:\n" +
        "\n".join(f"  {g}: README {documented.get(g)} vs sk {real.get(g)}"
                  for g in sorted(set(documented) | set(real))
                  if documented.get(g) != real.get(g)))
