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
         + sorted((ROOT / "scripts").glob("*.py"))
         + sorted((ROOT / "experiments").rglob("*.py")))

# path -> why this one is not the anti-pattern.
ALLOWED = {
    "src/sismokaos/cli.py":
        "sk dispatches to scripts/ by file path, so it puts that directory on "
        "the path for the tool it is about to exec. It is the dispatcher, not "
        "a tool reaching for a sibling.",
    "src/sismokaos/forecasting/cnn_lstm_forecast.py":
        "imports `seismic_cli.forecast` from a SIBLING REPO whose location "
        "comes from --data-downloader-root. That path is not knowable at "
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

DOC_ROOTS = ("docs", "src", "scripts", "tests", "experiments")
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


def test_sk_registers_every_script_and_groups_every_command():
    """`sk` is the front door, so a tool it does not list is a tool nobody finds.

    Three ways this drifts, all silent: a new script nobody registers, a
    registration left behind after the script is renamed (which fails only when
    someone runs it), and a command in `COMMANDS` that no group prints, so it
    exists but never appears in the listing.
    """
    from sismokaos.cli import COMMANDS, GROUPS

    registered = {v[0] for v in COMMANDS.values()}
    on_disk = {p.name for p in (ROOT / "scripts").glob("*.py")}
    grouped = {n for _, names in GROUPS for n in names}

    assert not on_disk - registered, \
        f"scripts/ holds {sorted(on_disk - registered)}, which `sk` does not list"
    assert not registered - on_disk, \
        f"`sk` points at {sorted(registered - on_disk)}, which scripts/ does not have"
    assert not set(COMMANDS) - grouped, \
        f"{sorted(set(COMMANDS) - grouped)} are registered but in no GROUPS section"
