"""Every tool in `scripts/` imports, and its `main()` is reachable.

`tests/test_imports.py` covers `src/`. Nothing covered `scripts/`, and that gap
had teeth: consolidating four copies of `haversine` onto the shared one removed
the local definition from all four files but added the import to only one. All
four still parsed, and three would have raised `NameError` on their first
distance calculation -- in the middle of a plan over tens of thousands of events.
A syntax check said clean. Importing the module is what catches it.

The `main()` check exists because `sk` dispatches by calling it: a tool that
imports fine but has no `main` is registered in `seismolib.cli` and broken only
when someone runs it.

`md2docx` is skipped when python-docx is absent -- it is a reporting convenience
run through `uv run --with python-docx`, not a pipeline dependency, so requiring
it here would make the suite fail on a machine that never renders a report.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Tools whose imports are not declared dependencies of this package. Each entry
# needs a reason: an unexplained skip here is how a broken tool stays broken.
OPTIONAL_DEPS = {
    "md2docx": "python-docx (run via `uv run --with python-docx`)",
}


def tools():
    return sorted(p.stem for p in SCRIPTS.glob("*.py") if p.name != "__init__.py")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", tools())
def test_tool_imports(name):
    """Importing must not raise, and must not run the tool's work."""
    try:
        _load(name)
    except ModuleNotFoundError as e:
        if name in OPTIONAL_DEPS:
            pytest.skip(f"{name} needs {OPTIONAL_DEPS[name]}: {e.name} missing")
        raise
    except SystemExit as e:
        pytest.fail(f"{name} called sys.exit({e.code}) at import time -- it runs "
                    f"work on import, which makes it untestable and makes an "
                    f"accidental import do real work")


@pytest.mark.parametrize("name", tools())
def test_tool_has_main(name):
    """`sk` dispatches by calling main(); a tool without one is registered broken."""
    try:
        mod = _load(name)
    except ModuleNotFoundError as e:
        if name in OPTIONAL_DEPS:
            pytest.skip(f"{name} needs {OPTIONAL_DEPS[name]}: {e.name} missing")
        raise
    assert callable(getattr(mod, "main", None)), (
        f"{name}.py has no callable main(); seismolib.cli dispatches by calling it")


def test_cli_registry_matches_disk():
    """Every `sk` command points at a file that exists, and vice versa."""
    from seismolib.cli import COMMANDS, GROUPS

    for cmd, (script, _desc) in COMMANDS.items():
        assert (SCRIPTS / script).exists(), (
            f"sk {cmd} -> scripts/{script}, which does not exist")

    grouped = {n for _g, names in GROUPS for n in names}
    assert grouped == set(COMMANDS), (
        f"the grouped listing and the command table disagree: "
        f"{grouped ^ set(COMMANDS)} appears in one but not the other")

    on_disk = set(tools())
    registered = {Path(s).stem for s, _ in COMMANDS.values()}
    unregistered = on_disk - registered
    assert not unregistered, (
        f"scripts/ holds {sorted(unregistered)} which `sk` does not expose -- "
        f"either register them in seismolib.cli or move them to experiments/")
