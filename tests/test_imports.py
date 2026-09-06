"""Every module imports, and importing it does not run an experiment.

`label_sweep.py` used to be a flat top-to-bottom script: importing it parsed
argv, read the 482k-row catalogue and ran a 140-cell sweep. That is not a style
preference -- it makes the module impossible to test and turns any accidental
import into a multi-minute job. This test is what keeps the next one from
appearing.

`EXECUTES_ON_IMPORT` is the documented exception list, not a suppression: each
entry is a script whose own docstring says it is `__main__`-only. Adding to it
should be a deliberate decision, and shrinking it is always an improvement.
"""

import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Flat scripts that run their work at module level, by design and by docstring.
EXECUTES_ON_IMPORT = {
    # Unpickles a full model object at import; documented as never-importable.
    "sismokaos.detection.cnn_run",
}

# Modules whose imports are not declared dependencies of this package. Each
# entry needs a reason: an unexplained skip here is how a broken tool stays
# broken. `sk docx` is a reporting convenience run through
# `uv run --with python-docx`, not a pipeline dependency, so requiring it would
# fail the suite on any machine that never renders a report.
OPTIONAL_DEPS = {
    "sismokaos.reporting.md2docx": ("docx", "python-docx"),
}


def modules():
    out = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        out.append(str(path.relative_to(SRC).with_suffix("")).replace("/", "."))
    return out


@pytest.mark.parametrize("name", modules())
def test_module_imports_without_running_anything(name):
    if name in EXECUTES_ON_IMPORT:
        pytest.skip(f"{name} is a flat __main__-only script")
    # Deliberately not `importorskip`: a module that fails to import is a
    # broken module, and turning that into a skip is how a missing `import re`
    # survives in `catalog.py` for weeks.
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        want, pkg = OPTIONAL_DEPS.get(name, (None, None))
        if e.name == want:
            pytest.skip(f"{name} needs {pkg}: {e.name} missing")
        raise
    except SystemExit as e:
        pytest.fail(f"{name} called sys.exit({e.code}) at import time -- it runs "
                    f"work on import, which makes it untestable and makes an "
                    f"accidental import do real work")


def test_every_module_exposes_the_main_sk_would_call():
    """A tool that imports but has no `main()` is registered broken.

    `sk` dispatches by calling `main()`. This covers every tool group, so a
    module dropped into one without an entry point fails here rather than when
    someone runs it.
    """
    import importlib

    missing = []
    for group in ("acquisition", "stations", "windows", "reporting", "tooling"):
        for name in modules():
            if not name.startswith(f"sismokaos.{group}."):
                continue
            try:
                mod = importlib.import_module(name)
            except ModuleNotFoundError as e:
                if OPTIONAL_DEPS.get(name, (None,))[0] == e.name:
                    continue
                raise
            if not callable(getattr(mod, "main", None)):
                missing.append(name)
    assert not missing, f"{missing} have no main() for `sk` to call"


def test_exception_list_has_no_stale_entries():
    """A module that has since been refactored should leave the list."""
    assert EXECUTES_ON_IMPORT <= set(modules())
    assert set(OPTIONAL_DEPS) <= set(modules())
