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
    "detection.cnn_run",
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
    importlib.import_module(name)


def test_exception_list_has_no_stale_entries():
    """A module that has since been refactored should leave the list."""
    assert EXECUTES_ON_IMPORT <= set(modules())
