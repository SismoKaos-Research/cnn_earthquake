"""Layout invariants: what may reach outside its own package, and how.

`sys.path` manipulation is how this repo used to paper over a layout problem,
and every instance of it hid something worth fixing. Three scripts added their
own directory so they could import a sibling script -- which is why
`cut_event_windows.py` and `continuous_false_alarms.py` could not be tested
together. Seven files in `src/detection/` added `src/`, which the editable
install has provided since the package existed. Four in `experiments/` added
the literal string `"src"`, which only resolves when the cwd happens to be the
repo root.

All of those are gone. The two that remain are here by name, each with the
reason it is not the same thing.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = (sorted((ROOT / "src").rglob("*.py"))
         + sorted((ROOT / "scripts").glob("*.py"))
         + sorted((ROOT / "experiments").rglob("*.py")))

# path -> why this one is not the anti-pattern.
ALLOWED = {
    "src/seismolib/cli.py":
        "sk dispatches to scripts/ by file path, so it puts that directory on "
        "the path for the tool it is about to exec. It is the dispatcher, not "
        "a tool reaching for a sibling.",
    "src/forecasting/cnn_lstm_forecast.py":
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
        f"\n\nShared code belongs in seismolib/, which every module can import "
        f"because the package is installed. If this one is genuinely different, "
        f"add it to ALLOWED in {Path(__file__).name} with the reason.")
