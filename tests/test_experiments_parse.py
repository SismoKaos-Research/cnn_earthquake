"""Everything in `experiments/` at least parses.

`tests/test_imports.py` covers `src/` and `tests/test_scripts_import.py` covers
`scripts/`; `experiments/` was covered by nothing, which is why its hardcoded
paths went stale unnoticed through two directory moves.

Parse, not import. These are analysis one-offs that read specific data files and
some do network work; importing them would run that. A syntax check is the part
that can be automated safely, and it is what catches a bad edit -- the haversine
consolidation broke three files that all still *parsed*, so this is a floor, not
a guarantee, and it says so.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiments"


def python_files():
    return sorted(p.relative_to(ROOT) for p in ROOT.rglob("*.py"))


@pytest.mark.parametrize("rel", [str(p) for p in python_files()])
def test_experiment_parses(rel):
    src = (ROOT / rel).read_text()
    ast.parse(src, filename=str(rel))


@pytest.mark.parametrize("rel", [str(p) for p in python_files()])
def test_experiment_has_a_docstring_saying_what_it_answers(rel):
    """These are the record of what was tried; an untitled one is unreadable."""
    tree = ast.parse((ROOT / rel).read_text(), filename=str(rel))
    doc = ast.get_docstring(tree)
    assert doc and len(doc.strip()) > 30, (
        f"experiments/{rel} has no module docstring. These files ARE the record "
        f"of what was tried and why; one without a question stated is scaffolding "
        f"nobody can evaluate later.")


def test_shell_runners_parse():
    """The reproduce runners are the provenance for published numbers."""
    import subprocess
    for sh in sorted(ROOT.rglob("*.sh")):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        assert r.returncode == 0, f"{sh.name} is not valid bash:\n{r.stderr}"
