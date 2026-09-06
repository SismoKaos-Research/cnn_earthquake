"""No module may reference a name nothing in the file ever binds.

This exists because of a bug the rest of the suite could not see. Moving the
continuous-scan machinery out of `src/sismokaos/continuous/cli.py` into
`sismokaos.continuous` carried the function bodies over verbatim but not three
things they depended on: `import math`, `from scipy import signal`, and the
`COMPONENT_ROLES` constant. A function's globals are its *defining* module's,
not its caller's, so the script's own copies did not rescue them.

Four functions were dead on arrival -- `pick_components`, `clip_spans`,
`clean_block`, `taper_vector`, which is the whole waveform path, i.e. `scan`,
`baseline`, and `cut_event_windows.py`. Everything still imported, the suite
stayed green, and the end-to-end check that was supposed to prove the move was
`coincidence`, which reads scored `.npz` files and never touches a waveform.

The check is deliberately blunt: a name loaded in a file must be bound
*somewhere* in that file, or be a builtin. That over-approximates what is in
scope, so it cannot flag a working file -- and it catches the entire class of
"the body moved, its imports did not", which is the risk every phase of this
restructure runs.
"""
import ast
import builtins
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = (sorted((ROOT / "src").rglob("*.py"))
         + sorted((ROOT / "experiments").rglob("*.py"))
         + sorted((ROOT / "tests").glob("*.py")))
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}


def _bound(tree):
    """Every name bound anywhere in the file, in any scope."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                            ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                out.add(n.name)
            a = getattr(n, "args", None)
            if a is not None:
                for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                    out.add(arg.arg)
                for arg in (a.vararg, a.kwarg):
                    if arg:
                        out.add(arg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchStar) and n.name:
            out.add(n.name)
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_undefined_names(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    known = _bound(tree) | BUILTINS
    missing = sorted({n.id for n in ast.walk(tree)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                      and n.id not in known})
    assert not missing, (
        f"{path.relative_to(ROOT)} uses {missing} but binds them nowhere -- "
        f"a body was moved without its import or its constant")
