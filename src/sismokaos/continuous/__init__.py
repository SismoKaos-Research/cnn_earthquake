"""Turning a continuous station archive into scored windows and alarms.

Extracted from `src/sismokaos/continuous/cli.py`, which had grown to 1,522
lines and was the most scientifically important tool in the repo while sitting
in a directory that cannot be imported. `cut_event_windows.py` needed six of
these functions and reached them with `sys.path.insert(__file__.parent)` -- a
script importing a sibling script by path manipulation, which is what made the
two impossible to test together and easy to break apart.

Nothing here is new. The bodies are the ones that produced the published
continuous-detection figures, moved verbatim; only two signatures changed, from
taking an argparse namespace to taking their parameters, because three CLIs call
them and their flags do not have to agree.

The library half is split by what it does to the data, in the order a scan uses
it:

    chunks        reading an archive, picking its three components, and
                  conditioning samples the way the training windows were made
    spans         the interval algebra that keeps a window off a data gap, and
                  that decides which stretch two stations both recorded
    association   predicted arrivals, measured SNR, and which windows a
                  catalogued event is allowed to excuse
    alarms        clustering scores into declarations, and matching them
                  across stations

Everything those four define is re-exported here, so `from sismokaos.continuous
import read_chunk` keeps working and no call site had to move.

The command half is one module per subcommand -- `baseline`, `scan`, `report`,
`coincidence`, `timing`, `verify` -- each owning both its argparse fragment and
its body, so a flag and the code that reads it are no longer 560 lines apart.
`cli` assembles them into the parser that `src/sismokaos/continuous/cli.py`
has always presented. They are deliberately NOT imported here: `scan` pulls in
torch and the detection package, and `from sismokaos.continuous import
component_segments` should not pay for that.
"""
from sismokaos.continuous.alarms import confirmed, declarations, load_scores
from sismokaos.continuous.association import (background_and_guards, load_snr,
                                              predicted_arrivals)
from sismokaos.continuous.chunks import (COMPONENT_ROLES, add_chunk_args,
                                         clean_block, component_segments,
                                         make_windows, pick_components,
                                         read_chunk, taper_vector)
from sismokaos.continuous.spans import (clip_spans, common_spans,
                                        coverage_spans, in_spans,
                                        intersect_spans, merge_intervals)

__all__ = [
    "COMPONENT_ROLES", "add_chunk_args", "background_and_guards", "clean_block",
    "clip_spans", "common_spans", "component_segments", "confirmed",
    "coverage_spans", "declarations", "in_spans", "intersect_spans",
    "load_scores", "load_snr", "make_windows", "merge_intervals",
    "pick_components", "predicted_arrivals", "read_chunk", "taper_vector",
]
