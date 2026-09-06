"""Selecting the checkpoints of one training arm, and refusing to guess.

Every eval script in this repo ensembles a directory of `.pth` files, and the
directory almost never holds one arm: `run_ponly_natural.sh` writes `1d`, `2d`
and `all` into the same `--save-dir`, so `trained_model_ponly_natural/` has 9
checkpoints spanning three architectures. A bare `glob("*.pth")` averages over
all of them, which is not an ensemble but a mixture of models answering
different questions. That has already cost two sets of checkpoints, quarantined
by hand, and one seed reported at 0.2480 -- an inverted model read as a training
outcome.

Four scripts grew their own near-identical anchored regex in response; this is
that regex, once, with the two refinements the copies lacked.

**Anchoring matters more than it looks.** `--branch-1d` takes `lstm`, `cnn` and
`cnn-lstm`, and `cnn` is a prefix of `cnn-lstm`. An unanchored `cnn` sweeps in
every `cnn-lstm` checkpoint, so the trailing underscore is load-bearing.

**Matching cannot be made strict enough on its own,** because the run tag has
grown over time. `--branch-1d` and `--seq-transform` were added after the 6 s
detector was trained, so its checkpoints are named `2d_linear_dataset_...` with
neither segment, and a fully anchored pattern rejects them. So the match stays
deliberately loose and the *result* is checked instead: group the survivors by
run identity and require exactly one. Old names resolve, and anything genuinely
ambiguous raises with the candidates named.
"""

import re
from pathlib import Path

# `{channels}_{fusion}_{branch_1d}_{seq_transform}_{dataset}_pid{pid}_seed{n}`.
# Stripping the pid/seed tail leaves the run identity -- everything that must
# match before two checkpoints may be averaged together.
RUN_SUFFIX = re.compile(r"_pid\d+_seed\d+\.pth$")


def run_identity(name):
    """The part of a checkpoint filename shared by every seed of one run."""
    return RUN_SUFFIX.sub("", Path(name).name)


def find_checkpoints(ckpt_dir, channels, fusion, branch=None, require_single_run=True):
    """Returns the checkpoints for one arm, sorted, or raises.

    Args:
        ckpt_dir: Directory of `.pth` checkpoints.
        channels: `1d`, `2d` or `all`.
        fusion: `linear` or `gate`.
        branch: `--branch-1d` value to require. None matches on channels and
            fusion alone, which is what checkpoints predating that flag need.
        require_single_run: Raise if the survivors span more than one run
            identity. Set False only to inspect a directory deliberately.

    Returns:
        Sorted list of `Path`s.

    Raises:
        FileNotFoundError: Nothing matched.
        ValueError: The match spans several runs and `require_single_run`.
    """
    pat = re.compile(rf"_{re.escape(channels)}_{re.escape(fusion)}_"
                     + (rf"{re.escape(branch)}_" if branch else ""))
    found = sorted(p for p in Path(ckpt_dir).glob("*.pth") if pat.search(p.name))
    if not found:
        raise FileNotFoundError(
            f"No checkpoints matching channels={channels} fusion={fusion}"
            + (f" branch-1d={branch}" if branch else "")
            + f" under {ckpt_dir}")

    runs = sorted({run_identity(p.name) for p in found})
    if require_single_run and len(runs) > 1:
        raise ValueError(
            f"{ckpt_dir} holds {len(runs)} distinct runs matching "
            f"channels={channels} fusion={fusion}"
            + (f" branch-1d={branch}" if branch else "")
            + ". Ensembling across them would mix architectures, datasets or "
              "amplitude transforms. Narrow it with --branch-1d, or point at a "
              "directory holding one run:\n  " + "\n  ".join(runs))
    return found
