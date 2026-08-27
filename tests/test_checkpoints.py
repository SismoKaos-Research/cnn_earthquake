"""Checkpoint selection -- the bug class that has cost this project the most.

A save dir routinely holds several arms (`run_ponly_natural.sh` writes `1d`,
`2d` and `all` into one directory), so "ensemble every .pth here" silently
averages models that answer different questions. These tests pin the three
things that keep that from happening: the trailing-underscore anchor, the
tolerance for old tag layouts, and the refusal to average across runs.
"""

import pytest

from seismolib.checkpoints import find_checkpoints, run_identity

# The tag as `cnn_lstm_classify.py` writes it today.
NEW = "best_cnnlstm_classify_{ch}_{fu}_{br}_{tf}_{ds}_pid{pid}_seed{seed}.pth"
# The tag before --branch-1d and --seq-transform existed. The 6 s detector
# checkpoints are named this way, so selection has to keep working on them.
OLD = "best_cnnlstm_classify_{ch}_{fu}_{ds}_pid{pid}_seed{seed}.pth"


def write(tmp_path, names):
    for n in names:
        (tmp_path / n).touch()
    return tmp_path


def new(ch="2d", fu="linear", br="cnn-lstm", tf="asinh", ds="dataset_A", pid=1, seed=42):
    return NEW.format(ch=ch, fu=fu, br=br, tf=tf, ds=ds, pid=pid, seed=seed)


def old(ch="2d", fu="linear", ds="dataset_A", pid=1, seed=42):
    return OLD.format(ch=ch, fu=fu, ds=ds, pid=pid, seed=seed)


def test_selects_only_the_named_arm(tmp_path):
    d = write(tmp_path, [new(ch=c, seed=s) for c in ("1d", "2d", "all") for s in (42, 43, 44)])
    got = find_checkpoints(d, "2d", "linear", "cnn-lstm")
    assert len(got) == 3
    assert all("_2d_linear_" in p.name for p in got)


def test_cnn_does_not_match_cnn_lstm(tmp_path):
    """`cnn` is a prefix of `cnn-lstm`; the trailing underscore is what separates them."""
    d = write(tmp_path, [new(br="cnn", pid=1, seed=42), new(br="cnn-lstm", pid=2, seed=42)])
    assert [p.name for p in find_checkpoints(d, "2d", "linear", "cnn")] == [new(br="cnn", pid=1, seed=42)]
    assert [p.name for p in find_checkpoints(d, "2d", "linear", "cnn-lstm")] == [new(br="cnn-lstm", pid=2, seed=42)]


def test_accepts_tags_written_before_branch_and_transform_existed(tmp_path):
    """A fully anchored pattern would reject the 6 s detector's own checkpoints."""
    d = write(tmp_path, [old(seed=s) for s in (42, 43, 44)])
    assert len(find_checkpoints(d, "2d", "linear")) == 3


def test_refuses_to_average_across_runs(tmp_path):
    d = write(tmp_path, [new(br="cnn"), new(br="cnn-lstm", pid=2)])
    with pytest.raises(ValueError) as e:
        find_checkpoints(d, "2d", "linear")
    # The message has to name the candidates, or the user cannot act on it.
    assert "cnn-lstm" in str(e.value) and "2 distinct runs" in str(e.value)


def test_seq_transform_is_part_of_run_identity(tmp_path):
    """An asinh checkpoint and a raw one are otherwise indistinguishable."""
    d = write(tmp_path, [new(tf="asinh"), new(tf="none", pid=2)])
    with pytest.raises(ValueError):
        find_checkpoints(d, "2d", "linear", "cnn-lstm")


def test_dataset_is_part_of_run_identity(tmp_path):
    """Two datasets is two experiments, never one ensemble."""
    d = write(tmp_path, [new(ds="dataset_A"), new(ds="dataset_B", pid=2)])
    with pytest.raises(ValueError):
        find_checkpoints(d, "2d", "linear", "cnn-lstm")


def test_reruns_of_the_same_arm_do_ensemble(tmp_path):
    """Different pid, same everything else: more seeds, not a different model."""
    d = write(tmp_path, [new(pid=1, seed=42), new(pid=2, seed=99)])
    assert len(find_checkpoints(d, "2d", "linear", "cnn-lstm")) == 2


def test_missing_arm_raises_file_not_found(tmp_path):
    d = write(tmp_path, [new(ch="1d")])
    with pytest.raises(FileNotFoundError):
        find_checkpoints(d, "2d", "linear", "cnn-lstm")


def test_empty_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_checkpoints(tmp_path, "2d", "linear")


def test_require_single_run_can_be_waived(tmp_path):
    d = write(tmp_path, [new(br="cnn"), new(br="cnn-lstm", pid=2)])
    assert len(find_checkpoints(d, "2d", "linear", require_single_run=False)) == 2


def test_result_is_sorted(tmp_path):
    d = write(tmp_path, [new(seed=s) for s in (44, 42, 43)])
    got = [p.name for p in find_checkpoints(d, "2d", "linear", "cnn-lstm")]
    assert got == sorted(got)


@pytest.mark.parametrize("name,expect", [
    (new(seed=42), new(seed=42).replace("_pid1_seed42.pth", "")),
    (old(seed=44), old(seed=44).replace("_pid1_seed44.pth", "")),
])
def test_run_identity_strips_only_pid_and_seed(name, expect):
    assert run_identity(name) == expect


def test_run_identity_accepts_a_full_path(tmp_path):
    assert run_identity(tmp_path / new()) == run_identity(new())
