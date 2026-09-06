"""Walk-forward splitting.

Hourly rows are not independent: neighbouring hours share the same earthquake,
so a random split leaks the answer across the boundary. These tests pin the
properties that make the split honest -- strict chronology, disjointness, and
an embargo that actually removes the rows adjacent to a boundary.
"""

import numpy as np

from sismokaos.splits import walk_forward_splits

IDX = np.arange(100)


def test_returns_the_requested_number_of_folds():
    assert len(walk_forward_splits(IDX, n_folds=3)) == 3


def test_splits_are_disjoint():
    for tr, va, te in walk_forward_splits(IDX, n_folds=3):
        assert not (set(tr) & set(va)) and not (set(va) & set(te)) and not (set(tr) & set(te))


def test_every_fold_is_strictly_chronological():
    """Train before val before test -- a model must never see the future."""
    for tr, va, te in walk_forward_splits(IDX, n_folds=3):
        assert tr.max() < va.min() < te.min()
        assert va.max() < te.min()


def test_training_window_expands_across_folds():
    folds = walk_forward_splits(IDX, n_folds=3)
    sizes = [len(tr) for tr, _, _ in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_each_fold_moves_forward_in_time():
    folds = walk_forward_splits(IDX, n_folds=3)
    starts = [te.min() for _, _, te in folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_embargo_removes_the_rows_adjacent_to_a_boundary():
    """Without it the first val row sits one hour after the last train row and
    is driven by the same event."""
    plain = walk_forward_splits(IDX, n_folds=2, embargo=0)
    gapped = walk_forward_splits(IDX, n_folds=2, embargo=5)
    for (tr, va, _), (gtr, gva, _) in zip(plain, gapped):
        assert gva.min() - gtr.max() > va.min() - tr.max()
        assert gva.min() > tr.max() + 5


def test_embargo_does_not_reorder_or_duplicate():
    for tr, va, te in walk_forward_splits(IDX, n_folds=3, embargo=3):
        for part in (tr, va, te):
            assert list(part) == sorted(part)
            assert len(set(part)) == len(part)


def test_label_balancing_equalises_positives_not_rows():
    """With labels the block edges follow the cumulative positive count, so a
    fold whose positives are clustered gets a wider row span, not fewer events."""
    labels = np.zeros(100, dtype=int)
    labels[:20] = 1                       # every positive at the start
    labels[90:] = 1
    folds = walk_forward_splits(IDX, n_folds=2, labels=labels)
    spans = [len(tr) for tr, _, _ in folds]
    assert spans[0] < 50                  # the dense head needs far fewer rows
    for tr, va, te in folds:
        assert tr.max() < va.min() < te.min()


def test_all_negative_labels_fall_back_to_equal_width_blocks():
    labels = np.zeros(100, dtype=int)
    with_labels = walk_forward_splits(IDX, n_folds=3, labels=labels)
    without = walk_forward_splits(IDX, n_folds=3)
    for (a, b, c), (d, e, f) in zip(with_labels, without):
        assert a.tolist() == d.tolist() and b.tolist() == e.tolist() and c.tolist() == f.tolist()


def test_indices_are_returned_not_positions():
    """`valid_end_indices` may be a sparse subset of the archive; the caller
    gets those values back, not 0..n."""
    sparse = np.arange(0, 200, 2)
    for tr, va, te in walk_forward_splits(sparse, n_folds=2):
        assert set(np.concatenate([tr, va, te])) <= set(sparse.tolist())
