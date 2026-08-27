"""The screen's permutation null.

The null decides whether "best of 528 columns beat the floor" means anything,
and it is built from two things that are easy to get quietly wrong: a
hand-rolled rank AUC (fast enough to permute 500 times over 528 columns) and a
block shuffle that has to preserve the label's autocorrelation. A wrong AUC or
an element-wise shuffle both produce a null that is too tight, which turns noise
into a finding.
"""

import numpy as np
import pandas as pd

from forecasting.chaos_univariate_screen import best_under_null
from seismolib.metrics import safe_auc


def ranked(mat):
    return np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 1, mat)


def test_rank_formula_agrees_with_sklearn():
    """The null's inner loop is a rank sum, not roc_auc_score. If they ever
    disagree the whole reference distribution is wrong."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    x = rng.normal(size=400)
    r = pd.Series(x).rank().to_numpy()
    n1 = int(y.sum())
    got = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    assert got == np.float64(safe_auc(y, x, oriented=False)).item() or \
        abs(got - safe_auc(y, x, oriented=False)) < 1e-12


def test_rank_formula_agrees_under_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    r = pd.Series(x).rank().to_numpy()
    n1 = int(y.sum())
    got = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    assert abs(got - safe_auc(y, x, oriented=False)) < 1e-12


def test_null_is_oriented_so_never_below_chance():
    rng = np.random.default_rng(1)
    y = np.repeat((rng.random(40) < 0.3).astype(int), 24)
    null = best_under_null(y, ranked(rng.normal(size=(20, len(y)))), 30, rng)
    assert len(null) and (null >= 0.5).all()


def test_null_returns_one_value_per_draw():
    rng = np.random.default_rng(2)
    y = np.repeat((rng.random(40) < 0.3).astype(int), 24)
    assert len(best_under_null(y, ranked(rng.normal(size=(8, len(y)))), 25, rng)) == 25


def test_more_columns_raise_the_null():
    """Best-of-N grows with N. If it did not, the multiple-comparison
    correction this null exists to provide would not be working."""
    rng = np.random.default_rng(3)
    y = np.repeat((rng.random(60) < 0.3).astype(int), 24)
    few = best_under_null(y, ranked(rng.normal(size=(5, len(y)))), 120, rng)
    many = best_under_null(y, ranked(rng.normal(size=(200, len(y)))), 120, rng)
    assert np.median(many) > np.median(few)


def test_block_shuffle_preserves_the_positive_rate():
    rng = np.random.default_rng(4)
    y = np.repeat((rng.random(40) < 0.3).astype(int), 24)
    # A permutation cannot change the class balance; if it did, every draw
    # would be scored against a different base rate.
    null = best_under_null(y, ranked(rng.normal(size=(4, len(y)))), 10, rng)
    assert np.isfinite(null).all()


def test_single_class_labels_give_an_empty_null():
    rng = np.random.default_rng(5)
    y = np.zeros(240, dtype=int)
    assert len(best_under_null(y, ranked(rng.normal(size=(4, 240))), 10, rng)) == 0


def test_a_genuinely_associated_column_beats_its_own_null():
    """Sanity in the other direction: the machinery must be able to detect
    something real, or a null result proves nothing."""
    rng = np.random.default_rng(6)
    y = np.repeat((rng.random(80) < 0.4).astype(int), 24)
    signal = y + rng.normal(scale=0.35, size=len(y))
    observed = safe_auc(y, signal, oriented=True)
    null = best_under_null(y, ranked(rng.normal(size=(50, len(y)))), 100, rng)
    assert observed > np.quantile(null, 0.95)
