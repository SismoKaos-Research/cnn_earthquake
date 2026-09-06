"""Metrics and floors.

`safe_auc(oriented=)` is the one that has actually gone wrong here: five
forecasting scripts reported a persistence floor without the orientation
correction, which understates the bar whenever the trivial predictor is
anti-correlated with the label -- and under Omori decay it usually is. A floor
of 0.43 is really a floor of 0.57, and the difference is the entire claimed
gain.
"""

import numpy as np
import pytest

from sismokaos.baselines import rate_persistence_auc
from sismokaos.metrics import (binary_report, majority_class_baseline,
                               persistence_baseline, predict_mean_baseline,
                               regression_report, safe_auc, safe_mcc)

Y = np.array([0, 0, 1, 1])
PERFECT = np.array([0.1, 0.2, 0.8, 0.9])
INVERTED = 1.0 - PERFECT


def test_safe_auc_perfect_and_inverted():
    assert safe_auc(Y, PERFECT) == 1.0
    assert safe_auc(Y, INVERTED) == 0.0


def test_oriented_flips_an_inverted_ranker():
    """A predictor known to be anti-correlated is exploitable by negating it,
    so the bar it sets is the flipped value, not the raw one."""
    assert safe_auc(Y, INVERTED, oriented=True) == 1.0


def test_oriented_leaves_a_correct_ranker_alone():
    assert safe_auc(Y, PERFECT, oriented=True) == 1.0


def test_oriented_never_reports_below_chance():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 40)
        if len(np.unique(y)) < 2:
            continue
        assert safe_auc(y, rng.normal(size=40), oriented=True) >= 0.5


def test_unoriented_reports_a_model_below_chance_as_it_is():
    """For a trained model, scoring below chance is a failure to surface --
    flipping it would hide an inverted model as a good one."""
    assert safe_auc(Y, INVERTED, oriented=False) < 0.5


def test_safe_auc_is_nan_on_a_single_class_split():
    assert np.isnan(safe_auc([1, 1, 1], [0.1, 0.5, 0.9]))
    assert np.isnan(safe_auc([0, 0, 0], [0.1, 0.5, 0.9], oriented=True))


def test_safe_auc_is_invariant_to_a_monotone_rescaling():
    """asinh is applied to the 1D channel; it must not move any AUC."""
    assert safe_auc(Y, PERFECT) == safe_auc(Y, np.arcsinh(PERFECT * 1e5))


def test_safe_mcc_is_nan_when_degenerate():
    assert np.isnan(safe_mcc([0, 1, 0, 1], [1, 1, 1, 1]))   # constant prediction
    assert np.isnan(safe_mcc([1, 1, 1, 1], [0, 1, 0, 1]))   # single-class truth
    assert safe_mcc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_binary_report_keys_and_confusion_matrix():
    r = binary_report(Y, PERFECT)
    assert r["n"] == 4 and r["roc_auc"] == 1.0
    assert r["confusion_matrix"] == [[2, 0], [0, 2]]
    for k in ("accuracy", "precision", "recall", "f1", "pr_auc", "mcc", "brier", "log_loss"):
        assert k in r


def test_binary_report_degrades_to_nan_not_an_exception():
    r = binary_report([1, 1, 1], [0.2, 0.6, 0.9])
    for k in ("roc_auc", "pr_auc", "mcc", "brier", "log_loss"):
        assert np.isnan(r[k]), k


def test_binary_report_threshold_is_inclusive():
    r = binary_report([0, 1], [0.5, 0.5], threshold=0.5)
    assert r["confusion_matrix"] == [[0, 1], [0, 1]]


def test_majority_class_baseline_uses_train_not_test():
    """The floor is what the training majority scores on test -- taking the
    majority from test would be fitting the bar to the answer."""
    maj, acc, _ = majority_class_baseline([0, 0, 0, 1], [1, 1, 1, 0])
    assert maj == 0
    assert acc == pytest.approx(0.25)


def test_majority_class_baseline_is_vacuous_when_balanced():
    """Exactly why baselines.py exists: 0.5 on a balanced split proves nothing."""
    _, acc, bal = majority_class_baseline([0, 0, 1, 1], [0, 0, 1, 1])
    assert acc == pytest.approx(0.5) and bal == pytest.approx(0.5)


def test_persistence_baseline_boundary_and_nan():
    pred = persistence_baseline([np.nan, 1.0, 3.0, 3.0001], horizon_days=3.0)
    assert pred.tolist() == [0, 1, 1, 0]        # NaN -> negative; d == horizon -> positive


def test_predict_mean_baseline_predicts_the_train_mean():
    """The floor is the train mean on every test row, not the test mean --
    the latter would be a constant fitted to the answer."""
    r = predict_mean_baseline([1.0, 2.0, 3.0], [2.0, 2.0])
    assert r["MAE"] == pytest.approx(0.0)
    assert predict_mean_baseline([0.0, 0.0, 0.0], [2.0, 2.0])["MAE"] == pytest.approx(2.0)


def test_regression_report_on_a_known_error():
    r = regression_report([1.0, 2.0], [1.5, 2.5])
    assert r["MAE"] == pytest.approx(0.5)
    assert r["RMSE"] == pytest.approx(0.5)
    assert r["max_error"] == pytest.approx(0.5)
    assert r["n"] == 2


def test_regression_report_r2_is_nan_on_one_sample():
    assert np.isnan(regression_report([1.0], [1.5])["R2"])


def test_rate_persistence_auc_flips_an_omori_anticorrelation():
    """Busy now implies calmer next, so the trailing count is anti-correlated
    with a rate-increase label. The bar is the flipped value."""
    labels = np.array([0, 0, 1, 1])
    trailing = np.array([9.0, 8.0, 2.0, 1.0])
    assert rate_persistence_auc(labels, trailing) == 1.0


def test_rate_persistence_auc_is_nan_on_a_single_class():
    assert np.isnan(rate_persistence_auc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])))
