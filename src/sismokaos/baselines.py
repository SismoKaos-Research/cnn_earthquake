"""Conditional floors -- what a trivial rule scores on the same data.

A majority-class baseline is vacuous on a balanced split, and comparing
against it has twice made a result look far stronger than it was. These
are the bars that actually have to be cleared."""

import numpy as np

from sismokaos.metrics import safe_auc


def rate_persistence_auc(labels: np.ndarray, trailing_counts: np.ndarray) -> float:
    """AUC of the trivial trailing-rate baseline for `label_hours_rate_change`.

    The trailing count is a legitimate backward-looking predictor, but its
    relationship to a rate-INCREASE label is inverted (Omori decay: busy now
    implies calmer next). Reports the achievable baseline as
    `max(auc, 1 - auc)`, since a forecaster free to choose the sign of a
    known-anti-correlated predictor gets the flipped value for free -- so
    that, not 0.5, is the bar a model has to clear.

    Args:
        labels: 0/1 rate-increase labels.
        trailing_counts: Trailing-window event counts, same length.

    Returns:
        Baseline AUC in [0.5, 1.0], or NaN if `labels` is single-class.
    """
    auc = safe_auc(labels, trailing_counts.astype(np.float64))
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
