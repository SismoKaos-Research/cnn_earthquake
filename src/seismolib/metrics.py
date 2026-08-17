"""Shared accuracy-metric reports and baselines for the classifiers/regressors
in this repo. Each `*_report` function returns a plain dict (safe to print,
log, or diff across runs); `print_report` gives every script the same
console format instead of a bespoke print block per script.

Not a runnable script -- imported only. Callers: every training/eval script
in src/ (cnn_lstm*.py, cnn_regression.py, cnn_riskclass.py, cnn_magclass.py,
cnn_ram_aux.py, cnn_groundmotion.py, groundmotion_baselines.py,
feature_lstm_forecast.py, raw_cnn_lstm_forecast.py,
raw100hz_cnn_lstm_forecast.py, riskclass_scalar.py, lgbm_cluster.py).
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, mean_absolute_error,
                             median_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)


def safe_auc(y, score, oriented=False):
    """Computes ROC-AUC, guarding against an undefined single-class split.

    Args:
        y: True binary labels.
        score: Predicted scores or probabilities for the positive class.
        oriented: If True, return `max(auc, 1 - auc)`. Use this for BASELINES,
            where the sign of the statistic is arbitrary -- a rule scoring 0.20
            is 0.80-accurate once you flip it, so reporting 0.20 as the bar
            understates it. Leave False for a trained model, where scoring
            below chance is a failure to surface, not a sign to flip.

    Returns:
        ROC-AUC as a float, or NaN if `y` contains only one class (AUC is
        undefined in that case).
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, score))
    return max(auc, 1.0 - auc) if oriented else auc


def safe_mcc(y, pred):
    """Computes Matthews correlation coefficient, guarding against degenerate input.

    Args:
        y: True binary labels.
        pred: Predicted binary labels.

    Returns:
        MCC as a float, or NaN if either `y` or `pred` contains only one
        class (MCC is undefined/degenerate in that case).
    """
    y, pred = np.asarray(y), np.asarray(pred)
    if len(np.unique(pred)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(matthews_corrcoef(y, pred))


def binary_report(y_true, y_score, y_pred=None, threshold=0.5):
    """Full metric set for a binary classifier.

    Accuracy/precision/recall/F1/ROC-AUC/PR-AUC/MCC/Brier/log-loss +
    confusion matrix.

    Args:
        y_true: True binary labels.
        y_score: Predicted positive-class probability.
        y_pred: Predicted binary labels. Defaults to thresholding `y_score`
            at `threshold` when None.
        threshold: Decision threshold used to derive `y_pred` from
            `y_score` when `y_pred` is not given.

    Returns:
        Dict with keys "n", "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier", "log_loss"
        (each a float, NaN where undefined on a single-class `y_true`), and
        "confusion_matrix" (list of lists).
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(np.int64)
    y_pred = np.asarray(y_pred)
    single_class = len(np.unique(y_true)) < 2
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "mcc": safe_mcc(y_true, y_pred),
        "brier": float("nan") if single_class else float(brier_score_loss(y_true, y_score)),
        "log_loss": float("nan") if single_class else float(
            log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def multiclass_report(y_true, y_pred, y_score=None, class_names=None):
    """Full metric set for a multiclass classifier.

    Accuracy/balanced-accuracy/macro+weighted precision-recall-F1/Cohen's
    kappa + confusion matrix.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_score: Optional per-class predicted probabilities, shape
            (n_samples, n_classes). Adds macro one-vs-rest AUC when given
            and `y_true` has more than one class present.
        class_names: Optional display names for the classes, in the same
            order as the sorted label set. Stored under "labels" for the
            caller to use when printing; falls back to the sorted label
            values themselves when None.

    Returns:
        Dict with keys "n", "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
        "cohen_kappa" (floats), "confusion_matrix" (list of lists),
        "labels" (list), and "macro_auc_ovr" (float, only present when
        `y_score` is given).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    report = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": class_names if class_names is not None else labels,
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            report["macro_auc_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr",
                                                           average="macro", labels=labels))
        except ValueError:
            report["macro_auc_ovr"] = float("nan")
    return report


def regression_report(y_true, y_pred):
    """Full metric set for a regressor.

    MAE/RMSE/R2/median-AE/max-error + residual std.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dict with keys "n" (int) and "MAE", "RMSE", "R2", "median_AE",
        "max_error", "resid_std" (floats; "R2" is NaN when fewer than 2
        samples).
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    resid = y_true - y_pred
    return {
        "n": len(y_true),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "median_AE": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(np.max(np.abs(resid))) if len(resid) else float("nan"),
        "resid_std": float(np.std(resid)) if len(resid) else float("nan"),
    }


def majority_class_baseline(y_train, y_test):
    """Floor: predicting the training set's majority class on every test row.

    Args:
        y_train: Training-set class labels, used to determine the majority
            class.
        y_test: Test-set class labels, used to score the floor prediction.

    Returns:
        Tuple of (majority_class, accuracy, balanced_accuracy).
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    labels = sorted(set(y_train.tolist()))
    maj = max(labels, key=lambda c: (y_train == c).sum())
    pred = np.full_like(y_test, maj)
    return maj, float((pred == y_test).mean()), float(balanced_accuracy_score(y_test, pred))


def predict_mean_baseline(y_train, y_test):
    """Floor: predicting the training set's mean on every test row (regression).

    Args:
        y_train: Training-set target values, used to compute the mean.
        y_test: Test-set target values, used to score the floor prediction.

    Returns:
        A `regression_report` dict (see `regression_report`) scoring the
        constant-mean prediction against `y_test`.
    """
    y_train, y_test = np.asarray(y_train, dtype=np.float64), np.asarray(y_test, dtype=np.float64)
    pred = np.full_like(y_test, y_train.mean())
    return regression_report(y_test, pred)


def persistence_baseline(days_since_prev, horizon_days):
    """Floor: predict positive iff a qualifying event occurred recently.

    Args:
        days_since_prev: Days since the previous qualifying event, per
            sample; NaN where no prior qualifying event is on record.
        horizon_days: Forecast horizon in days. A sample is predicted
            positive iff `days_since_prev <= horizon_days`.

    Returns:
        Array of int64 0/1 predictions, same length as `days_since_prev`.
        NaN entries predict negative (0) -- there is nothing to persist from.
    """
    d = np.asarray(days_since_prev, dtype=np.float64)
    return np.where(np.isnan(d), 0, (d <= horizon_days).astype(int)).astype(np.int64)


def print_report(name, report, digits=4):
    """Prints one of the `*_report` dicts in a consistent console format.

    Args:
        name: Header printed above the block (e.g. a fold/model label).
        report: A dict returned by `binary_report`, `multiclass_report`, or
            `regression_report`.
        digits: Decimal places used when printing float values.
    """
    print(f"\n--- {name} ---")
    for key, value in report.items():
        if key in ("confusion_matrix", "labels"):
            continue
        if isinstance(value, float):
            print(f"  {key:20s} {value:.{digits}f}")
        else:
            print(f"  {key:20s} {value}")
    if "confusion_matrix" in report:
        print("  confusion_matrix:")
        for row in report["confusion_matrix"]:
            print("   ", row)
