"""Shared accuracy-metric reports and baselines for the classifiers/regressors
in this repo. Each `*_report` function returns a plain dict (safe to print,
log, or diff across runs); `print_report` gives every script the same
console format instead of a bespoke print block per script.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, mean_absolute_error,
                             median_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)


def safe_auc(y, score):
    """ROC-AUC, or NaN when the split is single-class (undefined otherwise)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_mcc(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if len(np.unique(pred)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(matthews_corrcoef(y, pred))


def binary_report(y_true, y_score, y_pred=None, threshold=0.5):
    """Accuracy/precision/recall/F1/ROC-AUC/PR-AUC/MCC/Brier + confusion
    matrix for a binary classifier. `y_score` is the positive-class
    probability; `y_pred` defaults to thresholding it at `threshold`."""
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
    """Accuracy/balanced-accuracy/macro+weighted P-R-F1/Cohen's kappa +
    confusion matrix for a multiclass classifier. `y_score` (per-class
    probabilities) adds macro one-vs-rest AUC when given."""
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
    """MAE/RMSE/R2/median-AE/max-error + residual std for a regressor."""
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
    """Predicting the training set's majority class on every test row."""
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    labels = sorted(set(y_train.tolist()))
    maj = max(labels, key=lambda c: (y_train == c).sum())
    pred = np.full_like(y_test, maj)
    return maj, float((pred == y_test).mean()), float(balanced_accuracy_score(y_test, pred))


def predict_mean_baseline(y_train, y_test):
    """Predicting the training set's mean on every test row (regression floor)."""
    y_train, y_test = np.asarray(y_train, dtype=np.float64), np.asarray(y_test, dtype=np.float64)
    pred = np.full_like(y_test, y_train.mean())
    return regression_report(y_test, pred)


def persistence_baseline(days_since_prev, horizon_days):
    """Predict positive iff a qualifying event occurred within the previous
    `horizon_days`. NaN (no prior qualifying event on record) predicts
    negative -- there is nothing to persist from."""
    d = np.asarray(days_since_prev, dtype=np.float64)
    return np.where(np.isnan(d), 0, (d <= horizon_days).astype(int)).astype(np.int64)


def print_report(name, report, digits=4):
    """Consistent one-block console print for any of the `*_report` dicts above."""
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
