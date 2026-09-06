"""
Two-stage scalar risk classifier: the best-performing model found for the
three-class (noise / low-risk / high-risk) task.

It uses no image at all. That is the result, not an oversight: on this task
a gradient-boosted model over two physical scalars beats the CNN over
encoded windows by ~9 accuracy points, and the CNN's own failure mode was
station-specific overfitting (see the module docstring in `cnn_riskclass.py`
and report.md's discussion of amplitude as the recurring missing ingredient).

**Why two stages.** `distance_km` is UNDEFINED for noise windows -- there is
no event to measure from -- so in a flat three-class model "distance is
missing" separates noise almost perfectly by construction. That is an
artifact of dataset assembly, not signal, and it is worth about 10 accuracy
points of pure inflation (flat model 91.72% with distance, 81.55% without).
Splitting the decision removes that route entirely:

    Stage 1   noise vs earthquake     log_snr ONLY
    Stage 2   low_risk vs high_risk   log_snr + log_distance

Stage 2 trains only on earthquake windows, where distance is genuinely
measured, and there the pair is not two arbitrary features -- it is the
local-magnitude relation the rest of this project already relies on
(`cnn_regression.py`, report.md 4.5): observed amplitude plus distance
determines magnitude. Class probabilities are recombined by the chain rule,
so the three outputs are a proper distribution.

**Selection discipline.** Both stages' class-weight exponents are chosen by
station-grouped 5-fold CV on train+val pooled, never on test. The held-out
`val` split alone is unusable for ranking here: it is 736 windows from just
two noise stations, and it ranked the CNN (val MCC 0.873) above gradient
boosting (0.867) when their test MCCs are 0.599 and 0.851 respectively.
Every failure diagnosed on this task has been station generalization, so
the selection metric has to see many unseen stations, which is exactly what
GroupKFold over stations provides.

Usage:
    python riskclass_scalar.py --dataset-dir ../seismic_cli/data/dataset_riskclass_3s_v2

Not imported by anything else -- standalone script.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

RISK_CLASSES = ["00_noise", "01_low_risk", "02_high_risk"]
CLASS_TO_IDX = {c: i for i, c in enumerate(RISK_CLASSES)}

STAGE1_FEATURES = ["log_snr"]
STAGE2_FEATURES = ["log_snr", "log_distance"]


def _make_gb(**kw):
    """Builds one stage's HistGradientBoostingClassifier with shared hyperparameters.

    Args:
        **kw: Extra keyword arguments forwarded to
            `HistGradientBoostingClassifier` (e.g. `sample_weight` is passed
            to `.fit`, not here; this only builds the estimator).

    Returns:
        An unfitted `HistGradientBoostingClassifier`.
    """
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
        l2_regularization=5.0, early_stopping=True, random_state=42, **kw)


def _weights(y: np.ndarray, power: float) -> np.ndarray:
    """Computes inverse-frequency class weights raised to `power` (0 = unweighted).

    Args:
        y: Integer class labels.
        power: Exponent applied to the inverse-frequency weight; 0 gives
            uniform (unweighted) weights, 1 gives full inverse-frequency.

    Returns:
        Array of per-sample weights, same length as `y`.
    """
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
    w = (counts.sum() / np.maximum(counts, 1.0)) ** power
    return w[y]


def fit_two_stage(df: pd.DataFrame, power_stage1: float, power_stage2: float):
    """Fits the stage-1 (noise vs. earthquake) and stage-2 (low vs. high risk) models.

    Args:
        df: Training rows with columns 'y' (int class label, 0=noise,
            1=low_risk, 2=high_risk), plus `STAGE1_FEATURES`/
            `STAGE2_FEATURES` columns.
        power_stage1: Class-weight exponent for stage 1 (see `_weights`).
        power_stage2: Class-weight exponent for stage 2 (see `_weights`).

    Returns:
        Tuple of (stage1_model, stage2_model), both fitted
        `HistGradientBoostingClassifier`s.
    """
    is_eq = (df.y.values > 0).astype(int)
    m1 = _make_gb()
    m1.fit(df[STAGE1_FEATURES].to_numpy(float), is_eq,
           sample_weight=_weights(is_eq, power_stage1))

    eq = df[df.y > 0]
    is_hi = (eq.y.values == 2).astype(int)
    m2 = _make_gb()
    # HistGradientBoosting handles the residual real NaNs (stations missing
    # from the station catalog) natively, so no fill value is invented.
    m2.fit(eq[STAGE2_FEATURES].to_numpy(float), is_hi,
           sample_weight=_weights(is_hi, power_stage2))
    return m1, m2


def predict_two_stage(m1, m2, df: pd.DataFrame) -> np.ndarray:
    """Combines both stages' predictions into a 3-class probability distribution.

    Recombines by the chain rule: P(noise) = 1-P(eq), P(low) = P(eq)*(1-P(hi)),
    P(high) = P(eq)*P(hi).

    Args:
        m1: Fitted stage-1 model (noise vs. earthquake).
        m2: Fitted stage-2 model (low vs. high risk, earthquakes only).
        df: Rows to predict, with `STAGE1_FEATURES`/`STAGE2_FEATURES` columns.

    Returns:
        Array of shape (n_rows, 3), columns ordered as `RISK_CLASSES`
        (noise, low_risk, high_risk); each row sums to 1.
    """
    p_eq = m1.predict_proba(df[STAGE1_FEATURES].to_numpy(float))[:, 1]
    p_hi = m2.predict_proba(df[STAGE2_FEATURES].to_numpy(float))[:, 1]
    return np.column_stack([1.0 - p_eq, p_eq * (1.0 - p_hi), p_eq * p_hi])


def load(dataset_dir: str):
    """Loads the manifest and derives the log-distance feature and integer label.

    Args:
        dataset_dir: Directory from `seismic-cli generate-riskclass-dataset`
            containing manifest.csv.

    Returns:
        Manifest DataFrame with added 'log_distance' (log of distance_km,
        clipped to a 1 km floor; NaN where distance_km is absent) and 'y'
        (int class label from `CLASS_TO_IDX`) columns.
    """
    root = Path(dataset_dir)
    m = pd.read_csv(root / "manifest.csv")
    if "distance_km" not in m.columns:
        m["distance_km"] = np.nan
    m["log_distance"] = np.log(m["distance_km"].clip(lower=1.0))
    m["y"] = m.risk_class.map(CLASS_TO_IDX)
    return m


def main():
    """Selects class-weight powers by station-grouped CV, then fits and reports on test.

    Returns:
        None. Prints the CV selection, then test accuracy/macro-AUC/MCC/
        balanced accuracy, a confusion matrix, a classification report, and
        each stage's own AUC.
    """
    p = argparse.ArgumentParser(description="Two-stage scalar risk classifier.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory from `seismic-cli generate-riskclass-dataset`.")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    m = load(args.dataset_dir)
    dev = m[m.split.isin(["train", "val"])].copy()
    test = m[m.split == "test"].copy()
    print(f"dev {len(dev)} windows / {dev.station_key.nunique()} stations   "
          f"test {len(test)} windows / {test.station_key.nunique()} stations")

    gkf = GroupKFold(n_splits=args.folds)
    groups = dev.station_key.values
    best, best_score = None, -np.inf
    for pw1 in (0.0, 0.5, 0.75):
        for pw2 in (0.0, 0.5, 0.75, 1.0):
            scores = []
            for tr_i, va_i in gkf.split(dev, dev.y, groups):
                d_tr, d_va = dev.iloc[tr_i], dev.iloc[va_i]
                if d_tr.y.nunique() < 3 or d_va.y.nunique() < 2:
                    continue
                m1, m2 = fit_two_stage(d_tr, pw1, pw2)
                scores.append(recall_score(
                    d_va.y, predict_two_stage(m1, m2, d_va).argmax(1),
                    average="macro", zero_division=0))
            if scores and np.mean(scores) > best_score:
                best, best_score = (pw1, pw2), float(np.mean(scores))
    print(f"selected class-weight powers by station-grouped CV: "
          f"stage1={best[0]} stage2={best[1]} (CV balanced accuracy {best_score:.4f})")

    m1, m2 = fit_two_stage(dev, *best)
    P = predict_two_stage(m1, m2, test)
    pred = P.argmax(1)
    print("\n--- Test (evaluated once, after selection) ---")
    print(f"  Accuracy      {accuracy_score(test.y, pred) * 100:.2f}%")
    print(f"  Macro-AUC     {roc_auc_score(test.y, P, multi_class='ovr', average='macro'):.4f}")
    print(f"  MCC           {matthews_corrcoef(test.y, pred):+.4f}")
    print(f"  Balanced acc  {recall_score(test.y, pred, average='macro'):.4f}")
    print("\n  Confusion (rows=true, cols=pred):", RISK_CLASSES)
    print(confusion_matrix(test.y, pred))
    print("  per-class recall:",
          {c: round(float((pred[test.y.values == i] == i).mean()), 3)
           for i, c in enumerate(RISK_CLASSES)})
    print("\nClassification Report:")
    print(classification_report(test.y, pred, target_names=RISK_CLASSES, digits=4, zero_division=0))

    eq = test.y > 0
    print(f"\n  stage-1 AUC (noise vs earthquake): "
          f"{roc_auc_score((test.y > 0).astype(int), P[:, 1] + P[:, 2]):.4f}")
    print(f"  stage-2 AUC (low vs high, earthquakes only): "
          f"{roc_auc_score((test.y[eq] == 2).astype(int), m2.predict_proba(test[eq][STAGE2_FEATURES].to_numpy(float))[:, 1]):.4f}")


if __name__ == "__main__":
    main()
