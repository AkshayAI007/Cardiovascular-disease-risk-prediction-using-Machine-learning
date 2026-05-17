"""
Model evaluation: metrics computation, reporting, and cross-validation scoring.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from config import RANDOM_STATE, CV_FOLDS, SCORING_METRIC


# ── Single-split Metrics ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None,
                    threshold: float = 0.5) -> dict:
    if threshold != 0.5 and y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_report(y_true, y_pred, model_name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print("=" * 50)
    print(classification_report(y_true, y_pred,
                                 target_names=["No CHD", "CHD"]))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    print(f"  PPV (Precision):   {ppv:.4f}")
    print(f"  NPV:               {npv:.4f}")


# ── Cross-Validation Scoring ──────────────────────────────────────────────────

def cross_validate_model(model, X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision",
        "recall":    "recall",
        "f1":        "f1",
        "roc_auc":   "roc_auc",
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                             return_train_score=True, n_jobs=-1)
    rows = []
    for metric in scoring:
        rows.append({
            "Metric": metric,
            "Train Mean": results[f"train_{metric}"].mean(),
            "Train Std":  results[f"train_{metric}"].std(),
            "Val Mean":   results[f"test_{metric}"].mean(),
            "Val Std":    results[f"test_{metric}"].std(),
        })
    return pd.DataFrame(rows)


# ── Aggregate Comparison Table ────────────────────────────────────────────────

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    all_results: {model_name: {"y_true", "y_pred", "y_prob", "threshold"}}

    y_pred is already threshold-adjusted (tuned on val set during training),
    so we use it directly rather than recomputing at 0.5.
    ROC_AUC is computed from y_prob (threshold-independent).
    """
    rows = []
    for name, data in all_results.items():
        m = {
            "Accuracy":  accuracy_score(data["y_true"], data["y_pred"]),
            "Precision": precision_score(data["y_true"], data["y_pred"],
                                         zero_division=0),
            "Recall":    recall_score(data["y_true"], data["y_pred"],
                                      zero_division=0),
            "F1":        f1_score(data["y_true"], data["y_pred"],
                                  zero_division=0),
            "Threshold": data.get("threshold", 0.5),
        }
        if data.get("y_prob") is not None:
            m["ROC_AUC"] = roc_auc_score(data["y_true"], data["y_prob"])
        m["Model"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Model")
    col_order = [c for c in ["Accuracy", "Precision", "Recall",
                              "F1", "ROC_AUC", "Threshold"]
                 if c in df.columns]
    return df[col_order].sort_values("ROC_AUC", ascending=False)


# ── Clinical Utility ──────────────────────────────────────────────────────────

def net_benefit_curve(y_true, y_prob,
                      thresholds: np.ndarray = None) -> pd.DataFrame:
    """
    Decision-curve analysis: net benefit = TP/n - FP/n * (pt / (1-pt))
    where pt is the probability threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.50, 0.01)
    n = len(y_true)
    rows = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb = tp / n - (fp / n) * (pt / (1 - pt))
        rows.append({"threshold": pt, "net_benefit": nb})
    return pd.DataFrame(rows)
