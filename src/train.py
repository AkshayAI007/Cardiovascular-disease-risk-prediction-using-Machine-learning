"""
End-to-end training pipeline.

Run:
    python src/train.py
"""
import matplotlib
matplotlib.use("Agg")  # must be first — before any other import touches pyplot

import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import MODEL_DIR, RANDOM_STATE, REPORTS_DIR
from data_loader import load_data, basic_info, class_distribution
from preprocessing import preprocess_train
from models import (get_base_models, tune_model,
                    find_optimal_threshold, build_stacking_ensemble,
                    calibrate_model)
from evaluation import (compute_metrics, print_report,
                         cross_validate_model, build_comparison_table,
                         net_benefit_curve)
from visualization import (
    plot_class_distribution, plot_missing_heatmap,
    plot_continuous_distributions, plot_correlation_heatmap,
    plot_confusion_matrix, plot_roc_curves, plot_metrics_comparison,
    plot_feature_importance, plot_learning_curve, plot_calibration_curve,
)
from utils import set_seed, Timer, save_csv
from sklearn.model_selection import learning_curve, StratifiedKFold


def run_eda(df: pd.DataFrame) -> None:
    basic_info(df)
    class_distribution(df)
    plot_class_distribution(df)
    plot_missing_heatmap(df)
    cont_cols = [c for c in ["age", "cigsPerDay", "totChol", "sysBP",
                              "diaBP", "BMI", "heartRate", "glucose"]
                 if c in df.columns]
    plot_continuous_distributions(df, cont_cols)
    plot_correlation_heatmap(df.select_dtypes("number"), "raw")


def train_all_models(X_train, y_train,
                     X_val, y_val,
                     X_test, y_test) -> tuple[dict, dict]:
    """
    Train every model, tune threshold on VAL set (not test set),
    then evaluate final metrics on TEST set.
    """
    base_models = get_base_models()
    all_results: dict  = {}
    best_estimators: dict = {}

    for name, model in base_models.items():
        print(f"\n>>> Training: {name}")
        with Timer():
            tuned_model, _ = tune_model(name, model, X_train, y_train)

        has_proba = hasattr(tuned_model, "predict_proba")

        # ── Threshold tuned on VALIDATION set ────────────────────────────────
        if has_proba:
            val_prob = tuned_model.predict_proba(X_val)[:, 1]
            opt_threshold = find_optimal_threshold(y_val, val_prob)
        else:
            opt_threshold = 0.5

        # ── Final evaluation on TEST set (threshold fixed, never re-tuned) ──
        test_prob = tuned_model.predict_proba(X_test)[:, 1] if has_proba else None
        test_pred = ((test_prob >= opt_threshold).astype(int)
                     if test_prob is not None
                     else tuned_model.predict(X_test))

        print_report(y_test, test_pred, name)
        metrics = compute_metrics(y_test, test_pred, test_prob)
        metrics["Threshold"] = opt_threshold
        print(f"  Threshold (val-tuned): {opt_threshold:.2f} | "
              f"AUC: {metrics.get('ROC_AUC', 'N/A'):.4f}")

        plot_confusion_matrix(y_test, test_pred, name)
        if test_prob is not None:
            plot_calibration_curve(y_test, test_prob, name)

        all_results[name] = {
            "y_true":    y_test,
            "y_pred":    test_pred,
            "y_prob":    test_prob,
            "metrics":   metrics,
            "threshold": opt_threshold,
        }
        best_estimators[name] = tuned_model

    return all_results, best_estimators


def add_stacking_model(best_estimators: dict,
                       X_train, y_train,
                       X_val, y_val,
                       X_test, y_test,
                       all_results: dict) -> None:
    """Build a stacking ensemble from the top-3 models by ROC AUC."""
    sorted_names = sorted(
        all_results,
        key=lambda n: all_results[n]["metrics"].get("ROC_AUC", 0),
        reverse=True,
    )[:3]
    print(f"\n>>> Stacking ensemble from: {sorted_names}")

    base = [(n, best_estimators[n]) for n in sorted_names]
    stacker = build_stacking_ensemble(base)

    X_stack_train = pd.concat([
        pd.DataFrame(X_train), pd.DataFrame(X_val)
    ]).reset_index(drop=True)
    y_stack_train = pd.concat([y_train, y_val]).reset_index(drop=True)

    with Timer():
        stacker.fit(X_stack_train, y_stack_train)

    test_prob = stacker.predict_proba(X_test)[:, 1]
    opt_threshold = find_optimal_threshold(y_val,
                                            stacker.predict_proba(X_val)[:, 1])
    test_pred = (test_prob >= opt_threshold).astype(int)

    print_report(y_test, test_pred, "Stacking")
    metrics = compute_metrics(y_test, test_pred, test_prob)
    metrics["Threshold"] = opt_threshold
    plot_confusion_matrix(y_test, test_pred, "Stacking")
    plot_calibration_curve(y_test, test_prob, "Stacking")

    all_results["Stacking"] = {
        "y_true":    y_test,
        "y_pred":    test_pred,
        "y_prob":    test_prob,
        "metrics":   metrics,
        "threshold": opt_threshold,
    }
    best_estimators["Stacking"] = stacker


def select_best_model(all_results: dict) -> str:
    comparison = build_comparison_table(all_results)
    print("\n--- Model Comparison (test set, threshold-tuned predictions) ---")
    print(comparison.to_string())
    return comparison.index[0]


def save_artifacts(best_model, artifacts: dict,
                   model_name: str, threshold: float) -> None:
    bundle = {
        "model":        best_model,
        "fill_values":  artifacts["fill_values"],
        "stat_params":  artifacts["stat_params"],
        "scaler":       artifacts["scaler"],
        "feature_cols": artifacts["feature_cols"],
        "model_name":   model_name,
        "threshold":    threshold,
    }
    path = os.path.join(MODEL_DIR, "best_model_bundle.joblib")
    joblib.dump(bundle, path)
    print(f"\nModel bundle saved -> {path}")


def main() -> None:
    set_seed(RANDOM_STATE)

    # 1. Load & EDA
    df = load_data()
    run_eda(df)

    # 2. Leakage-safe preprocessing: train / val / test
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     artifacts) = preprocess_train(df)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # 3. Train + tune threshold on val set
    all_results, best_estimators = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. Stacking ensemble
    add_stacking_model(best_estimators, X_train, y_train,
                       X_val, y_val, X_test, y_test, all_results)

    # 5. Comparison charts
    plot_roc_curves(all_results)
    comparison_df = build_comparison_table(all_results).reset_index()
    plot_metrics_comparison(comparison_df)
    save_csv(comparison_df, "model_comparison.csv", REPORTS_DIR)

    # 6. Best model selection
    best_name  = select_best_model(all_results)
    best_model = best_estimators[best_name]
    best_threshold = all_results[best_name]["threshold"]
    print(f"\nBest model: {best_name}  threshold: {best_threshold:.2f}")

    # 7. Cross-validation on base model before calibration
    # sklearn's cross_validate needs to clone the estimator internally,
    # which requires a native sklearn class — run CV on the base model here.
    print(f"\n--- Cross-Validation: {best_name} ---")
    X_all = pd.concat([X_train, X_val, X_test])
    y_all = pd.concat([y_train, y_val, y_test])
    cv_df = cross_validate_model(best_model, X_all, y_all)
    print(cv_df.to_string(index=False))
    save_csv(cv_df, f"cv_{best_name}.csv", REPORTS_DIR)

    # 8. Feature importance (base model, before calibration wrapper)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_,
                        index=artifacts["feature_cols"])
        plot_feature_importance(imp, best_name)

    # 8b. Probability calibration (Platt scaling on val set)
    # Calibration adjusts the probability scale so that a predicted 70% risk
    # reflects a ~70% real-world frequency, which matters for clinical use.
    # The base model weights are unchanged; only the probability mapping shifts.
    # CV is done above on the uncalibrated model so sklearn can clone it freely.
    print(f"\n--- Calibrating: {best_name} (Platt scaling) ---")
    base_model_for_lc = best_model          # keep for learning curve (needs sklearn clone)
    best_model = calibrate_model(best_model, X_val, y_val)

    # Threshold was tuned on uncalibrated probs — re-tune on calibrated probs
    cal_val_prob = best_model.predict_proba(X_val)[:, 1]
    best_threshold = find_optimal_threshold(y_val, cal_val_prob)

    # Evaluate calibrated model on test set and update stored results
    cal_test_prob = best_model.predict_proba(X_test)[:, 1]
    cal_test_pred = (cal_test_prob >= best_threshold).astype(int)
    print_report(y_test, cal_test_pred, f"{best_name} (calibrated)")
    plot_calibration_curve(y_test, cal_test_prob, f"{best_name}_calibrated")
    all_results[best_name]["y_prob"]      = cal_test_prob
    all_results[best_name]["threshold"]   = best_threshold

    # 9. Learning curve on training data (SMOTE-resampled)
    # Uses base model — learning_curve needs to clone the estimator internally,
    # which requires a native sklearn class.
    print(f"\n--- Learning Curve: {best_name} ---")
    lc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sizes, tr_sc, val_sc = learning_curve(
        base_model_for_lc, X_train, y_train,
        cv=lc_cv, scoring="roc_auc",
        train_sizes=np.linspace(0.3, 1.0, 8),
        n_jobs=-1,
    )
    plot_learning_curve(tr_sc, val_sc, sizes, best_name)

    # 10. Decision-curve analysis (clinical utility)
    print(f"\n--- Decision-Curve Analysis: {best_name} ---")
    dca_df = net_benefit_curve(
        all_results[best_name]["y_true"],
        all_results[best_name]["y_prob"],
    )
    save_csv(dca_df, f"dca_{best_name}.csv", REPORTS_DIR)

    # 11. Persist everything needed for inference
    save_artifacts(best_model, artifacts, best_name, best_threshold)


if __name__ == "__main__":
    main()
