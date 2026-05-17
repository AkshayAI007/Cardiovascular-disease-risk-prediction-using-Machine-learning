"""
Model definitions, hyperparameter grids, and training utilities.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               AdaBoostClassifier,
                               StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from config import PARAM_GRIDS, CV_FOLDS, SCORING_METRIC, RANDOM_STATE


# ── Base Model Registry ───────────────────────────────────────────────────────

def get_base_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            device="cpu", n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVC": SVC(probability=True, random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    }


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

def tune_model(model_name: str, model, X_train, y_train):
    """
    Run GridSearchCV with stratified k-fold CV.
    Falls back to the base model if no grid is defined.
    """
    if model_name not in PARAM_GRIDS:
        print(f"No grid defined for {model_name}, using defaults.")
        model.fit(X_train, y_train)
        return model, {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRIDS[model_name],
        cv=cv,
        scoring=SCORING_METRIC,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"{model_name} -> best params: {grid.best_params_}  "
          f"CV {SCORING_METRIC}: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


# ── Stacking Ensemble ─────────────────────────────────────────────────────────

def build_stacking_ensemble(base_models: list[tuple]) -> StackingClassifier:
    """
    base_models: list of (name, fitted_estimator) tuples.
    Meta-learner is Logistic Regression.
    """
    meta = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=CV_FOLDS,
        passthrough=False,
        n_jobs=-1,
    )


# ── Threshold Optimiser ───────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob,
                            min_precision: float = 0.20) -> float:
    """
    Scan thresholds 0.10–0.90 and return the one that maximises F2-score
    (weights recall 2x over precision) subject to a minimum precision floor.

    Pure-recall optimisation is degenerate — it always drives the threshold
    to the minimum by flagging every patient as positive. F2 still prioritises
    recall heavily but cannot be gamed by flagging everyone (precision would
    collapse below min_precision, excluding that threshold from selection).
    """
    from sklearn.metrics import fbeta_score, precision_score
    import numpy as np

    thresholds = np.arange(0.10, 0.91, 0.01)
    best_t, best_score = thresholds[0], -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec < min_precision:
            continue
        score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t

    print(f"Optimal threshold (F2, min_prec={min_precision}): {best_t:.2f}  "
          f"score: {best_score:.4f}")
    return float(best_t)


# ── Probability Calibration ───────────────────────────────────────────────────

class _PlattCalibratedModel:
    """
    Thin wrapper that applies Platt scaling on top of a pre-fitted classifier.

    sklearn's CalibratedClassifierCV dropped cv='prefit' support in newer
    versions, so we implement the same logic directly:
      1. Run the base model to get raw probability scores.
      2. Fit a logistic regression that maps those scores to calibrated probs.
    Fully serialisable with joblib. Attribute lookups fall through to the
    base model so feature_importances_, classes_, etc. still work.
    """

    def __init__(self, base_model, platt_lr):
        self.base_model = base_model
        self.platt_lr   = platt_lr

    def predict_proba(self, X):
        import numpy as np
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.platt_lr.predict_proba(raw)

    def predict(self, X):
        return self.base_model.predict(X)

    def __getattr__(self, name):
        # Guard own instance attrs: if base_model/platt_lr aren't set yet
        # (e.g. during unpickling), fall through to AttributeError instead
        # of recursing into __getattr__ forever.
        if name in ("base_model", "platt_lr"):
            raise AttributeError(name)
        return getattr(self.base_model, name)


def calibrate_model(model, X_cal, y_cal) -> _PlattCalibratedModel:
    """
    Fit Platt scaling on a held-out calibration set.
    The base model weights are never modified — only the probability
    mapping (a single logistic regression) is adjusted.
    """
    raw_probs = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    platt.fit(raw_probs, y_cal)
    print(f"Platt calibration fitted on {len(y_cal)} samples.")
    return _PlattCalibratedModel(model, platt)
