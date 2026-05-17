"""
Feature engineering split into two phases:

  Phase 1 — Deterministic transforms (safe to apply before or after split,
             no statistics computed from data):
             create_pulse_pressure, create_age_group, create_bmi_category,
             create_smoking_intensity, log_transform

  Phase 2 — Fitted/statistical transforms (MUST be fitted on training data
             only, then applied to test/inference data):
             fit_outlier_clipper / apply_outlier_clipper
             fit_feature_selector / apply_feature_selector

  Master helpers:
             apply_deterministic_transforms(df)   ← Phase 1 only
             fit_statistical_transforms(X_train)  ← returns params dict
             apply_statistical_transforms(X, params) ← applies saved params
"""
import numpy as np
import pandas as pd
from config import LOG_TRANSFORM_COLS, TARGET_COL


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Deterministic (no data statistics needed)
# ═══════════════════════════════════════════════════════════════════════════════

def create_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """sysBP - diaBP is a validated cardiovascular risk marker."""
    df = df.copy()
    if "sysBP" in df.columns and "diaBP" in df.columns:
        df["pulsePressure"] = df["sysBP"] - df["diaBP"]
        df.drop(columns=["sysBP", "diaBP"], inplace=True)
    return df


def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into four clinical risk bands (ACC/AHA guidelines)."""
    df = df.copy()
    if "age" in df.columns:
        bins   = [0, 40, 50, 60, 100]
        labels = ["age_lt40", "age_40_50", "age_50_60", "age_60plus"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
        dummies = pd.get_dummies(df["age_group"], prefix="ag", drop_first=True)
        df = pd.concat([df.drop(columns=["age_group"]), dummies], axis=1)
    return df


def create_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """WHO BMI thresholds as binary flags."""
    df = df.copy()
    if "BMI" in df.columns:
        df["is_obese"]      = (df["BMI"] >= 30).astype(int)
        df["is_overweight"] = ((df["BMI"] >= 25) & (df["BMI"] < 30)).astype(int)
    return df


def create_smoking_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smoker flag × cigarettes/day interaction captures pack-year exposure
    more precisely than either feature alone.
    Detects the OHE column name dynamically (e.g. is_smoking_YES or
    is_smoking_1) so it works regardless of raw category labels.
    """
    df = df.copy()
    smoking_col = next(
        (c for c in df.columns if c.startswith("is_smoking_")), None
    )
    if smoking_col and "cigsPerDay" in df.columns:
        df["smoking_intensity"] = df[smoking_col].astype(float) * df["cigsPerDay"]
    return df


def log_transform(df: pd.DataFrame,
                  cols: list = LOG_TRANSFORM_COLS) -> pd.DataFrame:
    """log1p to reduce right-skew; clip(lower=0) guards against negative values."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def apply_deterministic_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Run all Phase-1 transforms in the correct order."""
    df = create_pulse_pressure(df)
    df = create_age_group(df)
    df = create_bmi_category(df)
    df = create_smoking_intensity(df)
    df = log_transform(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fitted/statistical (fit on X_train, apply separately to X_test)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_outlier_clipper(X_train: pd.DataFrame,
                        factor: float = 1.5) -> dict:
    """
    Compute IQR fences from training data.
    Returns a dict  {col: (lower_fence, upper_fence)}  for serialisation.
    """
    fences = {}
    for col in X_train.select_dtypes(include="number").columns:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        fences[col] = (q1 - factor * iqr, q3 + factor * iqr)
    return fences


def apply_outlier_clipper(X: pd.DataFrame, fences: dict) -> pd.DataFrame:
    """Winsorise using pre-computed training fences."""
    X = X.copy()
    for col, (lower, upper) in fences.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lower, upper=upper)
    return X


def fit_feature_selector(X_train: pd.DataFrame,
                          var_threshold: float = 0.01,
                          corr_threshold: float = 0.90) -> dict:
    """
    Decide which columns to drop based ONLY on training data.
    Returns a dict with 'low_var_cols' and 'high_corr_cols' to drop,
    and 'keep_cols' — the final column list.
    """
    low_var_cols = (X_train.var()[X_train.var() < var_threshold]
                    .index.tolist())

    remaining = X_train.drop(columns=low_var_cols, errors="ignore")
    corr = remaining.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_cols = [c for c in upper.columns if any(upper[c] > corr_threshold)]

    drop_cols = list(set(low_var_cols + high_corr_cols))
    keep_cols = [c for c in X_train.columns if c not in drop_cols]

    if low_var_cols:
        print(f"  Dropping low-variance: {low_var_cols}")
    if high_corr_cols:
        print(f"  Dropping high-correlation: {high_corr_cols}")

    return {"low_var_cols": low_var_cols,
            "high_corr_cols": high_corr_cols,
            "keep_cols": keep_cols}


def apply_feature_selector(X: pd.DataFrame, selector: dict) -> pd.DataFrame:
    """Keep only the columns that survived the training-data selection."""
    keep = [c for c in selector["keep_cols"] if c in X.columns]
    missing = [c for c in selector["keep_cols"] if c not in X.columns]
    if missing:
        print(f"  Warning — columns absent at inference: {missing}")
    return X[keep]


def fit_statistical_transforms(X_train: pd.DataFrame,
                                var_threshold: float = 0.01,
                                corr_threshold: float = 0.90) -> dict:
    """
    Fit all Phase-2 statistical transforms on X_train.
    Returns a single 'stat_params' dict for serialisation into the model bundle.
    """
    fences   = fit_outlier_clipper(X_train)
    selector = fit_feature_selector(X_train, var_threshold, corr_threshold)
    return {"fences": fences, "selector": selector}


def apply_statistical_transforms(X: pd.DataFrame,
                                  stat_params: dict) -> pd.DataFrame:
    """Apply pre-fitted Phase-2 params to X (train fold or test/inference)."""
    X = apply_outlier_clipper(X, stat_params["fences"])
    X = apply_feature_selector(X, stat_params["selector"])
    return X
