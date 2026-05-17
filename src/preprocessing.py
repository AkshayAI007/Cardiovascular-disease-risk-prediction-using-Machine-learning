"""
Data preprocessing: imputation, encoding, scaling, and SMOTE.

Correct order (no leakage):
  raw df
    → stratified split (train / val / test)
    → fit imputer on X_train  →  transform X_val, X_test
    → deterministic transforms (pulse pressure, age groups, log, etc.)
    → fit stat transforms on X_train (IQR fences, column selector)
       →  apply to X_val, X_test
    → fit scaler on X_train  →  transform X_val, X_test
    → SMOTE on X_train only

All fitted objects are returned inside 'artifacts' for serialisation and
reuse at inference time (predict.py).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from config import (
    TARGET_COL, MEDIAN_FILL_COLS, MODE_FILL_COLS,
    OHE_COLS, SMOTE_RANDOM_STATE, TEST_SIZE, RANDOM_STATE,
)
from feature_engineering import (
    apply_deterministic_transforms,
    fit_statistical_transforms,
    apply_statistical_transforms,
)


# ── Imputation ────────────────────────────────────────────────────────────────

def fit_imputer(X_train: pd.DataFrame) -> dict:
    """Compute fill values from training data only."""
    fill_values: dict = {}
    for col in MEDIAN_FILL_COLS:
        if col in X_train.columns:
            fill_values[col] = X_train[col].median()
    for col in MODE_FILL_COLS:
        if col in X_train.columns:
            fill_values[col] = X_train[col].mode()[0]
    return fill_values


def apply_imputer(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    df = df.copy()
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ── Categorical Encoding ──────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode OHE_COLS. Use drop_first=False at inference so a
    single-row DataFrame still produces all dummy columns."""
    df = df.copy()
    for col in OHE_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────

def fit_scaler(X_train: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    return X_scaled, scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Applied on training fold only — after scaling, after split."""
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=TARGET_COL)
    print(f"After SMOTE -> {y_res.value_counts().to_dict()}")
    return X_res, y_res


# ── Master pipeline ───────────────────────────────────────────────────────────

def preprocess_train(df: pd.DataFrame,
                     val_size: float = 0.20):
    """
    Run the full preprocessing pipeline with no data leakage.

    Returns
    -------
    X_train_res, y_train_res : SMOTE-resampled training features + labels
    X_val_scaled, y_val       : scaled validation set (for threshold tuning)
    X_test_scaled, y_test     : scaled test set (final evaluation only)
    artifacts                 : dict of all fitted objects for inference reuse
    """
    # 1 — three-way stratified split
    X_raw = df.drop(columns=[TARGET_COL])
    y     = df[TARGET_COL]

    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    relative_val = val_size / (1 - TEST_SIZE)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val,
        random_state=RANDOM_STATE, stratify=y_temp
    )

    # 2 — impute (fitted on training raw data)
    fill_values = fit_imputer(X_train_raw)
    X_train_imp = apply_imputer(X_train_raw, fill_values)
    X_val_imp   = apply_imputer(X_val_raw,   fill_values)
    X_test_imp  = apply_imputer(X_test_raw,  fill_values)

    # 3 — categorical encoding (deterministic, safe on all sets)
    #     Must happen before deterministic transforms that reference encoded names
    def _encode(X, y_series):
        tmp = X.copy()
        tmp[TARGET_COL] = y_series.values
        tmp = encode_categoricals(tmp)
        return tmp.drop(columns=[TARGET_COL])

    X_train_enc = _encode(X_train_imp, y_train)
    X_val_enc   = _encode(X_val_imp,   y_val)
    X_test_enc  = _encode(X_test_imp,  y_test)

    # 4 — deterministic feature engineering (no statistics, safe on all sets)
    X_train_det = apply_deterministic_transforms(X_train_enc)
    X_val_det   = apply_deterministic_transforms(X_val_enc)
    X_test_det  = apply_deterministic_transforms(X_test_enc)

    # 5 — fit statistical transforms on training data only
    stat_params = fit_statistical_transforms(X_train_det)
    X_train_sel = apply_statistical_transforms(X_train_det, stat_params)
    X_val_sel   = apply_statistical_transforms(X_val_det,   stat_params)
    X_test_sel  = apply_statistical_transforms(X_test_det,  stat_params)

    # 6 — scale (fitted on training data only)
    X_train_scaled, scaler = fit_scaler(X_train_sel)
    X_val_scaled  = apply_scaler(X_val_sel,  scaler)
    X_test_scaled = apply_scaler(X_test_sel, scaler)

    # 7 — SMOTE on training fold only
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    artifacts = {
        "fill_values": fill_values,
        "stat_params":  stat_params,
        "scaler":       scaler,
        "feature_cols": list(X_train_scaled.columns),
    }

    print(f"\nSplit sizes -> train: {len(X_train_res)}  "
          f"val: {len(X_val_scaled)}  test: {len(X_test_scaled)}")
    return (X_train_res, y_train_res,
            X_val_scaled, y_val,
            X_test_scaled, y_test,
            artifacts)


def preprocess_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Apply the exact same transforms used during training to new/unseen data.
    Uses only serialised statistics — never re-fits on inference data.
    """
    df = apply_imputer(df, artifacts["fill_values"])
    df = encode_categoricals(df, drop_first=False)
    df = apply_deterministic_transforms(df)
    df = apply_statistical_transforms(df, artifacts["stat_params"])

    # Align to training column set
    feature_cols = artifacts["feature_cols"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return apply_scaler(df, artifacts["scaler"])
