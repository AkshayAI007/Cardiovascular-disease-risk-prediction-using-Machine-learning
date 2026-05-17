"""
Shared helper utilities used across the project.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from config import MODEL_DIR, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_STATE) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_model(obj, filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(obj, path)
    logger.info(f"Saved -> {path}")
    return path


def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return joblib.load(path)


def save_csv(df: pd.DataFrame, filename: str, reports_dir: str) -> str:
    path = os.path.join(reports_dir, filename)
    df.to_csv(path, index=True)
    logger.info(f"CSV saved -> {path}")
    return path


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        logger.info(f"Elapsed: {self.elapsed:.2f}s")


# ── Data Sanity Checks ────────────────────────────────────────────────────────

def assert_no_leakage(train_idx, test_idx) -> None:
    overlap = set(train_idx) & set(test_idx)
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} shared indices!"
    logger.info("Leakage check passed.")


def assert_no_nulls(df: pd.DataFrame, stage: str = "") -> None:
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Null values found after {stage}: {nulls}"
    logger.info(f"Null check passed at stage: '{stage}'")


def assert_class_balance(y: pd.Series, min_ratio: float = 0.1) -> None:
    ratio = y.value_counts(normalize=True).min()
    assert ratio >= min_ratio, (
        f"Severe class imbalance: minority class = {ratio:.2%}")
    logger.info(f"Class balance OK — minority ratio: {ratio:.2%}")


# ── Feature Drift ─────────────────────────────────────────────────────────────

def detect_feature_drift(ref: pd.DataFrame, new: pd.DataFrame,
                          threshold: float = 0.2) -> list[str]:
    """
    Flag features whose mean has shifted by more than `threshold` std-devs
    relative to the reference distribution.
    """
    drifted = []
    for col in ref.columns:
        if col not in new.columns:
            continue
        ref_mean, ref_std = ref[col].mean(), ref[col].std()
        new_mean = new[col].mean()
        if ref_std > 0 and abs(new_mean - ref_mean) / ref_std > threshold:
            drifted.append(col)
    if drifted:
        logger.warning(f"Feature drift detected in: {drifted}")
    return drifted


# ── SHAP Explainability ───────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame):
    """Return SHAP values for tree-based models; falls back to KernelExplainer."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. pip install shap")
        return None

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)
    # For binary classification, take positive class values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values, explainer


def plot_shap_summary(shap_values, X: pd.DataFrame,
                       model_name: str, reports_dir: str) -> None:
    try:
        import shap
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X, show=False)
        path = os.path.join(reports_dir,
                            f"shap_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"SHAP plot saved -> {path}")
    except Exception as e:
        logger.warning(f"Could not save SHAP plot: {e}")
