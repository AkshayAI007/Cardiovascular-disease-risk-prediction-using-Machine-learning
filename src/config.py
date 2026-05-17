"""
Central configuration for all paths, hyperparameters, and constants.
"""
import os


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_cardiovascular_risk.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
TARGET_COL = "TenYearCHD"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_COLS = ["sex", "is_smoking", "education", "BPMeds",
                    "prevalentStroke", "prevalentHyp", "diabetes"]
CONTINUOUS_COLS  = ["age", "cigsPerDay", "totChol", "sysBP",
                    "diaBP", "BMI", "heartRate", "glucose"]

# Columns filled with median (continuous) vs mode (categorical)
MEDIAN_FILL_COLS = ["glucose", "totChol", "cigsPerDay", "BMI", "heartRate"]
MODE_FILL_COLS   = ["education", "BPMeds"]

# Columns one-hot encoded
OHE_COLS = ["sex", "is_smoking"]

# Log-transform targets (skewed continuous features)
LOG_TRANSFORM_COLS = ["cigsPerDay", "totChol", "BMI", "heartRate",
                      "glucose", "pulsePressure"]

# ── Class Imbalance ───────────────────────────────────────────────────────────
SMOTE_RANDOM_STATE = 42

# ── Model Hyperparameter Grids ────────────────────────────────────────────────
PARAM_GRIDS = {
    # l1_ratio=0 → L2,  l1_ratio=1 → L1  (saga supports both; no FutureWarning)
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "l1_ratio": [0, 1],
        "solver": ["saga"],
        "max_iter": [2000],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        # device is fixed at model init; no need to grid-search it
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "probability": [True],
    },
    "NaiveBayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5],
    },
}

CV_FOLDS = 5
SCORING_METRIC = "roc_auc"
