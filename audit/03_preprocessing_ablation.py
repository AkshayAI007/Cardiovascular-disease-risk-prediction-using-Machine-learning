"""
Audit step 3 — ablation of the preprocessing decisions.

Every variant is evaluated with 5x5 repeated stratified CV on the full dataset,
with ALL fitting (imputation, clipping, selection, scaling, resampling) done
inside each training fold. Variants A-D reuse the repo's own functions from
src/; E-J are plain sklearn/imblearn pipelines.

Run from the repo root (takes a few minutes):
    python audit/03_preprocessing_ablation.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             log_loss, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import feature_engineering as FE  # noqa: E402
import preprocessing as P  # noqa: E402
from config import TARGET_COL  # noqa: E402

df = pd.read_csv(REPO / "data_cardiovascular_risk.csv").drop(columns=["id"])
X_raw, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL].values
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)


def metrics(yt, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(logit, yt).coef_[0, 0]
    return {"AUC": roc_auc_score(yt, p), "PR_AUC": average_precision_score(yt, p),
            "Brier": brier_score_loss(yt, p), "LogLoss": log_loss(yt, p),
            "CalSlope": slope, "MeanPred-Obs": p.mean() - yt.mean()}


def summarise(name, rows):
    d = pd.DataFrame(rows)
    out = {"Variant": name, **{c: f"{d[c].mean():.3f} ± {d[c].std():.3f}" for c in d.columns}}
    print(out, flush=True)
    return out


# ── A-D: the repo's own preprocessing functions, fitted per fold ─────────────
def deterministic(d, keep_bp):
    if not keep_bp:
        return FE.apply_deterministic_transforms(d)
    d = d.copy()
    d["pulsePressure"] = d["sysBP"] - d["diaBP"]          # add PP, keep sysBP/diaBP
    for f in (FE.create_age_group, FE.create_bmi_category,
              FE.create_smoking_intensity, FE.log_transform):
        d = f(d)
    return d


def repo_prep(Xtr, Xte, fix_clip, keep_bp):
    fv = P.fit_imputer(Xtr)
    a = P.encode_categoricals(P.apply_imputer(Xtr, fv))
    b = P.encode_categoricals(P.apply_imputer(Xte, fv))
    a, b = deterministic(a, keep_bp), deterministic(b, keep_bp)
    sp = FE.fit_statistical_transforms(a)
    if fix_clip:  # clip only genuinely continuous columns
        binary = [c for c in a.columns if a[c].nunique() <= 2]
        sp["fences"] = {k: v for k, v in sp["fences"].items() if k not in binary}
    a, b = FE.apply_statistical_transforms(a, sp), FE.apply_statistical_transforms(b, sp)
    a, sc = P.fit_scaler(a)
    return a, P.apply_scaler(b, sc)


def run_repo(name, fix_clip, keep_bp, smote=True):
    rows = []
    for tr, te in CV.split(X_raw, y):
        with contextlib.redirect_stdout(io.StringIO()):
            a, b = repo_prep(X_raw.iloc[tr], X_raw.iloc[te], fix_clip, keep_bp)
        ytr = y[tr]
        if smote:
            a, ytr = SMOTE(random_state=42).fit_resample(a, ytr)
        # hyper-parameters chosen by the repo's GridSearchCV
        m = LogisticRegression(C=0.1, l1_ratio=1, solver="saga", max_iter=5000).fit(a, ytr)
        rows.append(metrics(y[te], m.predict_proba(b)[:, 1]))
    return summarise(name, rows)


# ── E-J: clean sklearn pipelines on raw clinical features ────────────────────
LOG = ["cigsPerDay", "totChol", "glucose", "BMI", "heartRate"]
LIN = ["age", "education", "sysBP", "diaBP"]
BIN = ["sex", "is_smoking", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]


def clean_X(X):
    X = X.copy()
    X["sex"] = (X["sex"] == "M").astype(int)
    X["is_smoking"] = (X["is_smoking"] == "YES").astype(int)
    return X


def make_ct():
    logp = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                     ("sc", StandardScaler())])
    linp = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler())])
    return ColumnTransformer([("log", logp, LOG), ("lin", linp, LIN),
                              ("bin", SimpleImputer(strategy="most_frequent"), BIN)])


def run_pipe(name, est, smote=False):
    Xc, rows = clean_X(X_raw), []
    for tr, te in CV.split(Xc, y):
        steps = [("ct", make_ct())]
        steps += [("smote", SMOTE(random_state=42))] if smote else []
        pipe = (ImbPipeline if smote else Pipeline)(steps + [("m", est)])
        pipe.fit(Xc.iloc[tr], y[tr])
        rows.append(metrics(y[te], pipe.predict_proba(Xc.iloc[te])[:, 1]))
    return summarise(name, rows)


if __name__ == "__main__":
    res = [
        run_repo("A  as-built: SMOTE, IQR-clips binaries, sysBP/diaBP -> PP", False, False),
        run_repo("B  A + don't clip binary columns", True, False),
        run_repo("C  B + keep sysBP/diaBP", True, True),
        run_repo("D  C without SMOTE", True, True, smote=False),
        run_pipe("E  clean LR, no resampling", LogisticRegression(max_iter=5000)),
        run_pipe("F  clean LR + class_weight=balanced",
                 LogisticRegression(max_iter=5000, class_weight="balanced")),
        run_pipe("G  clean LR + SMOTE inside pipeline", LogisticRegression(max_iter=5000), smote=True),
        run_pipe("H  HistGradientBoosting (depth 3, lr .05)",
                 HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                l2_regularization=1.0, min_samples_leaf=40,
                                                random_state=0)),
        run_pipe("I  XGBoost shallow + regularised",
                 XGBClassifier(max_depth=2, learning_rate=0.03, n_estimators=300, subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=10, reg_lambda=5,
                               eval_metric="logloss", n_jobs=4, random_state=0)),
        run_pipe("J  RandomForest (min_samples_leaf=20)",
                 RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=4,
                                        random_state=0)),
    ]
    print("\n" + pd.DataFrame(res).to_string(index=False))
