"""
Audit step 4 — show that running GridSearchCV on an already-SMOTE'd training
set (what src/train.py does) inflates CV scores and selects over-fit
hyper-parameters, versus SMOTE inside each CV fold or no resampling.

Uses the repo's own preprocess_train() so the train/test rows are identical
to the shipped pipeline.

Run from the repo root (takes a few minutes):
    python audit/04_smote_cv_leak.py
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from data_loader import load_data  # noqa: E402
from preprocessing import preprocess_train  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    Xtr_res, ytr_res, Xva, yva, Xte, yte, _ = preprocess_train(load_data())

# Recover the real (pre-SMOTE) training rows. Reproduce the repo's split to get
# y_train; imblearn's SMOTE returns the original rows first, then synthetic ones.
from sklearn.model_selection import train_test_split  # noqa: E402
from config import RANDOM_STATE, TARGET_COL, TEST_SIZE  # noqa: E402

_df = load_data()
_Xt, _, _yt, _ = train_test_split(_df.drop(columns=[TARGET_COL]), _df[TARGET_COL],
                                  test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                  stratify=_df[TARGET_COL])
_, _, _y_train, _ = train_test_split(_Xt, _yt, test_size=0.20 / (1 - TEST_SIZE),
                                     random_state=RANDOM_STATE, stratify=_yt)
n_real = len(_y_train)
assert (ytr_res.iloc[:n_real].values == _y_train.values).all()
Xtr, ytr = Xtr_res.iloc[:n_real].reset_index(drop=True), ytr_res.iloc[:n_real].reset_index(drop=True)
print(f"real train rows={len(ytr)} (pos={ytr.sum()}) | after SMOTE={len(ytr_res)} | test={len(yte)}\n")

cv = StratifiedKFold(5, shuffle=True, random_state=42)
grids = {
    "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=4),
                     {"n_estimators": [200], "max_depth": [5, 10, None], "min_samples_leaf": [1, 20]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss", n_jobs=4, random_state=42),
                {"max_depth": [2, 3, 7], "learning_rate": [0.05, 0.2], "n_estimators": [100, 300]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 11, 51], "weights": ["uniform", "distance"]}),
}

rows = []
for name, (est, grid) in grids.items():
    setups = {
        "SMOTE before CV (as-built)": (GridSearchCV(est, grid, cv=cv, scoring="roc_auc"), Xtr_res, ytr_res),
        "SMOTE inside CV": (GridSearchCV(ImbPipeline([("smote", SMOTE(random_state=42)), ("m", est)]),
                                         {f"m__{k}": v for k, v in grid.items()},
                                         cv=cv, scoring="roc_auc"), Xtr, ytr),
        "No resampling": (GridSearchCV(est, grid, cv=cv, scoring="roc_auc"), Xtr, ytr),
    }
    for setup, (gs, Xf, yf) in setups.items():
        gs.fit(Xf, yf)
        test_auc = roc_auc_score(yte, gs.predict_proba(Xte)[:, 1])
        rows.append({"Model": name, "Setup": setup, "CV_AUC": round(gs.best_score_, 3),
                     "Test_AUC": round(test_auc, 3), "Gap": round(gs.best_score_ - test_auc, 3),
                     "Selected params": gs.best_params_})
        print(rows[-1], flush=True)

print("\n" + pd.DataFrame(rows).to_string(index=False))

# Sampling noise on the repo's single 678-row test set
lr = LogisticRegression(C=0.1, l1_ratio=1, solver="saga", max_iter=5000).fit(Xtr_res, ytr_res)
p, yv = lr.predict_proba(Xte)[:, 1], yte.values
rng, boots = np.random.default_rng(0), []
for _ in range(2000):
    i = rng.integers(0, len(yv), len(yv))
    if yv[i].min() != yv[i].max():
        boots.append(roc_auc_score(yv[i], p[i]))
print(f"\nLR test AUC {roc_auc_score(yv, p):.3f}, 95% bootstrap CI "
      f"[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}] (n_pos={yv.sum()})")
