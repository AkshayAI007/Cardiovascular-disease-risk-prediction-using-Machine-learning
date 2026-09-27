"""
Audit step 5 — a reference ("how it should be done") pipeline, evaluated on the
SAME held-out test split the repo uses, so results are directly comparable with
the shipped bundle. Also runs behavioural (invariance / monotonicity) tests on
both models.

Run from the repo root:
    python audit/05_reference_pipeline.py
"""
import sys, io, contextlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, confusion_matrix
import joblib
from config import TEST_SIZE, RANDOM_STATE, TARGET_COL

df = pd.read_csv(REPO / "data_cardiovascular_risk.csv").drop(columns=["id"])
X = df.drop(columns=[TARGET_COL]); y = df[TARGET_COL]
X_dev, X_te, y_dev, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                            random_state=RANDOM_STATE, stratify=y)


class DomainCleaner(BaseEstimator, TransformerMixin):
    """Stateless, explicit, schema-checked encoding + domain-consistent fixes."""
    def fit(self, X, y=None):
        # learn smoker-specific cigsPerDay median (fitted -> train only)
        s = X["is_smoking"].astype(str).str.upper().isin(["YES", "1", "TRUE"])
        self.cigs_smoker_median_ = float(X.loc[s, "cigsPerDay"].median())
        return self

    def transform(self, X):
        X = X.copy()
        X["sex"] = X["sex"].astype(str).str.upper().map({"M": 1, "F": 0, "1": 1, "0": 0})
        X["is_smoking"] = X["is_smoking"].astype(str).str.upper().map(
            {"YES": 1, "NO": 0, "1": 1, "0": 0})
        if X[["sex", "is_smoking"]].isna().any().any():
            raise ValueError("unrecognised sex / is_smoking value")
        # cigsPerDay is only ever missing for smokers -> impute smoker median, never 0
        miss = X["cigsPerDay"].isna()
        X.loc[miss & (X.is_smoking == 1), "cigsPerDay"] = self.cigs_smoker_median_
        X.loc[X.is_smoking == 0, "cigsPerDay"] = 0.0
        return X


LOG = ["cigsPerDay", "totChol", "glucose", "BMI"]
LIN = ["age", "sysBP", "diaBP", "heartRate"]
BIN = ["sex", "is_smoking", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
# education deliberately excluded (socio-economic proxy; negligible signal)

pre = ColumnTransformer([
    ("log", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                      ("sc", StandardScaler())]), LOG),
    ("lin", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), LIN),
    ("bin", SimpleImputer(strategy="most_frequent"), BIN),
], verbose_feature_names_out=False)

model = Pipeline([
    ("clean", DomainCleaner()),
    ("pre", pre),
    # no SMOTE, no class_weight: keep probabilities on the real prevalence scale
    ("clf", CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=5000),
                                   method="sigmoid", cv=5)),
])

# ── model-selection estimate: CV on the development set only ────────────────
cv = StratifiedKFold(5, shuffle=True, random_state=0)
oof = cross_val_predict(model, X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
print(f"Dev-set 5-fold CV  AUC={roc_auc_score(y_dev, oof):.3f}  Brier={brier_score_loss(y_dev, oof):.4f}")

model.fit(X_dev, y_dev)
p_ref = model.predict_proba(X_te)[:, 1]

# ── shipped bundle on the same test rows ────────────────────────────────────
from preprocessing import preprocess_inference
with contextlib.redirect_stdout(io.StringIO()):
    bundle = joblib.load(REPO / "models" / "best_model_bundle.joblib")
    art = {k: bundle[k] for k in ("fill_values", "stat_params", "scaler", "feature_cols")}
    p_old = bundle["model"].predict_proba(preprocess_inference(X_te, art))[:, 1]


def boot_ci(yv, p, n=2000):
    rng = np.random.default_rng(0); b = []
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].min() != yv[i].max(): b.append(roc_auc_score(yv[i], p[i]))
    return np.percentile(b, [2.5, 97.5])


def report(name, p, yv=y_te.values):
    lo, hi = boot_ci(yv, p)
    out = {"Model": name, "AUC": round(roc_auc_score(yv, p), 3), "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
           "PR-AUC": round(average_precision_score(yv, p), 3),
           "Brier": round(brier_score_loss(yv, p), 4),
           "Mean pred": round(p.mean(), 3), "Obs rate": round(yv.mean(), 3),
           "Max pred": round(p.max(), 3)}
    for t in (0.075, 0.10, 0.20):
        tn, fp, fn, tp = confusion_matrix(yv, (p >= t).astype(int)).ravel()
        out[f"Sens@{t}"] = round(tp / (tp + fn), 3); out[f"Spec@{t}"] = round(tn / (tn + fp), 3)
    return out


print(pd.DataFrame([report("Shipped bundle (as-built)", p_old),
                    report("Reference pipeline", p_ref)]).T.to_string(header=False))

# ── behavioural tests ────────────────────────────────────────────────────────
base = dict(age=55, sex="M", is_smoking="NO", cigsPerDay=0, BPMeds=0, prevalentStroke=0,
            prevalentHyp=0, diabetes=0, totChol=220, sysBP=130, diaBP=85, BMI=26,
            heartRate=75, glucose=90, education=2)


def both(**kw):
    d = pd.DataFrame([{**base, **kw}])
    with contextlib.redirect_stdout(io.StringIO()):
        a = bundle["model"].predict_proba(preprocess_inference(d, art))[0, 1]
    return round(a, 3), round(model.predict_proba(d)[0, 1], 3)


tests = [("baseline", {}),
         ("diabetes=1, glucose=250", dict(diabetes=1, glucose=250)),
         ("BPMeds=1", dict(BPMeds=1)),
         ("prevalentStroke=1", dict(prevalentStroke=1)),
         ("BMI 35", dict(BMI=35)),
         ("BP 180/135 (same PP)", dict(sysBP=180, diaBP=135)),
         ("smoker 20/day", dict(is_smoking="YES", cigsPerDay=20)),
         ("female", dict(sex="F"))]
rows = [{"Scenario": n, "Shipped": both(**kw)[0], "Reference": both(**kw)[1]} for n, kw in tests]
print("\nBehavioural tests (predicted 10-yr CHD risk):")
print(pd.DataFrame(rows).to_string(index=False))

# single-row vs batch consistency for the reference model
batch = pd.DataFrame([{**base, "is_smoking": "YES", "cigsPerDay": 30}, base])
single = model.predict_proba(batch.iloc[[0]])[0, 1]
assert abs(model.predict_proba(batch)[0, 1] - single) < 1e-12
print("\nReference model: single-row == batch prediction  ✔")
