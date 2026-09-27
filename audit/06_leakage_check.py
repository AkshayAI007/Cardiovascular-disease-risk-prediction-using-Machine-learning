"""
Audit step 6 — data-leakage check.

Question: is the data split into train / validation / test FIRST, and is every
preprocessing statistic fitted on the TRAINING rows only? Then: is the
validation / test data kept out of every later decision (tuning, selection,
calibration, thresholds, reporting)?

  1. Split integrity (disjoint, stratified, no duplicate rows across splits)
  2. Call-order trace: instrument preprocess_train() and record which rows
     each fitted step actually receives
  3. Perturbation test: change val/test features -> fitted state must not move;
     change train features -> it must move (proves the test is sensitive)
  4. Target leakage: target never in the feature matrix, no single feature
     suspiciously predictive
  5. Downstream (src/train.py): where val/test rows re-enter fitting/selection
  6. The original notebook's order (fit + SMOTE before split) vs split-first

Run from the repo root:
    python audit/06_leakage_check.py
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
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import (StratifiedKFold, cross_val_score,  # noqa: E402
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import preprocessing as P  # noqa: E402
from config import RANDOM_STATE, TARGET_COL  # noqa: E402
from data_loader import load_data  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
df = load_data()
quiet = contextlib.redirect_stdout


def run_preprocess(frame):
    with quiet(io.StringIO()):
        return P.preprocess_train(frame)


def header(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ─────────────────────────────────────────────────────────────────────────────
# 2 (run first so we know the split). Instrument every fit/split call.
# ─────────────────────────────────────────────────────────────────────────────
events = []
_orig = {n: getattr(P, n) for n in
         ("train_test_split", "fit_imputer", "fit_statistical_transforms",
          "fit_scaler", "apply_smote")}


def _wrap(name):
    def inner(*args, **kw):
        out = _orig[name](*args, **kw)
        if name == "train_test_split":
            events.append((name, len(args[0]), (out[0].index, out[1].index)))
        else:
            events.append((name, len(args[0]), args[0].index))
        return out
    return inner


for n in _orig:
    setattr(P, n, _wrap(n))
Xtr_res, ytr_res, Xva, yva, Xte, yte, art = run_preprocess(df)
for n, f in _orig.items():
    setattr(P, n, f)

split1, split2 = events[0][2], events[1][2]
test_idx, val_idx = pd.Index(split1[1]), pd.Index(split2[1])
train_idx = pd.Index(split2[0])

# ─────────────────────────────────────────────────────────────────────────────
header("1. SPLIT INTEGRITY")
# ─────────────────────────────────────────────────────────────────────────────
y = df[TARGET_COL]
checks = {
    "train/val/test disjoint": not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx)
                                    or set(val_idx) & set(test_idx)),
    "union covers every row": len(train_idx) + len(val_idx) + len(test_idx) == len(df),
}
for k, v in checks.items():
    print(f"  [{PASS if v else FAIL}] {k}")
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    print(f"  {name:5s} n={len(idx):4d}  positives={int(y.loc[idx].sum()):3d}  rate={y.loc[idx].mean():.3f}")
feat = df.drop(columns=[TARGET_COL])
key = pd.util.hash_pandas_object(feat, index=False)
dup = len(set(key.loc[train_idx]) & (set(key.loc[val_idx]) | set(key.loc[test_idx])))
print(f"  [{PASS if dup == 0 else FAIL}] identical feature rows shared between train and val/test: {dup}")

# ─────────────────────────────────────────────────────────────────────────────
header("2. CALL ORDER INSIDE preprocess_train()  (which rows does each fit see?)")
# ─────────────────────────────────────────────────────────────────────────────
ok_all = True
for i, (name, n_rows, idx) in enumerate(events, 1):
    if name == "train_test_split":
        print(f"  {i}. {name:28s} input rows={n_rows:4d}  -> split")
        continue
    idx = pd.Index(idx)
    only_train = set(idx) <= set(train_idx) and len(idx) == len(train_idx)
    ok_all &= only_train
    print(f"  {i}. {name:28s} input rows={n_rows:4d}  "
          f"train={len(set(idx) & set(train_idx)):4d} val={len(set(idx) & set(val_idx)):3d} "
          f"test={len(set(idx) & set(test_idx)):3d}  [{PASS if only_train else FAIL}]")
first_fit = next(i for i, e in enumerate(events) if e[0] != "train_test_split")
print(f"  [{PASS if first_fit == 2 else FAIL}] both splits happen before the first fitted step")
print(f"  [{PASS if ok_all else FAIL}] every fitted step received exactly the {len(train_idx)} training rows")
print("  note: encode_categoricals / deterministic transforms are stateless (no statistics), "
      "applied per split")

# ─────────────────────────────────────────────────────────────────────────────
header("3. PERTURBATION TEST  (fitted state vs. contents of val/test)")
# ─────────────────────────────────────────────────────────────────────────────


NUM = [c for c in df.columns if c != TARGET_COL and pd.api.types.is_numeric_dtype(df[c])]
df_f = df.astype({c: float for c in NUM})        # same values, float dtype for in-place edits


def perturb(frame, rows):
    f, num = frame.copy(), NUM
    rng = np.random.default_rng(0)
    f.loc[rows, num] = f.loc[rows, num] * rng.uniform(0.2, 5.0, size=(len(rows), len(num)))
    f.loc[rows[::2], "glucose"] = np.nan         # different missingness pattern
    f.loc[rows, "sex"] = np.where(f.loc[rows, "sex"] == "M", "F", "M")
    return f                                      # labels untouched -> same split


def fitted_state(a):
    return {"fill_values": a["fill_values"],
            "fences": a["stat_params"]["fences"],
            "kept_cols": a["stat_params"]["selector"]["keep_cols"],
            "scaler_mean": np.round(a["scaler"].mean_, 12).tolist(),
            "scaler_scale": np.round(a["scaler"].scale_, 12).tolist()}


r_base = run_preprocess(df_f)
base_state = fitted_state(r_base[6])
r_vt = run_preprocess(perturb(df_f, val_idx.append(test_idx)))
r_tr = run_preprocess(perturb(df_f, train_idx))
same_vt = fitted_state(r_vt[6]) == base_state and r_vt[0].equals(r_base[0])
moved_tr = fitted_state(r_tr[6]) != base_state
val_moved = not np.allclose(r_vt[2].values, r_base[2].values)
print(f"  [{PASS if same_vt else FAIL}] perturbing val+test rows leaves imputer / IQR fences / "
      f"selector / scaler / SMOTE'd train matrix bit-identical")
print(f"  [{PASS if moved_tr else FAIL}] perturbing train rows DOES change the fitted state "
      f"(test is sensitive)")
print(f"  [{PASS if val_moved else FAIL}] perturbed val rows are transformed differently "
      f"(transform applied, not refitted)")

# ─────────────────────────────────────────────────────────────────────────────
header("4. TARGET LEAKAGE")
# ─────────────────────────────────────────────────────────────────────────────
print(f"  [{PASS if TARGET_COL not in art['feature_cols'] else FAIL}] "
      f"'{TARGET_COL}' not in the model's feature columns")
tr = df.loc[train_idx]
aucs = {}
for c in feat.columns:
    s = tr[c]
    s = (s == s.mode()[0]).astype(float) if s.dtype == object or str(s.dtype) == "str" else s
    s = s.fillna(s.median())
    a = roc_auc_score(tr[TARGET_COL], s)
    aucs[c] = max(a, 1 - a)
top = pd.Series(aucs).sort_values(ascending=False)
print(f"  [{PASS if top.iloc[0] < 0.8 else FAIL}] strongest single feature: {top.index[0]} "
      f"AUC={top.iloc[0]:.3f} (no feature is a proxy for the label)")
print("  top-5 univariate AUCs (train rows):", top.head(5).round(3).to_dict())

# ─────────────────────────────────────────────────────────────────────────────
header("5. DOWNSTREAM: where val/test re-enter fitting or selection (src/train.py)")
# ─────────────────────────────────────────────────────────────────────────────
n_train = len(train_idx)
X_real = Xtr_res.iloc[:n_train]                   # SMOTE returns originals first
y_real = ytr_res.iloc[:n_train]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
lr = LogisticRegression(C=0.1, l1_ratio=1, solver="saga", max_iter=5000)

# 5a. GridSearchCV folds reuse preprocessing fitted on ALL training rows
from importlib import util as _u  # noqa: E402
_spec = _u.spec_from_file_location("abl", REPO / "audit" / "03_preprocessing_ablation.py")
abl = _u.module_from_spec(_spec)
with quiet(io.StringIO()):
    _spec.loader.exec_module(abl)
Xraw_tr, y_tr = df.drop(columns=[TARGET_COL]).loc[train_idx], y.loc[train_idx].values
outside = cross_val_score(LogisticRegression(C=0.1, l1_ratio=1, solver="saga", max_iter=5000),
                          X_real, y_real, cv=cv, scoring="roc_auc").mean()
inside = []
for a, b in cv.split(Xraw_tr, y_tr):
    with quiet(io.StringIO()):
        A, B = abl.repo_prep(Xraw_tr.iloc[a], Xraw_tr.iloc[b], False, False)
    inside.append(roc_auc_score(y_tr[b], LogisticRegression(
        C=0.1, l1_ratio=1, solver="saga", max_iter=5000).fit(A, y_tr[a]).predict_proba(B)[:, 1]))
print(f"  5a. preprocessing fitted once on all train rows, then CV'd (no SMOTE): CV AUC "
      f"{outside:.4f} vs refit inside each fold {np.mean(inside):.4f}  "
      f"-> gap {outside - np.mean(inside):+.4f} (minor)")

# 5b. SMOTE before GridSearchCV
cv_smote = cross_val_score(XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.1,
                                         subsample=0.8, eval_metric="logloss", n_jobs=4,
                                         random_state=42), Xtr_res, ytr_res, cv=cv,
                           scoring="roc_auc").mean()
xgb = XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.1, subsample=0.8,
                    eval_metric="logloss", n_jobs=4, random_state=42).fit(Xtr_res, ytr_res)
print(f"  5b. SMOTE applied before GridSearchCV (XGBoost, selected params): CV AUC {cv_smote:.3f} "
      f"vs test AUC {roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1]):.3f}  -> MAJOR (see 04)")

# 5c. Stacking fitted on train + VAL, threshold then tuned on VAL
stack = StackingClassifier(
    estimators=[("lr", LogisticRegression(C=0.1, l1_ratio=1, solver="saga", max_iter=5000)),
                ("nb", GaussianNB()),
                ("ada", AdaBoostClassifier(n_estimators=200, learning_rate=1.5, random_state=42))],
    final_estimator=LogisticRegression(max_iter=1000), cv=5)
stack.fit(pd.concat([Xtr_res, Xva]).reset_index(drop=True), pd.concat([ytr_res, yva]).reset_index(drop=True))
p_val, p_te = stack.predict_proba(Xva)[:, 1], stack.predict_proba(Xte)[:, 1]
print(f"  5c. Stacking trained on train+VAL, threshold tuned on VAL: AUC on val (seen in training) "
      f"{roc_auc_score(yva, p_val):.3f} vs test {roc_auc_score(yte, p_te):.3f}  -> val is not held out")

# 5d. Winner (and stacking members) picked on the TEST set
p_lr = lr.fit(Xtr_res, ytr_res).predict_proba(Xte)[:, 1]
p_nb = GaussianNB().fit(Xtr_res, ytr_res).predict_proba(Xte)[:, 1]
rng, wins = np.random.default_rng(0), {"LogisticRegression": 0, "NaiveBayes": 0, "Stacking": 0}
yv = yte.values
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv))
    s = {"LogisticRegression": roc_auc_score(yv[i], p_lr[i]),
         "NaiveBayes": roc_auc_score(yv[i], p_nb[i]), "Stacking": roc_auc_score(yv[i], p_te[i])}
    wins[max(s, key=s.get)] += 1
print(f"  5d. best model + stacking members chosen by TEST AUC. Winner across 1,000 bootstrap "
      f"resamples of the test set: {', '.join(f'{k} {v / 10:.0f}%' for k, v in wins.items())}"
      f"  -> the 'winner' is noise; its test score is a max over candidates (optimistic)")

# 5e. Reported "cross-validation" mixes synthetic rows and the TEST set
n_syn = len(Xtr_res) - n_train
n_all = len(Xtr_res) + len(Xva) + len(Xte)
print(f"  5e. cross_validate_model(X_all): {n_all} rows = {n_train} real train + {n_syn} SYNTHETIC "
      f"({n_syn / n_all:.0%}) + {len(Xva)} val + {len(Xte)} TEST ({len(Xte) / n_all:.0%}); "
      f"positive rate {pd.concat([ytr_res, yva, yte]).mean():.2f} vs real {y.mean():.2f}")
print("  5f. Platt calibration fitted on VAL, then threshold re-tuned on the SAME val rows "
      "(val used twice; test untouched)")
print("  5g. learning_curve() runs on the SMOTE'd training set -> same synthetic-neighbour leak as 5b")

# ─────────────────────────────────────────────────────────────────────────────
header("6. ORIGINAL NOTEBOOK ORDER vs SPLIT-FIRST (same features, same models)")
# ─────────────────────────────────────────────────────────────────────────────
nb_cols = ["age", "education", "sex", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp",
           "diabetes", "totChol", "BMI", "heartRate", "glucose", "pulsePressure"]
cont = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]


def notebook_features(frame, fill):
    f = frame.fillna(fill).copy()
    f[cont] = np.log(f[cont] + 1)
    f["pulsePressure"] = f["sysBP"] - f["diaBP"]
    f["sex"] = (f["sex"] == "M").astype(int)
    return f[nb_cols]


def fill_from(frame):
    return {c: (frame[c].mode()[0] if c in ("education", "BPMeds") else frame[c].median())
            for c in ["glucose", "education", "BPMeds", "totChol", "cigsPerDay", "BMI", "heartRate"]}


models = {"LogReg": lambda: LogisticRegression(max_iter=5000),
          "XGBoost": lambda: XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=200,
                                           eval_metric="logloss", n_jobs=4, random_state=42)}
raw, yy = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
rows = []

# N0 — notebook: impute + scale + SMOTE on ALL rows, THEN split
Xa = StandardScaler().fit_transform(notebook_features(raw, fill_from(raw)))
Xs, ys = SMOTE(random_state=42).fit_resample(Xa, yy)
is_syn = np.r_[np.zeros(len(yy), bool), np.ones(len(ys) - len(yy), bool)]
a_tr, a_te, b_tr, b_te, s_tr, s_te = train_test_split(Xs, ys, is_syn, test_size=0.2, random_state=42)
for m, mk in models.items():
    p = mk().fit(a_tr, b_tr).predict_proba(a_te)[:, 1]
    rows.append({"Order": "N0 notebook: fit+SMOTE on all rows, then split", "Model": m,
                 "Test AUC": roc_auc_score(b_te, p), "Test acc": accuracy_score(b_te, p >= 0.5),
                 "Synthetic rows in test": f"{s_te.mean():.0%}"})

# N1 — impute + scale on all rows before split, SMOTE on train only
Xtr_, Xte_, ytr_, yte_ = train_test_split(Xa, yy, test_size=0.2, random_state=42, stratify=yy)
Xtr_s, ytr_s = SMOTE(random_state=42).fit_resample(Xtr_, ytr_)
for m, mk in models.items():
    p = mk().fit(Xtr_s, ytr_s).predict_proba(Xte_)[:, 1]
    rows.append({"Order": "N1 impute+scale on all rows, split, SMOTE on train", "Model": m,
                 "Test AUC": roc_auc_score(yte_, p), "Test acc": accuracy_score(yte_, p >= 0.5),
                 "Synthetic rows in test": "0%"})

# N2 — split FIRST, every statistic fitted on train only, no resampling
rtr, rte, ytr2, yte2 = train_test_split(raw, yy, test_size=0.2, random_state=42, stratify=yy)
fv = fill_from(rtr)
sc = StandardScaler().fit(notebook_features(rtr, fv))
Xtr2, Xte2 = sc.transform(notebook_features(rtr, fv)), sc.transform(notebook_features(rte, fv))
for m, mk in models.items():
    p = mk().fit(Xtr2, ytr2).predict_proba(Xte2)[:, 1]
    rows.append({"Order": "N2 split first, fit on train only (correct)", "Model": m,
                 "Test AUC": roc_auc_score(yte2, p), "Test acc": accuracy_score(yte2, p >= 0.5),
                 "Synthetic rows in test": "0%"})
out = pd.DataFrame(rows)
out[["Test AUC", "Test acc"]] = out[["Test AUC", "Test acc"]].round(3)
print(out.to_string(index=False))
print(f"\n  majority-class baseline accuracy on real data: {1 - yy.mean():.3f}")
