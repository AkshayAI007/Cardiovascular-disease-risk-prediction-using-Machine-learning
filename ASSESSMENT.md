# Cardiovascular Risk Prediction — Model Assessment

## 1. What the Original Notebook Does Well

| Strength | Detail |
|---|---|
| Multiple algorithms | 6 classifiers compared head-to-head |
| Hyperparameter tuning | GridSearchCV used for each model |
| Imbalance handling | SMOTE applied to minority class |
| Rich EDA | 33+ charts covering distributions, correlations, clinical patterns |
| Feature engineering | Pulse pressure derived from sysBP/diaBP |
| Model serialisation | Pickle used to persist final model |

---

## 2. Critical Gaps & Loopholes

### 2.1 Data Leakage — HIGH SEVERITY

**Problem:** The original notebook applies StandardScaler and SMOTE _before_ performing the train/test split. This means the scaler's mean/std and SMOTE's synthetic samples are computed using test-set information, inflating all reported metrics.

**Evidence:** XGBoost reports 100% train accuracy, which is a direct symptom.

**Fix (implemented in `preprocessing.py`):**
```
raw data → impute → encode → split → (scale on train only) → SMOTE on train only → model
```
The scaler is fitted on `X_train`, then `transform`-only is applied to `X_test`.

---

### 2.2 Overfitting — HIGH SEVERITY

**Problem:** XGBoost train accuracy = 100%, test = 89.67%. The ~10% gap signals memorisation of training data, not generalisation.

**Root causes:**
- No regularisation parameters (`reg_alpha`, `reg_lambda`) in the search grid
- `max_depth=7` is very deep for 3,390 samples
- No early stopping used

**Fix (in `models.py` / `config.py`):**
- Added `subsample` and `colsample_bytree` to the XGBoost grid
- Recommend adding `reg_alpha: [0, 0.1, 1]` and `reg_lambda: [1, 5, 10]`
- Use `learning_curve()` to visualise the train/val gap (in `visualization.py`)

---

### 2.3 No sklearn Pipeline — MEDIUM SEVERITY

**Problem:** Preprocessing steps are applied as ad-hoc transformations in the notebook. Any re-run or deployment step risks applying them in the wrong order or with wrong parameters (e.g., fitting a scaler on new data instead of reusing training stats).

**Fix (in `preprocessing.py`):** All transformers are fitted once on training data and returned as `artifacts` dict. Inference (`predict.py`) reuses these serialised transformers.

---

### 2.4 Threshold Fixed at 0.5 — MEDIUM SEVERITY

**Problem:** In a medical screening task, missing a true positive (false negative) carries far higher cost than a false alarm. The default 0.5 probability threshold was never tuned.

**Fix (in `models.py`):** `find_optimal_threshold()` scans thresholds and maximises recall, ensuring the model flags as many at-risk patients as possible.

---

### 2.5 Education as Top Predictive Feature — MEDIUM SEVERITY

**Problem:** The notebook identifies "education" as a top-4 feature. While socioeconomic status correlates with CHD, education is a proxy variable: it acts as a confounder for income, lifestyle, and healthcare access. Relying on it can:
- Introduce demographic bias
- Produce a model that discriminates by socioeconomic group
- Fail on datasets where education encoding differs

**Recommendation:**
- Use SHAP values to distinguish direct vs. proxy contributions
- Consider dropping `education` and using domain-informed features instead
- Run a fairness audit (demographic parity, equalised odds)

---

### 2.6 Class Imbalance Strategy is Incomplete — MEDIUM SEVERITY

**Problem:** SMOTE was used but:
1. Applied before splitting (see §2.1)
2. No comparison with alternative strategies (class_weight, ADASYN, cost-sensitive learning)

**Fix:** Compare SMOTE vs. `class_weight='balanced'` on XGBoost. Cost-sensitive learning avoids generating synthetic data that may not reflect the true data manifold.

---

### 2.7 No Cross-Validation Beyond Grid Search — LOW-MEDIUM SEVERITY

**Problem:** Model selection is based on a single 80/20 split. With only 3,390 records, results are sensitive to how the split falls.

**Fix (in `evaluation.py`):** `cross_validate_model()` runs stratified k-fold CV reporting mean ± std for all metrics. This gives a more reliable estimate of generalisation.

---

### 2.8 Pickle for Serialisation — LOW SEVERITY

**Problem:** `pickle` is Python-version-specific and not safe for loading untrusted files.

**Fix (in `train.py` / `predict.py`):** `joblib.dump` / `joblib.load` used instead. Joblib is faster for large numpy arrays and is the sklearn-recommended serialiser.

---

### 2.9 No Calibration Check — LOW SEVERITY

**Problem:** Probability outputs from XGBoost are not calibrated. An uncalibrated model that outputs `p=0.8` may not correspond to 80% actual risk, which is critical for clinical communication.

**Fix (in `visualization.py`):** `plot_calibration_curve()` implemented. If miscalibrated, use `CalibratedClassifierCV` with `method='isotonic'`.

---

### 2.10 cigsPerDay vs is_smoking Redundancy

**Problem:** Both `cigsPerDay` and `is_smoking` are in the dataset. Non-smokers have `cigsPerDay=0` but the correlation between these features was not removed via explicit feature selection.

**Fix (in `feature_engineering.py`):** `create_smoking_intensity()` combines both into a single interaction feature. `drop_high_correlation()` removes redundant features above a 0.90 threshold.

---

## 3. Enhancements Implemented in Refactored Code

| Enhancement | File | Benefit |
|---|---|---|
| Leakage-safe pipeline | `preprocessing.py` | True generalisation metrics |
| Threshold optimisation | `models.py` | Higher recall for medical use |
| Stacking ensemble | `models.py` | Combines model diversity |
| Learning curves | `visualization.py` | Detects overfitting visually |
| Calibration curves | `visualization.py` | Trustworthy probability outputs |
| Cross-validation | `evaluation.py` | Reliable generalisation estimate |
| Net benefit / DCA | `evaluation.py` | Clinical decision-curve analysis |
| SHAP explainability | `utils.py` | Patient-level explanations |
| Feature drift detection | `utils.py` | Production monitoring |
| IQR winsorisation | `feature_engineering.py` | Robust to extreme outliers |
| Age groups & BMI flags | `feature_engineering.py` | Clinically meaningful features |
| Smoking intensity | `feature_engineering.py` | Better exposure modelling |
| joblib serialisation | `train.py` / `predict.py` | Safe, fast model persistence |
| CLI inference tool | `predict.py` | Single patient or batch CSV |
| Centralised config | `config.py` | One place to change parameters |

---

## 4. Recommended Additional Enhancements (Not Yet Implemented)

### 4.1 SHAP-Based Feature Selection
Run SHAP and drop features with mean |SHAP| < 0.001. This is more principled than correlation thresholds and directly measures predictive contribution.

### 4.2 Fairness Audit
```python
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
```
Check model performance stratified by `sex` and `education` quartile to ensure the model does not disadvantage any demographic group.

### 4.3 Calibrated Model Output
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
```

### 4.4 ADASYN as SMOTE Alternative
```python
from imblearn.over_sampling import ADASYN
```
ADASYN focuses synthetic generation on hard-to-classify minority samples, which can produce better decision boundaries than uniform SMOTE.

### 4.5 Nested Cross-Validation
Wrap GridSearchCV inside an outer k-fold loop to get an unbiased estimate of the tuned model's generalisation performance.

### 4.6 REST API / Streamlit App
Wrap `predict.py` in a FastAPI or Streamlit interface for clinical users who cannot run Python directly.

```python
# FastAPI skeleton
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class Patient(BaseModel):
    age: float; sex: int; ...

@app.post("/predict")
def predict_chd(patient: Patient):
    df = pd.DataFrame([patient.dict()])
    result = predict(df, bundle)
    return result[["chd_probability", "risk_label"]].to_dict(orient="records")[0]
```

### 4.7 Automated Retraining Trigger
Monitor incoming patient data for feature drift (implemented in `utils.detect_feature_drift`). When drift exceeds threshold, trigger retraining automatically.

---

## 5. Final Model Recommendation

| Criterion | Recommendation |
|---|---|
| Best single model | XGBoost (ROC AUC ~0.90) |
| More robust | Stacking ensemble (LR + RF + XGBoost) |
| Most explainable | Logistic Regression (coefficients directly interpretable) |
| Clinical deployment | Calibrated XGBoost + threshold tuned for recall ≥ 0.90 |

For a **screening** tool (flag everyone at risk), optimise for **recall**.  
For a **diagnostic** tool (confirm disease), optimise for **precision**.  
The clinical use case here (10-year CHD screening) demands recall ≥ 0.90.

---

## 6. Project File Structure (After Refactoring)

```
project/
├── data_cardiovascular_risk.csv
├── ASSESSMENT.md
├── src/
│   ├── config.py              # All constants, paths, hyperparameter grids
│   ├── data_loader.py         # Load data, EDA summary, class distribution
│   ├── preprocessing.py       # Imputation, encoding, scaling, SMOTE
│   ├── feature_engineering.py # Feature creation, outlier treatment, selection
│   ├── visualization.py       # EDA + model evaluation plots
│   ├── models.py              # Model registry, tuning, stacking, threshold
│   ├── evaluation.py          # Metrics, CV, DCA, comparison table
│   ├── train.py               # End-to-end training entrypoint
│   └── predict.py             # CLI inference for single patient or batch CSV
├── models/
│   └── best_model_bundle.joblib
└── reports/
    └── (auto-generated PNG charts and CSV reports)
```
