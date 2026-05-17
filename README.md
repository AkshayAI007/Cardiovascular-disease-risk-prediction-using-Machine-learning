# Cardiovascular Disease Risk Prediction
### 10-Year CHD Risk Assessment · Framingham Heart Study · Machine Learning

---

## Overview

This project builds an end-to-end machine learning pipeline to predict an individual's **10-year risk of coronary heart disease (CHD)** using clinical and demographic data from the Framingham Heart Study cohort.

The pipeline covers everything from raw data to a deployed web application — leakage-safe preprocessing, nine trained classifiers, threshold optimisation, probability calibration, and a multi-tab Streamlit UI for both single-patient and batch predictions.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Framingham Heart Study (Kaggle) |
| Patients | 3,390 |
| Features | 15 clinical + demographic |
| Target | `TenYearCHD` — binary (0 = No CHD, 1 = CHD onset within 10 years) |
| Class balance | ~85% No CHD / ~15% CHD (imbalanced) |

**Features used:**

| Category | Features |
|---|---|
| Demographics | `age`, `sex`, `education` |
| Lifestyle | `is_smoking`, `cigsPerDay` |
| Medical history | `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` |
| Clinical measurements | `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose` |

---

## Project Structure

```
├── src/
│   ├── config.py              # Paths, hyperparameter grids, constants
│   ├── data_loader.py         # CSV loading and basic EDA
│   ├── preprocessing.py       # Imputation, encoding, scaling, SMOTE
│   ├── feature_engineering.py # Pulse pressure, age groups, BMI flags, log transforms
│   ├── models.py              # Model registry, GridSearchCV, threshold optimiser, calibration
│   ├── evaluation.py          # Metrics, cross-validation, comparison table, DCA
│   ├── visualization.py       # All report charts (ROC, calibration, learning curve, etc.)
│   ├── train.py               # End-to-end training pipeline
│   └── predict.py             # CLI inference — single patient or batch CSV
├── app_streamlit.py           # Streamlit web application
├── app_api.py                 # FastAPI REST endpoint
├── data_cardiovascular_risk.csv
├── models/
│   └── best_model_bundle.joblib   # Saved model + preprocessing artifacts
├── reports/                   # All generated charts and CSVs
└── requirements.txt
```

---

## Pipeline

### 1. Preprocessing (leakage-safe)

All fitting is done on the **training fold only** and applied to validation/test sets using saved parameters — no data leakage.

```
Raw data
  → Stratified 3-way split (train 60% / val 20% / test 20%)
  → Median/mode imputation (fitted on X_train)
  → One-hot encoding (sex, is_smoking)
  → Feature engineering (deterministic):
      · Pulse pressure  (sysBP − diaBP)
      · Age group bins  (ACC/AHA clinical bands)
      · BMI flags       (is_obese, is_overweight)
      · Smoking intensity interaction term
      · log1p transforms on skewed continuous features
  → IQR outlier clipping (fitted on X_train)
  → Low-variance and high-correlation feature removal (fitted on X_train)
  → StandardScaler (fitted on X_train)
  → SMOTE oversampling (training fold only)
```

### 2. Model Training

Nine classifiers trained with `GridSearchCV` (stratified 5-fold, AUC scoring):

| Model | Grid Searched |
|---|---|
| Logistic Regression | C, l1_ratio, solver |
| Random Forest | n_estimators, max_depth, min_samples |
| XGBoost | max_depth, learning_rate, n_estimators, subsample |
| KNN | n_neighbors, weights, metric |
| SVC | C, kernel |
| Naive Bayes | var_smoothing |
| Gradient Boosting | n_estimators, learning_rate, max_depth, subsample |
| AdaBoost | n_estimators, learning_rate |
| Stacking Ensemble | top-3 by AUC as base learners, LR meta-learner |

### 3. Threshold Optimisation

Thresholds are tuned on the **validation set** (never the test set) by maximising **F2-score** subject to a minimum precision floor of 0.20.

> Pure recall maximisation is degenerate — it always drives the threshold to zero by flagging every patient as high-risk. F2 weights recall 2× over precision but cannot be gamed by flagging everyone, since precision collapsing below 20% disqualifies that threshold.

### 4. Probability Calibration

The best model is post-processed with **Platt scaling** (logistic regression fitted on held-out validation probabilities). This ensures that a predicted "70% risk" reflects a real-world ~70% frequency rather than an arbitrary score — critical for clinical interpretation.

---

## Results

**Best model: Logistic Regression (calibrated)**

| Metric | Value |
|---|---|
| ROC-AUC | 0.714 |
| Sensitivity (Recall) | 72.6% |
| Specificity | 58.0% |
| Precision (PPV) | 23.4% |
| Decision threshold | 0.13 (post-calibration) |

**All model comparison (test set):**

| Model | AUC | Recall | Precision | Threshold |
|---|---|---|---|---|
| **LogisticRegression** | **0.714** | 73.5% | 22.7% | 0.42 |
| Stacking | 0.713 | 80.4% | 21.5% | 0.26 |
| NaiveBayes | 0.705 | 69.6% | 25.5% | 0.37 |
| AdaBoost | 0.666 | 52.9% | 26.1% | 0.49 |
| RandomForest | 0.647 | 67.6% | 20.7% | 0.23 |
| XGBoost | 0.607 | 43.1% | 24.2% | 0.10 |
| KNN | 0.582 | 51.9% | 17.4% | 0.29 |
| GradientBoosting | 0.557 | 42.2% | 19.9% | 0.10 |
| SVC | 0.556 | 41.2% | 17.9% | 0.15 |

> AUC ~0.71 is a solid result for a tabular medical dataset with 15% class prevalence and no deep feature engineering.

---

## Web Application

The Streamlit app provides a full clinical UI with three tabs:

**Tab 1 — Patient Assessment**
- Input form: demographics, lifestyle, medical history, clinical measurements
- Risk gauge chart with colour zones and decision threshold needle
- Risk classification: LOW / MODERATE / HIGH / VERY HIGH
- Per-factor status table against clinical reference ranges (BP stages, cholesterol, BMI, glucose)
- Tailored clinical recommendations based on the patient's specific risk factors

**Tab 2 — Batch Prediction**
- CSV upload with automatic preprocessing and imputation
- Summary metrics: total patients, high-risk count, mean probability
- Risk score distribution histogram and category pie chart
- Downloadable results CSV

**Tab 3 — Model Insights**
- Performance metric tiles
- All training report images: ROC curves, calibration curve, learning curve, confusion matrix
- Full model comparison table
- Pipeline summary

### Run the app

```bash
streamlit run app_streamlit.py
```

---

## CLI Inference

**Single patient:**
```bash
python src/predict.py \
  --age 55 --sex M --is_smoking 0 --cigsPerDay 0 \
  --BPMeds 0 --prevalentStroke 0 --prevalentHyp 1 --diabetes 0 \
  --totChol 250 --sysBP 140 --diaBP 90 --BMI 28.5 \
  --heartRate 75 --glucose 100 --education 2
```

**Batch CSV:**
```bash
python src/predict.py --csv path/to/new_patients.csv
```

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/AkshayAI007/Cardiovascular-disease-risk-prediction-using-Machine-learning.git
cd Cardiovascular-disease-risk-prediction-using-Machine-learning

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Train the model (generates model bundle + all report charts)
python src/train.py

# Launch the web app
streamlit run app_streamlit.py
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Three-way split (train / val / test) | Threshold tuning uses val set; test set touched only once for final evaluation — no leakage |
| SMOTE on training fold only | Synthetic samples must never appear in val/test — that would inflate metrics artificially |
| F2 threshold optimisation | Medical screening prioritises recall (catching sick patients) but pure recall optimisation is degenerate; F2 + precision floor gives the right trade-off |
| Platt calibration | Raw model scores are not interpretable as probabilities; calibration makes them meaningful for clinical communication |
| `device="cpu"` for XGBoost | Dataset (~3,000 rows) is too small for GPU to help; mixed device training/inference caused warnings |
| Logistic Regression as winner | Interpretable, well-calibrated, highest AUC — appropriate for a clinical risk tool |

---

## Disclaimer

This is a research and portfolio project. The model and application **do not constitute medical advice** and must not replace clinical judgment. Predictions should only be used to support, not substitute, the assessment of a qualified healthcare professional.
