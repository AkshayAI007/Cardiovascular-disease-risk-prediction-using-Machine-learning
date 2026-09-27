# ML Implementation Audit: Cardiovascular (10-year CHD) Risk Prediction

**Audited commit:** `3a9ce96` · **Scope:** data, preprocessing, modelling, evaluation, inference/serving
**Method:** code review, plus re-running the training pipeline in isolation, probing the shipped model bundle, the FastAPI app and the CLI, and running controlled ablations. Every number below comes from the scripts in this folder (see [Reproducing](#10-reproducing-this-audit)).

---

## 0. Verdict

**The model is not production-ready, and its individual predictions should not be relied on.**

The `src/` refactor fixes the notebook's main leak: scaling and SMOTE no longer happen before the train/test split. But it also contains defects that break the model for individual patients. Some are new (the IQR clipper, the variance filter, SMOTE before CV) and one is inherited from the notebook (pulse pressure replacing BP):

| # | Blocker | Effect on the shipped model |
|---|---|---|
| 1 | IQR clipping is applied to binary columns | `diabetes`, `BPMeds`, `is_obese` become constant 0 and are ignored. Glucose is capped at ~109 mg/dL. |
| 2 | An unscaled variance filter drops `prevalentStroke` | Prior stroke has no effect, even though stroke patients have a 45.5% observed CHD rate. |
| 3 | `sysBP`/`diaBP` are replaced by pulse pressure only | BP 180/135 scores the same as 130/85. |
| 4 | SMOTE is applied before `GridSearchCV` | CV AUC is 0.91–0.97 against test AUC 0.56–0.67, and the most over-fit hyper-parameters get selected. |
| 5 | The test set is used for model selection | The reported "best model" metrics are optimistically biased. |
| 6 | The CLI and the UI give wrong results | `--sex M` crashes and `--sex 1` scores a male as female. The web UI silently shows a hard-coded formula whenever the API fails. |

**Leakage check (§4).** The split does happen first, and every preprocessing statistic (imputer, IQR fences, feature selector, scaler, SMOTE) is fitted on the 2,034 training rows only. This was verified by tracing every fit call and by perturbing the val/test rows. The leaks come *after* preprocessing, in `src/train.py`: SMOTE before CV, model selection on the test set, stacking trained on the validation set, and a CV report that includes test rows. The original notebook fails the split-first rule outright: its XGBoost score of ~0.89 accuracy / 0.95 AUC drops to 0.85 / 0.63 when the split comes first.

Aggregate AUC (~0.71) hides the feature defects. The destroyed features are rare (22–100 patients each), so the overall score barely changes. The damage shows up in the highest-risk patients:

| Subgroup (whole dataset) | n | Observed CHD rate | Shipped model mean prediction |
|---|---:|---:|---:|
| All patients | 3390 | 15.1% | 14.9% |
| Diabetic | 87 | **37.9%** | 19.9% |
| Prior stroke | 22 | **45.5%** | 19.6% |
| On BP medication | 100 | **33.0%** | 23.2% |
| Glucose ≥ 126 mg/dL | 70 | **48.6%** | 19.7% |

These figures are partly in-sample, which flatters the model. The highest probability it outputs for any of the 3,390 patients is **0.39**.

---

## 1. What is done well

- The data is split into train/val/test first, and imputation, clipping, feature selection, scaling and SMOTE are fitted on the training split only, then reused at inference (`preprocessing.py`). Verified in §4.
- Splits are stratified. The threshold for the base models is tuned on a validation set, not the test set.
- Probability calibration and decision-curve analysis are attempted. Clinical metrics (sensitivity, specificity, PPV, NPV) are reported.
- Preprocessing parameters are serialised with the model. The API uses Pydantic validation and has a health endpoint.

The structure is sound. The problems are in which transforms are applied, what they are applied to, and how models are selected and evaluated.

---

## 2. Data audit

| Property | Value |
|---|---|
| Rows / features | 3,390 / 15 (+ `id`, dropped) |
| Target | `TenYearCHD`: 511 positives (**15.1%**) |
| Rows with ≥ 1 missing value | 463 (13.7%) |
| Missing | glucose 304 (9.0%), education 87, BPMeds 44, totChol 38, cigsPerDay 22, BMI 14, heartRate 1 |
| Support (training range) | age **32–70**, sysBP 83.5–295, glucose 40–394, totChol 107–696 |
| Rare binaries | prevalentStroke **22**, diabetes **87**, BPMeds **100** |
| Duplicates / impossible BP (dia ≥ sys) | 0 / 0 |

### Findings

| ID | Finding | Evidence | Consequence |
|---|---|---|---|
| D1 | `cigsPerDay` is missing **only for smokers** (22/22) | Median of all rows = **0**; median among smokers = **20** | Global-median imputation turns those 22 known smokers into 0-cigarette smokers. |
| D2 | Glucose is not missing at random with respect to diabetes | Missing rate 9.1% in non-diabetics vs 3.4% in diabetics; median glucose 78 vs **150** | A single global median (78) plus no missing-indicator discards information. |
| D3 | The strongest risk markers are rare | Stroke (n=22) → 45.5% CHD; diabetes (n=87) → 37.9%; BPMeds (n=100) → 33.0% | Any variance-based or IQR-based statistic treats these as noise or outliers (see P6, P7). |
| D4 | Narrow age support | Ages 32–70 only | The API accepts 20–100, so predictions at the extremes are unsupported extrapolation. |
| D5 | `education` is a socio-economic proxy; mode imputation fills **1** (the lowest level) | `fill_values['education'] = 1.0` | Fairness risk and negligible signal. Should be justified or dropped. |
| D6 | The binary 10-year label has no censoring information, no HDL, and comes from a historical, demographically narrow cohort | Kaggle subset of Framingham | Limits external validity. Any clinical use needs external validation and a comparison against an established risk score. |
| D7 | `smoking_intensity = is_smoking × cigsPerDay` equals `cigsPerDay` by construction | Non-smokers always have `cigsPerDay = 0` | The engineered feature is redundant, which is why the correlation filter drops it. |

---

## 3. Preprocessing: what is applied, and what should be

Order in `preprocess_train()`: split 60/20/20 → impute → one-hot → deterministic features → **fit IQR fences + feature selector** → clip → select → scale → **SMOTE** → `GridSearchCV`.

### P6: IQR clipping destroys binary features (Critical)

`fit_outlier_clipper()` computes 1.5×IQR fences for **every numeric column**, including 0/1 flags. For any flag with fewer than 25% positives, Q1 = Q3 = 0, so the fence is `[0, 0]` and every `1` is clipped to `0`. From the shipped bundle:

```
BPMeds           [0.000, 0.000]  <-- constant 0
prevalentStroke  [0.000, 0.000]  <-- constant 0
diabetes         [0.000, 0.000]  <-- constant 0
is_obese         [0.000, 0.000]  <-- constant 0
glucose (log)    [4.045, 4.700]  -> caps glucose at e^4.7 - 1 ≈ 109 mg/dL
```

The selector is fitted on the **unclipped** data but applied **after** clipping, so these columns survive selection as constants. Their LR coefficients are exactly `0.0000`, and glucose's is `0.0025`. Clipping glucose at 109 mg/dL also removes the diabetic range (median 150).

**Should be:** never clip binary columns. For continuous columns, enforce clinically plausible input limits as validation, and keep extreme-but-real values; they are signal (glucose 394 means uncontrolled diabetes). If winsorising, do it only on continuous columns at wide percentiles, inside the pipeline.

### P7: Variance and correlation filtering (High)

`VarianceThreshold`-style filtering at `var < 0.01` runs on **unscaled, mixed-unit** data (log features have variance around 0.02). It drops `prevalentStroke` (variance ≈ 0.006). The |r| > 0.9 correlation filter drops `is_smoking_YES` and `smoking_intensity`, and which one of a correlated pair it drops depends on column order.

**Should be:** none. There are 15 features and 511 events (≈ 34 events per variable), so regularisation handles redundancy. Feature choices should come from the clinical domain, not an order-dependent filter.

### P4: Feature engineering (High)

| Transform | Issue | Should be |
|---|---|---|
| `pulsePressure` **replaces** `sysBP`/`diaBP` | Systolic BP is one of the strongest CHD predictors. Absolute BP level is lost: **180/135 scores the same as 130/85** (0.146). | Keep `sysBP` (and `diaBP`). Add PP only if it helps in CV. |
| Age bins + continuous age | Redundant for a linear model; `ag_age_60plus` coefficient is 0 | Continuous age, or splines |
| `is_obese` / `is_overweight` | `is_obese` is zeroed by P6, and the net BMI effect becomes negative: BMI 26→35 **lowers** risk 0.146→0.127 | Continuous `log(BMI)` |
| `smoking_intensity` | Equals `cigsPerDay` (D7). Picks "the first `is_smoking_*` column", so in a mixed batch it can use `is_smoking_0`. This is a latent, batch-dependent bug. | Remove |
| `log1p` on skewed columns | Fine | Keep, inside a `ColumnTransformer` |

### P2/P3: Imputation and encoding (Medium; High at inference)

- Imputers are fitted on train only (✔), but they ignore D1/D2 and add no missing-indicators.
- Encoding uses ad-hoc `pd.get_dummies`: `drop_first=True` in training, `drop_first=False` at inference, and **silent zero-filling** of any expected column that is missing (`preprocess_inference`). That zero-fill is why a mismatched encoding never raises an error. It just scores the patient wrongly (see S2).

**Should be:** explicit mappings (`sex: {M:1, F:0}`, `is_smoking: {YES:1, NO:0}`) that reject unknown values. Use `SimpleImputer(add_indicator=True)`, or conditional imputation (smoker median for `cigsPerDay`, diabetes-aware for glucose), fitted inside the pipeline. **Fail loudly** on schema mismatch.

### P9: SMOTE (High)

Four separate problems:

1. **It is applied once, before `GridSearchCV`.** Synthetic points built from a sample in one CV fold land in the other folds, so CV rewards memorisation. Reproduced from `src/train.py`:

   | Model | CV AUC on SMOTE'd train (as-built) | Test AUC | Hyper-parameters selected |
   |---|---:|---:|---|
   | RandomForest | **0.967** | 0.647 | `max_depth=None, min_samples_leaf=1` |
   | XGBoost | **0.960** | 0.612 | `max_depth=7, n_estimators=300` |
   | GradientBoosting | **0.951** | 0.610 | `max_depth=5` |
   | KNN | **0.939** | 0.582 | `k=7, weights=distance` |
   | SVC | **0.928** | 0.556 | `rbf, C=10` |
   | AdaBoost | **0.913** | 0.666 | `lr=1.5, 200 estimators` |
   | NaiveBayes | 0.708 | 0.705 | — |
   | LogisticRegression | 0.724 | 0.714 | `C=0.1, L1` |

   A controlled re-run (`04_smote_cv_leak.py`) uses identical grids and identical train/test rows and changes only *where* SMOTE happens:

   | Model | Setup | CV AUC | Test AUC | CV − Test | Selected |
   |---|---|---:|---:|---:|---|
   | RandomForest | SMOTE before CV (as-built) | 0.966 | 0.642 | **0.325** | `max_depth=None, min_leaf=1` |
   | RandomForest | SMOTE inside CV | 0.701 | 0.684 | 0.017 | `max_depth=5, min_leaf=20` |
   | RandomForest | No resampling | 0.699 | **0.707** | −0.007 | `max_depth=5, min_leaf=20` |
   | XGBoost | SMOTE before CV (as-built) | 0.956 | 0.623 | **0.333** | `max_depth=7, lr=0.2, 300 trees` |
   | XGBoost | SMOTE inside CV | 0.690 | 0.681 | 0.009 | `max_depth=2, lr=0.05, 100 trees` |
   | XGBoost | No resampling | 0.697 | **0.696** | 0.000 | `max_depth=2, lr=0.05, 100 trees` |
   | KNN | SMOTE before CV (as-built) | 0.924 | 0.602 | **0.322** | `k=11, distance` |
   | KNN | SMOTE inside CV | 0.666 | 0.654 | 0.012 | `k=51, uniform` |
   | KNN | No resampling | 0.676 | **0.710** | −0.034 | `k=51, distance` |

   The flexible models consistently picked the high-capacity end of their grids: unlimited or deepest trees, the most estimators, the largest `C`, distance-weighted neighbours. The tree models are not weak on this data: with honest CV, a shallow regularised XGBoost scores **0.713** and RandomForest (min_leaf=20) **0.712** (§6.1).
2. **SMOTE interpolates binary columns.** 13.6% of synthetic `sex_M` values and 7.9% of `prevalentHyp` values are fractional. Mixed data needs `SMOTENC`, if oversampling is used at all.
3. **It shifts predicted probabilities by about +0.29** (mean predicted minus observed; see §6). Platt scaling has to undo this afterwards.
4. **It gives no discrimination benefit.** LR AUC is 0.721 without resampling vs 0.718 with SMOTE inside the pipeline.

**Should be:** no resampling. Train at the natural prevalence and handle the cost asymmetry with the decision threshold. If resampling is kept, put it inside an `imblearn.pipeline.Pipeline` so it runs within each CV training fold, and recalibrate afterwards.

### Step-by-step summary

| Step | As built | Should be |
|---|---|---|
| Split | 60/20/20 single split; selection, calibration and threshold all use 678 rows with **102 positives** | 80/20 dev/test. All tuning via (repeated) stratified CV on dev; test used once, with bootstrap CIs |
| Schema | None | Validate types, ranges (training support) and cross-field rules (`diaBP < sysBP`, non-smoker ⇒ `cigsPerDay = 0`) |
| Impute | Global median/mode; no indicators | Pipeline `SimpleImputer(add_indicator=True)` plus domain-conditional rules |
| Encode | `get_dummies`; drop_first differs between train and inference; silent zero-fill | Explicit 0/1 mapping; raise on unknown or missing |
| Features | PP replaces BP; bins/flags; redundant interaction | Raw clinical variables plus `log1p` on skewed ones |
| Outliers | 1.5×IQR on **all** numeric columns, including binaries | Input validation only; no clipping of binaries |
| Selection | Unscaled variance filter plus correlation filter | None (regularisation) |
| Scale | ✔ fitted on train | Same, inside the `Pipeline` |
| Imbalance | SMOTE before CV | None, or inside CV via `imblearn` |

---

## 4. Data-leakage check: split first, fit on train only

**Question:** is the data split into train / validation / test *before* any preprocessing, and is every preprocessing statistic learned from the training rows only?

**Answer:** yes for the preprocessing in `src/preprocessing.py`. No for what `src/train.py` does with the validation and test sets afterwards. No for the original notebook.

Evidence comes from `06_leakage_check.py`, which instruments `preprocess_train()` to record exactly which rows each fitted step receives, then re-runs it with the val/test rows deliberately corrupted.

### 4.1 Preprocessing (`src/preprocessing.py`): passes

| Check | Result | Evidence |
|---|---|---|
| Split before any fitting | ✅ PASS | Call trace: `split(3390 rows)` → `split(2712)` → `fit_imputer` → `fit_statistical_transforms` → `fit_scaler` → `apply_smote` |
| Each fitted step sees only training rows | ✅ PASS | Each received exactly **2,034 train / 0 val / 0 test** rows |
| Fitted state independent of val/test | ✅ PASS | Rescaling every numeric val/test value by ×0.2–5, flipping `sex` and blanking glucose leaves the imputer values, IQR fences, kept columns, scaler mean/scale and the SMOTE'd training matrix **bit-identical**. The same corruption applied to *train* rows does change them, so the test is sensitive. |
| Val/test are transformed, never refitted | ✅ PASS | Corrupted val rows come out different, using the stored training statistics |
| Inference reuses stored statistics only | ✅ PASS | `preprocess_inference()` has no fit calls |
| Splits disjoint and stratified | ✅ PASS | 2,034 / 678 / 678 rows; positive rate 15.1% / 15.0% / 15.0%; 0 identical feature rows shared across splits |
| No target leakage | ✅ PASS | `TenYearCHD` is not in the feature matrix. The strongest single feature is `sysBP` (AUC 0.685 on train), so no feature is a proxy for the label. |

Per step:

| Step | Learns statistics? | Fitted on | Applied to val/test as | Verdict |
|---|---|---|---|---|
| Stratified split 60/20/20 | — | — | — | ✅ first operation |
| Median/mode imputation | yes | train only | transform | ✅ |
| One-hot encoding (`sex`, `is_smoking`) | no | — (per split) | per split | ✅ no leak; consistency risk (P3) |
| Pulse pressure, age bins, BMI flags, `log1p` | no | — | per split | ✅ |
| IQR fences | yes | train only | transform | ✅ no leak, but applied to the wrong columns (P6) |
| Variance/correlation selector | yes | train only | transform | ✅ no leak, but harmful (P7) |
| `StandardScaler` | yes | train only | transform | ✅ |
| SMOTE | yes | train only | not applied | ✅ |

So the preprocessing problems in §3 are about *what* the transforms do, not *where* they are fitted.

### 4.2 After preprocessing (`src/train.py`): fails

| # | How val/test data leaks into a decision | Code | Measured effect | Severity |
|---|---|---|---|---|
| L1 | The SMOTE'd training set is fed to `GridSearchCV`. Synthetic rows interpolated from patients in one fold land in the other folds. | `train.py:63` → `models.tune_model` | XGBoost CV AUC **0.960** vs test **0.612** | High |
| L2 | The best model is picked by **test** AUC, and the stacking members are the top 3 by **test** AUC | `train.py:108-112, 147-151, 200` | Across 1,000 bootstrap resamples of the test set the "winner" changes: LR 48%, Stacking 34%, NaiveBayes 17%. The pick is noise, and the winner's test score is a maximum over candidates, so it is optimistic. | High |
| L3 | Stacking is trained on train **+ val**, then its threshold is tuned on val | `train.py:118-128` | AUC on val (seen during training) 0.725 vs test 0.713 | High |
| L4 | The saved "cross-validation" report runs on SMOTE'd train + val + **test** | `train.py:209-211` | 4,810 rows: **30% synthetic, 14% test**; 40% positive vs a real 15% | High (reporting) |
| L5 | Platt calibration and the re-tuned threshold are both fitted on the same val rows | `train.py:228-232` | The test set stays clean, but the threshold is fitted to calibration noise on 102 positives | Medium |
| L6 | `learning_curve()` runs on the SMOTE'd training set | `train.py:247` | Same mechanism as L1 | Medium |
| L7 | Preprocessing is fitted once on all training rows, then reused inside the `GridSearchCV` folds instead of being refitted per fold | `preprocess_train` → `tune_model` | CV AUC 0.7234 vs 0.7233 refitted per fold, **negligible here** | Low; still fix it by putting preprocessing in a `Pipeline` |
| L8 | EDA plots and feature decisions (log columns, pulse pressure) were made on the full dataset | `train.py:run_eda`, notebook | Not measurable; a researcher-choice leak | Low |

### 4.3 The original notebook: fails the split-first rule

The notebook imputes, scales and applies SMOTE to the **whole** dataset, then splits. Reproduced with its feature set:

| Order | Model | Test AUC | Test accuracy | Synthetic rows in "test" |
|---|---|---:|---:|---:|
| N0 Notebook: impute + scale + SMOTE on all rows, then split | XGBoost | **0.954** | **0.892** | 40% |
| N0 Notebook | LogReg | 0.734 | 0.675 | 40% |
| N1 Impute + scale on all rows, split, SMOTE on train only | XGBoost | 0.640 | 0.822 | 0% |
| N1 | LogReg | 0.737 | 0.692 | 0% |
| **N2 Split first, everything fitted on train (correct)** | XGBoost | 0.634 | 0.850 | 0% |
| **N2** | LogReg | 0.736 | 0.853 | 0% |

- **N0 reproduces the notebook's headline number** (XGBoost ≈ 0.89 test accuracy, and ASSESSMENT.md's "XGBoost ROC AUC ~0.90"). 40% of the "test" patients are synthetic blends of training patients, so a depth-7 XGBoost scores well by recognising them. Split first, the same model gets AUC **0.634** and accuracy **0.850**, which is no better than predicting "no CHD" for everyone (0.849).
- **N1 vs N2:** fitting the imputer and scaler before the split moves AUC by less than 0.01 here. Almost all of the notebook's inflation comes from applying SMOTE before the split. Fitting on train only still matters: it makes the test honest by construction rather than by luck.
- Logistic regression barely changes under N0 because a linear model can't memorise synthetic neighbours. Flexible models can, which is why the leak mainly shows up in the tree-model results.

### 4.4 The correct order

```
raw data
 └─ 1. split FIRST: development (80%) / test (20%), stratified      ← test set locked away
 └─ 2. on development data only: K-fold CV
        each fold: fit  impute → encode → scale → (resample) → model   on the fold's train part
                   apply (transform only)                              to the fold's validation part
        → choose model, hyper-parameters, calibration and threshold from CV scores only
 └─ 3. refit the whole Pipeline on all development data
 └─ 4. test set: transform + predict ONCE → report with confidence intervals
 └─ 5. inference: the same fitted Pipeline, transform only
```

In code, put every step that learns statistics inside one `sklearn` or `imblearn` `Pipeline`. Then `GridSearchCV`, `cross_val_score` and `CalibratedClassifierCV` refit it inside every fold automatically, which removes L1, L6 and L7 by construction. Select on CV rather than the test set (L2), keep stacking and threshold data separate (L3, L5), and never pass test rows to CV (L4). `05_reference_pipeline.py` follows this order.

---

## 5. Modelling, selection and evaluation

| ID | Severity | Finding | Where |
|---|---|---|---|
| M1 | High | **The best model is chosen by test-set AUC** (`select_best_model` sorts test results), and stacking members are the "top-3 by test AUC". The test set is no longer an unbiased estimate. | `train.py` `select_best_model`, `add_stacking_model` |
| M2 | High | **Stacking is fitted on train + val, then its threshold is tuned on val**, which is its own training data. | `train.py:add_stacking_model` |
| M3 | High | **The saved "cross-validation" report is invalid.** It runs on `concat(SMOTE'd train, val, test)`: synthetic rows at ~40% prevalence, plus the test set. It reports precision **0.62**, while real test precision is **0.23**. | `train.py` step 7, `reports/cv_LogisticRegression.csv` |
| M4 | Medium | The learning curve is computed on SMOTE'd data, so the README's "gap of 0.005 confirms no overfitting" is measured on synthetic samples. | `train.py` step 9 |
| M5 | Medium | **No uncertainty is reported.** The test AUC 95% CI is **[0.657, 0.767]**. LR (0.714), Stacking (0.713) and NaiveBayes (0.705) cannot be told apart. | — |
| M6 | Medium | Threshold = argmax F2 with an arbitrary precision floor of 0.20, on 102 positives. The grid starts at 0.10, and in the committed results XGBoost and GradientBoosting sit exactly on that boundary. After calibration, both the Platt fit and the threshold re-tune use the **same** validation set. The shipped threshold is **0.13**; the docs say **0.18**. | `models.find_optimal_threshold` |
| M7 | Medium | Custom `_PlattCalibratedModel`: `predict()` returns the uncalibrated base model's 0.5-threshold labels (on the SMOTE scale), which is inconsistent with `predict_proba()`. sklearn ≥ 1.6 supports `CalibratedClassifierCV(FrozenEstimator(model))`, so the custom class isn't needed. | `models.py` |
| M8 | Medium | Metrics missing for a risk model: Brier score, calibration slope/intercept, PR-AUC, subgroup performance. DCA is saved as CSV without treat-all/treat-none references. | `evaluation.py` |
| M9 | Low | 4 of the 17 LR coefficients are exactly zero: `BPMeds`, `diabetes` and `is_obese` because they are constant after clipping, and `ag_age_60plus` from the L1 penalty (`C=0.1`). `heartRate` and `glucose` are ≈ 0. | shipped bundle |

---

## 6. Controlled experiments

### 6.1 Ablation of preprocessing decisions

5 × 5 repeated stratified CV on all 3,390 rows, with **every fitted step inside the folds**. Values are mean ± SD over 25 folds (`03_preprocessing_ablation.py`). "Mean pred − obs" is calibration-in-the-large; 0 is perfect.

| Variant | AUC | PR-AUC | Brier | Mean pred − obs |
|---|---|---|---|---|
| **A** As-built (SMOTE, clips binaries, BP → PP), LR C=0.1 L1 | 0.713 ± 0.026 | 0.335 | 0.213 | **+0.290** |
| B  A without clipping binaries | 0.715 ± 0.026 | 0.337 | 0.212 | +0.287 |
| C  B + keep sysBP/diaBP | 0.717 ± 0.027 | 0.333 | 0.212 | +0.288 |
| D  C without SMOTE | 0.721 ± 0.025 | 0.338 | **0.117** | **0.000** |
| **E** Clean pipeline LR, no resampling | **0.721 ± 0.027** | **0.361** | **0.116** | 0.000 |
| F  E + `class_weight='balanced'` | 0.721 ± 0.027 | 0.354 | 0.210 | +0.289 |
| G  E + SMOTE *inside* the pipeline | 0.718 ± 0.027 | 0.351 | 0.211 | +0.286 |
| H  HistGradientBoosting (depth 3) | 0.695 ± 0.025 | 0.322 | 0.119 | −0.001 |
| I  XGBoost (depth 2, regularised) | 0.713 ± 0.026 | 0.338 | 0.117 | 0.000 |
| J  RandomForest (min_samples_leaf 20) | 0.712 ± 0.025 | 0.333 | 0.118 | 0.000 |

What the ablation shows:

- **The discrimination ceiling on this data is about 0.72 AUC.** Every sensible model lands within ±0.01 of it.
- **Resampling never helps AUC and always wrecks calibration.** Brier roughly doubles and mean risk is overstated by 29 points.
- The bug fixes (B, C) move aggregate AUC only slightly, because the affected patients are rare. Their value shows up in the behavioural tests below.

### 6.2 Shipped bundle vs reference pipeline on the repo's own test split

The reference pipeline is `05_reference_pipeline.py`: explicit encoding, domain-aware `cigsPerDay` imputation, median imputation with indicators, `log1p`, scaling, LR, and `CalibratedClassifierCV(cv=5)`. There is no SMOTE, clipping or filtering. It is trained on the same 80% the repo trains, tunes and calibrates on, and tested on the same 678 rows.

| Metric | Shipped bundle | Reference |
|---|---:|---:|
| AUC (95% bootstrap CI) | 0.714 [0.657, 0.767] | 0.719 [0.664, 0.774] |
| PR-AUC | 0.315 | 0.329 |
| Brier | 0.1186 | 0.1172 |
| Max predicted risk | **0.387** | 0.657 |
| Sens / Spec at 10% risk | 0.882 / 0.377 | 0.794 / 0.467 |
| Sens / Spec at 20% risk | 0.431 / 0.835 | 0.451 / 0.851 |

### 6.3 Behavioural tests (same patient, one factor changed)

Baseline patient: 55-year-old male, non-smoker, BP 130/85, cholesterol 220, BMI 26, glucose 90.

| Scenario | Shipped | Reference | Expected direction |
|---|---:|---:|---|
| Baseline | 0.146 | 0.148 | — |
| Diabetic, glucose 250 | **0.146** | 0.353 | ↑ |
| On BP medication | **0.146** | 0.150 | ↑ (weak) |
| Prior stroke | **0.146** | 0.217 | ↑ |
| BMI 35 | **0.127** ↓ | 0.146 | ↑ / flat |
| BP 180/135 | **0.146** | 0.272 | ↑ |
| Smoker, 20/day | 0.206 | 0.234 | ↑ |
| Female | 0.100 | 0.096 | ↓ |

This is the key point for production: **the aggregate metrics are nearly identical, but the shipped model gives clinically wrong answers for diabetics, stroke survivors and severe hypertensives.** Aggregate AUC is not an acceptance test. Behavioural tests like these need to be part of CI.

---

## 7. Inference and serving

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| S1 | Critical | **The web UI falls back to a hard-coded formula on any API failure**, including HTTP 4xx validation errors. When `/health` is up, the badge still says "API CONNECTED", and the result is labelled "LogReg" with a hard-coded 0.18 threshold, so a made-up risk is shown as the model's. | `Cardiovascular Risk Assessment.html` `demoPredict()` and the `catch(apiErr)` block |
| S2 | Critical | **The CLI mis-encodes sex.** The documented `--sex M` crashes (`int('M')` is evaluated eagerly as the `dict.get` default). `--sex 1` produces column `sex_1`, the model expects `sex_M`, and the silent zero-fill makes **every CLI patient female** (male 0.1097 vs the correct 0.1609). | `src/predict.py:build_single_patient_df` |
| S3 | High | The API maps `sex` to `"M"/"F"` but sends `is_smoking` as `0/1`, producing `is_smoking_0/1` instead of the trained `is_smoking_YES`. This is masked only because the correlation filter happens to drop that column. It will break on any retrain that keeps it. | `app_api.py:PatientInput.to_model_dict` |
| S4 | High | Input ranges exceed training support: age 20–100 (trained on 32–70), glucose up to 500. There are no cross-field checks, so `is_smoking=0, cigsPerDay=30` is accepted and scored as a heavy smoker. | `PatientInput` |
| S5 | Medium | `POST /predict/batch` with `[]` → **HTTP 500**. Batch size is unbounded, and the batch path has no error handling. | probe |
| S6 | Medium | **The artefact can't be loaded portably.** It pickles the custom class `models._PlattCalibratedModel`, and the repo-root `models/` directory shadows `src/models.py`, so `joblib.load` from the repo root fails with `AttributeError`. It was pickled with scikit-learn **1.8.0**, `requirements.txt` is unpinned (`>=`), and loading under 1.9.1 raises `InconsistentVersionWarning`. | probe |
| S7 | Medium | `requirements.txt` omits `matplotlib` and `seaborn`, which `train.py` imports, so the README's `pip install -r requirements.txt && python src/train.py` fails. | `src/visualization.py` imports |
| S8 | Medium | No model metadata in the bundle: git SHA, data hash, library versions, metrics, feature schema and training ranges are all missing. The API response has no model version. | `train.py:save_artifacts` |
| S9 | Medium | No tests and no CI. `utils.detect_feature_drift`, `compute_shap_values` and the `assert_*` checks are never called. Diagnostics use `print` rather than logging, and there is no request or prediction logging or monitoring. | repo |
| S10 | Low | CORS `*`, no auth or rate-limit. The model loads lazily on the first request. `config.py` creates directories as an import side effect. | `app_api.py`, `config.py` |

---

## 8. Documentation integrity

| Claim | Location | Reality |
|---|---|---|
| Decision threshold 0.18 | README, TECHNICAL_DOCUMENTATION | Shipped bundle: **0.13** |
| "Best single model: XGBoost (ROC AUC ~0.90)" | ASSESSMENT.md §5 | That number comes from the leaky notebook; XGBoost's test AUC here is **0.61** |
| Threshold "maximises recall" | ASSESSMENT.md §2.4 | Code maximises F2 with a precision floor |
| joblib replaces pickle to fix "not safe for loading untrusted files" | ASSESSMENT.md §2.8 | joblib **is** pickle underneath; loading an untrusted bundle can still execute code |
| Learning-curve gap 0.005 "confirms not overfitting" | README | Measured on SMOTE'd training data (M4) |
| "Calibrated output of 0.30 reflects ≈30% risk" | README / docs | Holds on average, not in high-risk subgroups (§0); outputs never exceed 0.39 |
| `app_streamlit.py` | README project tree | Does not exist |
| "Recommend recall ≥ 0.90" | ASSESSMENT.md §5 | Not what the code optimises for |

---

## 9. How it should be built

### 9.1 Reference design

```python
pipe = Pipeline([
    ("schema", DomainCleaner()),  # explicit 0/1 maps; raise on unknown; cigsPerDay rules
    ("pre", ColumnTransformer([
        ("log", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("log", FunctionTransformer(np.log1p)),
                          ("sc",  StandardScaler())]), ["cigsPerDay", "totChol", "glucose", "BMI"]),
        ("lin", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), ["age", "sysBP", "diaBP", "heartRate"]),
        ("bin", SimpleImputer(strategy="most_frequent"),
                ["sex", "is_smoking", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]),
    ])),
    ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=5000), method="sigmoid", cv=5)),
])
# Model selection: RepeatedStratifiedKFold on the 80% dev set; score = roc_auc AND neg_brier/log_loss.
# If resampling is truly wanted: imblearn.pipeline.Pipeline([... , ("smote", SMOTENC(...)), ("clf", ...)])
# Test set: evaluated ONCE, with bootstrap CIs, calibration plot, subgroup table, DCA vs treat-all/none.
# Persist ONE object (the Pipeline) + metadata; pin versions; verify on load.
```

A runnable version is in `05_reference_pipeline.py`.

### 9.2 Production checklist

| Area | Requirement |
|---|---|
| **Data contract** | Pandera or Pydantic schema for raw input: types, allowed categories, training-support ranges, cross-field rules. Reject; never zero-fill. |
| **Single artefact** | One `sklearn.Pipeline` from raw input to probability. Serialise with `skops` or joblib alongside `{git_sha, data_sha256, sklearn/xgboost versions, feature schema, training ranges, CV + test metrics with CIs, threshold, created_at}`. Pin a lock file. |
| **Selection** | CV on the dev set only (nested CV if reporting tuned performance). Choose the simplest model within 1 SE. Test used once. |
| **Imbalance** | Train at natural prevalence. Decide the threshold with clinicians, using risk bands (e.g. <5% / 5–7.5% / 7.5–20% / ≥20%) or DCA net-benefit, not F2. |
| **Calibration** | `CalibratedClassifierCV` (or `FrozenEstimator` + held-out set). Report slope, intercept and Brier, overall **and by subgroup**. |
| **Tests (CI)** | Unit tests per transform. **Behavioural tests** (monotonic in diabetes, sysBP, age, smoking; sex encoding; single row == batch). **Train/serve parity test** (the same raw row through the training path and the API gives identical features). Golden-prediction regression test. |
| **Serving** | Load the model at startup and fail fast. Model version in every response. 4xx for bad input, bounded batch size. Structured request/prediction logging. **No silent fallback predictions.** |
| **Monitoring** | Input drift (PSI or KS per feature), prediction-distribution drift, and calibration once outcomes arrive. Alerting and a retraining playbook. |
| **Governance** | Model card following TRIPOD+AI. Fairness and subgroup review (sex, age band, education). Benchmark against an established clinical risk score. External validation before any clinical use. |

### 9.3 Remediation plan

**P0: before any prediction is shown to anyone**
1. Remove the IQR clipper and the variance/correlation filters. Keep `sysBP`/`diaBP`. Remove `smoking_intensity`.
2. Remove SMOTE (or move it inside an `imblearn` pipeline). Retrain and recalibrate.
3. Fix inference encoding: explicit maps for `sex`/`is_smoking` in the API and CLI; raise on missing or unknown columns instead of zero-filling.
4. Remove the UI's `demoPredict` fallback, or restrict it to an explicit, unmistakably labelled demo toggle. Never use it on 4xx/5xx.
5. Clamp API ranges to the training support (age 32–70, etc.) or flag out-of-distribution requests.

**P1: make it trustworthy**
6. Refactor to a single `Pipeline` artefact with metadata and pinned dependencies. Fix `requirements.txt`.
7. Select models by CV on the dev set, touch the test set once, and report CIs. Drop the test-set-based stacking selection.
8. Add unit, behavioural and parity tests plus CI.
9. Replace F2 threshold search with a clinician-agreed threshold or risk bands, backed by DCA.

**P2: make it operable**
10. Subgroup, fairness and calibration reporting. Decide on `education`.
11. Logging, monitoring, a model registry and versioning.
12. Correct the docs (§8), write a model card, and run external validation.

---

## 10. Reproducing this audit

From the repo root, after `pip install -r requirements.txt matplotlib seaborn httpx`:

| Script | What it shows | Runtime |
|---|---|---|
| `python audit/01_data_profile.py` | Data profile, missingness patterns, consistency checks (§2) | seconds |
| `python audit/02_inference_probe.py` | Bundle internals (fences, coefficients), API/CLI behaviour, subgroup calibration, load failure (§0, §3, §7) | seconds |
| `python audit/03_preprocessing_ablation.py` | Ablation table (§6.1) | ~6 min |
| `python audit/04_smote_cv_leak.py` | CV-vs-test gap from SMOTE-before-CV; bootstrap CI (§3 P9, §5 M5) | ~3 min |
| `python audit/05_reference_pipeline.py` | Reference pipeline vs shipped bundle; behavioural tests (§6.2, §6.3) | seconds |
| `python audit/06_leakage_check.py` | Split-first / train-only fitting trace, perturbation test, downstream leaks, notebook comparison (§4) | ~2 min |

The CV-vs-test table in §3 P9 is from a clean re-run of `python src/train.py` at `3a9ce96` (scikit-learn 1.9.1, xgboost 3.2.0, imbalanced-learn 0.14.2). It matches `reports/model_comparison.csv` for every model except XGBoost (0.612 vs 0.605) and GradientBoosting (0.610 vs 0.557), which differ because of library versions.
