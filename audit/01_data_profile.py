"""
Audit step 1 — raw data profile, consistency checks and missingness patterns.

Run from the repo root:
    python audit/01_data_profile.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO / "data_cardiovascular_risk.csv")

print(f"shape: {df.shape}")
print(f"target: {df.TenYearCHD.value_counts().to_dict()}  prevalence={df.TenYearCHD.mean():.3f}")
na = df.isna().sum()
print("\nmissing values:\n" + na[na > 0].to_string())
print(f"rows with any NA: {df.isna().any(axis=1).sum()} ({df.isna().any(axis=1).mean():.1%})")
print(f"duplicate rows (excluding id): {df.drop(columns='id').duplicated().sum()}")

print("\nrange of continuous features:")
cont = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
print(df[cont].describe().T[["min", "50%", "max"]].to_string())
print("\nskew:\n" + df[cont].skew().round(2).to_string())

print("\nrare binary features (count of 1s):")
for c in ["prevalentStroke", "diabetes", "BPMeds", "prevalentHyp"]:
    print(f"  {c:16s} {int(df[c].sum()):5d}   CHD rate when 1: {df.loc[df[c] == 1, 'TenYearCHD'].mean():.3f}")

print("\nconsistency checks:")
print("  cigsPerDay NaN & smoker     :", int((df.cigsPerDay.isna() & (df.is_smoking == "YES")).sum()))
print("  cigsPerDay NaN & non-smoker :", int((df.cigsPerDay.isna() & (df.is_smoking == "NO")).sum()))
print("  median cigsPerDay (all)     :", df.cigsPerDay.median())
print("  median cigsPerDay (smokers) :", df.loc[df.is_smoking == "YES", "cigsPerDay"].median())
print("  diaBP >= sysBP              :", int((df.diaBP >= df.sysBP).sum()))
print("  glucose median | diabetes   :", df.groupby("diabetes").glucose.median().to_dict())
print("  glucose NaN rate | diabetes :", df.groupby("diabetes").glucose.apply(lambda s: round(s.isna().mean(), 3)).to_dict())

print("\nCHD rate when feature is missing vs present:")
for c in ["glucose", "education", "BPMeds", "totChol", "cigsPerDay", "BMI"]:
    m = df[c].isna()
    print(f"  {c:11s} missing={df.loc[m, 'TenYearCHD'].mean():.3f} (n={m.sum():3d})  "
          f"present={df.loc[~m, 'TenYearCHD'].mean():.3f}")
