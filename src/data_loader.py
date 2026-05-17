"""
Data loading and initial exploratory data analysis.
"""
import pandas as pd
import numpy as np
from config import DATA_PATH, TARGET_COL, CATEGORICAL_COLS, CONTINUOUS_COLS


DROP_COLS = ["id"]  # patient identifier — not a predictive feature

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.concat([missing, missing_pct], axis=1,
                    keys=["count", "%"])[missing > 0])
    print("\n--- Duplicates ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("=" * 60)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T


def class_distribution(df: pd.DataFrame) -> pd.Series:
    dist = df[TARGET_COL].value_counts(normalize=True) * 100
    print(f"\nTarget distribution (%):\n{dist.round(2)}")
    return dist


def unique_counts(df: pd.DataFrame) -> pd.Series:
    return df.nunique().sort_values()


def split_by_type(df: pd.DataFrame):
    """Return (categorical_cols, continuous_cols) that actually exist in df."""
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    con = [c for c in CONTINUOUS_COLS if c in df.columns]
    return cat, con


if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    print("\nDescriptive Statistics:")
    print(descriptive_stats(df))
    class_distribution(df)
    print("\nUnique values per column:")
    print(unique_counts(df))
