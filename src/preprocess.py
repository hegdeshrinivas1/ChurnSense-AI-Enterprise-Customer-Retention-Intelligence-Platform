"""Data cleaning and preprocessing for ChurnSense."""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from src.config import TARGET, ID_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES, DATA_RAW_FILE


def clean_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Clean raw Telco dataset: fix TotalCharges, strip whitespace, drop ID, encode target.

    Args:
        df: Input DataFrame.
        verbose: Whether to print before/after null and dtype summaries.

    Returns:
        Tuple of (X, y) where y is None if TARGET column is not in df.
    """
    df_clean = df.copy()

    # Capture before summary
    before_nulls = df_clean.isnull().sum().to_dict()
    before_dtypes = {col: str(dtype) for col, dtype in df_clean.dtypes.items()}

    # Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()

    # Drop ID column if present
    if ID_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[ID_COLUMN])

    # Strip whitespace from string/object cells
    str_cols = df_clean.select_dtypes(include=["object", "string", "str"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Fix TotalCharges whitespace / blank issue
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce").fillna(0.0)

    # Ensure numeric types
    for col in NUMERIC_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

    # Handle Target if present
    y: Optional[pd.Series] = None
    if TARGET in df_clean.columns:
        # Map Yes -> 1, No -> 0
        y = df_clean[TARGET].map({"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}).astype(int)
        X = df_clean.drop(columns=[TARGET])
    else:
        X = df_clean

    # Capture after summary
    after_nulls = X.isnull().sum().to_dict()
    after_dtypes = {col: str(dtype) for col, dtype in X.dtypes.items()}

    if verbose:
        print("=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Original shape: {df.shape} -> Cleaned feature shape: {X.shape}")
        print("\nBefore Nulls (columns with nulls or empty):")
        total_charges_spaces = (df["TotalCharges"] == " ").sum() if "TotalCharges" in df.columns else 0
        print(f" - TotalCharges whitespace count: {total_charges_spaces}")
        for col, count in before_nulls.items():
            if count > 0:
                print(f" - {col}: {count}")

        print("\nAfter Nulls (all columns):")
        null_count = sum(after_nulls.values())
        print(f" - Total null count in features: {null_count}")

        print("\nCleaned Feature Data Types:")
        for col, dt in after_dtypes.items():
            print(f" - {col:20s}: {dt}")
        if y is not None:
            print(f"\nTarget ({TARGET}) Distribution:")
            print(y.value_counts(normalize=True).rename({0: "No (0)", 1: "Yes (1)"}).to_string())
        print("=" * 60)

    return X, y


if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_RAW_FILE)
    X, y = clean_data(raw_df, verbose=True)
