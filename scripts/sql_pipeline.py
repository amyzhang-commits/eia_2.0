#!/usr/bin/env python3
"""
sql_pipeline.py

Phase 3/4: Data cleaning + SQL preparation utilities.

Functions
---------
schema_preview(df)
    Print schema overview + detect missingness representations.
sql_prep_columns(df, date_cols=None, replace_missing_values=None, ...)
    Clean column names + handle missingness + optional date creation + save CSV.
dtype_audit(df, print_results=True)
    Audit inferred vs coercible dtypes with missing % context.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Dict, List
import warnings

# Silence FutureWarnings from pandas operations
warnings.filterwarnings("ignore", category=FutureWarning)

# Project paths (match dataset_ingestion.py)
DATASETS_OUTPUT = Path(__file__).parent.parent / "datasets_merged"
DATASETS_OUTPUT.mkdir(exist_ok=True)


# ======================================================
# 🔄 Schema Preview Helper
# ======================================================


def schema_preview(df: pd.DataFrame):
    """
    Print a human-friendly schema overview for a merged DataFrame:
    - Columns + dtype
    - Global missingness representations (NaN, blanks, 'NA', '.', etc.)
    """
    print("🧬 Full Column Schema (name → dtype):\n")
    schema_df = df.dtypes.rename("dtype").to_frame()
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        print(schema_df.to_string())
    print()

    # Detect missingness representations (global)
    placeholders = ['NA', 'N/A', 'na', 'N/a', '-', '.']
    missing_repr = {}
    missing_repr['NaN/None'] = df.isnull().any().any()
    missing_repr['Empty string / whitespace'] = (
        df.astype(str).apply(lambda col: col.str.strip() == '').any().any()
    )
    for val in placeholders:
        missing_repr[val] = (df == val).any().any()

    print("🔎 Detected missingness representations in dataset:")
    for rep, present in missing_repr.items():
        if present:
            print(f"   • {rep}")
    print()


# ======================================================
# 📥 Phase 3: SQL prep column names (FutureWarning-safe)
# ======================================================


def sql_prep_columns(
    df: pd.DataFrame,
    date_cols: Optional[Dict[str, tuple]] = None,
    replace_missing_values: Optional[List[str]] = None,
    missing_marker: str = 'null_rec',
    save: bool = False,
    output_name: str = 'sql_ready'
) -> pd.DataFrame:
    """
    Prepare a DataFrame for SQL insertion:
    - Clean column names (lowercase, underscores, remove special chars)
    - Move any leading numbers in column names to the end
    - Convert blanks/empty strings to NaN
    - Optionally replace dataset-specific missingness representations with a marker
    - Optionally combine year/month columns into a single date column
    - Optionally save to CSV
    - Auto-prints final column names

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean
    date_cols : dict, optional
        Mapping for date creation: {'new_date_col': ('year_col', 'month_col')}
        Example: {'date': ('year', 'month')}
    replace_missing_values : list, optional
        List of values in the dataset to treat as missing and replace with missing_marker
        Example: ['NA', 'N/A', 'na', 'N/a', '-', '.']
    missing_marker : str, default 'null_rec'
        Value to use when replacing dataset-specific missingness
    save : bool, default False
        Whether to save the cleaned DataFrame to CSV
    output_name : str, default 'sql_ready'
        Base name for the saved CSV file

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for SQL insertion
    """
    # 0️⃣ Handle missingness (FutureWarning-safe)
    df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)

    if replace_missing_values:
        df = df.replace(replace_missing_values, missing_marker).infer_objects(copy=False)

    # 1️⃣ Clean column names
    def clean_col(col: str) -> str:
        col = col.strip()
        col = re.sub(r'\?', '', col)
        col = col.lower()
        col = col.replace(' ', '_')
        col = re.sub(r'[^a-z0-9_]', '', col)

        # Move leading numbers to the end
        match = re.match(r'^(\d+)(.*)', col)
        if match:
            prefix = match.group(2).strip('_')
            col = f"{prefix}_{match.group(1)}" if prefix else f"col_{match.group(1)}"
        return col

    # Clean all column names
    cleaned_cols = [clean_col(c) for c in df.columns]

    # Suffix duplicates: foo_bar, foo_bar_2, foo_bar_3, etc.
    seen = {}
    final_cols = []
    for col in cleaned_cols:
        if col not in seen:
            seen[col] = 1
            final_cols.append(col)
        else:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")

    df.columns = final_cols

    # 2️⃣ Optional date column creation
    if date_cols:
        for new_col, (year_col, month_col) in date_cols.items():
            if year_col in df.columns and month_col in df.columns:
                df[new_col] = pd.to_datetime(
                    df[year_col].astype(str) + '-' +
                    df[month_col].astype(str) + '-01',
                    errors='coerce'  # invalid combos become NaT
                )

    # 3️⃣ Auto-print final column names
    print("\n🧬 Final cleaned column names:")
    print("   " + ", ".join(df.columns.tolist()))
    print()

    # 4️⃣ Optionally save CSV
    if save:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATASETS_OUTPUT / f"{output_name}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 SQL-prepared CSV saved to: {output_file}\n")

    return df


# ======================================================
# 📥 Phase 4: SQL prep - dtypes (FutureWarning-safe)
# ======================================================


def dtype_audit(df: pd.DataFrame, print_results: bool = True) -> pd.DataFrame:
    """
    Inspect inferred vs coercible dtypes without mutating data.
    Adds context with missing/empty percentage.

    Returns a DataFrame with:
    - column
    - inferred_dtype
    - percent parseable as numeric
    - percent parseable as datetime
    - percent missing/NaN
    """
    records = []

    for col in df.columns:
        series = df[col]
        inferred = str(series.dtype)

        # Missing/NaN ratio
        missing_ratio = series.isna().mean()

        # Try numeric coercion safely
        coerced_numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = coerced_numeric.notna().mean()

        # Try datetime coercion safely
        coerced_datetime = pd.to_datetime(series, errors="coerce")
        datetime_ratio = coerced_datetime.notna().mean()

        records.append({
            "column": col,
            "inferred_dtype": inferred,
            "numeric_parseable_%": round(numeric_ratio * 100, 2),
            "datetime_parseable_%": round(datetime_ratio * 100, 2),
            "missing_%": round(missing_ratio * 100, 2)
        })

    audit_df = pd.DataFrame(records)
    if print_results:
        print("\n🔍 DTYPE AUDIT RESULTS:")
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
            print(audit_df.to_string(index=False))
        print()
    return audit_df


# ======================================================
# 🧪 Demo / Entry point
# ======================================================
if __name__ == "__main__":
    print("🧪 sql_pipeline.py demo")
    print("💡 Import this module and chain with dataset_ingestion:")
    print("   from dataset_ingestion import merge_folder_files")
    print("   from sql_pipeline import schema_preview, sql_prep_columns, dtype_audit")
    print("   df = merge_folder_files()")
    print("   schema_preview(df)")
    print("   df_clean = sql_prep_columns(df, save=True)")
