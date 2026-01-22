#!/usr/bin/env python3
"""
sql_pipeline.py

Phase 3/4: Data cleaning + SQL preparation utilities.

Functions
---------
schema_preview(df)
    Print schema overview + detect missingness representations.
    Returns per-column missingness breakdown (raw counts).
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


def schema_preview(
    df: pd.DataFrame,
    save: bool = True,
    output_name: str = 'schema_missingness_audit'
) -> pd.DataFrame:
    """
    Print a human-friendly schema overview for a merged DataFrame:
    - Columns + dtype
    - Returns per-column missingness breakdown (raw counts)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to audit
    save : bool, default True
        Whether to save the missingness audit to CSV
    output_name : str, default 'schema_missingness_audit'
        Base name for the saved CSV file

    Returns
    -------
    pd.DataFrame
        Per-column missingness counts with columns:
        - dtype, total_rows, null_nan, empty_whitespace
        - Additional columns for any detected placeholders
    """
    print("🧬 Full Column Schema (name → dtype):\n")
    schema_df = df.dtypes.rename("dtype").to_frame()
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        print(schema_df.to_string())
    print()

    # Per-column missingness breakdown (raw counts)
    placeholders = ['NA', 'N/A', 'na', 'N/a', '-', '.']
    records = []
    for col in df.columns:
        series = df[col]
        record = {
            'column': col,
            'dtype': str(series.dtype),
            'total_rows': len(series),
            'null_nan': int(series.isnull().sum()),
            'empty_whitespace': int((series.astype(str).str.strip() == '').sum()),
        }
        # Count each placeholder (only if present in this column)
        for ph in placeholders:
            count = int((series == ph).sum())
            if count > 0:
                record[f'"{ph}"'] = count

        records.append(record)

    missingness_df = pd.DataFrame(records).set_index('column')

    # Set display to show all rows in Jupyter
    pd.set_option('display.max_rows', None)

    # Optionally save to datasets_merged (pre-cleaned audit)
    if save:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATASETS_OUTPUT / f"{output_name}_{timestamp}.csv"
        missingness_df.to_csv(output_file)
        print(f"💾 Missingness audit saved to: {output_file}\n")

    return missingness_df


# ======================================================
# 📥 Phase 3: SQL prep column names (FutureWarning-safe)
# ======================================================


def sql_prep_columns(
    df: pd.DataFrame,
    date_cols: Optional[Dict[str, tuple]] = None,
    recode_values: Optional[Dict[str, str]] = None,
    save: bool = False,
    output_name: str = 'sql_ready'
) -> pd.DataFrame:
    """
    Prepare a DataFrame for SQL insertion:
    - Clean column names (lowercase, underscores, remove special chars)
    - Move any leading numbers in column names to the end
    - Convert empty strings/whitespace to 'not_reported' (preserves distinction from NaN)
    - NaN values stay as NaN (become NULL in SQL)
    - Optionally recode dataset-specific values (e.g., '.', '-') to meaningful strings
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
    recode_values : dict, optional
        Mapping of original values to new values for coded missingness
        Example: {'.': 'suppressed', '-': 'zero_or_na'}
    save : bool, default False
        Whether to save the cleaned DataFrame to CSV
    output_name : str, default 'sql_ready'
        Base name for the saved CSV file

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for SQL insertion
        - Empty strings → 'not_reported'
        - NaN → NULL in SQL
        - Coded values (e.g., '.', '-') → as-is, or remapped via recode_values
    """
    # 0️⃣ Handle missingness: empty → 'not_reported', NaN stays as NaN (→ NULL in SQL)
    df = df.replace(r'^\s*$', 'not_reported', regex=True).infer_objects(copy=False)

    # Optionally recode dataset-specific coded values
    if recode_values:
        df = df.replace(recode_values).infer_objects(copy=False)

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
# 📥 Phase 4: SQL prep - dtypes
# ======================================================


def dtype_audit(df: pd.DataFrame, ignore_values: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Inspect inferred vs coercible dtypes to guide SQL type mapping.
    Suggests dtype based on actual data values, ignoring missingness markers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to audit
    ignore_values : list, optional
        Values to ignore when suggesting dtype. Default: ['not_reported']

    Returns a DataFrame with:
    - column
    - inferred_dtype (pandas' current dtype)
    - suggested_dtype (what it should be, ignoring markers)
    - numeric_parseable_%
    """
    if ignore_values is None:
        ignore_values = ['not_reported']

    records = []

    for col in df.columns:
        series = df[col]
        inferred = str(series.dtype)

        # Filter out ignored values for analysis
        filtered = series[~series.isin(ignore_values) & series.notna()]

        # Try numeric coercion on filtered data
        coerced_numeric = pd.to_numeric(filtered, errors="coerce")
        numeric_ratio = coerced_numeric.notna().mean() if len(filtered) > 0 else 0

        # Keywords that indicate a column should be FLOAT (measurements/metrics)
        float_keywords = ['_mwh', '_mw', '_mmbtu', '_gallons', '_rate', '_volume', '_capacity', '_intensity']

        # Suggest dtype based on filtered values
        if len(filtered) == 0:
            suggested = "TEXT"  # all missing/ignored
        elif numeric_ratio >= 0.99:
            # Force FLOAT for measurement columns, even if data is whole numbers
            col_lower = col.lower()
            if any(kw in col_lower for kw in float_keywords):
                suggested = "FLOAT"
            elif (coerced_numeric.dropna() % 1 == 0).all():
                suggested = "INTEGER"
            else:
                suggested = "FLOAT"
        else:
            suggested = "TEXT"

        records.append({
            "column": col,
            "inferred_dtype": inferred,
            "suggested_dtype": suggested,
            "numeric_parseable_%": round(numeric_ratio * 100, 2),
        })

    audit_df = pd.DataFrame(records)

    # Set display to show all rows when returned in Jupyter
    pd.set_option('display.max_rows', None)

    return audit_df


# ======================================================
# 📥 Phase 5: Apply suggested dtypes
# ======================================================


def apply_sql_dtypes(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    marker_to_null: str = 'not_reported'
) -> pd.DataFrame:
    """
    Apply suggested dtypes from dtype_audit to the DataFrame.
    Converts 'not_reported' markers to proper nulls for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to convert
    audit_df : pd.DataFrame
        Output from dtype_audit() with 'column' and 'suggested_dtype' columns
    marker_to_null : str, default 'not_reported'
        String marker to convert to null before type conversion

    Returns
    -------
    pd.DataFrame
        DataFrame with SQL-friendly dtypes:
        - INTEGER → pandas Int64 (nullable)
        - FLOAT → pandas Float64 (nullable)
        - TEXT → string
    """
    df = df.copy()

    # Build mapping from audit
    dtype_map = dict(zip(audit_df['column'], audit_df['suggested_dtype']))

    converted = []
    failed = []

    for col in df.columns:
        if col not in dtype_map:
            continue

        suggested = dtype_map[col]

        try:
            if suggested == 'INTEGER':
                # Replace marker with NA, then convert to nullable integer
                df[col] = df[col].replace(marker_to_null, pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                converted.append((col, 'Int64'))

            elif suggested == 'FLOAT':
                # Replace marker with NA, then convert to nullable float
                df[col] = df[col].replace(marker_to_null, pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')
                converted.append((col, 'Float64'))

            elif suggested == 'TEXT':
                # Keep as string, preserve marker
                df[col] = df[col].astype('string')
                converted.append((col, 'string'))

        except Exception as e:
            failed.append((col, str(e)))

    print(f"✅ Converted {len(converted)} columns")
    if failed:
        print(f"⚠️  Failed to convert {len(failed)} columns:")
        for col, err in failed:
            print(f"   • {col}: {err}")

    return df


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
