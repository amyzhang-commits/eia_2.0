#!/usr/bin/env python3
"""
dataset_merger.py

Two-phase ingestion module for Excel/CSV datasets.

Phase 1: Schema Preview + Memory Estimation (NO merging)
         Column schemas, row counts, memory footprint, merge recommendations.
Phase 2: Merge files with coverage analysis and optional heatmap.

Designed for reproducibility, portability, and large public datasets (e.g. EIA).

Functions
---------
preview_folder_schema(folder_path=None)
    Inspect schemas + estimate memory usage without loading data.
convert_excel_to_csv(folder_path=None, output_folder=None, size_threshold_gb=0.5)
    Convert Excel files to CSV for memory-efficient chunked processing.
merge_folder_files(folder_path=None, save=False, show_heatmap=True, chunk_size=None)
    Memory-safe merge with coverage heatmap visualization.
"""

from pathlib import Path
from typing import Optional, Union, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import openpyxl
import psutil
import re
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ======================================================
# 📁 Project-relative paths
# ======================================================

FALLBACK_DATASETS_PATH = Path(__file__).parent.parent / "datasets_thermoelectric"
DATASETS_OUTPUT = Path(__file__).parent.parent / "datasets_merged"
DATASETS_OUTPUT.mkdir(exist_ok=True)

# ======================================================
# 🧠 Header detection logic
# ======================================================

def detect_header_row(
    file_path: Union[str, Path],
    nrows: int = 10,
    string_ratio_threshold: float = 0.5
) -> int:
    """Detect the most likely header row in a CSV or Excel file."""
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == ".xlsx":
            preview = pd.read_excel(file_path, header=None, nrows=nrows)
        elif file_path.suffix.lower() == ".csv":
            preview = pd.read_csv(file_path, header=None, nrows=nrows, encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported file type: {file_path.name}")

        for i, row in preview.iterrows():
            string_ratio = row.apply(lambda x: isinstance(x, str)).mean()
            if string_ratio > string_ratio_threshold:
                return i
        return 0
    except Exception as e:
        print(f"⚠️  Error detecting header in {file_path.name}: {e}")
        return 0

# ======================================================
# 📄 File row counters
# ======================================================

def get_total_rows_excel(file_path: Union[str, Path], header_row: int = 0) -> int:
    """Get exact data row count from Excel file (excluding header)."""
    file_path = Path(file_path)
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True)
        sheet = wb.active
        total_rows = sheet.max_row - header_row
        wb.close()
        return total_rows
    except Exception as e:
        print(f"⚠️  Error counting rows in {file_path.name}: {e}")
        return 0

def get_total_rows_csv(file_path: Union[str, Path], header_row: int = 0) -> int:
    """Get exact data row count from CSV file (excluding header)."""
    file_path = Path(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
            total_lines = sum(1 for _ in fin)
        return total_lines - 1  # Subtract header row
    except Exception as e:
        print(f"⚠️  Error counting rows in {file_path.name}: {e}")
        return 0

# ======================================================
# 📄 Unified file reader
# ======================================================

def read_file(file_path: Union[str, Path], chunk_size: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    """Read a single file with auto-detected header."""
    file_path = Path(file_path)
    header_row = detect_header_row(file_path)
    
    if verbose:
        print(f"Reading {file_path.name} → detected header row: {header_row}")

    try:
        if file_path.suffix.lower() == ".xlsx":
            # Excel doesn't support chunksize parameter
            return pd.read_excel(file_path, header=header_row)
        elif file_path.suffix.lower() == ".csv":
            if chunk_size:
                # Read in chunks and concatenate
                chunks = []
                for chunk in pd.read_csv(file_path, header=header_row, 
                                        encoding="utf-8", errors="replace", 
                                        chunksize=chunk_size):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_csv(file_path, header=header_row, encoding="utf-8", errors="replace")
    except Exception as e:
        if verbose:
            print(f"⚠️  Error reading {file_path.name}: {e}")
        return pd.DataFrame()

# ======================================================
# 🔍 Phase 1: Schema Preview + Memory Estimation
# ======================================================

def preview_folder_schema(folder_path: Optional[Union[str, Path]] = None) -> None:
    """
    Inspect all files without merging. Provides schema analysis and memory estimation.
    
    Outputs:
    - Column counts and header detection per file
    - Schema consistency warnings
    - Memory footprint estimation
    - Merge safety recommendations

    Parameters
    ----------
    folder_path : str or pathlib.Path, optional
        Folder containing CSV/Excel files. If None, uses FALLBACK_DATASETS_PATH.
    """
    folder = Path(folder_path) if folder_path else FALLBACK_DATASETS_PATH
    files = [f for f in folder.glob("*") 
             if f.suffix.lower() in [".xlsx", ".csv"] and not f.name.startswith(".")]

    if not files:
        raise ValueError(f"No valid files found in {folder}")

    print(f"\n🔍 Previewing schema in folder: {folder}\n")
    schemas = {}
    row_counts = {}
    mem_estimates_gb = {}
    large_excel_files = []

    for f in files:
        header_row = detect_header_row(f)
        
        try:
            # Get columns (zero rows = headers only)
            if f.suffix.lower() == ".xlsx":
                cols = pd.read_excel(f, header=header_row, nrows=0).columns.tolist()
                total_rows = get_total_rows_excel(f, header_row)
            else:
                cols = pd.read_csv(f, header=header_row, nrows=0, encoding="utf-8", errors="replace").columns.tolist()
                total_rows = get_total_rows_csv(f, header_row)

            schemas[f.name] = cols
            row_counts[f.name] = total_rows
            
            # Memory estimate: sample 100 rows → extrapolate
            if f.suffix.lower() == ".xlsx":
                sample = pd.read_excel(f, header=header_row, nrows=100)
            else:
                sample = pd.read_csv(f, header=header_row, nrows=100, encoding="utf-8", errors="replace")
            
            sample_bytes = sample.memory_usage(deep=True).sum()
            est_bytes = sample_bytes * (total_rows / 100) if total_rows > 0 else 0
            est_gb = est_bytes / 1e9
            mem_estimates_gb[f.name] = est_gb
            
            # Track large Excel files
            if f.suffix.lower() == ".xlsx" and est_gb > 0.5:
                large_excel_files.append((f.name, est_gb))
            
            print(f"  {f.name}: {len(cols)} cols | header: {header_row} | "
                  f"{total_rows:,} rows | ~{est_gb:.2f} GB")
        except Exception as e:
            print(f"⚠️  Error processing {f.name}: {e}")
            continue

    # Schema consistency
    all_columns = set().union(*schemas.values())
    total_columns = len(all_columns)
    print(f"\n📊 Total unique columns across files: {len(all_columns)}")    
    
    for name, cols in schemas.items():
        missing = all_columns - set(cols)
        if missing:
            print(f"  ⚠️  {name} missing: {sorted(list(missing))}")

    # Memory recommendation
    total_est_gb = sum(mem_estimates_gb.values())
    system_ram_gb = psutil.virtual_memory().total / 1e9
    
    print(f"\n💾 Estimated total memory: {total_est_gb:.2f} GB")
    print(f"🖥️  Available system RAM:   {system_ram_gb:.2f} GB")
    
    # Excel-specific warnings
    if large_excel_files:
        print(f"\n⚠️  EXCEL FILE WARNING:")
        print(f"    The following Excel files are large (>0.5 GB):")
        for fname, size in large_excel_files:
            print(f"      • {fname} (~{size:.2f} GB)")
        print(f"    Excel files cannot use chunked reading.")
        print(f"    RECOMMENDATION: Convert to CSV first using:")
        print(f"      convert_excel_to_csv()")
    
    if total_est_gb > 0.7 * system_ram_gb:
        print("\n⚠️   MEMORY RECOMMENDATION: Use chunked merge")
        print(f"     Set chunk_size={max(1000, int(10000 * (system_ram_gb / total_est_gb)))}")
        if large_excel_files:
            print(f"     (After converting Excel files to CSV)")
    else:
        print("\n✅   RECOMMENDATION: Direct merge safe")
        
    # Persistence recommendation (conditional)
    if total_columns > 30:
        print("\n🗄️  STORAGE RECOMMENDATION:")
        print(f"    Detected wide schema ({total_columns} columns).")
        print("    Use 'sql_prep()' to persist.")


    print("\n✅ Phase 1 complete. Review before merging.\n")

# ======================================================
# 🔄 Excel to CSV Converter
# ======================================================

def convert_excel_to_csv(
    folder_path: Optional[Union[str, Path]] = None,
    output_folder: Optional[Union[str, Path]] = None,
    size_threshold_gb: float = 0.5
) -> None:
    """
    Convert Excel files to CSV format for memory-efficient chunked processing.
    
    Useful when Excel files are too large for memory and need chunked reading.
    Converts files above the size threshold and saves them to output folder.
    
    Parameters
    ----------
    folder_path : str or pathlib.Path, optional
        Folder containing Excel files. If None, uses FALLBACK_DATASETS_PATH.
    output_folder : str or pathlib.Path, optional
        Where to save CSV files. If None, saves to same folder as Excel files.
    size_threshold_gb : float, default 0.5
        Minimum file size (GB) to trigger conversion. Files smaller than this are skipped.
    """
    folder = Path(folder_path) if folder_path else FALLBACK_DATASETS_PATH
    output = Path(output_folder) if output_folder else folder
    output.mkdir(exist_ok=True)
    
    excel_files = [f for f in folder.glob("*.xlsx") if not f.name.startswith(".")]
    
    if not excel_files:
        print(f"No Excel files found in {folder}")
        return
    
    print(f"\n🔄 Converting Excel files to CSV in: {folder}")
    print(f"📁 Output folder: {output}")
    print(f"📏 Size threshold: {size_threshold_gb} GB\n")
    
    converted_count = 0
    skipped_count = 0
    
    for f in excel_files:
        # Estimate file size
        header_row = detect_header_row(f)
        total_rows = get_total_rows_excel(f, header_row)
        
        try:
            sample = pd.read_excel(f, header=header_row, nrows=100)
            sample_bytes = sample.memory_usage(deep=True).sum()
            est_bytes = sample_bytes * (total_rows / 100) if total_rows > 0 else 0
            est_gb = est_bytes / 1e9
            
            if est_gb < size_threshold_gb:
                print(f"⏭️  Skipping {f.name} (~{est_gb:.2f} GB < {size_threshold_gb} GB threshold)")
                skipped_count += 1
                continue
            
            print(f"🔄 Converting {f.name} (~{est_gb:.2f} GB)...")
            
            # Read and convert
            df = pd.read_excel(f, header=header_row)
            output_file = output / f"{f.stem}.csv"
            df.to_csv(output_file, index=False)
            
            print(f"   ✅ Saved to: {output_file.name}")
            converted_count += 1
            
        except Exception as e:
            print(f"   ⚠️  Error converting {f.name}: {e}")
            continue
    
    print(f"\n✅ Conversion complete:")
    print(f"   • Converted: {converted_count} files")
    print(f"   • Skipped: {skipped_count} files")
    print(f"   • Output: {output}\n")

# ======================================================
# 🔗 Phase 2: Merge
# ======================================================

def merge_folder_files(
    folder_path: Optional[Union[str, Path]] = None,
    save: bool = False,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Merge all CSV/Excel files in a folder with optional chunked reading.
    
    Parameters
    ----------
    folder_path : str or pathlib.Path, optional
        Folder containing files to merge. If None, uses FALLBACK_DATASETS_PATH.
    save : bool, default False
        Whether to save merged DataFrame to CSV.
    chunk_size : int, optional
        Number of rows to read at a time for CSV files. Use for memory optimization.
        If None, reads entire files at once. Recommended: 10000-50000 for large files.
        Note: Excel files do not support chunking.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing all file data.
    """
    folder = Path(folder_path) if folder_path else FALLBACK_DATASETS_PATH
    files = [
        f for f in folder.glob("*")
        if f.suffix.lower() in [".xlsx", ".csv"] and not f.name.startswith(".")
    ]

    if not files:
        raise ValueError(f"No valid files found in {folder}")

    print(f"\n🔗 Merging files from: {folder}")
    if chunk_size:
        print(f"📦 Using chunked reading: {chunk_size:,} rows per chunk")

    # Merge all files with tqdm progress bar
    merged_df = pd.concat(
        (read_file(f, chunk_size=chunk_size, verbose=False) for f in tqdm(files, desc="Merging files")),
        ignore_index=True
    )

    print(f"📐 Merged DataFrame shape: {merged_df.shape}")
    print(f"✅ {len(files)} files merged\n")

    # --------------------------
    # Top 10 columns with most missing values (count + percentage)
    # --------------------------
    
    missing_counts = merged_df.isnull().sum()
    missing_pct = (missing_counts / len(merged_df)) * 100

    top_missing = pd.DataFrame({
        "Absent Count": missing_counts,
        "Absent Percentage": missing_pct
    }).sort_values("Absent Count", ascending=False).head(10)

    print("🔟 Top 10 Columns with Most Missing Values:\n")
    with pd.option_context("display.max_rows", None):
        print(top_missing.to_string())
    print()

    # --------------------------
    # Full schema overview (columns + dtypes)
    # --------------------------
    print("🧬 Full Column Schema (name → dtype):\n")
    schema_df = merged_df.dtypes.rename("dtype").to_frame()
    with pd.option_context("display.max_rows", None):
        print(schema_df.to_string())
        print()

    # --------------------------
    # Detect missingness representations
    # --------------------------
    def detect_missing_representations(df):
        placeholders = ['NA', 'N/A', 'na', 'N/a', '-', '.']
        missing_repr = {}
        missing_repr['NaN/None'] = df.isnull().any().any()
        missing_repr['Empty string / whitespace'] = (df.applymap(
            lambda x: isinstance(x, str) and x.strip() == ''
        )).any().any()
        for val in placeholders:
            missing_repr[val] = (df == val).any().any()
        return missing_repr

    missing_types = detect_missing_representations(merged_df)
    print("🔎 Detected missingness representations in dataset:")
    for rep, present in missing_types.items():
        if present:
            print(f" • {rep}")
    print()

    # Optionally save merged CSV
    if save:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATASETS_OUTPUT / f"merged_{folder.name}_{timestamp}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"💾 Merged CSV saved to: {output_file}\n")

    return merged_df
    
# ======================================================
# 📥 Phase 3: SQL prep
# ======================================================

def sql_prep(
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

    # ------------------------------
    # 0️⃣ Handle missingness
    # ------------------------------
    # Convert blanks/whitespace-only strings to NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Replace dataset-specific missing values with missing_marker
    if replace_missing_values:
        df = df.replace(replace_missing_values, missing_marker)

    # ------------------------------
    # 1️⃣ Clean column names
    # ------------------------------
    def clean_col(col: str) -> str:
        col = col.strip()
        col = re.sub(r'\?', '', col)
        col = col.lower()
        col = col.replace(' ', '_')
        col = re.sub(r'[^a-z0-9_]', '', col)

        # Move leading numbers to the end
        match = re.match(r'^(\d+)(.*)', col)
        if match:
            col = f"{match.group(2)}_{match.group(1)}"
        return col

    df = df.rename(columns=lambda c: clean_col(c))

    # ------------------------------
    # 2️⃣ Optional date column creation
    # ------------------------------
    if date_cols:
        for new_col, (year_col, month_col) in date_cols.items():
            if year_col in df.columns and month_col in df.columns:
                df[new_col] = pd.to_datetime(
                    df[year_col].astype(str) + '-' +
                    df[month_col].astype(str) + '-01',
                    errors='coerce'  # invalid combos become NaT
                )

    # ------------------------------
    # 3️⃣ Optionally save CSV
    # ------------------------------
    if save:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATASETS_OUTPUT / f"{output_name}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 SQL-prepared CSV saved to: {output_file}\n")

    return df

# ======================================================
# 🏁 Script execution
# ======================================================

if __name__ == "__main__":
    preview_folder_schema()
    
    # If Phase 1 warns about large Excel files, convert them first:
    # convert_excel_to_csv()
    
    # Then merge (with chunking if recommended):
    # merge_folder_files(chunk_size=10000)
    merge_folder_files()
