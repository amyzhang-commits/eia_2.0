#!/usr/bin/env python3
"""
dataset_merger.py

Two-phase ingestion module for Excel/CSV datasets.

Phase 1: Preview schema differences across files (NO merging)
Phase 2: Merge files with auto-detected headers and optional autosave.

Designed for reproducibility, portability, and large public datasets (e.g. EIA).
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm
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

    if file_path.suffix.lower() == ".xlsx":
        preview = pd.read_excel(file_path, header=None, nrows=nrows)
    elif file_path.suffix.lower() == ".csv":
        preview = pd.read_csv(file_path, header=None, nrows=nrows)
    else:
        raise ValueError(f"Unsupported file type: {file_path.name}")

    for i, row in preview.iterrows():
        string_ratio = row.apply(lambda x: isinstance(x, str)).mean()
        if string_ratio > string_ratio_threshold:
            return i
    return 0

# ======================================================
# 📄 Unified file reader
# ======================================================

def read_file(
    file_path: Union[str, Path],
    verbose: bool = False
) -> pd.DataFrame:
    """Read a single file with auto-detected header."""
    file_path = Path(file_path)
    header_row = detect_header_row(file_path)

    if verbose:
        print(f"Reading {file_path.name} → detected header row: {header_row}")

    if file_path.suffix.lower() == ".xlsx":
        return pd.read_excel(file_path, header=header_row)
    elif file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, header=header_row)
    else:
        raise ValueError(f"Unsupported file type: {file_path.name}")

# ======================================================
# 🔍 Phase 1: Schema Preview
# ======================================================

def preview_folder_schema(folder_path: Optional[Union[str, Path]] = None) -> None:
    folder = Path(folder_path) if folder_path else FALLBACK_DATASETS_PATH
    files = [
        f for f in folder.glob("*")
        if f.suffix.lower() in [".xlsx", ".csv"] and not f.name.startswith(".")
    ]

    if not files:
        raise ValueError(f"No valid files found in {folder}")

    print(f"\n🔍 Previewing schema in folder: {folder}\n")
    schemas = {}

    for f in files:
        header_row = detect_header_row(f)
        cols = (
            pd.read_excel(f, header=header_row, nrows=0).columns.tolist()
            if f.suffix.lower() == ".xlsx"
            else pd.read_csv(f, header=header_row, nrows=0).columns.tolist()
        )
        schemas[f.name] = cols
        print(f"{f.name}: {len(cols)} columns | detected header row: {header_row}")

    all_columns = set().union(*schemas.values())
    print(f"\n📊 Total unique columns across files: {len(all_columns)}\n")

    for name, cols in schemas.items():
        missing = all_columns - set(cols)
        if missing:
            print(f"⚠️ {name} missing columns: {sorted(missing)}")

    print("\n✅ Schema preview complete.\n")

# ======================================================
# 🔗 Phase 2: Merge + Coverage Summary
# ======================================================

def merge_folder_files(
    folder_path: Optional[Union[str, Path]] = None,
    save: bool = False,
    show_heatmap: bool = True
) -> pd.DataFrame:
    folder = Path(folder_path) if folder_path else FALLBACK_DATASETS_PATH
    files = [
        f for f in folder.glob("*")
        if f.suffix.lower() in [".xlsx", ".csv"] and not f.name.startswith(".")
    ]

    if not files:
        raise ValueError(f"No valid files found in {folder}")

    print(f"\n🔗 Merging files from: {folder}")

    merged_df = pd.concat(
        (read_file(f) for f in tqdm(files, desc="Merging files")),
        ignore_index=True
    )

    print(f"\n📐 Merged DataFrame shape: {merged_df.shape}")
    print(f"✅ {len(files)} files merged\n")

    # Coverage summary
    absent_count = merged_df.isnull().sum()
    absent_pct = (absent_count / len(merged_df)) * 100
    coverage_df = pd.DataFrame({
        "Absent Count": absent_count,
        "Absent Percentage": absent_pct
    }).loc[merged_df.columns]

    with pd.option_context("display.max_rows", None):
        print("📊 Coverage Summary:\n")
        print(coverage_df.to_string())
        print()

    # Heatmap
    if show_heatmap:
        absent_mask = merged_df.isnull()
        cmap = ListedColormap(["#00008B", "#D3D3D3"])  # present, absent

        plt.figure(figsize=(16, 8))
        sns.heatmap(absent_mask, cmap=cmap, cbar=False, yticklabels=False)
        plt.title("Coverage Heatmap")
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        save_path = DATASETS_OUTPUT / f"coverage_heatmap_{folder.name}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Coverage heatmap saved to: {save_path}\n")
        plt.show()

    if save:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATASETS_OUTPUT / f"merged_{folder.name}_{timestamp}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"💾 Merged CSV saved to: {output_file}\n")

    # Top 10 columns with most absence
    top_missing = (
        coverage_df
        .sort_values("Absent Count", ascending=False)
        .head(10)
    )

    print("🔟 Top 10 Columns with Most Missing Values:\n")
    print(top_missing.to_string())
    print()

    return merged_df


# ======================================================
# 🏁 Script execution
# ======================================================

if __name__ == "__main__":
    preview_folder_schema()
    merge_folder_files()

