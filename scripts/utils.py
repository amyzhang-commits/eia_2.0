#!/usr/bin/env python3
"""
utils.py

General utility functions for data analysis workflows.

Functions
---------
save_df(df, filename, folder)
    Save DataFrame to CSV with existence check.
coverage_summary(df)
    Return DataFrame with missing count + percentage per column.
column_profile(df)
    Generate numerical stats + categorical summary (unique count, mode).
coverage_heatmap(df, filename, folder)
    Display coverage heatmap, auto-saves by default.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


# ======================================================
# Default output folders (project-relative)
# ======================================================

FALLBACK_PLOTS_FOLDER = Path(__file__).parent.parent / "visualizations"
FALLBACK_DATA_FOLDER = Path(__file__).parent.parent / "data_exports"


# ======================================================
# Save DataFrame
# ======================================================

def save_df(
    df: pd.DataFrame,
    filename: str,
    folder: Optional[Union[str, Path]] = None
) -> Path:
    """
    Save DataFrame to CSV with existence confirmation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Name of the output CSV file.
    folder : str or Path, optional
        Output folder. If None, uses FALLBACK_DATA_FOLDER.

    Returns
    -------
    Path
        Full path to the saved file.
    """
    folder = Path(folder) if folder else FALLBACK_DATA_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    full_path = folder / filename

    df.to_csv(full_path, index=False)

    # Confirm file was created
    if full_path.exists():
        print(f"✅ Saved: {full_path}")
    else:
        print(f"⚠️  Error: {filename} was not saved.")

    return full_path


# ======================================================
# Coverage summary
# ======================================================

def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with missing count and percentage per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        Summary with 'missing_count' and 'missing_%' columns.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_%": missing_pct.round(2)
    })

    # Set display to show all rows in Jupyter
    pd.set_option('display.max_rows', None)

    return summary_df


# ======================================================
# Column profile (numerical + categorical summaries)
# ======================================================

def column_profile(df: pd.DataFrame) -> tuple:
    """
    Generate summary profiles for all columns:
    - Numerical: descriptive statistics (describe().T) + missing info
    - Categorical: unique count + mode

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to profile.

    Returns
    -------
    tuple (numerical_summary, categorical_summary)
        numerical_summary : pd.DataFrame
            Transposed describe() output with missing counts.
        categorical_summary : pd.DataFrame
            One row per column with unique_count, mode, mode_count, mode_pct.
    """
    # --- Numerical summary ---
    num_cols = df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns
    if len(num_cols) > 0:
        numerical_summary = df[num_cols].describe().T
        numerical_summary['missing_count'] = df[num_cols].isnull().sum()
        numerical_summary['missing_pct'] = (numerical_summary['missing_count'] / len(df) * 100).round(2)
    else:
        numerical_summary = pd.DataFrame()

    # --- Categorical summary (one row per column) ---
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    cat_records = []

    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        total = len(df)
        n_unique = df[col].nunique(dropna=False)
        mode_val = vc.index[0] if len(vc) > 0 else None
        mode_count = vc.iloc[0] if len(vc) > 0 else 0

        cat_records.append({
            'column': col,
            'unique_count': n_unique,
            'mode': mode_val,
            'mode_count': mode_count,
            'mode_pct': round(mode_count / total * 100, 2),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': round(df[col].isnull().sum() / total * 100, 2)
        })

    categorical_summary = pd.DataFrame(cat_records).set_index('column')

    # Set display to show all rows in Jupyter
    pd.set_option('display.max_rows', None)

    return numerical_summary, categorical_summary


# ======================================================
# Coverage heatmap
# ======================================================

def coverage_heatmap(
    df: pd.DataFrame,
    save: bool = True,
    filename: Optional[str] = None,
    folder: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 8),
    title: str = "Coverage Heatmap",
    dpi: int = 300,
    transpose: bool = False
) -> Optional[Path]:
    """
    Display coverage heatmap showing missing values pattern.
    Auto-saves by default.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to visualize.
    save : bool, default True
        Whether to save the heatmap to file.
    filename : str, optional
        Output filename. If None, generates timestamped name.
    folder : str or Path, optional
        Output folder. If None, uses FALLBACK_PLOTS_FOLDER.
    figsize : tuple, default (16, 8)
        Figure size (width, height).
    title : str, default "Coverage Heatmap"
        Plot title.
    dpi : int, default 300
        Resolution for saved image.
    transpose : bool, default False
        If True, columns on y-axis (horizontal labels), rows on x-axis.

    Returns
    -------
    Path or None
        Path to saved file if save=True, else None.
    """
    absent_mask = df.isnull()
    cmap = ListedColormap(["#00008B", "#D3D3D3"])  # present (dark blue), absent (gray)

    if transpose:
        absent_mask = absent_mask.T
        # Auto-scale height: ~0.25 inches per column, minimum 8
        auto_height = max(8, len(df.columns) * 0.25)
        fig, ax = plt.subplots(figsize=(figsize[0], auto_height))
        sns.heatmap(absent_mask, cmap=cmap, cbar=False, xticklabels=False, yticklabels=True, ax=ax)
        ax.set_xlabel("Rows")
        ax.set_ylabel("Columns")
        ax.tick_params(axis='y', labelsize=8)  # Smaller font to fit
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(absent_mask, cmap=cmap, cbar=False, yticklabels=False, ax=ax)
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")

    ax.set_title(title)

    saved_path = None
    if save:
        # Generate filename if not provided
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coverage_heatmap_{timestamp}.png"

        # Build path
        folder = Path(folder) if folder else FALLBACK_PLOTS_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        saved_path = folder / filename

        # Save FIRST (before show clears the figure)
        fig.savefig(saved_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved: {saved_path}")

    plt.show()
    return saved_path


# ======================================================
# Demo / Entry point
# ======================================================

if __name__ == "__main__":
    print("🧪 utils.py - General utility functions")
    print("\nAvailable functions:")
    print("  save_df(df, filename, folder)")
    print("  coverage_summary(df)")
    print("  column_profile(df)  # returns (numerical_df, categorical_df)")
    print("  coverage_heatmap(df, filename, folder)  # auto-saves by default")
    print(f"\nDefault folders:")
    print(f"  Plots: {FALLBACK_PLOTS_FOLDER}")
    print(f"  Data:  {FALLBACK_DATA_FOLDER}")
