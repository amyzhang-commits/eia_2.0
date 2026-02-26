# EIA 2.0: U.S. Thermoelectric Cooling Water Analysis (2014–2024)

A ground-up rebuild of an [earlier project](https://amyzhang-commits.github.io/project-powerplants.html)
analyzing water usage by U.S. electricity power plants.

## Research Topic

Most U.S. electricity generation is combustion-based and requires large volumes
of water for cooling. Thermoelectric cooling data provides a clearer picture of
electricity infrastructure's environmental footprint — and including water source
information makes that impact concrete rather than abstract.

**Data source:** [EIA Thermoelectric Cooling Data](https://www.eia.gov/electricity/data/water/)
— annual Excel files (2014–2024), merged into a single dataset of ~900K rows, 70 columns.

## EIA v1 → v2

[EIA v1](https://amyzhang-commits.github.io/project-powerplants.html) focused on
auditing suspicious water metrics, concluding with a custom suspicion score to
surface high-risk categories (complex configurations, solar thermal, nuclear) and
proposing ML-based anomaly detection for water-use reporting.

**EIA v2 reorients the approach:**

1. **A different analytical sequence** so that:
   - Data missingness as a feature of schema organization — not a data quality
     red flag — is surfaced earlier and more efficiently
   - Time to analysis is dramatically reduced
   - Focus shifts to the first-order problem of understanding and predicting
     water usage, rather than the second-order problem of anomaly detection
     (which requires the former to be robust anyway)
2. **Refactored notebook scripts** for greater legibility and reproducibility

**Result:** Phase 1 (data wrangling, profiling, cleaning) was completed in two
weeks — down from two months in v1, with the first week dedicated to building
reusable helper modules.

## Status

**Phase 1 — Complete:** Data wrangling, profiling, and cleaning. Output:
`analysis_ready_df.csv`, ready for Phase 2.

**Phase 2 — In Progress:** Exploratory data analysis and visualization.

## Project Structure

*Forthcoming.*

## Setup & Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/amyzhang-commits/eia_2.0.git
cd eia_2.0

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies and helper modules
pip install -e .
```

**Data:** Download the annual thermoelectric cooling Excel files (2014–2024) from
[EIA](https://www.eia.gov/electricity/data/water/) and place them in
`datasets_thermoelectric/`.

**Run notebooks in order:**
1. `01_data_ingestion.ipynb` — merges raw files, exports to `datasets_merged/`
2. `02_profile_analysis.ipynb` — confirms grain, melts schema
3. `03_forensic_clean_for_analysis.ipynb` — produces `analysis_ready_df.csv`

## Notebooks

### `01_data_ingestion.ipynb`
Merges annual EIA thermoelectric cooling files into a single ground-truth table
(~900K rows, 70 columns, 2014–2024). Python automations handle header alignment,
column uniqueness checks, memory estimation, missingness representation, and dtype
inspection pre-merge. The resulting dataset is also persisted in SQL as a secondary
environment for schema exploration.

### `02_profile_analysis.ipynb`
Confirms the observation grain of the dataset. The process involves untangling
relationships between columns imported from a different EIA dataset at a different
grain — metric attributes repeated across prefixed column sets. Observations are
confirmed at: `plant_code`-`generator_id`-`boiler_id`-`cooling_id`-`year`-`month`.
The dataset's complexity becomes clear as naive assumptions are challenged:
"Unoperable" equipment relationships can still carry meaningful water and fuel
metrics; in-service dates can post-date observations, as EIA tracks planned
projects. The central accomplishment is confirming the grain and melting the schema
— which revealed that original missingness was systematic, not noise.

### `03_forensic_clean_for_analysis.ipynb`
Produces `analysis_ready_df` from the melted dataset. The dataset enters this
notebook with four distinct water reporting states: both withdrawal and consumption
volumes reported, withdrawal only, consumption only, and neither. Withdrawal-only
and consumption-only rows are validated as structurally sound. Negative water metric
rows are investigated: plant-level patterns suggest operational aberrations rather
than systematic error, and the signals for these anomalies are preserved in other
features, making removal low-risk. All neither-metric rows are removed: investigation
surfaces these as structurally expected absences — dry-cooled or non-Schedule-8
plants rather than missing data.

### `02_decision_appendix.ipynb`
Working research trail for `02_profile_analysis`: exploratory hypotheses, domain
questions, intermediate findings, and AI-assisted reasoning that informed but did
not make it into the main notebook.

### `03_decision_appendix.ipynb`
Working research trail for `03_forensic_clean_for_analysis`: detailed plant-level
investigations, hypothesis testing (including SCD Type 2 and sector/fuel-based
missingness), domain research, and the full analytical process behind each cleaning
decision.

## Helper Modules

Located in `scripts/`.

### `dataset_pipeline.py`
- `datasets_overview(folder_path)` — Inspect schemas and estimate memory usage
  without loading data.
- `convert_excel_to_csv(folder_path, output_folder, size_threshold_gb)` — Convert
  Excel files to CSV for memory-efficient chunked processing.
- `merge_folder_files(folder_path, save, chunk_size, show_preview)` — Memory-safe
  merge with summary of top 10 columns with most missing values.

### `sql_pipeline.py`
- `schema_preview(df)` — Print schema overview, detect missingness representations,
  return per-column missingness breakdown.
- `sql_prep_columns(df, date_cols, replace_missing_values, ...)` — Clean column
  names, handle missingness, optional date creation, save CSV.
- `dtype_audit(df, print_results)` — Audit inferred vs. coercible dtypes with
  missing % context.

### `utils.py`
- `save_df(df, filename, folder)` — Save DataFrame to CSV with existence check.
- `coverage_summary(df)` — Return DataFrame with missing count and percentage per
  column.
- `column_profile(df)` — Generate numerical stats and categorical summary.
- `coverage_heatmap(df, filename, folder)` — Display coverage heatmap, auto-saves
  by default.
