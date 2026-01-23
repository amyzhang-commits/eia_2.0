# EIA 2.0

We're out to salvage and seriously upgrade [an earlier project](https://github.com/amyzhang-commits/eia_water_anomaly_detection).

Starting with modularizing the data ingestion pipeline:
- **`dataset_pipeline.py`** — automates many of the necessary header-blank-space and column alignment checks necessary before dataset merges
- **`sql_pipeline.py`** — prepares the newly merged dataset for import into SQL as a secondary environment for validation checks, and for sometimes faster querying on disk
- **`utils.py`** — general utility functions for coverage analysis and profiling
