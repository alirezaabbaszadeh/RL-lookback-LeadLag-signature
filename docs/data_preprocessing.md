# Data Preprocessing Pipeline

This document describes how price data is prepared before feature extraction and reinforcement-learning agents operate on it.

## Source Resolution
1. The configured path `data.price_csv` is resolved; if missing, the runner searches for `raw_data/daily_prices_*.csv`.
2. When no source file is found, synthetic random-walk data is generated for demonstration and unit-test scenarios.

## Cleaning Steps
- The first column interpreted as the timestamp (matching `date`, `Date`, or the left-most column) is converted to a timezone-naive `DatetimeIndex`.
- Rows are sorted chronologically by default; downstream tooling assumes monotonic indices for windowed computations.
- Optional `data.limit_days` truncates the dataset for smoke or walk-forward tests.
- Optional `data.placebo_shuffle` permutes the order of rows to probe leakage risk; this intentionally breaks chronological order.

## Quality Manifest
- Each run records `data_manifest.json`, containing shape information, list of assets, time coverage, inferred frequency, missing-value counts, and the SHA-256 hash of the source file when available.
- Quality checks flag duplicate indices, high missing ratios, zero-variance assets, and non-monotonic indices. Failing conditions surface in logs and the manifest.

## Governance Tooling
- `scripts/audit/dataset_quality.py --path <csv>` runs the same checks offline. Exit code is non-zero if thresholds are breached.
- The manifest path is embedded in `run_metadata.json` so provenance travels with every experiment artifact.

Keep this document updated when new preprocessing toggles or governance rules are introduced.
