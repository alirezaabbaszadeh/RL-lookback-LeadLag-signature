# Data Preprocessing Pipeline

This document describes how price data is prepared before feature extraction and reinforcement-learning agents operate on it.

## Source Resolution
1. The configured path `data.price_csv` is resolved. If the file is absent, the runner searches for `raw_data/daily_prices_*.csv` and falls back to `raw_data/daily_price.csv` when available.
2. When no CSV is found, a deterministic synthetic price panel (three assets, 300 daily observations) is generated. The random generator is seeded from `run.seed` so smoke tests remain reproducible.

## Cleaning Steps
- Timestamp detection follows the same order as the code in `training/run_support.read_prices`: explicit `date`, explicit `Date`, otherwise the left-most column is treated as the index.
- Rows are sorted chronologically; downstream tooling assumes monotonic indices for windowed computations.
- Optional `data.limit_days` truncates the dataset for smoke or walk-forward tests.
- Optional `data.placebo_shuffle` permutes the order of rows to probe leakage risk; this intentionally breaks chronological order but preserves reproducibility through the configured seed.
- Additional preprocessing helpers (`preprocessing_data/preprocessing.py`) provide:
  - `resample_crypto_data` for OHLC resampling with explicit aggregation semantics.
  - `selected_uni` to filter the asset universe based on monthly inclusion lists and recent data availability.

## Quality Manifest
- Each run records `data_manifest.json`, containing shape information, list of assets, time coverage, inferred frequency, missing-value counts, SHA-256 hash of the source file when available, and embedded quality flags from `governance/dataset.run_quality_checks`.
- Quality checks flag duplicate indices, high missing ratios, zero-variance assets, and non-monotonic indices. Failing conditions surface in logs and the manifest for governance review.

## Governance Tooling
- `scripts/audit/dataset_quality.py --path <csv>` runs the same checks offline. Exit code is non-zero if thresholds are breached.
- The manifest path is embedded in `run_metadata.json` so provenance travels with every experiment artefact.

Keep this document updated when new preprocessing toggles or governance rules are introduced.
