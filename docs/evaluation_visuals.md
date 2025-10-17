Evaluation Visuals and Reports

This document outlines the required visuals and summary tables for evaluating lead-lag signature experiments and RL lookback strategies.

- Signal Strength Over Time
  - Line plots of `mean_abs_matrix`, `max_abs_matrix`, and `row_sum_range`.
  - Purpose: assess magnitude and variability of inferred relationships.

- Stability Over Time
  - Line plots of `stability_matrix_corr` and `stability_rowsum_corr` (t vs t-1).
  - Purpose: gauge temporal consistency of signals.

- Distribution Summaries
  - Histograms or KDE for `mean_abs_matrix` and `row_sum_std`.
  - Purpose: identify skew/outliers across the backtest period.

- Cross-Scenario Comparison
  - Bar/box plots of key metrics aggregated per scenario (per-seed means with CI).
  - Include bootstrap 95% CI and, when multiple scenarios exist, pairwise Welch test p-values.

- Run Metadata Snapshot
  - Tabulate seeds, git commit, environment info per run (from `run_metadata.json`).

Artifacts

- Per-run: `metrics_timeseries.csv`, `summary.csv`, `fig_signal_strength.png`, `fig_stability.png`.
- Multi-seed aggregate: `stats.csv`, `significance.csv` (bootstrap CI), `welch.csv` (pairwise Welch tests).

Notes

- Visual generation is optional in headless environments; plots are skipped when matplotlib is unavailable.
- Statistical significance uses bootstrap CI by default; Welch tests are added when multiple scenarios are present.

