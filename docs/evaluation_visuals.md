Evaluation Visuals and Reports

This document outlines the visuals and summary tables produced by the current evaluation stack. Use it as a checklist when refreshing artefacts under `results/` or compiling reports.

- Signal Strength Over Time
  - Line plots of `mean_abs_matrix`, `max_abs_matrix`, and `row_sum_range`.
  - Purpose: assess magnitude and variability of inferred relationships.

- Stability Over Time
  - Line plots of `stability_matrix_corr` and `stability_rowsum_corr` (t vs t-1).
  - Purpose: gauge temporal consistency of signals.

- Distribution Summaries
  - Optional histograms or KDE for `mean_abs_matrix` and `row_sum_std` when visual tooling is available.
  - Purpose: identify skew/outliers across the backtest period. (Not generated automatically by the core scripts.)

- Cross-Scenario Comparison
  - The `evaluation.aggregate.aggregate` helper produces CSV summaries instead of figures: `comparison_summary.csv`, `comparison_table.tex` (or `.csv` fallback), and `significance_<metric>*.csv` with Welch p-values, Benjamini–Hochberg `q_value`, and Cohen's _d_.
  - Plotting remains optional; pairwise tables power most of the comparative analysis.

- Run Metadata Snapshot
  - Tabulate seeds, git commit, environment info per run (from `run_metadata.json`).

Artifacts

- Per-run: `metrics_timeseries.csv`, `summary.csv`, optional `fig_signal_strength.png`, `fig_stability.png` (skipped when Matplotlib is unavailable).
- Multi-seed aggregate (when `runner_multiseed` is invoked across several seeds): `stats.csv`, `significance.csv` (bootstrap CI), and `welch.csv` when multiple scenarios feed the aggregator.
- Cross-scenario aggregate: `results/aggregate/comparison_summary.csv`, `comparison_table.tex`/`.csv`, `significance_<metric>.csv`, and `significance_<metric>_pairs.csv`.

Notes

- Visual generation is optional in headless environments; plots are skipped when Matplotlib is unavailable.
- Statistical significance in `runner_multiseed` comes from bootstrap CIs; cross-scenario Welch tests are handled by `evaluation.aggregate`.

