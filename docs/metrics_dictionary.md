# Metrics Dictionary

The experiment stack emits metrics at three levels: per-step time series, per-run summaries, and multi-run aggregates. This page documents the contract expected by downstream tools (dashboards, audits, reports).

## Core Time-Series Metrics
- `mean_abs_matrix`: mean absolute value of off-diagonal elements in each lead-lag matrix.
- `max_abs_matrix`: maximum absolute value of off-diagonal elements.
- `row_sum_range`: difference between the maximum and minimum row sums across assets.
- `row_sum_std`: standard deviation of row sums across assets.
- `stability_matrix_corr`: correlation between flattened off-diagonal values at times _t_ and _t-1_.
- `stability_rowsum_corr`: correlation between row sums at times _t_ and _t-1_.
- `n_assets`: number of assets present at each time step.
- `signal`: optional dynamic baseline signal strength (logged when dynamic runners execute).
- `lookback`: active lookback window recorded by dynamic and RL runners.

## Per-Run Summaries
Summaries written to `summary.csv` include the following columns per metric:
- `mean`, `median`, `std`, `max`, `min`

The values come directly from `evaluation.metrics.summarize_metrics`, so adding new statistics requires updating that helper. `metrics_timeseries.csv` retains the raw timeline and may include runner-specific additions (such as `signal`, `lookback`, or `decisions`) when those fields are recorded by the environment.

## Multi-Run Aggregates
`runner_multiseed.py` aggregates per-run summaries into:
- `stats.csv`: bootstrap-enhanced aggregates with columns `<stat>_mean`, `<stat>_std`, `<stat>_ci95`, `<stat>_boot_low`, `<stat>_boot_high`.
- `significance.csv`: human-friendly subset with confidence bounds.
- `welch.csv`: pairwise Welch _t_-test results for each metric/column combination when summaries from multiple scenarios are grouped together.

Cross-scenario comparisons produced by `evaluation.aggregate.aggregate` emit:
- `results/aggregate/comparison_summary.csv`: wide-form view of the mean statistics per scenario.
- `results/aggregate/comparison_table.tex` (or `.csv` fallback when LaTeX is unavailable): tabular report for manuscripts.
- `results/aggregate/significance_<metric>.csv` and `significance_<metric>_pairs.csv`: Welch p-values with Benjamini–Hochberg adjusted `q_value` plus Cohen's _d_.

## Observability Hooks
- Structured logs embed contextual fields (`module`, `run_name`, `seed`, `scenario`, etc.) for ingestion by log processors.
- The lightweight console dashboard (`observability/dashboard.py`) inspects `summary.csv` files and renders the columns listed above. Add new metrics here to have them surfaced automatically.

## Notes
- Confidence intervals default to bootstrap estimates with 2,000 resamples; change the constant in `runner_multiseed._bootstrap_ci` to adjust.
- Any additional metric appended to `summary.csv` or aggregates must be numeric to participate in statistical comparisons; non-numeric data should remain in metadata JSON.
