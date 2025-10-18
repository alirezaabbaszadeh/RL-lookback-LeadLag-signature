# Phase 1 — Artifact Contracts & Lineage

Scope
- Define schemas and contracts for key artifacts and verify against samples.

Contracts (finalized v1)
- summary.csv
  - columns: metric, mean, median, std, max, min
  - producer: training/run_scenario.py
  - consumer: training/runner_multiseed.py (aggregation), reporting/generate_report.py
- metrics_timeseries.csv
  - index: date; columns: computed metrics (e.g., mean_abs_matrix, row_sum_std, stability_matrix_corr, ...)
  - producer: training/run_scenario.py
  - consumer: evaluation/aggregate.py (pairwise tests), plotting utilities
- stats.csv (aggregate)
  - columns: scenario, metric, count, <stat>_mean, <stat>_std, <stat>_ci95, <stat>_boot_low, <stat>_boot_high
  - expected <stat> in {mean, median, std, max, min}; not all present for every metric
  - producer: training/runner_multiseed.py
  - consumer: reporting/compare_scenarios.py, reporting/generate_report.py
- significance.csv (aggregate)
  - columns: scenario, metric, mean_boot_low/high, median_boot_low/high, std_boot_low/high, ... , note
  - producer: training/runner_multiseed.py
  - status: key truncation for *_boot_low/high fixed (suffix parsing corrected on 2025-10-18)
- welch.csv (aggregate, optional)
  - columns: metric, column, scenario_a, scenario_b, mean_a, mean_b, n_a, n_b, t_stat, df, p_value
  - producer: training/runner_multiseed.py (only when multiple scenarios in one aggregation)

Validation notes
- Sample aggregates under results/manual/*_aggregate match the schema above; older files may still show truncated keys.
- reporting/compare_scenarios.py consumes stats.csv via (metric == <name>, mean_mean).


Next steps
- Validate against `results/manual/*` and note deviations.
- Harmonize compare tooling (evaluation/aggregate.py vs reporting/compare_scenarios.py).
