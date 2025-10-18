# Metrics Dictionary

This dictionary enumerates the core and aggregate metrics produced by the
pipeline and expected RL metrics for observability.

Core metrics (per time step)
- `mean_abs_matrix`: mean absolute value of off-diagonal entries of the lead-lag matrix.
- `max_abs_matrix`: maximum absolute value of off-diagonal entries.
- `row_sum_range`: max(row sums) − min(row sums) across assets.
- `row_sum_std`: standard deviation of row sums across assets.
- `stability_matrix_corr`: correlation of flattened off-diagonal matrix values between t and t−1.
- `stability_rowsum_corr`: correlation of row sums between t and t−1.
- `n_assets`: number of assets present in the matrix at each time step.

Aggregate summary columns (per metric in summary/aggregate tables)
- `<metric>_mean`, `<metric>_median`, `<metric>_std`, `<metric>_max`, `<metric>_min`
  appear in `summary.csv` (per-run) and as columns in `stats.csv` (aggregated), often with
  additional columns for confidence ranges: `<stat>_ci95`, `<stat>_boot_low`, `<stat>_boot_high`.

RL metrics (expected)
- `reward_episode_mean`, `reward_step_mean`, `policy_loss`, `value_loss`, `entropy`.
- Action dynamics: `action_change_rate`, decision state such as `lookback_mean`, `lookback_std`.
- Logging of these metrics is performed when RL runner is active and MLflow is available.

Notes
- All downstream tools assume aggregated stats follow the `stats.csv` contract
  (rows indexed by `metric`, columns for `<stat>_mean`, etc.).
- Confidence intervals use bootstrap by default; pairwise tests across scenarios
  may be reported in `welch.csv` where applicable.

