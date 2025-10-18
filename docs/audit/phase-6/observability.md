# Phase 6 — Observability & Profiling (Plan)

Metrics dictionary (expanded)
- Core metrics: mean_abs_matrix, max_abs_matrix, row_sum_range, row_sum_std,
  stability_matrix_corr, stability_rowsum_corr.
- Aggregate stats: <metric>_[mean|median|std|max|min] as summary columns; per-run summaries in summary.csv.
- RL metrics (expected): reward_episode_mean, reward_step_mean, policy_loss, value_loss,
  entropy, action_change_rate, lookback_mean, lookback_std.

See `docs/metrics_dictionary.md` for full definitions.

Logging policy
- ≥ 10 metrics per run; consistent naming and units.
- MLflow (optional): log metrics and artifacts when available; document TRACKING_URI and enable/disable via env.
- Headless plotting: skip or fallback gracefully; document behavior.

Profiling policy
- cProfile and wall-clock timers around signature pipeline and training loop.
- Artifact retention: keep last N profiles per scenario; rotate older ones.

Acceptance
- Dictionary published; ≥ 10 metrics logged per run.
- profiles/* present per run; retention/rotation documented.

Evidence
- Metrics inventory: `docs/audit/phase-6/metrics_inventory.csv` (≥ 35 logged summary metrics per run in recent samples).
 - MLflow toggle: set `MLFLOW_ENABLED=0/1` to disable/enable logging; supported in training runners.
 - Profiles retention: override with env `PROFILES_MAX_KEEP`, default 10; implemented in `reporting/profiling.py`.
 - Headless plotting: enable `metrics.headless: true` in config to skip figure generation.
