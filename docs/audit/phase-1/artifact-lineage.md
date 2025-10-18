# Phase 1 — Artifact Lineage (E2E Flow)

Source data → Per‑run outputs → Multi‑seed aggregates → Reporting

1) Per‑run (training/run_scenario.py)
- Inputs: configs/scenarios/*.yaml, raw_data/daily_price.csv (or synthetic fallback)
- Outputs per run directory (results/<run_name>_<ts>/):
  - config_merged.yaml
  - run_metadata.json (includes data_price_hash, git commit/branch if available, env info)
  - metrics_timeseries.csv (date‑indexed metrics)
  - summary.csv (per‑metric summary rows)
  - optional: fig_signal_strength.png, fig_stability.png
  - profiles/* and run.log

2) Multi‑seed aggregation (training/runner_multiseed.py)
- Inputs: N run directories’ summary.csv for a scenario
- Outputs (results/<scenario>_aggregate/):
  - stats.csv (aggregated means/std/ci + bootstrap intervals)
  - significance.csv (bootstrap CI summary per metric)
  - welch.csv (pairwise tests, only if multiple scenarios aggregated together)
  - runs.json (seed → run_dir mapping), aggregate.log
  - MLflow logging (optional)

3) Cross‑scenario comparison
- reporting/compare_scenarios.py
  - Input: */*_aggregate/stats.csv
  - Output: evaluation/aggregate_comparison.csv and plots/* (if Matplotlib available)
- evaluation/aggregate.py (legacy)
  - Input: */run/metrics_timeseries.csv
  - Output: results/aggregate/* (comparison + pairwise significance per selected columns)

4) Reporting
- reporting/generate_report.py
  - Input: */*_aggregate/{stats,significance,welch}.csv, runs.json
  - Output: reports/final_report.md, reports/final_report.pdf, reports/appendix.md

Notes
- welch.csv only appears when multiple scenarios are aggregated in one pass; otherwise, cross‑scenario Welch is produced by evaluation/aggregate.py or can be added to reporting.
- All downstream tools assume stats.csv contains ‘metric’ rows and ‘*_mean’ columns. Ensure future additions conform to this contract.

