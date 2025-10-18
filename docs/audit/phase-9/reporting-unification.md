# Phase 9 - Reporting Unification (Completed 2025-10-19)

## Unified Workflow
- `training/runner_multiseed.py` is the canonical entry point for multi-seed aggregation. It writes:
  - `stats.csv` with bootstrap confidence intervals,
  - `significance.csv` with readable intervals,
  - `welch.csv` for pairwise t-tests when scenarios share metrics.
- `reporting/compare_scenarios.py` consumes the aggregated CSVs and generates comparison plots/tables. The CLI expects a directory containing `stats.csv` and outputs standard artefacts under `results/aggregate/`.
- Redundant scripts were retired; documentation now references the single CLI (`README.md`, `docs/deployment/kaggle_setup.md`, `CONTRIBUTING.md`). Existing aggregate outputs (e.g., `results/aggregate/comparison_summary.csv`) were produced with this unified path.

## Reproduce
```bash
python hydra_main.py --scenarios fixed_30 rl_ppo --multi_seed_enabled --seeds 42 52 62 --output_root results/manual
python reporting/compare_scenarios.py --aggregate results/manual/fixed_30_aggregate
```

## Outcome
There is one supported reporting workflow for statistical comparisons, and it is fully documented. Phase 9 is **completed**.
