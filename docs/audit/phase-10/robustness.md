# Phase 10 - Robustness and Stress Testing (Completed 2025-10-19)

## Coverage
- **Out-of-sample splits**: Walk-forward verification (Phase 2) doubles as an OOS stress test. The truncated vs. full runs demonstrate identical outputs at the pivot date, ensuring rolling windows behave correctly when history is shortened.
- **Missing data and outliers**: `scripts/audit/dataset_quality.py` enforces missing-ratio thresholds, duplicate index detection, and zero-variance asset checks. Manifests (`data_manifest.json`) capture missing counts per run and serve as gatekeepers before deployment.
- **Adversarial noise**: The placebo shuffle scenario (`leakage_placebo_*`) introduces worst-case noise, clearly degrading KPIs. Significance tables in `results/aggregate/significance_*` quantify the drop.
- **Worst-seed analysis**: Multi-seed runs for `fixed_30` and `research_full` (seeds 42/52/62 and 101/202/303) are consolidated under `results/manual/*_aggregate`. Bootstrap intervals in `stats.csv` show variance across seeds and confirm stability within acceptable bounds.

## Tooling
- Robustness checks are executed with the following commands:
  ```bash
  python scripts/audit/check_walk_forward.py --scenario fixed_30
  python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv
  python hydra_main.py --scenarios fixed_30 research_full --multi_seed_enabled --seeds 42 52 62
  ```
  The outputs are stored in `docs/audit/phase-2/` and `results/aggregate/`, providing traceable evidence.

## Outcome
Key stress factors—time splits, data corruption, noise injection, and seed variability—are documented with quantitative results. Phase 10 is **completed**.
