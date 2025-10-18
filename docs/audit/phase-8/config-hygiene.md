# Phase 8 - Scenario Discovery and Config Hygiene (Completed 2025-10-19)

## Discovery
- `hydra_main.get_available_scenarios()` now merges presets from `SCENARIO_PRESETS` and both `configs/scenario/` and `configs/scenarios/` directories. Current output:
  `['abl_lite_gpu', 'abl_random', 'abl_server', 'abl_smoke', 'dynamic_adaptive', 'fast_smoke', 'fixed_30', 'fixed_90', 'rl_ppo']`.
- `hydra_main.validate_scenario_cfg` enforces required keys (`name`, `path`, `runner`) and confirms YAML paths exist before execution.

## Validation
- Smoke command for every preset:
  ```bash
  python hydra_main.py --scenario <name> --output_root results/smoke
  ```
  (optional RL-only dependencies are documented in `requirements.txt`; the Kaggle requirements already list `iisignature` and `dcor`).
- `scripts/audit/validate_artifacts.py` scans `results/` for runs and aggregates, flagging missing summaries, metadata hashes, or truncated bootstrap columns. Existing results (`results/manual`, `results/aggregate`) pass without errors.

## Outcome
No dangling paths or stale presets remain; every scenario is discoverable, validated, and ready for execution in CI, local, or Kaggle environments. Phase 8 is **completed**.
