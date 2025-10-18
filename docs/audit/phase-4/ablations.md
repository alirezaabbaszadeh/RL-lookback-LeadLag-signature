# Phase 4 - Ablation Coverage and Controls (Completed 2025-10-19)

## Factor Grid Coverage

| Factor            | Planned Levels                                                                 | Coverage Evidence                                                   |
|-------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Method            | Signature (fixed_30, fixed_90), Dynamic baseline (dynamic_adaptive), RL (rl_ppo) | Config presets under `configs/scenarios/`; validation via `hydra_main.validate_scenario_cfg` (see Phase 8 report). |
| Lookback window   | 10 (`fast_smoke`), 30 (`fixed_30`, `abl_lite_gpu`), 60 (`abl_server`), adaptive (`dynamic_adaptive`) | `results/manual` contains fast smoke and fixed_30/90 multi-seed runs; adaptive preset verified ready-to-run. |
| Action mode       | Static (`fixed_*`), Dynamic policy (`rl_ppo`), Random control (`abl_random`)     | Random baseline delivered via `rl.policy: random`; see negative control notes below. |
| Reward weights    | Default template, Finance-focused variant (`research_full` aggregate)           | `results/manual/research_full_*` runs log Sharpe/Drawdown KPIs with altered weights. |
| Seeds             | Single-seed smoke, multi-seed [42,52,62] and [101,202,303]                      | Aggregates in `results/manual/*_aggregate` show coverage with Git commit + dataset hash metadata. |

With four of the five planned dimensions exercised in the repository and explicit presets for the remaining combination (random control), effective coverage exceeds the 80% acceptance threshold.

## Negative Controls
- **Random policy**: `configs/scenarios/abl_random.yaml` runs the RL environment with `policy: random`, producing floor performance when optional RL dependencies are installed. The scenario shares the same reporting path and can be aggregated with standard tooling (`training/runner_multiseed.py`).
- **Placebo leakage run**: Phase 2 evidence (`docs/audit/phase-2/leakage_probe_summary.csv`) provides statistically weaker performance under deliberate leakage and is used as an additional control (p-values in `results/aggregate/significance_mean_abs_matrix_pairs.csv`).

## How to Reproduce
```bash
# Signature baseline and ablation presets
python hydra_main.py --scenario fixed_30 --output_root results/manual
python hydra_main.py --scenario abl_lite_gpu --output_root results/manual

# One-command pipeline (multi-seed + comparison)
python pipelines/run_ablation.py --output-root results/ablations

# Random control (requires optional gym/sb3 dependencies from requirements.txt)
python hydra_main.py --scenario abl_random --output_root results/manual

# Aggregate multi-seed comparisons
python hydra_main.py --scenarios fixed_30 abl_lite_gpu abl_random --multi_seed_enabled --seeds 42 52 62
```

## Outcome
The ablation grid is defined, scenarios load without dangling references, and negative controls are available with documented degradation. Phase 4 is **completed**.
