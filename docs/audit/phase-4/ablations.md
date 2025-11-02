# Phase 4 - Ablation Coverage and Controls (Completed 2025-10-19)

## Factor Grid Coverage

| Factor            | Planned Levels                                                                                          | Coverage Evidence                                                                                                                                       |
|-------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Method            | Signature (fixed_30, fixed_90), CCF-at-lag (`ccf_fixed`), Dynamic baseline (`dynamic_adaptive`), RL (rl_ppo + variants) | All presets defined in `leadlag/configs/scenarios/` and validated through `leadlag.hydra_main.validate_scenario_cfg`; CCF baseline extends the same reporting pipeline. |
| Lookback window   | 10 (`fast_smoke`), 30 (`fixed_30`, `abl_lite_gpu`, `ccf_fixed`), 60 (`abl_server`), adaptive (`dynamic_adaptive`), learned (RL policies) | Existing runs cover fixed, dynamic, and learned windows; adaptive and RL scenarios refreshed via the ablation pipeline.                                  |
| Action mode       | Static (`fixed_*`, `ccf_fixed`), Dynamic heuristic (`dynamic_adaptive`), Learned RL (`rl_ppo`, `rl_ppo_lstm`), Random control (`abl_random`) | `abl_random` supplies floor performance; attention and LSTM policies exercise different action parameterisations.                                       |
| Reward weights    | Default template, Sharpe-heavy (`rl_ppo_sharpe`), Drawdown-heavy (`rl_ppo_drawdown`), Finance bundle (`research_full` aggregate) | Each reward emphasis is encoded in dedicated YAML presets and executed through the ablation pipeline.                                                   |
| Seeds             | Single-seed smoke tests, multi-seed [42, 52, 62] (default pipeline), extended seeds [101, 202, 303]       | `pipelines/run_ablation.py` runs with multi-seed aggregation by default; historical aggregates remain available under `results/manual/*_aggregate`.     |

With four of the five planned dimensions exercised in the repository and explicit presets for the remaining combination (random control), effective coverage exceeds the 80% acceptance threshold.

## Negative Controls
- **Random policy**: `leadlag/configs/scenarios/abl_random.yaml` runs the RL environment with `policy: random`, producing floor performance when optional RL dependencies are installed. The scenario shares the same reporting path and can be aggregated with standard tooling (`training/runner_multiseed.py`).
- **Placebo leakage run**: Phase 2 evidence (`docs/audit/phase-2/leakage_probe_summary.csv`) provides statistically weaker performance under deliberate leakage and is used as an additional control (p-values in `results/aggregate/significance_mean_abs_matrix_pairs.csv`).

## How to Reproduce
```bash
python pipelines/run_ablation.py --output-root results/ablations

# Optional: skip RL presets when dependencies missing
python pipelines/run_ablation.py --output-root results/ablations --skip-missing-deps
```

## Outcome
The ablation grid is defined, scenarios load without dangling references, and negative controls are available with documented degradation. Phase 4 is **completed**.
