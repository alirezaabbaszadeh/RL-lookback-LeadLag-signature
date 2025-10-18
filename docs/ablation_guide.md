# Ablation Scenarios and Usage

This repository provides three ablation-oriented scenarios to support quick
pipeline checks, realistic runs on modest GPUs, and deeper experiments on
server-class runners (e.g., Kaggle).

- `abl_smoke` — fastest pipeline integrity check (no RL), minimal metrics,
  no plots; intended to surface errors quickly.
- `abl_lite_gpu` — RL-enabled with light settings for weak GPUs; reduced
  feature dimensions and timesteps.
- `abl_server` — RL-enabled, multi-seed, larger timesteps for more stable
  results on stronger machines.

## How to run

Single scenario (auto mode, no Hydra required):

```
python hydra_main.py --scenario abl_smoke --output_root results
```

Multiple scenarios (ablation sweep):

```
python hydra_main.py --scenarios abl_smoke abl_lite_gpu abl_server --output_root results --multi_seed_enabled
```

Notes:
- `abl_smoke` ignores multi-seed to stay fast.
- `abl_server` enables multi-seed by default (seeds 101, 202, 303).
- All three scenarios have corresponding YAMLs under `configs/scenarios/` and
  presets in `hydra_main.py` so they work with or without Hydra installed.

## Suggested study grid

- Fixed vs RL: run `fixed_30`, `fixed_90`, `abl_lite_gpu`, `abl_server`.
- Dynamic baseline: add `dynamic_adaptive` for adaptive lookback.
- Aggregate and compare with `reporting/compare_scenarios.py`.

