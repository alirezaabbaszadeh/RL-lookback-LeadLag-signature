# Ablation Scenarios and Usage

> **Metadata**
> - Last updated: 2025-02-15
> - Maintainer: Experimentation Working Group
> - Status: Published
> - Source of truth: `docs/ablation_guide.md`

Three ablation-focused scenarios provide coverage from smoke checks to server-grade RL studies. They share logging conventions with the broader platform and align with the configuration patterns described in the [Hydra configuration reference](config_reference.md).

- `abl_smoke` — fastest pipeline integrity check (no RL), minimal metrics, no plots; intended to surface errors quickly.
- `abl_lite_gpu` — RL-enabled with light settings for modest GPUs; reduced feature dimensions and timesteps.
- `abl_server` — RL-enabled, multi-seed, larger timesteps for more stable results on accelerator-equipped machines.

## Running scenarios with the packaged CLI

Single scenario (auto-detects descriptor defaults):

```bash
leadlag --scenarios abl_smoke --results-root results/ablations
```

Full ablation sweep:

```bash
leadlag --scenarios abl_smoke abl_lite_gpu abl_server --results-root results/ablations
```

Notes:
- `abl_smoke` ignores multi-seed to remain fast.
- `abl_server` enables multi-seed by default (seeds 101, 202, 303).
- All three scenarios live under `leadlag/configs/scenario/` and inherit the same aggregation pipeline as baseline runs.

## Hydra overrides

Adjust seeds, runners, or custom parameters with Hydra overrides:

```bash
python -m leadlag.hydra_main \
  scenarios='[abl_smoke, abl_lite_gpu, abl_server]' \
  multi_seed.enabled=true \
  output_root=results/ablations_multiseed
```

To experiment with alternative RL hyperparameters, clone the YAML under `leadlag/configs/scenarios/`, adjust values (for example `rl.total_timesteps`), and reference the new file through an inline descriptor:

```bash
python -m leadlag.hydra_main \
  scenarios='[{name: abl_lite_gpu_custom, path: leadlag/configs/scenarios/abl_lite_gpu_custom.yaml, runner: rl}]' \
  output_root=results/ablations_custom
```

Hydra writes merged configs to `<output_root>/<scenario>/config_merged.yaml`, making it easy to diff adjustments against the stock ablation presets.

## Suggested study grid

- Fixed vs RL: run `fixed_30`, `fixed_90`, `abl_lite_gpu`, and `abl_server`.
- Dynamic baseline: add `dynamic_adaptive` for adaptive lookback comparisons.
- Aggregate and compare with `leadlag-compare` or `python -m leadlag.reporting.compare_scenarios` once results are collected.
