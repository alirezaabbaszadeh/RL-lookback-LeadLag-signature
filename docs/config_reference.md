# Hydra Configuration Reference

> **Metadata**
> - Last updated: 2025-02-15
> - Maintainer: Documentation Working Group
> - Status: Published
> - Source of truth: `docs/config_reference.md`

This reference outlines how Hydra drives scenario execution for the LeadLag-signature RL project. Use it alongside the [Reproducibility guide](repro.md) when preparing local runs, CI smoke tests, or Kaggle notebooks.

## Directory Overview

```text
configs/
├── config.yaml            # base entry point for leadlag.hydra_main
└── scenario/              # reusable scenario descriptors (Hydra/YAML mode)
    ├── fixed_30.yaml
    ├── fixed_90.yaml
    ├── dynamic_adaptive.yaml
    └── rl_ppo.yaml
```

## Core `config.yaml` fields

| Key | Type | Description |
| --- | --- | --- |
| `defaults.scenario` | str | Hydra default pointing to a file in `configs/scenario/`. Override with `python -m leadlag.hydra_main scenario=fixed_90`. |
| `output_root` | str | Root directory for run artefacts (defaults to `results/`; override with `output_root=/tmp/run`). |
| `multi_seed.enabled` | bool | Enables multi-seed aggregation when `true`. Scenario descriptors can override this flag. |
| `multi_seed.seeds` | list[int] | Default seeds used when multi-seed aggregation is enabled. |
| `scenarios` | list[str \| dict] | Optional sequence of scenarios to run back-to-back. Entries may be scenario names or inline dicts matching the columns below. |

## Scenario descriptors (`configs/scenario/*.yaml`)

| Key | Type | Description |
| --- | --- | --- |
| `name` | str | Scenario identifier surfaced in logs and aggregate outputs. |
| `path` | str | Path to the YAML file describing analysis parameters. |
| `runner` | str | One of `scenario`, `dynamic`, or `rl`, selecting the execution path. |
| `multi_seed.enabled` | bool | Optional override of the global multi-seed toggle. |
| `multi_seed.seeds` | list[int] | Optional seed list scoped to this scenario. |

## Built-in presets

Hydra fallback mode exposes several named presets via `scenario=<name>` even without Hydra installed.

| Preset | Purpose | Key settings |
| --- | --- | --- |
| `fast_smoke` | Ultra-fast smoke test | Synthetic data when CSV inputs are missing, `lookback=10`, plots disabled, single seed. |
| `fixed_30` | Research baseline | `lookback=30`, multi-seed `[42, 52, 62]`, generates full reporting artefacts. |
| `dynamic_adaptive` | Rule-based adaptive baseline | Configures dynamic lookback windows with min/max/step guards. |
| `rl_ppo` | Lightweight RL | Stable-Baselines3 PPO with `total_timesteps=2000` and reduced network sizes. |

## CLI usage patterns

### Package CLI (`leadlag`)

The packaged `leadlag` command discovers descriptors, applies scenario defaults, and runs aggregations:

```bash
leadlag --list
leadlag --scenarios fixed_30 rl_ppo --results-root results
leadlag --dry-run --format json --include rl
```

Key flags:
- `--results-root` chooses where merged configs, manifests, and metrics are written (defaults to `LEADLAG_RESULTS_ROOT` or `results`).
- `--include` / `--exclude` filter scenario filenames by substring.
- `--max-scenarios` caps the number of scenarios executed after filtering.
- `--runner {auto,scenario,dynamic,rl}` overrides automatic runner selection.
- `--status` reports run status for an existing results directory without re-executing scenarios.
- `--skip-existing` avoids re-running scenarios that already succeeded under the current results root.

### Hydra module entry point

When you need Hydra overrides or inline compositions, call the module entry point directly:

```bash
python -m leadlag.hydra_main \
  scenario=fixed_30 \
  multi_seed.enabled=true \
  multi_seed.seeds='[11, 22, 33]' \
  output_root=results/custom
```

Use inline dictionaries to run ad-hoc scenarios without creating YAML files:

```bash
python -m leadlag.hydra_main \
  scenarios='[{name: custom, path: configs/scenarios/fixed_30.yaml, runner: scenario}]'
```

Hydra writes merged configs to `<output_root>/<scenario>/config_merged.yaml`, enabling diffs between presets and overrides.

For deeper orchestration patterns, refer back to the [Reproducibility guide](repro.md) and the [Ablation guide](ablation_guide.md).
