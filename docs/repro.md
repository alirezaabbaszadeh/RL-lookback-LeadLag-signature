# Reproducibility Guide

> **Metadata**
> - Last updated: 2025-02-15
> - Maintainer: Experimentation Working Group
> - Status: Published
> - Source of truth: `docs/repro.md`

This guide documents the end-to-end workflow for reproducing baseline and RL experiments in the LeadLag-signature RL project. Follow it in conjunction with the [Hydra configuration reference](config_reference.md) and [Ablation guide](ablation_guide.md).

## Environment setup

```bash
conda env create -f environment.yml
conda activate leadlag
```

- Python 3.10 is the validated interpreter across CI, Kaggle, and local development.
- Optional extras can be installed via `pip install .[rl]`, `pip install .[signature]`, or `pip install .[mlflow]` after activating the environment.
- `requirements.txt`, `requirements-rl.txt`, and `requirements-kaggle.txt` remain available for fully pinned offline installs.

## Wheel packaging and CLI smoke check

Run this sequence from a clean checkout to confirm that the wheel bundles all
configs and that the console scripts resolve correctly. The verification mirrors
what Kaggle notebooks do when they install the project from a wheel cache.

```bash
python -m pip install build  # one-time dependency
python -m build              # produces dist/leadlag_signature_rl-<ver>.whl

python -m venv .venv_packaging
source .venv_packaging/bin/activate
pip install dist/leadlag_signature_rl-*.whl

# CLI entry points should now resolve without touching the source tree
leadlag --help
leadlag-full-suite --help

deactivate
rm -rf .venv_packaging
```

If any of the help commands fail, inspect `pyproject.toml` packaging metadata
and ensure new modules/config files live under `src/leadlag/` so they are picked
up by the wheel builder.

## Quick runs with the packaged CLI

Single scenario (uses the descriptor defaults for seeds and outputs artefacts under `results/` by default):

```bash
leadlag --scenarios fixed_30
```

Multiple scenarios with aggregation and explicit output root:

```bash
leadlag --scenarios fixed_30 rl_ppo --results-root runs/repro
```

Preview what will run without executing:

```bash
leadlag --dry-run --format json --include rl
```

Set `LEADLAG_RESULTS_ROOT` to change the default results directory when `--results-root` is omitted.

## Hydra overrides and scripting

For Hydra-style overrides (custom seeds, inline descriptors, parameter sweeps) call the module directly:

```bash
python -m leadlag.hydra_main \
  scenario=fixed_30 \
  multi_seed.enabled=true \
  multi_seed.seeds='[42, 52, 62]' \
  output_root=results/fixed_30_multiseed
```

Additional examples:

```bash
# Run sequential scenarios defined inline
python -m leadlag.hydra_main \
  scenarios='[fixed_30, rl_ppo]' \
  output_root=results/batch

# Validate a scenario without executing it
python -m leadlag.hydra_main --cfg job --resolve
```

Artefacts per run include `config_merged.yaml`, `run_metadata.json`, `data_manifest.json`, `metrics_timeseries.csv`, `summary.csv`, and optional figures or profiling data. When multi-seed aggregation is enabled, Hydra also produces `stats.csv`, `significance.csv`, `welch.csv`, and `runs.json`.

## Optional MLflow integration

When `mlflow` is installed, the orchestration layer logs metrics and artefacts automatically. Configure the target instance via:

```bash
export MLFLOW_TRACKING_URI=<tracking-url>
export MLFLOW_EXPERIMENT_NAME="LeadLag Signature"
```

Re-run the desired scenario commands and the CLI will stream metrics to the configured experiment.

## Docker workflow

```bash
docker build -t leadlag-rl:latest .
docker run --rm -it \
  -v "$(pwd)":/workspace \
  leadlag-rl:latest \
  leadlag --scenarios fixed_30 --results-root /workspace/results
```

Mount additional data volumes or override the entry command as needed.

## Troubleshooting checklist

- **Dataset quality**: `python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv`
- **Governance manifests**: ensure `data_manifest.json` exists per run before promoting results.
- **Plotting issues**: plots are optional; headless environments still produce CSV outputs.
- **Optional dependencies**: install `iisignature` for signature-specific tests and `pip install -r requirements-rl.txt` for Stable-Baselines3 scenarios.
- **Binary compatibility**: stick to the provided Conda environment or dependency pins to avoid `numpy.dtype size changed` errors.
- **Tests**: `pytest -q` verifies that optional dependencies are wired correctly once installed.

## End-to-end reproducibility checklist

1. `conda env create -f environment.yml && conda activate leadlag`
2. `python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv`
3. `leadlag --scenarios fixed_30 --results-root results`
4. Inspect `results/` for artefacts; re-run with Hydra overrides if custom seeds or paths are required.
5. (Optional) Configure MLflow variables and re-run scenarios to log experiments.

## Scenario Driver CLI (`leadlag`)

For bulk execution across all descriptors and aggregation tooling, rely on the packaged command:

```bash
# Execute every scenario discovered in leadlag/configs/scenario/
leadlag --results-root results

# Preview scenarios without running them
leadlag --dry-run --include rl

# Fail-fast and store logs elsewhere
leadlag --results-root runs/2024-10-22 --stop-on-error --log-level DEBUG
```

Key operational flags that interact with `--results-root`:

- `--status` inspects the resolved results directory (either from `--results-root` or `LEADLAG_RESULTS_ROOT`) and summarises existing runs without launching new work. Point it at historical artefacts to verify reproducibility before re-running anything.
- `--skip-existing` checks the same results root for successful scenario folders and skips them during execution. Use a new `--results-root` when you need a clean rerun from scratch.
- `--validate <scenario>` performs schema validation for the specified descriptor (name or path) and exits. It does not create output under the results root but helps confirm that a configuration is runnable before scheduling it.
- `--log-path` overrides where the driver writes its structured log. By default the file lives at `<results-root>/main.log`, so changing `--results-root` automatically relocates the log unless you pin an explicit path.

The CLI writes structured logs to `<results-root>/main.log` (unless `--log-path` is provided) and renders JSON envelopes when `--format json` is set, making it suitable for automation in CI or Kaggle notebooks.
