# Reactor v3 LeadLag Platform

GPU-first reinforcement-learning research stack for lead–lag signature trading.
Reactor v3 standardises the code layout, Hydra configuration surface, and Kaggle
notebook workflow so that a single pipeline produces reproducible, paper-grade
artefacts.

## Highlights

- **Scenario driver CLI** – The `leadlag` console entry point lists packaged
  scenarios, validates configs, runs selected workloads, and reports aggregated
  status. Companion commands such as `leadlag-full-suite`, `leadlag-ablation`,
  and reporting utilities remain available for scripted automation.
- **Deterministic output contract** – Every CLI honours `--format text|json`
  (with `--json` as a temporary alias) and emits a stable envelope containing
  the command line, parsed arguments, results, and discovered artefacts.
- **GPU aware** – Defaults to CUDA with AMP enabled; automatically falls back to
  CPU when unavailable.
- **Leakage-safe CV** – Purged/embargoed walk-forward splits, ready for nested
  tuning.
- **Journal-ready bundles** – The Kaggle orchestrator creates
  `multi_stage_artifacts/` and `multi_stage_artifacts.zip` with stage manifests,
  audit logs, and paper outputs that match TMLR reviewer expectations out of the
  box.
- **Paper-grade statistics** – HAC confidence intervals, PSR/DSR, SPA-lite, and a
  model confidence set automatically exported to `paper_outputs_root` after each
  run.
- **Kaggle-native** – One notebook path with Internet + GPU ON; all artefacts
  stored inside `/kaggle/working`.

See [`docs/reactor_v3_refactor_plan.md`](docs/reactor_v3_refactor_plan.md) for the
full roadmap and governance milestones.

## Installation

The project is packaged via `pyproject.toml`. Install it in editable mode for
local development or scripts:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras expose the RL stack, signature methods, and MLflow exporters
defined in [`pyproject.toml`](pyproject.toml):

```bash
# enable Stable-Baselines3 + signature features during local research
python -m pip install -e '.[rl,signature]'

# append MLflow utilities when experiment tracking is required
python -m pip install -e '.[mlflow]'
```

When mirroring the CI/Kaggle environment, reuse the curated lockfiles:

```bash
# base research stack (no RL extras)
python -m pip install -r requirements.txt

# Kaggle-ready bundle with notebook-friendly pins
python -m pip install -r requirements-kaggle.txt

# Reinforcement learning agents and dependencies used in pipelines
python -m pip install -r requirements-rl.txt

# Developer tooling (formatters, linters, pytest, build helpers)
python -m pip install -r requirements-dev.txt
```

See [`docs/kaggle_artifacts.md`](docs/kaggle_artifacts.md) for notebook-specific
pinning strategies and [`docs/config_reference.md`](docs/config_reference.md)
for configuration conventions shared across environments.

## Repository Layout

```
src/leadlag/
  main.py                      # Primary CLI dispatcher (exposed as `leadlag`)
  cli/                         # Shared CLI plumbing and JSON/text formatters
  driver/                      # Scenario discovery, filtering, execution
  pipelines/run_full_suite.py  # Legacy Hydra launcher (still exported)
  env/                         # t→t+1 execution with cost modelling
  features/                    # signature & lead–lag transforms
  cv/                          # walk-forward purged/embargoed splits
  eval/                        # sharpe/sortino/HAC/SPA/MCS utilities
  reporting/                   # metrics writers and reporting CLIs
  utils/repro.py               # device selection, seed control, manifests
src/leadlag/configs/
  config.yaml                  # defaults (agent, data, features, hardware)
  scenarios/*.yaml             # Packaged scenario definitions
  agent/                       # PPO, DQN, A2C, SAC, TD3 presets
  features/                    # base/signature/leadlag toggles
  data/                        # universe + dataset directory profiles
  split/walk_forward_purged.yaml
  training/{smoke,base,paper}.yaml
  hardware/{gpu,auto}.yaml
```

Support files:
- `requirements*.txt` – dependency pins for Kaggle notebooks and supporting
  tooling.
- `tests/` – unit coverage for CV, stats, features, and reporting utilities.
- `notebooks/` – Jupyter notebooks mirroring the Kaggle flow for offline
  exploration.

## Command-Line Quickstart

The [`leadlag`](src/leadlag/main.py) console script is the canonical entry
point. It provides scenario discovery, validation, execution, and status
reporting with consistent text/JSON envelopes:

```bash
# inspect packaged scenarios and metadata
leadlag --list

# preview the scenarios that would run without executing workloads
leadlag --dry-run --include dynamic --max-scenarios 2 --format text

# execute specific scenarios and capture a JSON status envelope
leadlag --scenarios dynamic_adaptive rl_ppo --results-root results --format json

# report run status (works for orchestrated pipelines too)
leadlag --status --results-root results --format json
```

Compatibility shims remain available for legacy scripts:

```bash
# mirrors `leadlag` (deprecated wrapper kept for older notebooks)
python main.py --status --results-root results --format text
```

Hydra-centric experiments can be launched directly through
[`hydra_main.py`](hydra_main.py) or the underlying module:

```bash
# run a packaged scenario with optional multi-seed execution
python -m leadlag.hydra_main --scenario fixed_30 --output_root results

# or when Hydra is installed, leverage the native launcher
python hydra_main.py --scenario dynamic_adaptive --multi_seed_enabled --seeds 11 21 31
```

Pipeline entry points under [`pipelines/`](pipelines) expose higher-level
workflows used in CI and Kaggle submissions:

```bash
# full suite orchestration (Hydra overrides accepted)
leadlag-full-suite results_root=results/full_suite training=smoke

# targeted ablations defined in pipelines/run_ablation.py
leadlag-ablation --output-root results/ablations --single-seed

# module form for custom schedulers
python -m pipelines.run_full_suite results_root=results/full_suite training=smoke
```

Reporting utilities live in [`reporting/`](reporting) and include commands such
as `leadlag-report`, `leadlag-compare`, and `leadlag-plot-balance`. Detailed
guides for datasets, configuration, metrics, ablations, and governance live in
[`docs/`](docs/), e.g. [`docs/ablation_guide.md`](docs/ablation_guide.md),
[`docs/metrics_dictionary.md`](docs/metrics_dictionary.md), and
[`docs/repro.md`](docs/repro.md).

## Kaggle Notebook Workflow (Journal Submission)

Use this notebook outline when preparing reviewer artefacts for **TMLR**. All
steps assume **GPU + Internet ON** and run entirely inside `/kaggle/working`.

1. **Verify the runtime**
   ```python
   import torch

   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   !nvidia-smi -L
   ```
2. **Prime the environment and prefetch wheels** – copy/paste this cell (adjust
   the dataset name if you attached the repo under a different alias). The
   `requirements-rl.txt` extras cover Stable-Baselines3, while the Gymnasium
   1.x + Dopamine stack is only needed when the optional sanity stage is
   enabled:
   ```bash
   %%bash
   set -e
   WORK=/kaggle/working
   cd "$WORK"

   if [ ! -f kaggle/run_all.py ]; then
     cp -r /kaggle/input/leadlag-signature/* "$WORK"/
   fi

   python -m pip install --upgrade pip
   mkdir -p wheelhouse .cache/pip
   python -m pip download -d wheelhouse -r requirements-kaggle.txt
   python -m pip download -d wheelhouse -r requirements-rl.txt
   python -m pip download -d wheelhouse "dopamine-rl==4.1.2" "gymnasium==1.0.0"

   export PIP_CACHE_DIR=/kaggle/working/.cache/pip
   export PIP_FIND_LINKS=/kaggle/working/wheelhouse
   export PIP_NO_INDEX=1
   ```
3. **Run the grand orchestrator** – builds virtualenvs per stage and bundles the
   artefacts reviewers will download:
   ```python
   !python kaggle/run_all.py
   ```
   Use `--no-prefetch` to skip the wheel download step or `--artifacts-root` to
   change the output location.
4. **Inspect the aggregated status** – every CLI exposes the JSON envelope
   contract. After the orchestrator finishes, capture the overall status to aid
   reviewer notes or debugging:
   ```python
   !leadlag --status --results-root /kaggle/working/results --format json
   ```
   The `success` flag and `errors` list reflect the pipeline outcome and match
   what CI validates in this repository. Redirect the command to
   `/kaggle/working/run_status.json` if you want to bundle the envelope with
   submission artefacts.
5. **Surface the paper outputs** – the orchestrator already copies
   `paper_outputs/` inside `multi_stage_artifacts/full_suite/`, but you can
   mirror them into the top level notebook directory for quick inspection:
   ```python
   import shutil, pathlib

   src = pathlib.Path("/kaggle/working/multi_stage_artifacts/full_suite/paper_outputs")
   dst = pathlib.Path("/kaggle/working/paper_outputs")
   if src.exists():
       shutil.copytree(src, dst, dirs_exist_ok=True)
       print("Paper outputs copied to", dst)
   else:
       print("paper_outputs missing – inspect stage logs")
   ```
6. **Download submission artefacts** – from the Kaggle sidebar, download
   `/kaggle/working/multi_stage_artifacts.zip` (reviewers unpack this file) and
   optionally `/kaggle/working/paper_outputs/` if you mirrored the tables.

The zipped bundle contains all stage manifests, logs, metrics, anonymised data
cards, and status summaries required by the journal.

### Long-only experiments

Set `env.allow_short=false` in your Hydra overrides to clamp the synthetic
trading environment to the long-only range `[0, env.max_abs_position]`. Actions
and random exploration will respect the bound while costs and metrics continue
to accumulate normally. For example:

```bash
leadlag-full-suite training=smoke env.allow_short=false
```

## Submission Bundle Layout

`multi_stage_artifacts/` mirrors what the orchestrator zips for reviewers:

- `summary.json` – stage-level status, durations, and log pointers.
- `full_suite/` – Hydra pipeline outputs including `results/`,
  `paper_outputs/`, audit scans, and generated reports.
- `sb3_leadlag/` – production Stable-Baselines3 runs on the LeadLag
  environment (`metrics_timeseries.csv`, `summary.csv`, `model.zip`, manifests).
- `dopamine/` – Gymnasium 1.x sanity checks with iteration statistics and logs.

Within `full_suite/paper_outputs/` you will find the canonical statistics:

- `all_metrics_raw.csv` – concatenated metrics for every run.
- `psr_dsr_pvalues.csv`, `hac_sharpe_confidence_intervals.csv`, SPA tables,
  plots, and `paper_results.md`.
- `paper_status.txt` – single-line readiness summary copied into reviewer notes.

If you run `scripts/reproduce_all.sh` locally (the command executed inside the
`full_suite` stage), the same directories appear under your configured
`RES`/`OUT` paths.

## Repository Hygiene

Cleaning local artefacts keeps the repository lightweight and helps CI mirrors
stay reproducible:

- `make clean` – remove Python caches, Ruff/Mypy state, and the JSON-formatted
  smoke artefacts under `tmp_cli_json_run/`.
- `make distclean` – perform `make clean` plus delete the virtual environment,
  build outputs, and any cached smoke results or `results/` directories.
- `rm -rf results multi_stage_artifacts wheelhouse paper_outputs` – optional
  manual sweep for local experiment artefacts. Ensure anything you need is
  archived before running this command.
- `leadlag --status --format json` – confirm that `results_root` no longer
  contains stale runs once the cleanup is complete.

## Testing

Run the lightweight smoke suite:

```bash
pytest
```

For CI environments without CUDA, override `hardware=auto` and `training=smoke`
to keep runtimes bounded. A quick contract check such as `leadlag --status
--format json --dry-run` verifies that CLI output continues to conform to the
shared envelope.
