# LeadLag Signature RL Platform

End-to-end research environment for analysing lead-lag signatures, experimenting with adaptive lookback policies, and benchmarking reinforcement-learning agents on financial time-series data. The stack is designed for reproducible experiments across local machines, Kaggle notebooks, and CI smoke runs.

---

## Key Capabilities

- **Signature-based analytics** - modular feature extraction, distance-correlation variants, and lead-lag matrix visualisation (`models/`, `evaluation/`).
- **Flexible experimentation** - Hydra-driven configuration for deterministic scenarios, dynamic baselines, and RL policies (`configs/`, `training/`).
- **Research automation** - multi-seed orchestration, reporting utilities, synthetic meta-RL datasets, and offline RL tooling (`training/runner_multiseed.py`, `research/meta_rl`, `research/offline_rl`).
- **Governance & reproducibility** - dataset manifests, quality audits, structured logging, and smoke tests tailored for cloud execution (`governance/`, `scripts/smoke_kaggle.py`, `docs/data_preprocessing.md`).

---

## Repository Layout

| Path | Description |
|------|-------------|
| `configs/` | Base Hydra configs plus scenario presets (`fixed_30`, `dynamic_adaptive`, `rl_ppo`, `fast_smoke`, ...). |
| `envs/leadlag_env.py` | Gymnasium-compatible environment wrapping the lead-lag analyser for RL agents. |
| `evaluation/` | Metrics, summaries, visualisation helpers, and statistical comparisons. |
| `governance/` | Dataset hashing, manifests, and quality checks shared by all runners. |
| `observability/` | Structured logging utilities and a CLI dashboard for run inspection. |
| `research/meta_rl/` | Synthetic regime generator, meta-RL agent baseline, and transfer evaluator. |
| `research/offline_rl/` | Trajectory logger and behaviour cloning trainer for offline pipelines. |
| `scripts/` | Execution helpers (`run_experiment.*`), audits, and Kaggle smoke tests. |
| `kaggle/starter.py` | All-in-one entrypoint for Kaggle notebooks (scenario + meta/offline options). |
| `docs/` | Detailed guides covering reproducibility, deployment, roadmap, and metrics. |
| `tests/` | Pytest suite validating analyzers, policies, rewards, and runners. |

---

## Quick Start

```bash
pip install -r requirements.txt        # or use requirements-kaggle.txt for the lean stack
python hydra_main.py --scenario fixed_30 --output_root results
```

Multi-scenario or multi-seed runs:

```bash
python hydra_main.py   --scenarios fixed_30 rl_ppo   --multi_seed_enabled   --seeds 42 52 62   --output_root results
```

Generated artifacts include merged configs, dataset manifests, metrics timelines, summary tables, plots, and optional MLflow logs.

---

## Kaggle Deployment

1. Install the lightweight dependency set:
   ```bash
   pip install -r requirements-kaggle.txt
   ```
2. Use the starter helper inside the Kaggle notebook (internet disabled):
   ```bash
   python kaggle/starter.py      --scenario fixed_30      --run-meta-rl      --run-offline      --output-root /kaggle/working
   ```
3. Follow the detailed packaging checklist in `docs/deployment/kaggle_setup.md` (dataset preparation, governance checks, notebook template, final review).

### Kaggle Multi-Stack Automation

When you need to exercise mutually incompatible RL stacks (e.g. SB3 0.29.x vs. Dopamine 1.x) inside a single Kaggle runtime, use the orchestrator:

```bash
python kaggle/run_multi_stage.py
```

- Installs each stack with forced `pip` reinstalls, runs the corresponding stage script, captures logs + models under `/kaggle/working/multi_stage_artifacts/<stage>/`, and uninstalls the packages before moving on.
- Stages are registered in `kaggle/run_multi_stage.py` and implemented under `kaggle/stages/`. Extend them or add new ones by editing the registry.
- Run individual stages via `python kaggle/run_multi_stage.py --stage sb3_kaggle` and list available entries with `--list`.
- Every execution emits `summary.json` (timings, status, log paths) and stage-specific `requirements.txt`, making it easy to reproduce or debug downstream notebooks.
- The default registry includes:
  - `sb3_kaggle` - Stable-Baselines3 PPO smoke test compatible with Kaggle pins.
  - `dopamine` - Dopamine + Gymnasium 1.x validation via random rollouts.
  - `leadlag_hydra` - Runs project Hydra scenarios (single or multi-seed) and keeps outputs in the stage artifact folder.
  - `sb3_leadlag` - Production SB3 training on LeadLag env (SB3 2.1.0 + Gymnasium 0.29.1; PPO, PPO-LSTM, attention policy) with env overrides for device/timesteps.
  - `full_suite` - Runs `pipelines/run_full_suite.py` (includes ablation pipeline, audits, reports) to avoid duplicates.

### One-Command Grand Run

To execute the complete suite (baselines, ablations, RL, audits, reports) plus the Dopamine stack in a single command on Kaggle with Internet ON:

```bash
python kaggle/run_all.py
```

- Prefetches wheels for fast installs, uses a local pip cache, then runs `full_suite`, `sb3_leadlag` (production RL), and `dopamine` via the orchestrator. Outputs live under `/kaggle/working/multi_stage_artifacts/` and a bundled `multi_stage_artifacts.zip` is created for download.
- Prefetches wheels for fast installs, uses a local pip cache, then runs `full_suite`, `sb3_leadlag` (production RL), and `dopamine` via the orchestrator. Outputs live under `/kaggle/working/multi_stage_artifacts/` and a bundled `multi_stage_artifacts.zip` is created for download. See `docs/deployment/kaggle_setup.md` for a copy/paste notebook cell.

Env Overrides (advanced)

- `leadlag_hydra` stage: `LEADLAG_SCENARIOS`, `LEADLAG_SEEDS`, `LEADLAG_MULTI_SEED`.
- `sb3_leadlag` stage: `SB3_DEVICE` (`cuda`/`cpu`/`auto`), `SB3_TIMESTEPS`, `SB3_N_STEPS`, `SB3_BATCH_SIZE`, `SB3_LR`, `SB3_EVAL_FREQ`, `SB3_VERBOSE`, `SB3_SEED`.

Recommended Kaggle settings: Internet ON, GPU for RL workloads.

---

## Ablation Pipeline

Generate a full ablation suite (baseline, dynamic, RL, and random controls) with one command:

```bash
python pipelines/run_ablation.py --output-root /kaggle/working/ablations
```

- Uses multi-seed aggregation by default (`--seeds 42 52 62`). Add `--single-seed` for quick smoke runs.
- Covers signature baselines (`fixed_30`, `fixed_90`), CCF baseline (`ccf_fixed`), dynamic heuristic (`dynamic_adaptive`), RL variants (attention, Sharpe-heavy, drawdown-heavy, PPO-LSTM), and the random control (`abl_random`).
- RL-focused scenarios require optional dependencies (`stable-baselines3`, `torch`, `sb3-contrib`). Install them or run the script with `--skip-missing-deps` to skip RL presets automatically.
- Outputs are stored under `<output-root>/<scenario>_*` plus comparison CSV/plots inside `<output-root>/ablation_comparison/`.

---

## Full Experiment & Audit Suite

Run the complete set of experiments and audits with one command:

```bash
python pipelines/run_full_suite.py --output-root /kaggle/working/full_suite
```

- Executes dataset-quality checks, baseline scenario(s), meta/offline RL baselines, leakage probes, walk-forward verification, the ablation suite, aggregate comparisons, **portfolio balance plots**, and final report generation.
- Key toggles: `--baseline-seeds`, `--baseline-single-seed`, `--ablation-scenarios`, `--ablation-single-seed`, `--skip-ablation`, `--skip-meta-offline`, `--skip-audit`, `--skip-report`, `--skip-baseline`.
- Install optional RL dependencies (`stable-baselines3`, `torch`, `sb3-contrib`) or use `--skip-optional-deps` to automatically skip RL workloads.
- Outputs are organised under `/core`, `/meta_rl`, `/offline`, `/ablations`, `/robustness`, `/aggregate_comparison`, `/evaluation/plots/balance`, `/reports`, and `/audit` beneath the chosen output root.

### Portfolio Balance Charts
- The full suite automatically calls `reporting/plot_balance_history.py` and stores equity-curve plots under `evaluation/plots/balance` (all runs, per-scenario, per-method, per-lookback).

### Run Logs
- Each full-suite execution writes a summary JSON under `/logs` (command line, parsed arguments, dependency status, start/end timestamps, duration, and validated scenarios).
- This makes it easy to compare runs later or attach metadata to reports without adding extra notebook cells.
- To rerun or customise these plots separately:
  ```bash
  python reporting/plot_balance_history.py       --results-root /kaggle/working/full_suite       --out /kaggle/working/full_suite/evaluation/plots/balance       --start-balance 100000
  ```
- Use `--max-lines` to limit the number of overlays in the global chart when the number of runs is very large.

---

## Governance & Smoke Tests

- Dataset audit:
  ```bash
  python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv
  ```
- Kaggle-compatible smoke run (fast scenario, meta-RL, offline baseline):
  ```bash
  python scripts/smoke_kaggle.py --output-root dist/kaggle_smoke --keep-meta-rl --keep-offline
  ```
- Full test suite:
  ```bash
  pytest -q
  ```
All runners emit `data_manifest.json` and structured logs, allowing rapid validation of data provenance and run context.

---

## Release Workflow

1. Ensure the repository is clean (`git status`), smoke tests pass, and the governance audit succeeds.
2. Update `CHANGELOG.md` with the latest roadmap entries (`docs/future_roadmap.pseudo`). Add release bullet points if needed.
3. Tag and publish:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```
4. Draft GitHub release notes summarising major modules and include links to Kaggle instructions.

---

## Helpful References

- `docs/repro.md` - reproducibility guide (conda + Docker + CLI usage).
- `docs/data_preprocessing.md` - data cleaning steps and governance tooling.
- `docs/deployment/kaggle_setup.md` - Kaggle-ready checklist and notebook structure.
- `docs/future_roadmap.pseudo` - roadmap, status tracker, and change log history.

Questions or contributions? Stay tuned for `CONTRIBUTING.md` and `CHANGELOG.md`, or open an issue once this repository is published. Happy experimenting!
