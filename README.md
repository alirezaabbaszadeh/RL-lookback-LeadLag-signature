# LeadLag Signature RL Platform

End-to-end research environment for analysing lead–lag signatures, experimenting with adaptive lookback policies, and benchmarking reinforcement-learning agents on financial time-series data. The stack is designed for reproducible experiments across local machines, Kaggle notebooks, and CI smoke runs.

---

## Key Capabilities

- **Signature-based analytics** – modular feature extraction, distance-correlation variants, and lead–lag matrix visualisation (`models/`, `evaluation/`).
- **Flexible experimentation** – Hydra-driven configuration for deterministic scenarios, dynamic baselines, and RL policies (`configs/`, `training/`).
- **Research automation** – multi-seed orchestration, reporting utilities, synthetic meta-RL datasets, and offline RL tooling (`training/runner_multiseed.py`, `research/meta_rl`, `research/offline_rl`).
- **Governance & reproducibility** – dataset manifests, quality audits, structured logging, and smoke tests tailored for cloud execution (`governance/`, `scripts/smoke_kaggle.py`, `docs/data_preprocessing.md`).

---

## Repository Layout

| Path | Description |
|------|-------------|
| `configs/` | Base Hydra configs plus scenario presets (`fixed_30`, `dynamic_adaptive`, `rl_ppo`, `fast_smoke`, …). |
| `envs/leadlag_env.py` | Gym-compatible environment wrapping the lead–lag analyser for RL agents. |
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
python hydra_main.py \
  --scenarios fixed_30 rl_ppo \
  --multi_seed_enabled \
  --seeds 42 52 62 \
  --output_root results
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
   python kaggle/starter.py \
     --scenario fixed_30 \
     --run-meta-rl \
     --run-offline \
     --output-root /kaggle/working
   ```
3. Follow the detailed packaging checklist in `docs/deployment/kaggle_setup.md` (dataset preparation, governance checks, notebook template, final review).

---

## Ablation Pipeline

Generate a full ablation suite (baseline, dynamic, RL, and random controls) with one command:

```bash
python pipelines/run_ablation.py --output-root /kaggle/working/ablations
```

- Uses multi-seed aggregation by default (`--seeds 42 52 62`). Add `--single-seed` for quick smoke runs.
- RL-focused scenarios (`abl_lite_gpu`, `abl_server`, `rl_ppo`) require optional dependencies (`stable-baselines3`, `torch`). Install them or run the script with `--skip-missing-deps`.
- Outputs are stored under `<output-root>/<scenario>_*` plus comparison CSV/plots inside `<output-root>/ablation_comparison/`.

---

## Full Experiment & Audit Suite

Run the complete set of experiments and audits with one command:

```bash
python pipelines/run_full_suite.py --output-root /kaggle/working/full_suite
```

- Executes dataset-quality checks, baseline scenario(s), meta/offline RL baselines, leakage probes, walk-forward verification, the ablation suite, aggregate comparisons, and final report generation.
- Toggle sections with flags: `--skip-ablation`, `--skip-meta-offline`, `--skip-audit`, `--skip-report`, `--skip-baseline`.
- Use `--skip-optional-deps` to auto-skip RL ablation presets when `stable-baselines3`/`torch` are unavailable.
- Outputs are organised under `/core`, `/ablations`, `/robustness`, `/aggregate_comparison`, and `/reports` beneath the chosen output root.

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
2. Update `CHANGELOG.md` (to be generated) with the latest roadmap entries (`docs/future_roadmap.pseudo`). Add release bullet points if needed.
3. Tag and publish:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```
4. Draft GitHub release notes summarising major modules and include links to Kaggle instructions.

---

## Helpful References

- `docs/repro.md` – reproducibility guide (conda + Docker + CLI usage).
- `docs/data_preprocessing.md` – data cleaning steps and governance tooling.
- `docs/deployment/kaggle_setup.md` – Kaggle-ready checklist and notebook structure.
- `docs/future_roadmap.pseudo` – roadmap, status tracker, and change log history.

Questions or contributions? Stay tuned for `CONTRIBUTING.md` and `CHANGELOG.md` in the release branch, or open an issue once this repository is published. Happy experimenting!
