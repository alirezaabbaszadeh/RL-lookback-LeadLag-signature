# Reactor v3 Refactor Roadmap

This document captures the agreed end-to-end migration plan for the Reactor v3
release. The goal is to provide a single, Kaggle-native execution path capable
of producing paper-grade artefacts under strict reproducibility constraints.

## Design Principles

- **GPU-first** – CUDA is the preferred device; the pipeline automatically
  falls back to CPU when necessary and disables mixed precision accordingly.
- **Deterministic & Fair** – Every configured agent consumes the same number of
  environment interactions, with strict seed control applied via
  `utils.repro.set_all_seeds`.
- **No leakage** – All walk-forward cross-validation splits apply purge and
  embargo logic, and the API is ready for nested validation.
- **Paper-grade statistics** – Reporting includes HAC confidence intervals,
  probabilistic/deflated Sharpe ratios, SPA/Reality-Check approximations, and a
  lightweight model confidence set routine.
- **Standard outputs** – Every run writes a canonical `metrics.csv` schema along
  with equity curves, trade logs, and a JSON manifest.
- **Kaggle-native** – Installs resolve from a pinned GitHub commit and all run
  artefacts live under `/kaggle/working`.

## Target Layout

```
src/leadlag/
  pipelines/
    run_full_suite.py
  env/
    trading_env.py
  features/
    signature.py
    leadlag.py
  cv/
    purged.py
  eval/
    stats.py
    stats_cli.py
  reporting/
    metrics_writer.py
  utils/
    repro.py
conf/
  config.yaml
  agent/{ppo,dqn,a2c,sac,td3}.yaml
  features/{base,signature,leadlag,signature_leadlag}.yaml
  data/{sp500_sector,crypto_top,...}.yaml
  split/walk_forward_purged.yaml
  training/{smoke,base,paper}.yaml
  hardware/{gpu,auto}.yaml
```

This structure already exists in the repository and will remain the single
supported code path. Any new functionality must plug into this layout.

## Kaggle Notebook Contract

A single Kaggle notebook (GPU + Internet ON) is the official execution surface.
Every action—from installation to statistics export—occurs inside the notebook
using Hydra overrides to control intensity.

1. **Environment check** – Validate CUDA availability via `torch.cuda` and
   `nvidia-smi`.
2. **Install** – `pip install` the project from a pinned Git commit and add core
   scientific dependencies.
3. **Configure** – Choose `training=smoke` or `training=paper`, point
   `data.dataset_dir` at the attached dataset, and select `hardware=gpu` for CUDA
   enforcement.
4. **Run grid** – Execute experiments exclusively through
   `python -m leadlag.pipelines.run_full_suite ...`. All overrides are handled by
   Hydra.
5. **Collect metrics** – Concatenate every `metrics.csv` into
   `/kaggle/working/paper_outputs/all_metrics_raw.csv`.
6. **Statistics export** – Invoke `python -m leadlag.eval.stats_cli` to generate
   confidence intervals, p-value tables, and plots.

Outputs are written beneath `/kaggle/working/results/<run_id>/` for individual
runs and `/kaggle/working/paper_outputs/` for aggregated artefacts.

## Hydra Configuration Highlights

- `conf/hardware/gpu.yaml` selects CUDA, enables AMP, and configures vectorised
  environments (`n_envs=8`).
- `conf/split/walk_forward_purged.yaml` wires the leakage-safe walk-forward
  splitting strategy with optional nested validation.
- `conf/training/{smoke,paper}.yaml` control total environment steps, seeds, and
  window counts. `paper` mode expands seeds and windows for full experiments.
- `conf/config.yaml` holds defaults and exposes the canonical override surface
  (`agent`, `features`, `data`, `split`, `training`, `hardware`).

## Phase Milestones

1. **R0 – Hygiene & Layout** – Formatting, `pre-commit`, README refresh.
2. **R1 – Single Entry Point** – All runs funnel through
   `pipelines/run_full_suite.py`.
3. **R2 – Purged CV** – Implement `WalkForwardPurged` and dataset manifests.
4. **R3 – Trading Realism** – Enforce t→t+1 execution with fees/slippage.
5. **R4 – Features** – Signature and lead–lag transforms with cache hooks.
6. **R5 – RL Agents & Fairness** – SB3 adapters with equal env-step budgets.
7. **R6 – Standardised Outputs** – Canonical metrics writer and manifests.
8. **R7 – Statistics** – HAC CIs, PSR/DSR, SPA-lite, MCS, bootstrap utilities.
9. **R8 – Generalisation** – Multi-market profiles and regime analysis.
10. **R9 – Kaggle Notebook** – The sole documented execution path.
11. **R10 – Reproducibility** – Run manifests, CI smoke tests, tagged release.

Each milestone must leave the notebook path operational and reproducible.

## Risks and Mitigations

- **Kaggle timeouts** – Default to `training=smoke` during development.
- **GPU OOM** – Tune `hardware.n_envs`, PPO batch sizes, and signature depth.
- **Version drift** – Pin Git commit/versions and capture manifests in
  `run_manifest.json`.
- **Data dependencies** – Ship datasets via Kaggle Datasets and validate early.

---

This roadmap should be referenced for all future work on Reactor v3. Deviations
must be documented alongside rationale and any required updates to the Kaggle
notebook workflow.
