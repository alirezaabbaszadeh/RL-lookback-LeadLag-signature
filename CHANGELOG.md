# Changelog

All notable changes will be documented in this file. Dates follow YYYY-MM-DD.

## Unreleased
- Initial GitHub packaging: cleaned README, added Kaggle deployment guide, smoke tests, and governance docs.
- Provide `requirements-kaggle.txt`, `kaggle/starter.py`, and `scripts/smoke_kaggle.py` for lean deployments.
- Added end-to-end orchestration pipelines (`pipelines/run_ablation.py`, `pipelines/run_full_suite.py`) to cover ablations, audits, meta/offline baselines, and reporting in a single run.

## 2025-10-18
- Completed Observability initiative: structured logging context, metrics dictionary refresh, CLI dashboard.
- Completed DataGovernance initiative: dataset manifests, preprocessing documentation, quality audit script.
- Finalised AdvancedResearch module:
  - Added synthetic regime datasets, meta-RL agent baseline, and transfer analysis outputs.
  - Added offline trajectory logger, behaviour cloning trainer, and offline vs. online benchmark tooling.
- Automated research reporting pipeline (ER-02) including `generate_report.py` and final report artifacts.

## 2025-10-17
- Experiment Orchestrator upgrades: multi-seed aggregation statistics, Welch tests, scenario validation helpers.
- Evaluation Reporting enhancements: required visuals documentation, comparison tooling, automated plots.
- InfrastructureQuality upgrades: CI workflow, lint configuration, environment spec, Dockerfile, reproducibility guide.

## 2025-10-16
- SignatureCore refactor and feature pipeline (batch API, YAML exposure, compression options).
- RLEngine improvements: environment upgrade, policy suite (PPO-LSTM, attention), reward templates with tests.
- Extended quality gates covering financial and safety metrics; added status tracker tooling.

## 2025-10-12
- Added fast smoke preset and CLI fallback improvements for quick validation.

## 2025-10-11
- Introduced signature modules, Hydra config, and multi-seed runner.
