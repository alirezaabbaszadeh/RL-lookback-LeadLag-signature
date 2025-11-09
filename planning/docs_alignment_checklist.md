# Documentation Alignment Checklist

## Inventory Snapshot

### README.md
- Provides project highlights, repository layout, and Kaggle orchestration workflow, including JSON-output CLIs and multi-stage artifact bundling expectations.【F:README.md†L8-L170】

### docs/
- `docs/config_reference.md` – Hydra configuration map describing defaults, scenario descriptors, and CLI usage patterns.【F:docs/config_reference.md†L11-L97】
- `docs/repro.md` – Reproducibility workflow covering environment setup, CLI usage, Hydra overrides, and troubleshooting guidance.【F:docs/repro.md†L1-L120】
- `docs/ablation_guide.md` – Ablation scenario rundown with CLI examples and override tips for exploratory studies.【F:docs/ablation_guide.md†L1-L55】
- `docs/advanced_research_plan.md` – Roadmap for Meta-RL and offline RL initiatives with implementation hooks and validation goals.【F:docs/advanced_research_plan.md†L1-L39】
- `docs/data_preprocessing.md` – Data ingestion, cleaning, and governance manifest documentation for upstream inputs.【F:docs/data_preprocessing.md†L1-L23】
- `docs/metrics_dictionary.md` – Contract for emitted metrics across time-series, summaries, and aggregates.【F:docs/metrics_dictionary.md†L1-L40】
- `docs/kaggle_artifacts.md` – Stage-by-stage artifact map for the Kaggle orchestrator bundle.【F:docs/kaggle_artifacts.md†L1-L56】

### research/
- `research/tmlr_source_extracts.md` – Snapshot of current TMLR submission requirements and policy excerpts.【F:research/tmlr_source_extracts.md†L1-L20】
- `research/offline_rl/` – Offline RL utilities exposing deprecated entrypoints that delegate to `leadlag.research.offline_rl` for trajectory capture and behavior cloning training.【F:research/offline_rl/train_offline.py†L1-L16】【F:research/offline_rl/log_trajectories.py†L1-L8】

### reporting/
- `src/leadlag/reporting/generate_report.py` – CLI for producing Markdown/PDF research reports from aggregated results.【F:src/leadlag/reporting/generate_report.py†L128-L200】
- `src/leadlag/reporting/compare_scenarios.py` – CLI for aggregating multi-seed stats, generating comparison CSVs, and optional plots.【F:src/leadlag/reporting/compare_scenarios.py†L13-L120】
- `src/leadlag/reporting/status_summary.py` – CLI that parses roadmap pseudo-documents to summarize open status items.【F:src/leadlag/reporting/status_summary.py†L1-L139】

### pipelines/
- `src/leadlag/pipelines/run_full_suite.py` – Hydra-driven end-to-end pipeline that materializes metrics, manifests, and paper outputs under configured results roots.【F:src/leadlag/pipelines/run_full_suite.py†L1157-L1195】
- `src/leadlag/pipelines/run_ablation.py` – Orchestrator for ablation scenarios with dependency checks, multi-seed execution, and comparison report integration.【F:src/leadlag/pipelines/run_ablation.py†L23-L200】

## Alignment Gaps & Follow-Up Tasks

- [ ] `docs/config_reference.md` lists a `fast_smoke` preset that no longer exists in the packaged scenario catalog; only `dynamic_adaptive`, `fixed_30`, `fixed_90`, and `rl_ppo` remain under `leadlag/configs/scenario/`. Update or replace the preset guidance.【F:docs/config_reference.md†L43-L52】【88a954†L1-L2】
- [ ] `docs/ablation_guide.md` states that ablation descriptors live in `leadlag/configs/scenario/` and that `abl_server` defaults to seeds 101/202/303, but the actual YAML lives under `leadlag/configs/scenarios/` with single-seed defaults and the ablation runner defaults to seeds 42/52/62. Refresh location and seed details.【F:docs/ablation_guide.md†L29-L33】【F:src/leadlag/configs/scenarios/abl_smoke.yaml†L1-L12】【F:src/leadlag/configs/scenarios/abl_server.yaml†L1-L33】【F:src/leadlag/pipelines/run_ablation.py†L153-L180】
- [ ] `docs/advanced_research_plan.md` and `docs/kaggle_artifacts.md` reference Meta-RL scripts and outputs (`research/meta_rl/*`, `runs/meta_rl`) that are absent from the repository and Kaggle orchestrator stages, which only execute `full_suite`, `sb3_leadlag`, and `dopamine`. Either reintroduce the Meta-RL tooling or update the documentation to match the current stack.【F:docs/advanced_research_plan.md†L5-L20】【F:docs/kaggle_artifacts.md†L14-L32】【13b65a†L1-L2】【F:src/leadlag/kaggle/run_all.py†L133-L149】
- [ ] Document the `leadlag-report` and roadmap status summary CLIs so users can discover report generation and roadmap-audit tooling. Both scripts expose packaged entry points but lack coverage in README/docs.【F:src/leadlag/reporting/generate_report.py†L128-L200】【F:src/leadlag/reporting/status_summary.py†L96-L139】
