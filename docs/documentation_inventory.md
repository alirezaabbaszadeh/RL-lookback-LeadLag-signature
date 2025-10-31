# Documentation Inventory

The table below summarizes the current documentation assets, their owners (where known), purpose, and maintenance status as of this audit.

| Path | Owner | Purpose | Status |
| --- | --- | --- | --- |
| `README.md` | — | Primary overview covering capabilities, repo layout, quickstart commands, CLI entry points, and Kaggle orchestration tips. | Current. Comprehensive orientation for new users. |
| `README_FA.md` | — | Farsi translation of the main README with the same onboarding and Kaggle deployment guidance for Persian speakers. | Current. Mirrors English guidance. |
| `CONTRIBUTING.md` | — | Contributor setup, coding standards, quality checks, and release expectations. | Current. Clear workflow for contributors. |
| `CHANGELOG.md` | — | Historical release notes and current “Unreleased” changes. | Needs formatting cleanup—the final bullet includes a literal `\n` join. |
| `docs/ablation_guide.md` | — | How-to for the three ablation presets plus recommended study grids. | Current. Focused scenario reference. |
| `docs/audit/` | — | README and task index describing the multi-phase audit programme and phase catalog. | Current. Organises extensive audit workflow. |
| `docs/adr/ADR-001-hydra-config.md` | — | Architecture decision record adopting Hydra for configuration, with context and consequences. | Current. Captures rationale for config system. |
| `docs/advanced_research_plan.md` | — | Execution plan for Meta-RL and offline RL initiatives with objectives, validation metrics, and next steps. | Current. Tracks research backlog completion. |
| `docs/config_reference.md` | — | Reference for Hydra config keys, scenario descriptors, and preset CLI usage (with bilingual notes). | Current. Helpful when editing configs. |
| `docs/data_preprocessing.md` | — | Describes data sourcing, cleaning, manifests, and governance tooling. | Current. Matches observed pipeline behaviour. |
| `docs/deployment/kaggle_setup.md` | — | Detailed Kaggle “one-command” orchestration guide, notebook cell, and troubleshooting advice. | Current, though overlaps with README’s Kaggle section. |
| `docs/evaluation_visuals.md` | — | Defines required evaluation plots, tables, and aggregate artifacts. | Current. Aligns with reporting outputs. |
| `archive/2025-10-19-roadmap/docs/future_roadmap.pseudo` | `roadmap_bot` (per metadata) | Formal roadmap with quality gates, module owners, initiatives, and status tracker. | Archived 2025-10-19. Retained for historical reference. |
| `docs/kaggle_artifacts.md` | — | Inventory of files produced by `kaggle/run_all.py`, annotated by artifact type. | Current. Matches deployment outputs. |
| `docs/metrics_dictionary.md` | — | Metric definitions spanning time-series, run summaries, and aggregates, plus observability hooks. | Current. Useful for analytics consumers. |
| `docs/standards.md` | Documentation Working Group | Shared style guide covering headings, metadata, terminology, and review workflow. | Draft. Circulated for feedback via `docs/standards_feedback_request.md`. |
| `docs/standards_feedback_request.md` | Documentation Working Group | Shared issue doc used to collect stakeholder feedback on documentation standards. | Open. Comment window closes 2025-11-07. |
| `docs/repro.md` | — | Reproducibility guide covering Conda/Docker setup, scenario execution, outputs, MLflow, and troubleshooting. | Current. Complements README quickstart. |
| `reporting/` | — | Legacy CLI wrappers that delegate to `leadlag.reporting.*` modules with deprecation warnings. | Redundant; wrappers marked deprecated pending removal. |
| `archive/2025-10-17-research-report/reports/final_report.md` | — | Generated research report summarising 2025 experiments and statistics. | Archived 2025-10-17. Superseded by future runs. |
| `archive/2025-10-17-research-report/reports/appendix.md` | — | Reproducibility appendix for the same 2025 run, with environment and artifact inventory. | Archived 2025-10-17 with the rest of the campaign outputs. |
| `archive/2025-10-17-research-report/reports/final_report.pdf` | — | PDF version of the 2025 report (binary artifact). | Archived 2025-10-17 with the Markdown sources. |
| `reports/README.md` | — | Placeholder describing where active reports will appear. | Current. Notes that historical outputs moved to the archive. |
| `notebooks/LeadLag_signature.ipynb` | — | Notebook for data preparation and lead-lag analysis; installs dependencies inline and loads CSVs from a fixed relative path. | Likely outdated—assumes `../LeadLag_signature` path and manual pip installs. |

