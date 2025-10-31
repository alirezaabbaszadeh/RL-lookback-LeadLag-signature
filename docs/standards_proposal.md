# Documentation Standards & Hierarchy Proposal

> **Metadata**
> - Last updated: 2025-11-02
> - Maintainer: Documentation Working Group
> - Status: Draft for stakeholder review
> - Source of truth: `docs/standards_proposal.md`

This proposal consolidates the in-flight documentation standards draft (`docs/standards.md`) and the
reporting hierarchy outline into a single artefact for cross-team review. It provides a compact
summary of the standards, highlights notable changes from current practice, and enumerates the
proposed directory layout for documentation and reports. Stakeholder feedback on this proposal will
unlock the subsequent migration and editing workstream.

## 1. Overview

- **Objective.** Establish a consistent writing style, metadata expectations, and structural
  taxonomy for all knowledge assets under `docs/`, `reports/`, and `reporting/`.
- **Scope.** Applies to narrative guides, reference material, planning artefacts, deployment
  runbooks, governance documentation, and experiment reports.
- **Decision drivers.** Fragmented tone, missing metadata, and overlapping folder hierarchies have
  slowed onboarding and audit readiness. This proposal seeks approval to harmonise conventions prior
  to restructuring files.

## 2. Documentation standards (draft)

The sections below abridge the key requirements from `docs/standards.md`. Final wording will be
aligned once the proposal is approved.

### 2.1 Metadata block

Every Markdown document must begin with a metadata callout containing:

1. **Last updated** — ISO-8601 format, refreshed upon substantive change.
2. **Maintainer** — Accountable person or team.
3. **Status** — `Draft`, `In review`, `Published`, or `Deprecated`.
4. **Source of truth** — Canonical repository path.

Optional entries (e.g., review cadence) may follow after the required quartet.

### 2.2 Structure and tone

- Use a single H1 for the document title, H2 for major sections, and H3 for subsections; avoid
  skipping heading levels and reserve deeper nesting for appendices only.
- Open with a short purpose statement to orient readers.
- Employ Title Case for headings unless referencing literals such as configuration keys.
- Provide context around stale data or legacy artefacts so the reader understands currency.
- Mirror bilingual phrasing conventions already present in `docs/config_reference.md` (translation in
  parentheses directly after the English text).

### 2.3 Formatting conventions

- Wrap commands, file paths, and configuration keys in backticks; prefer fenced code blocks with
  explicit language hints for multi-line examples.
- Use hanging indents for wrapped commands, spaces (no tabs) inside code fences, and descriptive
  placeholders such as `<output_root>`.
- Bullet lists use hyphens; ordered lists restart at `1.` for clarity; tables include header
  separators and specify units or clarifying notes.
- Use blockquote callouts (`>`) for metadata, warnings, or contextual notes.

### 2.4 Terminology alignment

Adopt and maintain the following terms:

| Preferred term | Usage guidance |
| -------------- | -------------- |
| LeadLag-signature RL project | Canonical project name; avoid variants such as "Lead Lag". |
| ExperimentOrchestrator | Treat as a proper noun when referencing orchestration components. |
| multi-seed | Lower case, hyphenated, including noun forms. |
| Hydra configuration | Capitalise Hydra; describe YAML entries as "scenario descriptors". |
| Kaggle orchestration guide | Reference `docs/deployment/kaggle_setup.md` when pointing to Kaggle resources. |
| Observability hooks | Lowercase plural, matching `docs/metrics_dictionary.md`. |

### 2.5 Governance & review cadence

- Documents under `reports/` refresh after each experimental campaign; metadata must reflect the
  latest revision date and campaign ID.
- Audit artefacts (`docs/audit/`) record phase owners and acceptance criteria inside their metadata
  block.
- When retiring content, add `Status: Deprecated` and link to the superseding resource.

## 3. Proposed hierarchy alignment

The table below consolidates the current folder inventory and the proposed target layout that
accompanies the standards draft.

```text
docs/
├── ablation_guide.md
├── advanced_research_plan.md
├── audit/
│   ├── phase-0/
│   ├── phase-1/
│   └── … (phases 2–12)
├── config_reference.md
├── data_preprocessing.md
├── deployment/
├── documentation_inventory.md
├── evaluation_visuals.md
├── future_roadmap.pseudo
├── kaggle_artifacts.md
├── metrics_dictionary.md
├── repro.md
├── standards.md
└── standards_feedback_request.md

reports/
├── appendix.md
├── final_report.md
└── final_report.pdf
```

### 3.1 Target layout summary

| Proposed path | Purpose & contents | Migration notes |
| ------------- | ------------------ | --------------- |
| `docs/guides/` | Step-by-step guides, runbooks, onboarding material. | Move `ablation_guide.md`, `repro.md`, `data_preprocessing.md`; create `docs/guides/README.md`. |
| `docs/reference/` | Configuration and terminology references. | Move `config_reference.md`, `metrics_dictionary.md`; align tables with reference format. |
| `docs/roadmap/` | Forward-looking plans and decision records. | Move `future_roadmap.pseudo` (convert to Markdown), `advanced_research_plan.md`, contents of `docs/adr/`. |
| `docs/audit/` | Retained structure for phase-based audit logs. | Add index linking phases. |
| `docs/deployment/` | Retained for environment-specific runbooks. | Ensure metadata compliance only. |
| `docs/inventory/` | Meta-documentation, standards, governance. | Move `documentation_inventory.md`, `standards.md`, `standards_feedback_request.md`; rename `documentation_inventory.md` to `index.md`. |
| `docs/archives/` | Deprecated or superseded artefacts. | Apply `Status: Deprecated` metadata when relocating files. |
| `reports/current/` | Active campaign reports and appendices. | Move `final_report.md`, `appendix.md`, `final_report.pdf`; add README with campaign identifiers. |
| `reports/archives/` | Historical report sets. | Adopt naming convention `YYYY-QX-<descriptor>.md` for archived files. |

### 3.2 Migration checklist

- Update relative links for all relocated documents.
- Adjust generation scripts (e.g., `reporting/generate_report.py`) to reflect new paths.
- Annotate archived assets with `Status: Deprecated` and link to replacement locations.
- Introduce index pages (`README.md` or `index.md`) where new top-level directories are created.

## 4. Approval required

Stakeholders are asked to review the standards and hierarchy together and provide one of the
following responses by **2025-11-07**:

- ✅ **Approve** — Ready to proceed with documentation updates and file migrations.
- ✏️ **Request changes** — Identify gaps, blockers, or clarifications needed before adoption.
- 🕒 **Needs more time** — Indicate when feedback will be available if the deadline cannot be met.

## 5. Next steps after approval

1. Publish updated standards in `docs/standards.md` (adjusting language if reviewers requested
   tweaks).
2. Execute the directory migration plan, updating links, scripts, and metadata.
3. Archive superseded content and add deprecation notes where appropriate.
4. Announce the completed changes in the documentation changelog and engineering weekly update.

For questions or clarifications, contact the Documentation Working Group in the `#docs-wg` Slack
channel.
