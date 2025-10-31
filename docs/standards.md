# Documentation Standards

> **Metadata**
> - Last updated: 2025-10-31
> - Maintainer: Documentation Working Group
> - Status: Draft for feedback
> - Source of truth: `docs/standards.md`

These standards establish a common style for all narrative documents under `docs/`, `reports/`,
`reporting/`, and other knowledge-sharing folders. They reflect conventions observed in
`docs/ablation_guide.md`, `docs/config_reference.md`, `docs/documentation_inventory.md`, and
`reports/final_report.md`, and capture desired adjustments to close gaps (for example, ensuring every
file begins with an explicit heading and metadata block).

## Required metadata block

Every document must include a metadata block directly under the H1 title using the callout pattern
above. The block must list the following keys in this exact order:

1. **Last updated** — ISO-8601 date (`YYYY-MM-DD`). Update whenever the substance of the document
   changes.
2. **Maintainer** — Team, working group, or individual accountable for accuracy.
3. **Status** — One of `Draft`, `In review`, or `Published`.
4. **Source of truth** — Relative path to the canonical location if the content is mirrored.

Optional fields such as “Review cadence” or “Related issues” may follow, but the required quartet must
always be present.

## Heading hierarchy

- Use a single `#` H1 for the document title (e.g., `# Reproducibility Guide`).
- Use `##` H2 sections for top-level topics (Environment, Methodology, Run Metadata, etc.).
- Use `###` H3 for subtopics (e.g., individual scenarios, audit phases).
- Avoid skipping levels (no `###` immediately below an H1) and reserve `####` only when absolutely
  necessary for deeply nested tables or appendices.
- Keep titles in Title Case unless the heading is a literal config key or CLI flag.

## Content and narrative conventions

- Lead with a one- or two-sentence summary explaining the purpose of the document.
- When referencing repository paths or commands, wrap them in backticks (e.g.,
  ``python hydra_main.py``) as demonstrated in `docs/config_reference.md`.
- Provide context for legacy or stale artifacts (e.g., call out that `reports/final_report.md` reflects
  2025 experiments and needs refreshing) so readers understand currency.
- For bilingual content, place the non-English translation immediately after the English phrase in
  parentheses, mirroring the format already used in `docs/config_reference.md`.

## Code and command formatting

- Use fenced code blocks with explicit language hints (for example, ```` ```bash ```` for shell
  commands, ```` ```python ```` for Python snippets, and ```` ```text ```` when displaying directory
  trees).
- Keep inline commands short; use multiline blocks for anything longer than one command or when
  showing sample output.
- Prefer descriptive placeholders (e.g., `<output_root>`) instead of ellipses for values readers must
  supply.
- Align wrapped commands using a hanging indent (four spaces) to preserve readability in plain text.
- Avoid mixing tabs and spaces; use spaces exclusively within Markdown code blocks.

## Lists, tables, and callouts

- Bullet lists should use hyphens (`-`) with two-space hanging indents when text wraps, matching
  `docs/ablation_guide.md`.
- Ordered procedures should be numbered lists starting at `1.` even if Markdown renders all numbers
  identically.
- Tables must include header separators (`| --- |`) and align with existing sizing conventions. Provide
  units or clarifying text in the Description column where applicable.
- Use blockquote callouts (`>`) for metadata, warnings, or context-sensitive notes instead of bolded
  inline labels.

## Approved terminology

Use the following preferred terms consistently:

- **LeadLag-signature RL project** — canonical project name (avoid "Lead Lag" or "Lead-Lag" variants).
- **ExperimentOrchestrator** — capitalised as a proper noun when referencing the orchestration
  pipeline.
- **multi-seed** — hyphenated, lower case, for both adjective and noun forms.
- **Hydra configuration** — capitalise Hydra; refer to YAML descriptors as "scenario descriptors".
- **Kaggle orchestration guide** — reference the document `docs/deployment/kaggle_setup.md` instead of
  inventing new labels.
- **Observability hooks** — plural with lowercase, matching `docs/metrics_dictionary.md` usage.

If new terminology is introduced, add it to this section to prevent drift across documents.

## Cross-references and linking

- Prefer relative links (e.g., `[Ablation guide](ablation_guide.md)`) when connecting documents within
  the same folder.
- When referencing material outside the repository (e.g., MLflow dashboards), include the access path
  and authentication requirements in parentheses.
- For generated artifacts like `reports/final_report.md`, add a note indicating the generation script
  (e.g., `reporting/generate_report.py`) and data vintage.

## Review and governance

- Documents in `reports/` must be revisited whenever a new experimental campaign concludes; update the
  metadata block to reflect the revision date and campaign identifier.
- Repository-wide audits (under `docs/audit/`) should note the phase owner and acceptance criteria link
  within the metadata block.
- Retire or archive files by adding a `Status: Deprecated` annotation in the metadata and linking to
  the replacement resource.

## Feedback workflow

- Proposed changes should be discussed via the shared tracking issue documented in
  `docs/standards_feedback_request.md` before merging substantial revisions.
- Major updates (new sections, renamed headings, expanded terminology) require approval from the
  Documentation Working Group.
- Minor fixes (typos, link updates) may merge with a single reviewer sign-off, but still require the
  metadata “Last updated” field to be refreshed.
