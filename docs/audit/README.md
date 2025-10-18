Audit Program (Reviewer‑Grade)

This folder organizes a multi‑phase audit aimed at surfacing reviewer‑grade
gaps, risks, and improvements across data, evaluation, ablations, RL methods,
reproducibility, observability, governance, and packaging.

- Each phase lives under `docs/audit/phase-<n>/` with:
  - `tasks.yaml` — live task list with status and acceptance criteria
  - `report.md` — findings, evidence, and recommendations
- A consolidated index is kept in `docs/audit/TASK_INDEX.md`.
- Use `python scripts/audit/list_tasks.py` to print status across phases.

