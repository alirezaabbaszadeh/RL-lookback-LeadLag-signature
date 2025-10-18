# Phase 2 — Data Integrity & Leakage (Plan)

Goals
- Ensure no look-ahead, sample leakage, or target leakage across the pipeline.

Planned probes
- Placebo test: shuffle future windows; expect metric degradation vs. true chronology.
- Walk-forward verification: enforce monotone time indices and past-only windows.
- CCF-at-lag policy: confirm lag handling never consumes future information.

Evidence to collect
- Side-by-side metrics for placebo vs. normal runs.
- Snippets of window construction code paths and time index checks.
- Unit tests asserting monotone indices and no future access.

Acceptance criteria
- Placebo tests significantly degrade key metrics.
- All windows are past-only; no future timestamps consumed.

Results (initial)
- Summary saved at `docs/audit/phase-2/leakage_probe_summary.csv` comparing control vs placebo on stability and signal metrics.
  Use `scripts/audit/leakage_probes.py` to regenerate on demand.
- Walk-forward check saved at `docs/audit/phase-2/walk_forward_check.csv` (pivot equality between full vs truncated runs; expected deltas near zero).
