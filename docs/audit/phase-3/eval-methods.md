# Phase 3 — Evaluation Methodology & Statistical Rigor

Design
- Multi-seed (N ≥ 3) per scenario and per ablation point.
- Bootstrap CIs (block bootstrap desirable for time-correlated series) for key metrics.
- Welch tests across scenarios; report raw p-values and FDR-adjusted q-values (Benjamini–Hochberg).
- Report effect sizes (Cohen's d) and, where feasible, power estimates for primary claims.

Implementation
- Long-form pairwise significance with BH/q-values and Cohen's d is produced by `evaluation/aggregate.py` into
  `results/aggregate/significance_<metric>_pairs.csv`.
- Matrix form of p-values remains available at `results/aggregate/significance_<metric>.csv` for quick visual checks.
- A summary tool (`scripts/audit/eval_summary.py`) collects all pairs and writes reviewer-ready CSV under
  `docs/audit/phase-3/significant_pairs.csv` (q<0.05 by default).

Artifacts
- Per-scenario aggregates: `*_aggregate/stats.csv`, `*_aggregate/significance.csv` (bootstrap CI summary).
- Cross-scenario significance: `results/aggregate/significance_<metric>.csv` and `significance_<metric>_pairs.csv`.
- Comparison table/plots: `evaluation/aggregate_comparison.csv` and `evaluation/plots/*` when available.

Acceptance criteria
- CIs present; raw p-values and q-values reported; effect sizes (Cohen's d) included in pairwise outputs.
- (Optional) Power ≥ 0.8 for key hypotheses at chosen alpha.
