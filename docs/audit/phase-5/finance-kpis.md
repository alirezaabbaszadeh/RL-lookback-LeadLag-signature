# Phase 5 — Financial KPIs & Risk Modeling

Pipeline
- Script `evaluation/finance_kpis.py` scans run directories and derives daily returns from an
  available column (prefers `portfolio_return`, `strategy_return`, `reward`, etc.; otherwise uses
  `% change` of `mean_abs_matrix` as a proxy).
- For each run, it computes:
  - Annualized return ((1 + returns).prod() ** (252/len) − 1)
  - Sharpe ratio (sqrt(252) × mean / std)
  - Max drawdown (min(cumulative / running_max − 1))
- Results written to `evaluation/finance_kpis.csv`.

Current status
- KPIs generated for existing runs (see `evaluation/finance_kpis.csv`).
- Some runs lacking adequate returns history are skipped (placebo runs show NaNs due to volatility).

Next steps
- Integrate more realistic PnL series (e.g., using RL reward or executed positions) to replace
  fallback derived from `mean_abs_matrix`.
- Surface KPIs in the reporting pipeline (appendix and final report).
- Add CI checks to ensure KPIs exist for primary research scenarios.

