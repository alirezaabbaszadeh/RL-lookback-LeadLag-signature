# Repository Audit — 2025-02-14

## Hydra + Packaging Status
- `leadlag.hydra_main` and `leadlag.pipelines.run_full_suite` both point Hydra at the in-package config tree, keeping source and wheel execution aligned.【F:src/leadlag/hydra_main.py†L474-L512】【F:src/leadlag/pipelines/run_full_suite.py†L989-L1016】
- The canonical defaults live in `src/leadlag/configs/config.yaml`, matching the roadmap groups discovered by the tree scan.【F:src/leadlag/configs/config.yaml†L1-L45】
- A direct compose smoke test passes against the packaged configs, confirming override coverage for PPO + signature lead-lag with GPU hardware.【cf403d†L1-L19】
- Wheel build + install succeeds; the installed distribution exposes 41 YAML configs and the module entry point runs cleanly via `python -m leadlag.pipelines.run_full_suite --cfg job`.【d5853e†L1-L152】【28c002†L1-L13】【0fb01d†L1-L96】

## Scientific + Reporting Components Present
- Purged walk-forward splitting (with embargo) is implemented and covered by targeted leakage guards in `tests/cv/test_purged.py`.【F:src/leadlag/cv/purged.py†L1-L113】【F:tests/cv/test_purged.py†L1-L71】
- The synthetic trading environment enforces t→t+1 fills, configurable costs/slippage, turnover/exposure accounting, and has regression tests for cost scaling and long-only constraints.【F:src/leadlag/env/trading_env.py†L1-L134】【F:tests/test_trading_env.py†L1-L200】
- Metrics ingestion/reporting relies on the canonical `MetricsWriter` schema, and the stats CLI already emits HAC intervals, PSR/DSR, SPA, and MCS artifacts with test coverage for the CLI workflow.【F:src/leadlag/reporting/metrics_writer.py†L1-L135】【F:src/leadlag/eval/stats_cli.py†L73-L180】【F:tests/test_main_cli.py†L1-L115】

## Follow-up Task Queue (derived from audit)
1. **[B2] Add a no-peek timing guard** — the feature stack builder currently derives predictors from raw returns without asserting that feature timestamps trail decision times; introduce an explicit alignment check plus a failing synthetic test that flips ordering to prove the guard.【F:src/leadlag/pipelines/run_full_suite.py†L278-L335】
2. **[G1] One-click reproduce script** — no `scripts/reproduce_*.sh` exists yet; add `scripts/reproduce_all.sh` to orchestrate the Kaggle paper run (full suite + stats CLI + artifact collation).【2741f8†L1-L2】
3. **[G2] Anonymous packaging workflow** — there is no `scripts/pack_openreview.sh`; implement the packaging script and companion README so Kaggle/CI can produce the review-safe ZIP payload.【e52d8f†L1-L2】

Documenting these gaps keeps the planning backlog anchored in the current code state while highlighting the next concrete deliverables for the Kaggle wheel path.
