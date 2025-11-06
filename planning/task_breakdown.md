# Detailed Task Breakdown for Hydra Packaging and Related Enhancements

This document captures the structured breakdown for the upcoming refactor, mirroring the scope requested in the latest roadmap update. Each phase (P0–P6) lists explicit file artefacts, commands, and acceptance checks so that individual PRs can stay vertical and reproducible.

## P0 — Hydra / Packaging Blockers

### P0-T1: Package the Hydra Config Tree
- **File tree**
  ```
  src/leadlag/configs/
    config.yaml
    agent/
      ppo.yaml
      dqn.yaml
      a2c.yaml
      sac.yaml
      td3.yaml
    data/
      sp500_sector.yaml
      crypto_top.yaml
    features/
      base.yaml
      signature.yaml
      leadlag.yaml
      signature_leadlag.yaml
    split/
      walk_forward_purged.yaml
    training/
      base.yaml
      smoke.yaml
      paper.yaml
    hardware/
      gpu.yaml
      auto.yaml
    reporting/
      base.yaml
    rewards/
      default.yaml
    scenario/
      fixed_30.yaml
      fixed_90.yaml
      dynamic_adaptive.yaml
      rl_ppo.yaml
  ```
- **Seed config**: `src/leadlag/configs/config.yaml`
  ```yaml
  defaults:
    - agent: ppo
    - data: sp500_sector
    - features: base
    - split: walk_forward_purged
    - training: smoke
    - hardware: gpu
    - _self_
  env:
    action_space: discrete3
  policy:
    name: mlp
  window:
    lookback: 128
  target:
    horizon: 1
  costs:
    fee_bps: 1
  slippage:
    bps: 2
  logging:
    run_id: ${now:%Y%m%d}-${agent.name}-${hydra.job.num}
  ```
- **Acceptance**: All defaults compose without error and group overrides resolve (see P0-T5 test guard).

### P0-T2: Include YAMLs in Distribution Artifacts
- **`pyproject.toml` adjustments**
  ```toml
  [build-system]
  requires = ["setuptools>=68", "wheel", "build"]
  build-backend = "setuptools.build_meta"

  [tool.setuptools]
  include-package-data = true

  [tool.setuptools.package-data]
  leadlag = ["configs/**/*.yaml"]

  [tool.setuptools.packages.find]
  where = ["src"]
  ```
- **Optional `MANIFEST.in`**
  ```
  recursive-include src/leadlag/configs *.yaml
  ```
- **Local smoke**
  ```bash
  python -m build
  pip install -U dist/*.whl
  python - <<'PY'
  import importlib.resources as ir
  p = ir.files('leadlag').joinpath('configs')
  ys = list(p.rglob('*.yaml'))
  print('yaml_count', len(ys))
  assert len(ys) >= 10
  PY
  ```

### P0-T3: Single Hydra Entry Point
- **Module**: `src/leadlag/pipelines/run_full_suite.py`
  ```python
  import hydra
  from omegaconf import DictConfig
  from hydra.core.hydra_config import HydraConfig

  @hydra.main(version_base=None, config_path="../configs", config_name="config")
  def main(cfg: DictConfig):
      print("Config sources:", [s.provider for s in HydraConfig.get().runtime.config_sources])
      # TODO: orchestrate data → features → training → evaluation → artifacts

  if __name__ == "__main__":
      main()
  ```
- **Acceptance**
  ```bash
  python -m leadlag.pipelines.run_full_suite --cfg job
  ```
  Should start without `ConfigNotFoundError` and print composed defaults.

### P0-T4: Remove Stale `configs/` References
- **Command**
  ```bash
  rg -n "configs/" -S | rg -v "src/leadlag/configs"
  ```
- Replace repo-root lookups with packaged paths and adjust any `initialize_config_dir` usage accordingly.
- **Acceptance**: Command returns no hits; smoke tests remain green.

### P0-T5: Compose & Wheel Smoke Tests
- **Test guard**: `tests/test_hydra_compose.py`
  ```python
  import pytest
  pytest.importorskip("hydra")
  from hydra import initialize, compose

  def test_defaults_compose():
      with initialize(version_base=None, config_path="../src/leadlag/configs"):
          cfg = compose(config_name="config")
      assert cfg.agent is not None

  def test_common_overrides():
      with initialize(version_base=None, config_path="../src/leadlag/configs"):
          cfg = compose(
              config_name="config",
              overrides=[
                  "agent=ppo",
                  "training=smoke",
                  "hardware=gpu",
                  "data=sp500_sector",
                  "split=walk_forward_purged",
                  "features=signature_leadlag",
              ],
          )
      assert cfg.features is not None
  ```
- **CI snippet**
  ```yaml
  - name: Build wheel & compose smoke
    run: |
      python -m build
      pip install -U dist/*.whl
      python -m pytest -q tests/test_hydra_compose.py
      python -m leadlag.pipelines.run_full_suite --cfg job
  ```
- **Acceptance**: CI succeeds; entry point runnable from the wheel.

## P1 — Leakage-Free Splits (Nested-ready)

### P1-T1: Purged Walk-Forward Splitter
- **Module**: `src/leadlag/cv/purged.py`
  ```python
  from dataclasses import dataclass
  import numpy as np
  from typing import Iterator, Tuple

  @dataclass
  class WalkForwardPurged:
      n_splits: int = 6
      embargo_frac: float = 0.01

      def split(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
          fold = n // self.n_splits
          embargo = max(0, int(self.embargo_frac * n))
          for k in range(self.n_splits):
              t0 = k * fold
              t1 = n if k == self.n_splits - 1 else (k + 1) * fold
              mask = np.ones(n, dtype=bool)
              mask[max(0, t0 - embargo):min(n, t1 + embargo)] = False
              train_idx = np.where(mask)[0]
              test_idx = np.arange(t0, t1)
              yield train_idx, test_idx
  ```
- Extend with `PurgedKFold` for nested tuning.

### P1-T2: Hydra Profile & Split Export
- **Config**: `src/leadlag/configs/split/walk_forward_purged.yaml`
  ```yaml
  scheme: walk_forward_purged
  n_splits: 6
  embargo_frac: 0.01
  nested_tuning:
    enabled: true
    inner_n_splits: 3
    inner_embargo_frac: 0.01
  ```
- Pipeline must log `results/<run_id>/splits.csv` with columns: `window, train_start, train_end, test_start, test_end, embargo_frac`.

### P1-T3: Tests
- **Unit test**: `tests/test_purged_cv.py`
  ```python
  import numpy as np
  from leadlag.cv.purged import WalkForwardPurged

  def test_no_overlap_with_embargo():
      N = 1000
      cv = WalkForwardPurged(5, 0.02)
      for train, test in cv.split(N):
          assert np.intersect1d(train, test).size == 0
  ```
- **Acceptance**: Guard passes; `splits.csv` generated per run.

## P2 — Trading Realism

### P2-T1: t→t+1 Execution
- Ensure fills use **next bar open**; log both signal and execution timestamps in `env/trading_env.py`.

### P2-T2: Costs & Slippage
- Apply fees/slippage per transition:
  ```python
  commission = abs(delta_pos) * exec_price * cfg.costs.fee_bps * 1e-4
  slippage = abs(delta_pos) * exec_price * cfg.slippage.bps * 1e-4
  pnl = prev_pos * (next_close - curr_close) - commission - slippage
  ```
- Provide Hydra presets for bps tiers; aggregate by evaluation window.

### P2-T3: Constraints & Metrics
- Config entries: `env.leverage_cap`, `env.allow_short`.
- Track `Turnover = sum(|Δposition|)` and `Exposure = mean(|position|)`; persist to metrics rows.
- **Tests**: synthetic uptrend verifying higher bps reduce net PnL; ensure execution uses `next_open`.

## P3 — Reporting Unification

### P3-T1: Canonical `metrics.csv`
- **Writer**: `src/leadlag/reporting/metrics_writer.py`
  ```python
  from pathlib import Path
  import pandas as pd

  SCHEMA = [
      "experiment_id",
      "agent",
      "action_space",
      "policy",
      "features_signature",
      "signature_depth",
      "features_leadlag",
      "time_channel",
      "lookback",
      "horizon",
      "universe",
      "timeframe",
      "split_scheme",
      "cost_fee_bps",
      "slippage_bps",
      "reward",
      "seed",
      "window_index",
      "Sharpe",
      "Sortino",
      "MaxDD",
      "Turnover",
      "PnL",
      "Exposure",
  ]

  def write_metrics(out_dir, rows):
      out = Path(out_dir)
      out.mkdir(parents=True, exist_ok=True)
      df = pd.DataFrame(rows)
      for col in SCHEMA:
          if col not in df:
              df[col] = None
      df = df[SCHEMA]
      df.to_csv(out / "metrics.csv", index=False)
      return df
  ```

### P3-T2: Schema Guard
- **Test**: `tests/test_metrics_schema.py`
  ```python
  import pandas as pd
  from leadlag.reporting.metrics_writer import SCHEMA

  def test_schema_columns(tmp_path):
      csv_path = tmp_path / "metrics.csv"
      pd.DataFrame([{col: None for col in SCHEMA}]).to_csv(csv_path, index=False)
      df = pd.read_csv(csv_path)
      assert list(df.columns) == SCHEMA
  ```

### P3-T3: Aggregator Script
- **Script**: `scripts/aggregate_metrics.py`
  ```python
  import glob
  import os
  import pandas as pd

  rows = []
  for path in glob.glob("/kaggle/working/results/*/metrics.csv"):
      df = pd.read_csv(path)
      df["run_dir"] = os.path.dirname(path)
      rows.append(df)
  pd.concat(rows, ignore_index=True).to_csv(
      "/kaggle/working/paper_outputs/all_metrics_raw.csv", index=False
  )
  ```
- **Acceptance**: Every run writes a valid `metrics.csv`; aggregation yields `all_metrics_raw.csv`.

## P4 — Fairness & Reproducibility

### P4-T1: Seed Utility
- **Module**: `src/leadlag/utils/repro.py`
  ```python
  import os
  import random
  import numpy as np

  def set_all_seeds(seed=0, cudnn_deterministic=True):
      os.environ["PYTHONHASHSEED"] = str(seed)
      random.seed(seed)
      np.random.seed(seed)
      try:
          import torch

          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = cudnn_deterministic
          torch.backends.cudnn.benchmark = not cudnn_deterministic
      except Exception:
          pass
  ```

### P4-T2: Training Budget Manifest
- Enforce `training.total_env_steps` across agents.
- **Utility**: `src/leadlag/utils/manifest.py`
  ```python
  import json
  import subprocess
  import sys

  def write_manifest(path, seed, env_steps_reported, env_steps_actual, extra=None):
      commit = subprocess.getoutput("git rev-parse --short HEAD") or "unknown"
      info = {
          "commit": commit,
          "seed": seed,
          "env_steps_reported": env_steps_reported,
          "env_steps_actual": env_steps_actual,
          "python": sys.version,
      }
      if extra:
          info.update(extra)
      with open(path, "w", encoding="utf-8") as fh:
          json.dump(info, fh, indent=2)
  ```

### P4-T3: Hydra Profiles for Hardware & Training
- **`hardware/gpu.yaml`**
  ```yaml
  device: cuda
  mixed_precision: amp
  n_envs: 8
  num_workers: 4
  pin_memory: true
  ```
- **`training/smoke.yaml`**
  ```yaml
  total_env_steps: 100000
  seeds: [0, 1]
  windows: 2
  ```
- **`training/paper.yaml`**
  ```yaml
  total_env_steps: 500000
  seeds: [0, 1, 2, 3, 4, 5, 6]
  windows: 6
  ```
- **Acceptance**: Logs show equal budgets; each run writes `run_manifest.json`.

## P5 — Stats, Calibration, Multi-seed

### P5-T1: Statistics Library & CLI
- **Library**: `src/leadlag/eval/stats.py`
  ```python
  import numpy as np
  import pandas as pd

  def sharpe(returns, rf=0.0): ...
  def sortino(returns, rf=0.0): ...
  def max_drawdown(equity): ...
  def hac_confint_sharpe(returns, lags=None): ...
  def psr_dsr(returns): ...
  def spa_reality_check(table_of_utilities, benchmark_col): ...
  def mcs(models_metrics_df, alpha=0.1): ...
  ```
- **CLI**: `src/leadlag/eval/stats_cli.py` summarising metrics into `/paper_outputs/` artefacts.

### P5-T2: Calibration CLI
- **Module**: `src/leadlag/eval/calibration_cli.py`
  - Computes CRPS, pinball losses, PIT histogram, prediction interval coverage.
  - Outputs `calibration.csv`, `pit_hist.png`, `coverage_plot.png`.

### P5-T3: Multi-seed Aggregation
- Run multi-seed orchestration; consolidate into `/paper_outputs/aggregate.csv` with mean/std/CI per metric.
- **Acceptance**: Stats & calibration outputs materialise alongside summary markdown (`paper_results.md`).

## P6 — One-click Reproduction & Packaging

### P6-T1: End-to-End Reproduce Script
- **Script**: `scripts/reproduce_all.sh`
  ```bash
  set -euo pipefail
  ENTRY="python -m leadlag.pipelines.run_full_suite"
  OUT="/kaggle/working/paper_outputs"
  RES="/kaggle/working/results"
  $ENTRY training=paper hardware=gpu split=walk_forward_purged
  python -m leadlag.eval.stats_cli --results "$RES" --out "$OUT" --benchmark Sharpe
  python -m leadlag.eval.calibration_cli --results "$RES" --out "$OUT"
  python scripts/aggregate_metrics.py
  ```

### P6-T2: Anonymous Packaging
- **Script**: `scripts/pack_openreview.sh`
  ```bash
  set -euo pipefail
  ZIP="artifact_anonymous.zip"
  ROOT="$(git rev-parse --show-toplevel)"
  find "$ROOT" -name ".git" -prune -o -print | grep -vE "(names|emails|affiliations)" >/dev/null
  zip -Xr "$ZIP" src/leadlag configs scripts README_ANON.md LICENSE
  ```
- Include `README_ANON.md` with anonymised run steps; ensure zip ≤ 100 MB.

### P6-T3: Kaggle Anonymous Path
- Notebook flow:
  1. Upload `artifact_anonymous.zip` as dataset.
  2. Install via `pip install artifact_anonymous.zip` or `pip install -e .` after unzip.
  3. Execute `scripts/reproduce_all.sh`.
- Outputs land in `/kaggle/working/paper_outputs/`.

## Global PR Checklist (Repeat in Each PR)
- [ ] Wheel smoke: `python -m build && pip install dist/*.whl`
- [ ] Entry point: `python -m leadlag.pipelines.run_full_suite --cfg job`
- [ ] Tests updated: `pytest -q`
- [ ] Artifacts captured in `${hydra:run.dir}` or `/kaggle/working/...`
- [ ] Documentation refreshed where relevant
- [ ] `rg "configs/"` clean of repo-root references

## Risk Guards for CI
- Leakage guard (no overlap with embargo)
- Budget guard (equal `total_env_steps`)
- Latency/cost guard (t→t+1 + bps sensitivity)
- Schema guard (`metrics.csv` columns)
- Wheel guard (build→install→entry point)

## Optional Enhancements
- Baseline ARIMA/ETS modules with Hydra presets
- HTML report consolidating metrics & figures

