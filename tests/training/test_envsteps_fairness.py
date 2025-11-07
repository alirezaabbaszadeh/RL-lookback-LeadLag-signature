from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from leadlag.eval.stats import PerformanceSummary, compute_equity_curve
from leadlag.env.trading_env import TradeMetrics
from leadlag.pipelines import run_full_suite
from leadlag.pipelines.run_full_suite import ENV_STEP_TOLERANCE
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row


def _base_cfg(results_root: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "results_root": str(results_root),
            "paper_outputs_root": str(results_root / "paper"),
            "logging": {"run_id": "fairness", "append_seed_window": False},
            "hardware": {"n_envs": 1},
            "training": {"total_env_steps": 12, "periods_per_year": 252},
            "window": {"lookback": 4},
            "features": {},
            "agent": {"name": "random"},
            "data": {},
            "split": {},
        }
    )


def _simulation_payload(returns: pd.Series, env_steps: int, requested_steps: int) -> dict[str, object]:
    summary = PerformanceSummary(
        sharpe=0.5,
        sortino=0.4,
        max_drawdown=-0.1,
        pnl=float(returns.sum()),
        returns=returns,
    )
    equity = compute_equity_curve(summary.returns)
    trade_metrics = TradeMetrics(pnl=summary.pnl, turnover=0.1, exposure=0.2, env_steps=env_steps, costs=0.01)
    history = pd.DataFrame({"reward": returns.to_numpy(dtype=float)})
    prices = pd.DataFrame(
        {"AssetA": np.linspace(1.0, 2.0, num=len(returns) + 1)},
        index=pd.date_range("2024-01-01", periods=len(returns) + 1, freq="D"),
    )

    return {
        "returns": summary.returns,
        "equity": equity,
        "summary": summary,
        "trade_metrics": trade_metrics,
        "positions": None,
        "trades": None,
        "cost_series": None,
        "reward_returns": summary.returns,
        "reward_trade_metrics": trade_metrics,
        "history": history,
        "env_steps": env_steps,
        "requested_env_steps": requested_steps,
        "n_envs": 1,
        "prices": prices,
        "dataset_path": None,
        "feature_stack": {},
        "feature_frame": pd.DataFrame(),
        "feature_time_meta": {},
        "agent_info": {"name": "stub"},
        "dataset_length": len(prices),
    }


def test_envsteps_manifest_matches_requested(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    cfg = _base_cfg(results_root)
    results_root.mkdir(parents=True)

    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))
    returns = pd.Series([0.01, -0.005, 0.007], dtype=float)
    simulation = _simulation_payload(returns, env_steps=12, requested_steps=12)

    run_dir1 = run_full_suite._prepare_directories(results_root, "run-001")
    row = build_metadata_row(
        run_dir1.name,
        OmegaConf.to_container(cfg, resolve=True),
        {"sharpe": 0.5, "sortino": 0.4, "max_drawdown": -0.1, "pnl": float(returns.sum())},
        seed=1,
        window_idx=0,
        turnover=0.1,
        exposure=0.2,
        costs=0.01,
        env_steps=12,
    )
    metrics_writer.write_row(run_dir1 / "metrics.csv", row)
    run_full_suite._write_artifacts(run_dir1, cfg, seed=1, window_idx=0, simulation=dict(simulation), metrics_writer=metrics_writer)

    manifest1 = json.loads((run_dir1 / "run_manifest.json").read_text(encoding="utf-8"))
    assert abs(manifest1["env_steps_reported"] - manifest1["env_steps_actual"]) <= ENV_STEP_TOLERANCE
    assert manifest1["requested_env_steps"] == 12

    run_dir2 = run_full_suite._prepare_directories(results_root, "run-002")
    row2 = build_metadata_row(
        run_dir2.name,
        OmegaConf.to_container(cfg, resolve=True),
        {"sharpe": 0.5, "sortino": 0.4, "max_drawdown": -0.1, "pnl": float(returns.sum())},
        seed=2,
        window_idx=0,
        turnover=0.1,
        exposure=0.2,
        costs=0.01,
        env_steps=12,
    )
    metrics_writer.write_row(run_dir2 / "metrics.csv", row2)
    run_full_suite._write_artifacts(run_dir2, cfg, seed=2, window_idx=0, simulation=dict(simulation), metrics_writer=metrics_writer)
    manifest2 = json.loads((run_dir2 / "run_manifest.json").read_text(encoding="utf-8"))
    assert abs(manifest2["env_steps_reported"] - manifest2["env_steps_actual"]) <= ENV_STEP_TOLERANCE


def test_envsteps_warning_on_mismatch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    results_root = tmp_path / "results"
    cfg = _base_cfg(results_root)
    results_root.mkdir(parents=True)

    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))
    returns = pd.Series([0.02, -0.01, 0.015], dtype=float)
    simulation = _simulation_payload(returns, env_steps=8, requested_steps=20)

    run_dir = run_full_suite._prepare_directories(results_root, "run-warn")
    row = build_metadata_row(
        run_dir.name,
        OmegaConf.to_container(cfg, resolve=True),
        {"sharpe": 0.6, "sortino": 0.5, "max_drawdown": -0.2, "pnl": float(returns.sum())},
        seed=3,
        window_idx=0,
        turnover=0.1,
        exposure=0.2,
        costs=0.02,
        env_steps=8,
    )
    metrics_writer.write_row(run_dir / "metrics.csv", row)

    caplog.set_level("WARNING", logger="leadlag.pipelines.run_full_suite")
    run_full_suite._write_artifacts(run_dir, cfg, seed=3, window_idx=0, simulation=simulation, metrics_writer=metrics_writer)

    assert any("Env step mismatch" in record.getMessage() for record in caplog.records)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["env_steps_reported"] == 20
    assert manifest["env_steps_actual"] == 8
