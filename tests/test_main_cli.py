from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from leadlag.eval import stats_cli
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row


@pytest.fixture
def results_fixture(tmp_path):
    results_root = tmp_path / "results"
    run_dir = results_root / "demo-run"
    run_dir.mkdir(parents=True)

    returns = pd.Series([0.01, -0.005, 0.007, 0.002], dtype=float)
    returns.to_frame(name="returns").to_csv(run_dir / "returns.csv", index=False)

    cfg = {
        "agent": {"name": "ppo"},
        "env": {"action_space": "discrete3"},
        "policy": {"name": "mlp"},
        "features": {"signature": {"enabled": False, "depth": 0}, "leadlag": {"enabled": False}, "time_channel": False},
        "window": {"lookback": 128},
        "target": {"horizon": 1},
        "data": {"universe": "demo", "timeframe": "5m"},
        "split": {"scheme": "walk_forward_purged"},
        "costs": {"fee_bps": 1},
        "slippage": {"bps": 2},
        "reward": {"name": "sharpe"},
    }
    metrics_writer = MetricsWriter(cfg)
    metrics_row = build_metadata_row(
        "demo-run",
        cfg,
        {"sharpe": 1.0, "sortino": 0.9, "max_drawdown": -0.1, "pnl": returns.sum()},
        seed=0,
        window_idx=0,
        turnover=0.0,
        exposure=0.0,
        costs=0.0,
        env_steps=len(returns),
    )
    metrics_writer.write_row(run_dir / "metrics.csv", metrics_row)

    return results_root, metrics_row


def test_stats_cli_generates_expected_artifacts(tmp_path, monkeypatch, results_fixture):
    results_root, metrics_row = results_fixture
    out_dir = tmp_path / "paper"

    argv = [
        "stats-cli",
        "--results",
        str(results_root),
        "--out",
        str(out_dir),
        "--spa-iterations",
        "20",
    ]

    monkeypatch.setattr(sys, "argv", argv)
    stats_cli.main()

    expected_files = [
        out_dir / "all_metrics_raw.csv",
        out_dir / "summary_table.csv",
        out_dir / "best_per_agent.csv",
        out_dir / "advanced_metrics.csv",
        out_dir / "psr_dsr_pvalues.csv",
        out_dir / "hac_sharpe_confidence_intervals.csv",
        out_dir / "spa_results.csv",
        out_dir / "spa_pvalues.csv",
        out_dir / "mcs.json",
        out_dir / "paper_results.md",
        out_dir / "forest_hac_ci.png",
        out_dir / "heatmap_agent_timeframe.png",
    ]
    for path in expected_files:
        assert path.exists()

    with (out_dir / "mcs.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "members" in payload

    advanced = pd.read_csv(out_dir / "advanced_metrics.csv")
    assert {"run_id", "sharpe", "psr", "dsr", "hac_sharpe_lower", "hac_sharpe_upper", "n_obs"}.issubset(
        set(advanced.columns)
    )

    spa = pd.read_csv(out_dir / "spa_results.csv")
    assert {"run_id", "spa_pvalue", "spa_sup_pvalue"}.issubset(set(spa.columns))


def test_stats_workflow_function_interface(tmp_path, results_fixture):
    results_root, _ = results_fixture
    out_dir = tmp_path / "paper_fn"

    artifacts = stats_cli.run_workflow(
        results_root,
        out_dir,
        periods=252,
        spa_iterations=20,
        block_length=None,
        seed=123,
    )

    assert (out_dir / "all_metrics_raw.csv").exists()
    assert (out_dir / "paper_results.md").exists()
    assert "summary_markdown" in artifacts
