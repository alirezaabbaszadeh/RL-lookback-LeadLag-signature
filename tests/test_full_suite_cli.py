from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from leadlag.eval import stats as stats_mod
from leadlag.pipelines import run_full_suite
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row


def _compose_config(tmp_path: Path):
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(config_dir), job_name="test-suite"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"results_root={tmp_path / 'results'}",
                f"paper_outputs_root={tmp_path / 'paper'}",
                "logging.run_id=test",
                "training.seeds=[0]",
                "training.windows=1",
                "hardware.n_envs=1",
                "training.total_env_steps=100",
            ],
        )
    return cfg


def test_simulate_episode_produces_metrics(tmp_path):
    cfg = _compose_config(tmp_path)
    run_dir = run_full_suite._prepare_directories(Path(cfg.results_root), "test-s00-w00")
    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))

    simulation = run_full_suite._simulate_episode(cfg, seed=0, window_idx=0)
    run_full_suite._write_artifacts(run_dir, cfg, 0, 0, simulation, metrics_writer)

    metrics_path = run_dir / "metrics.csv"
    equity_path = run_dir / "equity.csv"
    returns_path = run_dir / "returns.csv"
    assert metrics_path.exists()
    assert equity_path.exists()
    assert returns_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    assert not metrics_df.empty
    assert "Sharpe" in metrics_df.columns
    assert "EnvSteps" in metrics_df.columns
    assert metrics_df.loc[0, "EnvSteps"] == simulation["env_steps"]

    equity_df = pd.read_csv(equity_path)
    returns_df = pd.read_csv(returns_path)
    assert len(equity_df) == len(returns_df)


def test_build_metadata_row_matches_config(tmp_path):
    cfg = _compose_config(tmp_path)
    metrics = {"sharpe": 1.0, "sortino": 1.2, "max_drawdown": -0.1, "pnl": 0.05}
    env_steps = int(cfg.training.total_env_steps)
    row = build_metadata_row(
        "test-run",
        OmegaConf.to_container(cfg, resolve=True),
        metrics,
        seed=0,
        window_idx=0,
        env_steps=env_steps,
    )
    assert row["experiment_id"] == "test-run"
    assert row["agent"] == cfg.agent.name
    assert row["Sharpe"] == metrics["sharpe"]


def test_hac_confidence_interval_returns_bounds(tmp_path):
    cfg = _compose_config(tmp_path)
    simulation = run_full_suite._simulate_episode(cfg, seed=0, window_idx=0)
    lower, upper = stats_mod.hac_confidence_interval(
        simulation["returns"], periods_per_year=cfg.training.periods_per_year
    )
    assert lower <= upper
