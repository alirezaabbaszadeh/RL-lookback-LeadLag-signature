from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from leadlag.env.trading_env import SyntheticTradingEnvironment


def _compose_config(tmp_path) -> DictConfig:
    config_dir = (
        Path(__file__).resolve().parents[1] / "src" / "leadlag" / "configs"
    )
    with initialize_config_dir(config_dir=str(config_dir), job_name="test-trading-env", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"results_root={tmp_path / 'results'}",
                "training.total_env_steps=16",
                "training.seeds=[0]",
                "training.windows=1",
                "hardware.n_envs=1",
                "costs.fee_bps=0",
                "slippage.bps=0",
            ],
        )
    return cfg


def test_t_plus_one_execution(tmp_path):
    env = SyntheticTradingEnvironment(
        lookback=8,
        horizon=4,
        fee_bps=0.0,
        slippage_bps=0.0,
        seed=123,
    )
    base_returns = np.array([0.05, 0.04, 0.01], dtype=float)
    actions = np.array([1.0, -1.0, 0.0], dtype=float)

    realised = env.simulate_returns(
        total_steps=3,
        actions=actions,
        base_returns=base_returns,
        initial_position=0.0,
    )

    assert realised[0] == pytest.approx(0.0)
    # Position switches to +1 for the second return, then -1 for the third
    assert realised[1] == pytest.approx(base_returns[1])
    assert realised[2] == pytest.approx(-base_returns[2])

    metrics = env.last_metrics[0]
    assert metrics.env_steps == 3
    assert metrics.pnl == pytest.approx(realised.sum())
    assert metrics.costs == pytest.approx(0.0)


def test_commission_and_slippage_costs(tmp_path):
    env = SyntheticTradingEnvironment(
        lookback=8,
        horizon=4,
        fee_bps=50.0,
        slippage_bps=50.0,
        seed=456,
    )
    base_returns = np.array([0.02, 0.03], dtype=float)
    actions = np.array([1.0, 0.0], dtype=float)

    realised = env.simulate_returns(
        total_steps=2,
        actions=actions,
        base_returns=base_returns,
        initial_position=0.0,
    )

    # No return on first step, second step incurs costs for moving to +1
    expected_cost = (env.fee_bps + env.slippage_bps) / 10000.0
    assert realised[0] == pytest.approx(0.0)
    assert realised[1] == pytest.approx(base_returns[1] - expected_cost)

    metrics = env.last_metrics[0]
    assert metrics.costs == pytest.approx(expected_cost)
    assert metrics.turnover > 0.0


def test_hydra_fee_knobs_affect_pnl(tmp_path):
    cfg = _compose_config(tmp_path)

    steps = 6
    base_returns = np.full(steps, 0.02, dtype=float)
    actions = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)

    low_cost_env = SyntheticTradingEnvironment(
        lookback=int(cfg.window.lookback),
        horizon=int(cfg.target.horizon),
        fee_bps=float(cfg.costs.fee_bps),
        slippage_bps=float(cfg.slippage.bps),
        seed=789,
        max_abs_position=float(cfg.env.max_abs_position),
        initial_position=float(cfg.env.initial_position),
    )
    high_cost_env = SyntheticTradingEnvironment(
        lookback=int(cfg.window.lookback),
        horizon=int(cfg.target.horizon),
        fee_bps=25.0,
        slippage_bps=25.0,
        seed=789,
        max_abs_position=float(cfg.env.max_abs_position),
        initial_position=float(cfg.env.initial_position),
    )

    low_returns = low_cost_env.simulate_returns(
        total_steps=steps,
        actions=actions,
        base_returns=base_returns,
        initial_position=0.0,
    )
    high_returns = high_cost_env.simulate_returns(
        total_steps=steps,
        actions=actions,
        base_returns=base_returns,
        initial_position=0.0,
    )

    assert high_returns.sum() < low_returns.sum()
    assert high_cost_env.last_metrics[0].costs > low_cost_env.last_metrics[0].costs


def test_long_only_environment_clips_negative_positions():
    env = SyntheticTradingEnvironment(
        lookback=8,
        horizon=4,
        fee_bps=0.0,
        slippage_bps=0.0,
        seed=321,
        max_abs_position=2.0,
        initial_position=-1.0,
        allow_short=False,
    )

    actions = np.array([-2.0, -1.0, 0.5, 1.5], dtype=float)
    env.simulate_returns(
        total_steps=actions.size,
        actions=actions,
        base_returns=np.zeros(actions.size, dtype=float),
        initial_position=-0.5,
    )
    trade_path = env.last_paths[0]
    assert np.all(trade_path.positions >= 0.0)

    env.simulate_returns(total_steps=5)
    random_path = env.last_paths[0]
    assert np.all(random_path.positions >= 0.0)
