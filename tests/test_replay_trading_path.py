import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from leadlag.pipelines.run_full_suite import _replay_trading_path


def _build_prices(base_returns: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=base_returns.size + 1, freq="D")
    levels = [100.0]
    for ret in base_returns:
        levels.append(levels[-1] * (1.0 + float(ret)))
    values = np.column_stack([levels, levels])
    return pd.DataFrame(values, index=index, columns=["AssetA", "AssetB"])


def test_replay_trading_path_uses_recorded_positions():
    base_returns = np.array([0.01, -0.02, 0.015, 0.0], dtype=float)
    prices = _build_prices(base_returns)
    history_index = prices.index[1:]
    history = pd.DataFrame(
        {
            "lookback": [10, 12, 18, 20],
            "trading_signal": [-1.0, -0.5, 0.5, 1.0],
        },
        index=history_index,
    )

    cfg = OmegaConf.create(
        {
            "env": {"max_abs_position": 1.0, "allow_short": True, "initial_position": 0.0},
            "costs": {"fee_bps": 10.0},
            "slippage": {"bps": 0.0},
        }
    )

    realized = _replay_trading_path(cfg, prices, history)
    assert realized is not None

    expected_positions = pd.Series([-1.0, -0.5, 0.5, 1.0], index=history_index, dtype=float)
    expected_trades = expected_positions.diff().fillna(expected_positions.iloc[0]).astype(float)
    cost_rate = (10.0 + 0.0) / 10000.0
    expected_costs = expected_trades.abs() * cost_rate
    expected_returns = expected_positions.values * base_returns - expected_costs.values

    expected_positions.name = realized.positions.name
    expected_trades.name = realized.trades.name
    expected_costs.name = realized.costs.name

    pd.testing.assert_series_equal(realized.positions, expected_positions)
    pd.testing.assert_series_equal(realized.trades, expected_trades)
    pd.testing.assert_series_equal(realized.costs, expected_costs)
    np.testing.assert_allclose(realized.returns.values, expected_returns)

    metrics = realized.metrics
    assert metrics.env_steps == expected_positions.size
    np.testing.assert_allclose(metrics.turnover, expected_trades.abs().mean())
    np.testing.assert_allclose(metrics.costs, expected_costs.sum())
    np.testing.assert_allclose(metrics.pnl, expected_returns.sum())
