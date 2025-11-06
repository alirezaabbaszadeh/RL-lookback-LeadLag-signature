import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from leadlag.pipelines.run_full_suite import (
    _replay_trading_path,
    _summarize_trade_history,
)


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

    signals = history["trading_signal"].astype(float)
    initial_position = float(cfg.env.initial_position)
    expected_positions = signals.shift(1).fillna(initial_position).iloc[1:]
    expected_trades = expected_positions.diff().fillna(
        expected_positions.iloc[0] - initial_position
    ).astype(float)
    cost_rate = (10.0 + 0.0) / 10000.0
    price_proxy = prices.mean(axis=1)
    execution_prices = price_proxy.reindex(expected_positions.index).astype(float)
    execution_prices = execution_prices.ffill().bfill()
    expected_costs = expected_trades.abs() * execution_prices * cost_rate
    base_returns_series = prices.pct_change().mean(axis=1).fillna(0.0).iloc[1:]
    base_returns_series = base_returns_series.reindex(expected_positions.index)
    expected_returns = expected_positions.values * base_returns_series.values - expected_costs.values

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


def test_summarize_trade_history_scales_costs_with_price():
    index = pd.date_range("2022-01-01", periods=3, freq="D")
    history = pd.DataFrame(
        {
            "trading_signal": [-1.0, 0.0, 1.0],
        },
        index=index,
    )
    returns = pd.Series([0.0, 0.0, 0.0], index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "AssetA": [100.0, 102.0, 105.0],
            "AssetB": [100.0, 102.0, 105.0],
        },
        index=index,
    )

    cfg = OmegaConf.create(
        {
            "env": {
                "max_abs_position": 1.0,
                "allow_short": True,
                "initial_position": 0.0,
            },
            "costs": {"fee_bps": 25.0},
            "slippage": {"bps": 5.0},
        }
    )

    metrics = _summarize_trade_history(cfg, history, returns, prices=prices)

    initial_position = float(cfg.env.initial_position)
    signals = history["trading_signal"].astype(float)
    positions = signals.clip(-1.0, 1.0)
    trades = positions.diff().fillna(positions.iloc[0] - initial_position)
    traded_notional = trades.abs() * prices.mean(axis=1)
    cost_rate = (cfg.costs.fee_bps + cfg.slippage.bps) / 10000.0
    expected_costs = traded_notional.sum() * cost_rate

    assert metrics.costs == pytest.approx(expected_costs)
