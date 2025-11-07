from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leadlag.reporting.metrics_writer import (
    METRICS_COLUMNS,
    METRICS_DTYPES,
    MetricsWriter,
    build_metadata_row,
    enforce_metrics_schema,
)


@pytest.fixture
def metrics_config() -> dict[str, object]:
    return {
        "agent": {"name": "ppo"},
        "env": {"action_space": "discrete"},
        "policy": {"name": "mlp"},
        "features": {
            "signature": {"enabled": True, "depth": 2},
            "leadlag": {"enabled": False},
            "time_channel": True,
        },
        "window": {"lookback": 32},
        "target": {"horizon": 1},
        "data": {"universe": "demo", "timeframe": "5m"},
        "split": {"scheme": "walk_forward"},
        "costs": {"fee_bps": 1.5},
        "slippage": {"bps": 0.5},
        "reward": {"name": "sharpe"},
    }


@pytest.fixture
def metrics_row(metrics_config: dict[str, object]) -> dict[str, object]:
    return build_metadata_row(
        "demo-run",
        metrics_config,
        {"sharpe": 1.23, "sortino": 1.11, "max_drawdown": -0.2, "pnl": 0.45},
        seed=7,
        window_idx=1,
        turnover=0.05,
        exposure=0.4,
        costs=0.02,
        env_steps=128,
    )


def test_metrics_schema_enforced(tmp_path: Path, metrics_config: dict[str, object], metrics_row: dict[str, object]) -> None:
    writer = MetricsWriter(metrics_config)
    out_path = tmp_path / "metrics.csv"
    writer.write_row(out_path, metrics_row)

    raw_frame = pd.read_csv(out_path)
    coerced = enforce_metrics_schema(raw_frame)

    assert list(coerced.columns) == METRICS_COLUMNS
    dtype_map = {name: str(coerced[name].dtype) for name in METRICS_COLUMNS}
    assert dtype_map == METRICS_DTYPES

    assert coerced.loc[0, "experiment_id"] == "demo-run"
    assert coerced.loc[0, "seed"] == metrics_row["seed"]
    assert pytest.approx(coerced.loc[0, "Sharpe"], rel=1e-9) == metrics_row["Sharpe"]

    frame = writer.dataframe([metrics_row])
    assert list(frame.columns) == METRICS_COLUMNS
    assert {name: str(frame[name].dtype) for name in METRICS_COLUMNS} == METRICS_DTYPES
