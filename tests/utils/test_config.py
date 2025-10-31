from __future__ import annotations

from typing import Any, Dict

import pytest

from leadlag.utils.config import deep_update


@pytest.mark.parametrize(
    "base, overrides, expected",
    [
        (
            {"analysis": {"lookback": 10, "method": "leadlag"}},
            {"analysis": {"lookback": 20}},
            {"analysis": {"lookback": 20, "method": "leadlag"}},
        ),
        (
            {"run": {"seed": 42}, "rl": {"policy": {"name": "ppo"}}},
            {"rl": {"policy": {"name": "sac", "kwargs": {"lr": 1e-3}}}},
            {"run": {"seed": 42}, "rl": {"policy": {"name": "sac", "kwargs": {"lr": 1e-3}}}},
        ),
        (
            {"rl": {"policy": {"name": "ppo"}}},
            {"rl": {"policy": "random"}},
            {"rl": {"policy": "random"}},
        ),
    ],
)
def test_deep_update_merges(base: Dict[str, Any], overrides: Dict[str, Any], expected: Dict[str, Any]):
    result = deep_update(base, overrides)
    assert result is base
    assert result == expected


def test_deep_update_preserves_other_branches():
    base: Dict[str, Any] = {
        "analysis": {"lookback": 10, "method": "leadlag"},
        "run": {"seed": 7},
    }
    overrides: Dict[str, Any] = {
        "analysis": {"window": 30},
        "metrics": ["signal_strength"],
    }

    deep_update(base, overrides)

    assert base["analysis"] == {"lookback": 10, "method": "leadlag", "window": 30}
    assert base["run"] == {"seed": 7}
    assert base["metrics"] == ["signal_strength"]
