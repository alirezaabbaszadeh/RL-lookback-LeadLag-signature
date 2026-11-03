"""Unit tests for DynamicOnlineContinualLearner."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from leadlag.alpha.dynamic_online_learner import (
    DynamicOnlineContinualLearner,
    MarketRegime,
)


def test_learner_initialization():
    learner = DynamicOnlineContinualLearner(num_assets=10, lookback_memory=24)
    assert learner.num_assets == 10
    assert learner.lookback_memory == 24
    assert learner.state.step == 0
    assert len(learner.state.cumulative_events) == 0


def test_streaming_observe_and_dollar_neutrality():
    np.random.seed(42)
    learner = DynamicOnlineContinualLearner(num_assets=10, lookback_memory=24)
    
    for t in range(50):
        r_t = np.random.normal(0.0005, 0.01, size=10)
        res = learner.observe_and_adapt(timestamp=t, realized_returns=r_t)
        assert res["step"] == t + 1
        assert "adapted_weights" in res
        weights = res["adapted_weights"]
        # Dollar-neutrality check: sum of weights approx 0
        assert np.isclose(np.sum(weights), 0.0, atol=1e-5)
        # L1 norm budget <= 1.0
        assert np.sum(np.abs(weights)) <= 1.0 + 1e-5


def test_event_detection_and_regime_transition():
    np.random.seed(101)
    learner = DynamicOnlineContinualLearner(
        num_assets=5,
        lookback_memory=20,
        vol_surge_threshold=1.8,
        w1_threshold=0.015,
    )

    # 1. Warm-up with low volatility
    for t in range(30):
        r_t = np.random.normal(0.0, 0.002, size=5)
        learner.observe_and_adapt(timestamp=f"2024-01-01 {t:02d}:00", realized_returns=r_t)

    assert learner.state.current_regime in [
        MarketRegime.LOW_VOLATILITY_DRIFT,
        MarketRegime.BULL_EXPANSION,
    ]

    # 2. Inject massive volatility surge (shock event)
    shock_r = np.array([-0.05, 0.06, -0.07, 0.04, -0.06])
    res_shock = learner.observe_and_adapt(
        timestamp="2024-01-02 07:00",
        realized_returns=shock_r,
    )
    assert res_shock["total_events_recorded"] > 0
    assert learner.state.current_regime in [
        MarketRegime.HIGH_VOLATILITY_CHOP,
        MarketRegime.OOD_ANOMALY,
        MarketRegime.BEAR_CAPITULATION,
    ]

    # 3. Test export event log using tempfile
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "events.json"
        learner.export_event_log(out_file)
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["total_steps"] == 31
        assert data["total_events_detected"] > 0
