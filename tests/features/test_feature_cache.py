from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from omegaconf import OmegaConf

from leadlag.pipelines import run_full_suite


def _make_prices() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = np.linspace(1.0, 2.1, num=12)
    frame = pd.DataFrame({"AssetA": data, "AssetB": data[::-1]}, index=index)
    return frame


def test_feature_cache_hits_for_repeated_runs(monkeypatch, tmp_path):
    prices = _make_prices()

    counters = {"signature": 0, "leadlag": 0}

    original_signature = run_full_suite.compute_signature_features
    original_leadlag = run_full_suite.compute_lead_lag

    def counting_signature(series, depth):
        counters["signature"] += 1
        return original_signature(series, depth)

    def counting_leadlag(series):
        counters["leadlag"] += 1
        return original_leadlag(series)

    monkeypatch.setattr(run_full_suite, "compute_signature_features", counting_signature)
    monkeypatch.setattr(run_full_suite, "compute_lead_lag", counting_leadlag)

    features_cfg = OmegaConf.create(
        {
            "cache": {"enabled": True, "dir": str(tmp_path)},
            "signature": {"enabled": True, "depth": 2},
            "leadlag": {"enabled": True},
            "time_channel": True,
        }
    )

    stack_args = dict(universe="demo", timeframe="1h", lookback=8, seed=101)

    first = run_full_suite._build_feature_stack(prices, features_cfg, **stack_args)
    assert counters == {"signature": 1, "leadlag": 1}

    second = run_full_suite._build_feature_stack(prices, features_cfg, **stack_args)
    assert counters == {"signature": 1, "leadlag": 1}

    for key in first:
        assert key in second
        assert_array_equal(first[key], second[key])

    third = run_full_suite._build_feature_stack(prices, features_cfg, **{**stack_args, "seed": 202})
    assert counters == {"signature": 2, "leadlag": 2}

    for key in third:
        assert key in first
        assert_array_equal(third[key], first[key])


def test_feature_cache_respects_feature_toggles(monkeypatch, tmp_path):
    prices = _make_prices()

    counters = {"leadlag": 0}
    original_leadlag = run_full_suite.compute_lead_lag

    def counting_leadlag(series):
        counters["leadlag"] += 1
        return original_leadlag(series)

    monkeypatch.setattr(run_full_suite, "compute_lead_lag", counting_leadlag)

    stack_args = dict(universe="demo", timeframe="1h", lookback=8, seed=111)

    base_features_cfg = OmegaConf.create(
        {
            "cache": {"enabled": True, "dir": str(tmp_path)},
            "leadlag": {"enabled": False},
            "time_channel": False,
        }
    )

    base_stack = run_full_suite._build_feature_stack(prices, base_features_cfg, **stack_args)
    assert counters == {"leadlag": 0}
    assert set(base_stack) == {"returns"}

    toggled_cfg = OmegaConf.create(
        {
            "cache": {"enabled": True, "dir": str(tmp_path)},
            "leadlag": {"enabled": True},
            "time_channel": True,
        }
    )

    toggled_stack = run_full_suite._build_feature_stack(prices, toggled_cfg, **stack_args)
    assert counters == {"leadlag": 1}
    assert "returns" in toggled_stack
    assert "leadlag" in toggled_stack
    assert "time_channel" in toggled_stack
