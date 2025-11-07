import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from leadlag.pipelines import run_full_suite
from leadlag.utils import NoPeekError


def _minimal_cfg():
    return OmegaConf.create(
        {
            "training": {
                "total_env_steps": 8,
                "periods_per_year": 252,
                "seeds": [123],
                "windows": 1,
            },
            "window": {"lookback": 4},
            "hardware": {"n_envs": 1, "device": "cpu"},
            "agent": {"library": "random", "name": "random"},
            "data": {},
            "features": {},
            "env": {},
            "logging": {"run_id": "guard-test", "append_seed_window": False},
            "results_root": "results",
            "paper_outputs_root": "paper_outputs",
            "split": {"n_splits": 2, "scheme": "walk_forward"},
        }
    )


def test_simulate_episode_raises_when_prices_peek(monkeypatch):
    cfg = _minimal_cfg()

    misordered = pd.DataFrame(
        {"AssetA": [1.0, 1.1, 1.2]},
        index=pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-03"]),
    )

    monkeypatch.setattr(
        run_full_suite,
        "_load_price_data",
        lambda _cfg, _seed: (misordered, None),
    )

    with pytest.raises(NoPeekError):
        run_full_suite._simulate_episode(cfg, seed=123, window_idx=0)


def test_simulate_episode_raises_when_feature_times_peek(monkeypatch):
    cfg = _minimal_cfg()

    ordered = pd.DataFrame(
        {"AssetA": [1.0, 1.1, 1.2, 1.3]},
        index=pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
        ),
    )

    monkeypatch.setattr(
        run_full_suite,
        "_load_price_data",
        lambda _cfg, _seed: (ordered, None),
    )

    def fake_feature_stack(*_args, **_kwargs):
        stack = {"returns": np.zeros((len(ordered.index) - 1, 1), dtype=float)}
        frame = pd.DataFrame(
            {
                "t_feat": ordered.index[1:],
            },
            index=ordered.index[1:],
        )
        return stack, frame

    monkeypatch.setattr(run_full_suite, "_build_feature_stack", fake_feature_stack)

    with pytest.raises(NoPeekError):
        run_full_suite._simulate_episode(cfg, seed=123, window_idx=0)
