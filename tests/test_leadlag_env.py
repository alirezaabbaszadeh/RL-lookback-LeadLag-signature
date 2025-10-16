import numpy as np
import pytest

try:
    import pandas as pd
except ValueError:
    pytest.skip("Pandas binary incompatible with current numpy", allow_module_level=True)

from envs.leadlag_env import LeadLagEnv
from models.LeadLag_main import LeadLagConfig


def _make_prices(rows: int = 180) -> "pd.DataFrame":
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    levels = 100 + np.cumsum(rng.normal(scale=0.01, size=(rows, 3)), axis=0)
    return pd.DataFrame(levels, index=index, columns=list("ABC"))


def _leadlag_cfg() -> LeadLagConfig:
    return LeadLagConfig(
        method="ccf_at_lag",
        correlation_method="pearson",
        lookback=30,
        update_freq=1,
        use_parallel=False,
        num_cpus=1,
        quantiles=4,
        show_progress=False,
        Scaling_Method="mean-centering",
        sig_method="levy",
        lag=1,
    )


def test_relative_action_decreases_lookback():
    env = LeadLagEnv(
        price_df=_make_prices(),
        leadlag_config=_leadlag_cfg(),
        min_lookback=10,
        max_lookback=40,
        discrete_actions=True,
        action_mode="relative",
        relative_step=5,
        random_start=False,
        ema_alpha=None,
    )

    obs = env.reset()
    assert obs.shape == (env.obs_dim,)

    start_lookback = env._current_lookback
    _obs, _reward, _done, _info = env.step(0)  # action 0 => -relative_step
    assert env._current_lookback == max(env.min_lookback, start_lookback - env.relative_step)


def test_hybrid_discrete_actions_cover_extremes():
    env = LeadLagEnv(
        price_df=_make_prices(),
        leadlag_config=_leadlag_cfg(),
        min_lookback=8,
        max_lookback=32,
        discrete_actions=True,
        action_mode="hybrid",
        relative_step=4,
        random_start=False,
        ema_alpha=0.2,
    )

    env.reset()
    env.step(0)  # set to min
    assert env._current_lookback == env.min_lookback

    env.step(len(env._hybrid_actions) - 1)  # set to max
    assert env._current_lookback == env.max_lookback


def test_random_start_respects_episode_bounds():
    env = LeadLagEnv(
        price_df=_make_prices(),
        leadlag_config=_leadlag_cfg(),
        min_lookback=12,
        max_lookback=36,
        discrete_actions=True,
        action_mode="absolute",
        episode_length=25,
        random_start=True,
        random_seed=123,
    )

    starts = []
    for _ in range(5):
        env.reset()
        starts.append(env._episode_start_index)
    assert all(start >= env._min_start_index for start in starts)
    max_allowed = len(env._dates) - env._episode_length
    assert all(start <= max_allowed for start in starts)
