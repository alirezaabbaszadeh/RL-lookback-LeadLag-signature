import numpy as np  # Required for tests to use numpy namespace
import pandas as pd

from leadlag.eval import stats


def test_sharpe_sortino_and_hac_ci():
    returns = [0.01, 0.015, -0.005, 0.012, 0.02, -0.003, 0.018]
    sharpe = stats.annualized_sharpe(returns, periods_per_year=252)
    sortino = stats.sortino_ratio(returns, periods_per_year=252)
    assert sharpe > 0
    assert sortino >= sharpe
    lower, upper = stats.hac_sharpe_confidence_interval(returns, periods_per_year=252)
    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert lower <= upper


def test_psr_and_dsr_behaviour():
    returns = [0.01, 0.012, 0.011, -0.002, 0.009, 0.013, 0.01]
    psr = stats.probabilistic_sharpe_ratio(returns, periods_per_year=252)
    dsr_single = stats.deflated_sharpe_ratio(returns, periods_per_year=252, num_trials=1)
    dsr_many = stats.deflated_sharpe_ratio(returns, periods_per_year=252, num_trials=10)
    assert 0 <= psr <= 1
    assert 0 <= dsr_single <= 1
    assert dsr_many <= dsr_single + 1e-12


def test_stationary_bootstrap_shape():
    rng = np.random.default_rng(123)
    sample = np.arange(10, dtype=float)
    boot = stats.stationary_bootstrap(sample, block_length=3, rng=rng)
    assert boot.shape == sample.shape
    assert not np.array_equal(boot, sample)


def test_spa_and_mcs_outputs():
    returns_map = {
        "a": pd.Series([0.01, -0.002, 0.015, 0.008, 0.011], dtype=float),
        "b": pd.Series([0.005, 0.004, 0.006, 0.003, 0.007], dtype=float),
    }
    spa = stats.spa_reality_check(returns_map, iterations=50, seed=42)
    assert {"run_id", "spa_pvalue", "spa_sup_pvalue"}.issubset(set(spa.columns))
    assert len(spa) == 2
    mcs = stats.model_confidence_set(returns_map, iterations=50, seed=42)
    assert isinstance(mcs, list)
    assert len(mcs) >= 1
