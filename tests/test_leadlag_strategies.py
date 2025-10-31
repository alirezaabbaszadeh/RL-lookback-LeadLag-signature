import numpy as np
import pytest

from leadlag.models.config import LeadLagConfig, SKLEARN_AVAILABLE
from leadlag.models.strategies import CorrelationCalculator, LeadLagStrategyFactory


def _base_config(method: str, **overrides) -> LeadLagConfig:
    params = dict(
        method=method,
        correlation_method=overrides.pop("correlation_method", "pearson"),
        lookback=3,
        update_freq=1,
        use_parallel=False,
        num_cpus=1,
        quantiles=4,
        show_progress=False,
        Scaling_Method="mean-centering",
        sig_method=overrides.pop("sig_method", "custom"),
    )
    params.update(overrides)
    return LeadLagConfig(**params)


def _align_arrays(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return a[:-lag], b[lag:]
    if lag < 0:
        lag_abs = abs(lag)
        return a[lag_abs:], b[:-lag_abs]
    return a, b


def _manual_cross_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    a, b = _align_arrays(x, y, lag)
    if a.size == 0 or b.size == 0:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _manual_ccf_at_lag(data_pair: np.ndarray, lag: int) -> float:
    x, y = data_pair[:, 0], data_pair[:, 1]
    corr_xy = _manual_cross_corr(x, y, lag)
    corr_yx = _manual_cross_corr(y, x, lag)
    return corr_xy - corr_yx


def _manual_auc_measure(data_pair: np.ndarray, max_lag: int) -> float:
    lags = np.arange(1, max_lag + 1)
    lags = np.r_[-lags, lags]
    corrs = np.array([_manual_cross_corr(data_pair[:, 0], data_pair[:, 1], lag) for lag in lags])
    pos = np.abs(corrs[lags > 0]).sum()
    neg = np.abs(corrs[lags < 0]).sum()
    if pos + neg == 0:
        return 0.0
    return float(np.sign(pos - neg) * max(pos, neg) / (pos + neg))


def _manual_max_lag(data_pair: np.ndarray, max_lag: int) -> float:
    lags = np.arange(1, max_lag + 1)
    lags = np.r_[-lags, lags]
    corrs = np.array([_manual_cross_corr(data_pair[:, 0], data_pair[:, 1], lag) for lag in lags])
    pos_vals = np.abs(corrs[lags > 0])
    neg_vals = np.abs(corrs[lags < 0])
    leading = pos_vals.max() if pos_vals.size else 0.0
    lagging = neg_vals.max() if neg_vals.size else 0.0
    if leading > lagging:
        return float(leading)
    if leading < lagging:
        return float(-lagging)
    return 0.0


def test_ccf_at_lag_strategy_matches_manual() -> None:
    cfg = _base_config("ccf_at_lag", lag=1)
    strategy = LeadLagStrategyFactory(cfg).create()
    data_pair = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 5.0]])
    expected = _manual_ccf_at_lag(data_pair, 1)
    assert strategy.compute(data_pair) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_ccf_auc_strategy_balances_lag_masses() -> None:
    cfg = _base_config("ccf_auc", max_lag=2)
    strategy = LeadLagStrategyFactory(cfg).create()
    data_pair = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 0.1], [3.0, -0.2], [4.0, -0.3]])
    expected = _manual_auc_measure(data_pair, 2)
    assert strategy.compute(data_pair) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_ccf_max_lag_strategy_prefers_strongest_direction() -> None:
    cfg = _base_config("ccf_at_max_lag", max_lag=2)
    strategy = LeadLagStrategyFactory(cfg).create()
    data_pair = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 1.0], [4.0, 1.5]])
    expected = _manual_max_lag(data_pair, 2)
    assert strategy.compute(data_pair) == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_mutual_information_correlation_handles_discrete_inputs() -> None:
    cfg = _base_config("ccf_at_lag", lag=1, correlation_method="mutual_information")
    calculator = CorrelationCalculator(cfg)
    x = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    y = np.array([1, 1, 0, 0, 2, 2], dtype=float)
    result = calculator.cross_correlation(x, y, lag=0)
    assert result >= 0.0
