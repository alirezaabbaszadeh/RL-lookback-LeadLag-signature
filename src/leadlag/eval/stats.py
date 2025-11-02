"""Performance statistics for trading experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class PerformanceSummary:
    """Summary of basic performance statistics."""

    sharpe: float
    sortino: float
    max_drawdown: float
    pnl: float
    returns: pd.Series


def _to_series(data: Iterable[float]) -> pd.Series:
    series = pd.Series(list(data), dtype=float)
    series = series.dropna()
    if series.empty:
        raise ValueError("returns series is empty")
    return series


def _excess_returns(
    returns: Iterable[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    series = _to_series(returns)
    if risk_free_rate != 0:
        rf_per_period = risk_free_rate / periods_per_year
        series = series - rf_per_period
    return series


def compute_equity_curve(returns: Iterable[float], start: float = 1.0) -> pd.Series:
    series = _to_series(returns)
    equity = series.add(1).cumprod() * start
    return equity


def sharpe_ratio(
    returns: Iterable[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Return the per-period Sharpe ratio for a series of returns."""

    excess = _excess_returns(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return excess.mean() / std


def annualized_sharpe(
    returns: Iterable[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Return the annualised Sharpe ratio."""

    ratio = sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    return math.sqrt(periods_per_year) * ratio


def sortino_ratio(
    returns: Iterable[float],
    *,
    target: float = 0.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Return the annualised Sortino ratio with an optional target return."""

    excess = _excess_returns(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    target_per_period = target / periods_per_year
    downside = excess[excess < target_per_period]
    if downside.empty:
        return float("inf")
    downside_std = (downside - target_per_period).std(ddof=1)
    if downside_std == 0:
        return float("inf")
    return math.sqrt(periods_per_year) * (excess.mean() - target_per_period) / downside_std


def max_drawdown(equity: Iterable[float]) -> float:
    series = pd.Series(list(equity), dtype=float)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdowns = (series - running_max) / running_max
    return drawdowns.min()


def _newey_west_covariance(residuals: np.ndarray, max_lag: int) -> np.ndarray:
    n, k = residuals.shape
    cov = residuals.T @ residuals / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        front = residuals[lag:]
        back = residuals[:-lag]
        gamma = front.T @ back / n
        cov += weight * (gamma + gamma.T)
    return cov


def hac_sharpe_confidence_interval(
    returns: Iterable[float],
    *,
    alpha: float = 0.05,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """HAC (Newey-West) confidence interval for the annualised Sharpe ratio."""

    excess = _excess_returns(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    arr = excess.to_numpy()
    n = arr.shape[0]
    if n < 3:
        return (float("nan"), float("nan"))
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return (float("nan"), float("nan"))
    sr_period = mean / std
    sr_annual = sr_period * math.sqrt(periods_per_year)

    if max_lag is None:
        max_lag = max(1, int(1.3221 * n ** (1 / 5)))

    moments = np.column_stack((arr, arr**2))
    centered = moments - moments.mean(axis=0, keepdims=True)
    long_run = _newey_west_covariance(centered, max_lag)
    gradient = np.array([
        (std**2 + mean**2) / std**3,
        -0.5 * mean / std**3,
    ])
    variance = gradient.T @ long_run @ gradient / n
    variance = float(max(variance, 0.0))
    se_period = math.sqrt(variance)
    se_annual = se_period * math.sqrt(periods_per_year)
    z = stats.norm.ppf(1 - alpha / 2)
    lower = sr_annual - z * se_annual
    upper = sr_annual + z * se_annual
    return lower, upper


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    *,
    benchmark: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probability that the Sharpe ratio exceeds ``benchmark``."""

    excess = _excess_returns(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    n = len(excess)
    if n < 3:
        return float("nan")
    sr_period = sharpe_ratio(
        excess,
        risk_free_rate=0.0,
        periods_per_year=periods_per_year,
    )
    skewness = stats.skew(excess, bias=False)
    kurt = stats.kurtosis(excess, bias=False, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - skewness * sr_period + ((kurt - 1) / 4) * sr_period**2))
    sigma_sr = denom / math.sqrt(n - 1)
    if sigma_sr == 0:
        return float("nan")
    z_score = (sr_period - benchmark) / sigma_sr
    return stats.norm.cdf(z_score)


def deflated_sharpe_ratio(
    returns: Iterable[float],
    *,
    benchmark: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    num_trials: int = 1,
) -> float:
    """Deflated Sharpe ratio following Bailey & López de Prado (2014)."""

    if num_trials < 1:
        raise ValueError("num_trials must be positive")
    excess = _excess_returns(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    n = len(excess)
    if n < 3:
        return float("nan")
    sr_period = sharpe_ratio(excess, risk_free_rate=0.0, periods_per_year=periods_per_year)
    skewness = stats.skew(excess, bias=False)
    kurt = stats.kurtosis(excess, bias=False, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - skewness * sr_period + ((kurt - 1) / 4) * sr_period**2))
    sigma_sr = denom / math.sqrt(n - 1)
    if sigma_sr == 0:
        return float("nan")
    if num_trials == 1:
        z_score = (sr_period - benchmark) / sigma_sr
        return stats.norm.cdf(z_score)
    sr_star = benchmark + sigma_sr * stats.norm.ppf(1 - 1 / num_trials)
    z_score = (sr_period - sr_star) / sigma_sr
    return stats.norm.cdf(z_score)


def stationary_bootstrap(returns: ArrayLike, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Apply the stationary bootstrap (Politis & Romano, 1994)."""

    arr = np.asarray(returns, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return arr
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    indices: list[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(1.0 / block_length))
        for offset in range(length):
            indices.append((start + offset) % n)
            if len(indices) >= n:
                break
    return arr[np.array(indices[:n])]


def _prepare_returns_matrix(returns_map: Mapping[str, Iterable[float]]) -> tuple[list[str], np.ndarray]:
    if not returns_map:
        return [], np.empty((0, 0))
    series_list = []
    names: list[str] = []
    min_length = None
    for name, values in returns_map.items():
        series = _to_series(values)
        if series.empty:
            continue
        names.append(name)
        series_list.append(series.to_numpy())
        length = len(series)
        min_length = length if min_length is None else min(min_length, length)
    if not series_list or min_length is None:
        return [], np.empty((0, 0))
    trimmed = [series[:min_length] for series in series_list]
    matrix = np.column_stack(trimmed)
    return names, matrix


def spa_reality_check(
    returns_map: Mapping[str, Iterable[float]],
    *,
    benchmark: float = 0.0,
    periods_per_year: int = 252,
    iterations: int = 500,
    block_length: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Perform the SPA reality check for a collection of return series."""

    names, matrix = _prepare_returns_matrix(returns_map)
    if matrix.size == 0:
        return pd.DataFrame(columns=["run_id", "sharpe", "t_stat", "spa_pvalue", "spa_sup_pvalue"])

    n, k = matrix.shape
    benchmark_per_period = benchmark / periods_per_year
    centered = matrix - benchmark_per_period
    means = centered.mean(axis=0)
    stds = centered.std(axis=0, ddof=1)
    t_stats = np.zeros(k)
    sharpe_scores = np.zeros(k)
    for idx in range(k):
        std = stds[idx]
        if std > 0:
            t_stats[idx] = math.sqrt(n) * means[idx] / std
        else:
            t_stats[idx] = 0.0
        sharpe_scores[idx] = annualized_sharpe(matrix[:, idx], periods_per_year=periods_per_year)

    if block_length is None:
        block_length = max(1, int(np.sqrt(n)))

    rng = np.random.default_rng(seed)
    boot_t = np.zeros((iterations, k))
    centered_resid = centered - means
    for b in range(iterations):
        for idx in range(k):
            boot = stationary_bootstrap(centered_resid[:, idx], block_length, rng)
            std = stds[idx] if stds[idx] > 0 else 1.0
            boot_mean = boot.mean()
            boot_std = boot.std(ddof=1)
            denom = boot_std if boot_std > 0 else std
            boot_t[b, idx] = math.sqrt(n) * boot_mean / denom

    sup_obs = t_stats.max()
    sup_boot = boot_t.max(axis=1)
    sup_pvalue = (1 + np.sum(sup_boot >= sup_obs)) / (iterations + 1)
    individual_pvalues = (1 + np.sum(boot_t >= t_stats, axis=0)) / (iterations + 1)

    records = []
    for name, sharpe_val, t_val, p_val in zip(names, sharpe_scores, t_stats, individual_pvalues):
        records.append(
            {
                "run_id": name,
                "sharpe": sharpe_val,
                "t_stat": t_val,
                "spa_pvalue": p_val,
                "spa_sup_pvalue": sup_pvalue,
            }
        )
    return pd.DataFrame.from_records(records)


def model_confidence_set(
    returns_map: Mapping[str, Iterable[float]],
    *,
    alpha: float = 0.1,
    periods_per_year: int = 252,
    iterations: int = 500,
    block_length: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Compute a simple Model Confidence Set based on bootstrap Sharpe differences."""

    names, matrix = _prepare_returns_matrix(returns_map)
    if matrix.size == 0:
        return []
    if len(names) == 1:
        return names

    n, _ = matrix.shape
    sharpe_scores = np.array(
        [annualized_sharpe(matrix[:, idx], periods_per_year=periods_per_year) for idx in range(matrix.shape[1])]
    )
    best_idx = int(np.argmax(sharpe_scores))
    observed_diff = sharpe_scores[best_idx] - sharpe_scores

    if block_length is None:
        block_length = max(1, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    boot_diffs = np.zeros((iterations, len(names)))
    mean_vector = matrix.mean(axis=0)
    centered = matrix - mean_vector
    for b in range(iterations):
        boot_means = []
        for idx in range(len(names)):
            boot = stationary_bootstrap(centered[:, idx], block_length, rng) + mean_vector[idx]
            boot_means.append(annualized_sharpe(boot, periods_per_year=periods_per_year))
        boot_means = np.array(boot_means)
        boot_diffs[b] = boot_means[best_idx] - boot_means

    pvalues = (1 + np.sum(boot_diffs >= observed_diff, axis=0)) / (iterations + 1)
    members = [name for name, pval in zip(names, pvalues) if pval > alpha]
    if not members:
        members = [names[best_idx]]
    return members


def summarize_performance(
    returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> PerformanceSummary:
    series = _excess_returns(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    sharpe = annualized_sharpe(series, periods_per_year=periods_per_year)
    sortino = sortino_ratio(series, periods_per_year=periods_per_year)
    equity = compute_equity_curve(series + (risk_free_rate / periods_per_year))
    mdd = max_drawdown(equity)
    pnl = float(series.sum())
    return PerformanceSummary(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        pnl=pnl,
        returns=series,
    )


def equity_dataframe(equity: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"t": np.arange(len(equity)), "equity": equity.values})


def returns_dataframe(returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"t": np.arange(len(returns)), "returns": returns.values})


def export_equity(path: Path, equity: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    equity_dataframe(equity).to_csv(path, index=False)


def export_returns(path: Path, returns: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    returns_dataframe(returns).to_csv(path, index=False)
