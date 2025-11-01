"""Performance statistics for trading experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
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


def compute_equity_curve(returns: Iterable[float], start: float = 1.0) -> pd.Series:
    series = _to_series(returns)
    equity = series.add(1).cumprod() * start
    return equity


def annualized_sharpe(returns: Iterable[float], periods_per_year: int = 252) -> float:
    series = _to_series(returns)
    if series.std(ddof=1) == 0:
        return 0.0
    return math.sqrt(periods_per_year) * series.mean() / series.std(ddof=1)


def sortino_ratio(returns: Iterable[float], periods_per_year: int = 252) -> float:
    series = _to_series(returns)
    downside = series[series < 0]
    if downside.empty:
        return float("inf")
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return float("inf")
    return math.sqrt(periods_per_year) * series.mean() / downside_std


def max_drawdown(equity: Iterable[float]) -> float:
    series = pd.Series(list(equity), dtype=float)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdowns = (series - running_max) / running_max
    return drawdowns.min()


def hac_confidence_interval(
    returns: Iterable[float],
    alpha: float = 0.05,
    periods_per_year: int = 252,
    max_lag: int | None = None,
) -> tuple[float, float]:
    series = _to_series(returns)
    arr = series.to_numpy()
    n = arr.shape[0]
    if n < 2:
        return (float("nan"), float("nan"))
    demeaned = arr - arr.mean()
    if max_lag is None:
        max_lag = int(np.sqrt(n))
        max_lag = max(1, max_lag)
    gamma0 = np.dot(demeaned, demeaned) / n
    nw = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cov = np.dot(demeaned[:-lag], demeaned[lag:]) / n
        nw += 2.0 * weight * cov
    mean = arr.mean()
    se = math.sqrt(nw / n)
    z = stats.norm.ppf(1 - alpha / 2)
    lower = mean - z * se
    upper = mean + z * se
    return lower * periods_per_year, upper * periods_per_year


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    series = _to_series(returns)
    n = len(series)
    if n < 2:
        return float("nan")
    sharpe = annualized_sharpe(series, periods_per_year)
    skewness = stats.skew(series, bias=False)
    kurt = stats.kurtosis(series, bias=False, fisher=False)
    denom = math.sqrt(1 - skewness * sharpe + ((kurt - 1) / 4) * sharpe**2)
    if denom == 0:
        return float("nan")
    z_score = (sharpe - benchmark) * math.sqrt(n - 1) / denom
    return stats.norm.cdf(z_score)


def deflated_sharpe_ratio(
    returns: Iterable[float],
    benchmark: float = 0.0,
    periods_per_year: int = 252,
    num_trials: int = 1,
) -> float:
    series = _to_series(returns)
    n = len(series)
    if n < 2:
        return float("nan")
    sharpe = annualized_sharpe(series, periods_per_year)
    skewness = stats.skew(series, bias=False)
    kurt = stats.kurtosis(series, bias=False, fisher=False)
    denom = math.sqrt(1 - skewness * sharpe + ((kurt - 1) / 4) * sharpe**2)
    if denom == 0:
        return float("nan")
    if num_trials <= 1:
        z_score = (sharpe - benchmark) * math.sqrt(n - 1) / denom
        return stats.norm.cdf(z_score)
    adjusted = benchmark + denom / math.sqrt(n - 1) * stats.norm.ppf(1 - 1 / num_trials)
    z_score = (sharpe - adjusted) * math.sqrt(n - 1) / denom
    return stats.norm.cdf(z_score)


def stationary_bootstrap(returns: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = returns.shape[0]
    if n == 0:
        return returns
    indices = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(1.0 / max(block_length, 1)))
        for i in range(length):
            indices.append((start + i) % n)
            if len(indices) >= n:
                break
    return returns[np.array(indices[:n])]


def spa_reality_check(
    returns_map: Mapping[str, Iterable[float]],
    benchmark: float = 0.0,
    periods_per_year: int = 252,
    iterations: int = 500,
    block_length: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(seed)
    sharpe_scores: Dict[str, float] = {}
    returns_cache: Dict[str, np.ndarray] = {}
    for key, values in returns_map.items():
        series = _to_series(values)
        sharpe_scores[key] = annualized_sharpe(series, periods_per_year)
        returns_cache[key] = series.to_numpy()
    if not sharpe_scores:
        return pd.DataFrame(columns=["run_id", "sharpe", "spa_pvalue"])
    observed_best = max(sharpe_scores.values())
    if block_length is None:
        block_length = max(1, int(np.sqrt(min(len(v) for v in returns_cache.values()))))
    exceed_count = 0
    for _ in range(iterations):
        boot_scores = []
        for values in returns_cache.values():
            boot = stationary_bootstrap(values, block_length, rng)
            boot_scores.append(annualized_sharpe(boot, periods_per_year))
        if boot_scores and max(boot_scores) >= observed_best:
            exceed_count += 1
    pvalue = (exceed_count + 1) / (iterations + 1)
    for key, score in sharpe_scores.items():
        records.append({"run_id": key, "sharpe": score, "spa_pvalue": pvalue})
    return pd.DataFrame.from_records(records)


def model_confidence_set(
    returns_map: Mapping[str, Iterable[float]],
    alpha: float = 0.1,
    periods_per_year: int = 252,
) -> list[str]:
    sharpe_scores = {
        key: annualized_sharpe(values, periods_per_year)
        for key, values in returns_map.items()
    }
    if not sharpe_scores:
        return []
    threshold = np.quantile(list(sharpe_scores.values()), 1 - alpha)
    return [key for key, score in sharpe_scores.items() if score >= threshold]


def summarize_performance(
    returns: Iterable[float],
    periods_per_year: int = 252,
) -> PerformanceSummary:
    series = _to_series(returns)
    sharpe = annualized_sharpe(series, periods_per_year)
    sortino = sortino_ratio(series, periods_per_year)
    equity = compute_equity_curve(series)
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
