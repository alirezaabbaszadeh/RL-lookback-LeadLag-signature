"""Lead-lag correlation strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from leadlag.models.config import (
    DCOR_AVAILABLE,
    SKLEARN_AVAILABLE,
    LeadLagConfig,
)
from leadlag.models.leadlag.signature_extractor import SignatureConfig, SignatureExtractor

try:  # pragma: no cover - optional dependency
    import dcor  # type: ignore
except ImportError:  # pragma: no cover - fallback
    dcor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when numba missing
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _pearson_corr_numba(x: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover - compiled
        n = x.shape[0]
        if n == 0:
            return np.nan

        mean_x = 0.0
        mean_y = 0.0
        for i in range(n):
            mean_x += x[i]
            mean_y += y[i]
        mean_x /= n
        mean_y /= n

        numerator = 0.0
        denom_x = 0.0
        denom_y = 0.0
        for i in range(n):
            dx = x[i] - mean_x
            dy = y[i] - mean_y
            numerator += dx * dy
            denom_x += dx * dx
            denom_y += dy * dy

        if denom_x == 0.0 or denom_y == 0.0:
            return 0.0
        return numerator / (np.sqrt(denom_x) * np.sqrt(denom_y))


class LeadLagStrategy(Protocol):
    """Protocol for lead-lag correlation strategies."""

    def compute(self, data_pair: np.ndarray) -> float:
        ...


@dataclass
class CorrelationCalculator:
    """Helper responsible for computing cross-correlations under different metrics."""

    config: LeadLagConfig

    def cross_correlation(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if not np.any(valid_mask):
                return np.nan
            x = x[valid_mask]
            y = y[valid_mask]

        if lag > 0:
            if lag >= len(x):
                return np.nan
            x_lagged = x[:-lag]
            y_aligned = y[lag:]
        elif lag < 0:
            lag_abs = -lag
            if lag_abs >= len(y):
                return np.nan
            x_lagged = x[lag_abs:]
            y_aligned = y[:-lag_abs]
        else:
            x_lagged = x
            y_aligned = y

        if len(x_lagged) == 0 or len(y_aligned) == 0:
            return np.nan

        method = self.config.correlation_method
        if method == "pearson":
            if NUMBA_AVAILABLE:
                return float(_pearson_corr_numba(x_lagged, y_aligned))
            std_x = np.std(x_lagged)
            std_y = np.std(y_aligned)
            if std_x == 0 or std_y == 0:
                return 0.0
            return float(np.corrcoef(x_lagged, y_aligned)[0, 1])
        if method in {"kendall", "spearman"}:
            combined_df = pd.DataFrame({"x": x_lagged, "y": y_aligned})
            return float(combined_df.corr(method=method).iloc[0, 1])
        if method == "distance":
            if not DCOR_AVAILABLE:
                raise ImportError("dcor package required for distance correlation")
            return float(dcor.distance_correlation(x_lagged, y_aligned))
        if method == "mutual_information":
            return float(self._mutual_information(x_lagged, y_aligned))
        if method == "squared_pearson":
            std_x = np.nanstd(x_lagged**2)
            std_y = np.nanstd(y_aligned**2)
            if std_x == 0 or std_y == 0:
                return 0.0
            return float(np.corrcoef(x_lagged**2, y_aligned**2)[0, 1])
        raise NotImplementedError(f"Correlation method {method} not implemented")

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn package required for mutual information")

        from sklearn.feature_selection import mutual_info_classif  # type: ignore

        x_quantiles = pd.qcut(x, q=self.config.quantiles, labels=False, duplicates="drop")
        y_quantiles = pd.qcut(y, q=self.config.quantiles, labels=False, duplicates="drop")

        x_quantiles = np.asarray(x_quantiles, dtype=float)
        y_quantiles = np.asarray(y_quantiles, dtype=float)

        valid_mask = ~(np.isnan(x_quantiles) | np.isnan(y_quantiles))
        if not np.any(valid_mask):
            return float("nan")

        x_valid = x_quantiles[valid_mask]
        y_valid = y_quantiles[valid_mask]
        if len(x_valid) == 0:
            return float("nan")

        return float(
            mutual_info_classif(x_valid.reshape(-1, 1), y_valid, discrete_features=True, random_state=0)[0]
        )


@dataclass
class CcfAtLagStrategy:
    config: LeadLagConfig
    calculator: CorrelationCalculator

    def compute(self, data_pair: np.ndarray) -> float:
        x, y = data_pair[:, 0], data_pair[:, 1]
        lag = int(self.config.lag or 0)
        corr_xy = self.calculator.cross_correlation(x, y, lag)
        corr_yx = self.calculator.cross_correlation(y, x, lag)
        return corr_xy - corr_yx


@dataclass
class CcfAucStrategy:
    config: LeadLagConfig
    calculator: CorrelationCalculator

    def compute(self, data_pair: np.ndarray) -> float:
        x, y = data_pair[:, 0], data_pair[:, 1]
        lags = np.arange(1, int(self.config.max_lag or 0) + 1)
        lags = np.r_[-lags, lags]
        correlations = np.array([self.calculator.cross_correlation(x, y, lag) for lag in lags])
        pos_mask = lags > 0
        neg_mask = lags < 0
        A = np.abs(correlations[pos_mask]).sum()
        B = np.abs(correlations[neg_mask]).sum()
        if A + B == 0:
            return 0.0
        return float(np.sign(A - B) * max(A, B) / (A + B))


@dataclass
class CcfMaxLagStrategy:
    config: LeadLagConfig
    calculator: CorrelationCalculator

    def compute(self, data_pair: np.ndarray) -> float:
        x, y = data_pair[:, 0], data_pair[:, 1]
        lags = np.arange(1, int(self.config.max_lag or 0) + 1)
        lags = np.r_[-lags, lags]
        correlations = np.array([self.calculator.cross_correlation(x, y, lag) for lag in lags])
        pos_mask = lags > 0
        neg_mask = lags < 0
        pos_values = np.abs(correlations[pos_mask])
        neg_values = np.abs(correlations[neg_mask])

        leadingness = self._nanmax_with_default(pos_values)
        laggingness = self._nanmax_with_default(neg_values)
        if leadingness > laggingness:
            return float(leadingness)
        if leadingness < laggingness:
            return float(-laggingness)
        return 0.0

    @staticmethod
    def _nanmax_with_default(values: np.ndarray) -> float:
        if values.size == 0 or np.all(np.isnan(values)):
            return 0.0
        return float(np.nanmax(values))


class SignatureStrategy:
    def __init__(self, config: LeadLagConfig) -> None:
        self.config = config
        self._extractor: SignatureExtractor | None = None

    def compute(self, data_pair: np.ndarray) -> float:
        return float(self._get_extractor().compute(data_pair))

    def _get_extractor(self) -> SignatureExtractor:
        if self._extractor is None:
            extractor_config = SignatureConfig(
                order=2,
                scaling_method=getattr(self.config, "Scaling_Method", "mean-centering"),
                sig_method=getattr(self.config, "sig_method", "custom"),
            )
            self._extractor = SignatureExtractor(extractor_config)
        return self._extractor

    def __call__(self, data_pair: np.ndarray) -> float:
        return self.compute(data_pair)


class LeadLagStrategyFactory:
    """Factory that builds the appropriate strategy for a configuration."""

    def __init__(self, config: LeadLagConfig) -> None:
        self.config = config

    def create(self) -> LeadLagStrategy:
        method = self.config.method
        if method == "signature":
            return SignatureStrategy(self.config)
        calculator = CorrelationCalculator(self.config)
        if method == "ccf_at_lag":
            return CcfAtLagStrategy(self.config, calculator)
        if method == "ccf_auc":
            return CcfAucStrategy(self.config, calculator)
        if method == "ccf_at_max_lag":
            return CcfMaxLagStrategy(self.config, calculator)
        raise NotImplementedError(f"Method {method} not implemented")


__all__ = [
    "LeadLagStrategy",
    "LeadLagStrategyFactory",
    "CorrelationCalculator",
    "CcfAtLagStrategy",
    "CcfAucStrategy",
    "CcfMaxLagStrategy",
    "SignatureStrategy",
]
