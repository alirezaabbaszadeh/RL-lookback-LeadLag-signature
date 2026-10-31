"""Lead-lag matrix construction utilities."""

from __future__ import annotations

import itertools
from typing import Callable, Mapping, Tuple

import numpy as np
import pandas as pd


def compute_levy_matrix_vectorized(
    log_returns: np.ndarray,
    *,
    standardize: bool = True,
) -> np.ndarray:
    """Compute the exact Order-2 Path Signature Lévy Area matrix for all pairs simultaneously.

    For a window of increments ``Z`` of shape ``(L, N)``, the cumulative path is
    ``X_t = (1 / sqrt(L)) * sum_{u <= t} Z_u`` with ``X_0 = 0``. By Chen's identity,
    the second-order iterated integral for all ``(i, j)`` pairs is given by the single
    matrix product ``S2 = X_prev.T @ Z / L``, and the skew-symmetric Lévy area is:

        ``M = 0.5 * (S2 - S2.T)``

    where ``M[i, j] > 0`` indicates asset ``i`` leads asset ``j``.
    """
    arr = np.asarray(log_returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError("log_returns must be a 2D array of shape (L, N)")
    L, N = arr.shape
    if L < 2 or N < 2:
        return np.zeros((N, N), dtype=float)

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if standardize:
        mu = clean.mean(axis=0, keepdims=True)
        sd = clean.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z = (clean - mu) / sd
    else:
        Z = clean

    X_cum = np.cumsum(Z, axis=0)
    X_prev = np.vstack([np.zeros((1, N), dtype=Z.dtype), X_cum[:-1]])
    S2 = X_prev.T @ Z
    return 0.5 * (S2 - S2.T) / float(L)


def build_matrix(
    log_returns: pd.DataFrame, measure_fn: Callable[[np.ndarray], float]
) -> pd.DataFrame:
    """Construct antisymmetric lead-lag matrix using provided measurement function."""

    if log_returns.empty or log_returns.shape[1] < 2:
        return pd.DataFrame(index=log_returns.columns, columns=log_returns.columns, dtype=float)

    columns = log_returns.columns.tolist()
    data = log_returns.values
    n_assets = len(columns)
    matrix = np.zeros((n_assets, n_assets), dtype=float)

    for i, j in itertools.combinations(range(n_assets), 2):
        pair_data = data[:, [i, j]]
        if np.all(np.isnan(pair_data[:, 0])) or np.all(np.isnan(pair_data[:, 1])):
            continue
        value = measure_fn(pair_data)
        if np.isnan(value):
            continue
        matrix[i, j] = value
        matrix[j, i] = -value

    return pd.DataFrame(matrix, index=columns, columns=columns)


def build_matrices_batch(
    windows: Mapping[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    measure_fn: Callable[[np.ndarray], float],
) -> dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
    """Vectorized helper to build matrices for a batch of windows."""
    results: dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}
    for window_key, log_returns in windows.items():
        results[window_key] = build_matrix(log_returns, measure_fn)
    return results


__all__ = ["build_matrices_batch", "build_matrix", "compute_levy_matrix_vectorized"]
