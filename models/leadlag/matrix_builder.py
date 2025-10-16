"""Lead-lag matrix construction utilities."""

from __future__ import annotations

import itertools
from typing import Callable, Mapping, Tuple

import numpy as np
import pandas as pd


def build_matrix(log_returns: pd.DataFrame, measure_fn: Callable[[np.ndarray], float]) -> pd.DataFrame:
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
