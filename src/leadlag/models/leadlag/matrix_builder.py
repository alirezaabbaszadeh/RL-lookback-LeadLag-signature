"""Lead-lag matrix construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LevyPrefixCache:
    """Precomputed prefix tensors for O(1)-in-L windowed Lévy area evaluation (Theorem 7 Part B).

    Attributes:
        X_cum: Cumulative prefix sum of increments, shape (T + 1, N), with X_cum[0] = 0.
        A_prefix: Prefix raw skew Lévy area tensor, shape (T + 1, N, N).
        Q_cum: Cumulative prefix sum of squared increments, shape (T + 1, N), for O(1) variance.
        Y_cum: Cumulative prefix sum of time-weighted increments k * R_k, shape (T + 1, N),
            for O(1) Abel summation centering corrections.
        T: Total number of time steps (rows).
        N: Total number of assets (columns).
    """

    X_cum: np.ndarray
    A_prefix: np.ndarray
    Q_cum: np.ndarray
    Y_cum: np.ndarray
    T: int
    N: int


def prepare_levy_prefix_cache(
    log_returns: np.ndarray,
) -> LevyPrefixCache:
    """Precompute cumulative prefix tensors over series of shape (T, N) for O(1)-in-L window queries.

    This implements the preprocessing step for Theorem 7 Part B (Chen-Abel Bilinear Prefix Identity).
    In a single vectorized O(T * N^2) pass, it precomputes:
        1. X_t = sum_{k=0}^{t-1} R_k (first-order cumulative path)
        2. S_t = sum_{u=0}^{t-1} X_u R_u^T and A_t = 0.5 * (S_t - S_t^T) (second-order iterated tensor)
        3. Q_t = sum_{k=0}^{t-1} R_k^2 (quadratic prefix for O(1) sample variance)
        4. Y_t = sum_{k=0}^{t-1} k * R_k (linear-time-weighted prefix for O(1) Abel centering)

    Args:
        log_returns: 2D array of shape (T, N) representing asset return increments.

    Returns:
        LevyPrefixCache containing precomputed prefix tensors.
    """
    arr = np.asarray(log_returns, dtype=float)
    if arr.ndim != 2:
        raise ValueError("log_returns must be a 2D array of shape (T, N)")
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    T, N = clean.shape

    X_cum = np.vstack([np.zeros((1, N), dtype=float), np.cumsum(clean, axis=0)])
    Q_cum = np.vstack([np.zeros((1, N), dtype=float), np.cumsum(clean**2, axis=0)])
    k_vec = np.arange(T, dtype=float)[:, None]
    Y_cum = np.vstack([np.zeros((1, N), dtype=float), np.cumsum(k_vec * clean, axis=0)])

    # Vectorized outer products for iterated integral tensor S_cum:
    # S_cum[t] = sum_{u=0}^{t-1} X_u R_u^T
    X_prev = X_cum[:-1]
    terms = np.einsum("ti,tj->tij", X_prev, clean)
    S_cum = np.vstack([np.zeros((1, N, N), dtype=float), np.cumsum(terms, axis=0)])
    A_prefix = 0.5 * (S_cum - np.swapaxes(S_cum, 1, 2))

    return LevyPrefixCache(
        X_cum=X_cum,
        A_prefix=A_prefix,
        Q_cum=Q_cum,
        Y_cum=Y_cum,
        T=T,
        N=N,
    )


def compute_levy_prefix_chen_abel(
    log_returns: Optional[np.ndarray] = None,
    a: int = 0,
    b: Optional[int] = None,
    *,
    prefix_cache: Optional[LevyPrefixCache] = None,
    standardize: bool = True,
) -> np.ndarray:
    """Evaluate the standardized skew Lévy area matrix over window [a, b) in O(1) in L.

    =============================================================================
    THEOREM 7 PART B: Chen-Abel O(1)-in-L Bilinear Prefix Identity
    =============================================================================
    For a multivariate path with return increments R_t in R^N over any sub-window
    [a, b) with length L = b - a >= 2, the exact standardized second-order Path
    Signature Lévy Area matrix:
        M(a, b) = 1 / (2 * L) * (S2 - S2.T)
    where S2 = sum_{a <= k < u < b} Z_k Z_u^T with Z_u = (R_u - mu) / sigma,
    can be evaluated in strictly O(N^2) operations -- INDEPENDENT of window length L --
    using precomputed prefix sums and outer products via two structural identities:

    1. Chen's Concatenation Identity (Iterated Integrals / Signature Splitting):
       For concatenations [0, b) = [0, a) * [a, b), the second-order iterated integral
       splits bilinearly:
           S2_{[0, b)} = S2_{[0, a)} + S2_{[a, b)} + (X_a - X_0) (X_b - X_a)^T
       Projecting onto the Lie algebra of skew-symmetric matrices (0.5 * (S2 - S2.T))
       eliminates the quadratic self-outer-product (since X_a X_a^T is symmetric):
           A^{raw}_{[a, b)} = A_b - A_a - 0.5 * (X_a X_b^T - X_b X_a^T)
       where X_t = sum_{k=0}^{t-1} R_k and A_t is the precomputed prefix skew tensor.

    2. Abel Summation by Parts Centering Identity:
       Mean-centering Z_u = R_u - mu introduces cross-terms with mu = (X_b - X_a) / L:
           sum_{a <= k < u < b} (R_k - mu)(R_u - mu)^T
       The quadratic constant term sum mu mu^T = L(L-1)/2 * mu mu^T is symmetric and
       vanishes identically upon antisymmetrization.
       The linear cross-terms sum (u - a) R_u resolve via discrete Abel summation to:
           T = sum_{u=a}^{b-1} (u - a) R_u = (Y_b - Y_a) - a * (X_b - X_a)
       where Y_t = sum_{k=0}^{t-1} k * R_k.
       The skew-symmetric correction to the area is exactly:
           Delta A^{Abel} = T mu^T - mu T^T
       Thus, the exact centered skew area is:
           A^{centered}_{[a, b)} = A^{raw}_{[a, b)} + (T mu^T - mu T^T)

    3. Standardization Scaling:
       Sample variances sigma_j^2 = (sum R_{u, j}^2 - L * mu_j^2) / (L - 1) are obtained
       in O(1) from quadratic prefix Q_t. The final standardized matrix is:
           M(a, b) = (A^{centered}_{[a, b)} / (sigma sigma^T)) / L
       matching compute_levy_matrix_vectorized(log_returns[a:b], standardize=True)
       to IEEE 754 machine precision (< 1e-13).

    Args:
        log_returns: 2D array of shape (T, N). Optional if prefix_cache is supplied.
        a: Start index of the window (inclusive, 0-based).
        b: End index of the window (exclusive, 0-based). If None, defaults to T.
        prefix_cache: Precomputed LevyPrefixCache. If None, built from log_returns.
        standardize: Whether to standardize increments per asset over window [a, b).

    Returns:
        Skew-symmetric matrix M of shape (N, N) where M[i, j] > 0 indicates asset i
        leads asset j.
    """
    if prefix_cache is None:
        if log_returns is None:
            raise ValueError("Either log_returns or prefix_cache must be provided.")
        prefix_cache = prepare_levy_prefix_cache(log_returns)

    X_cum = prefix_cache.X_cum
    A_prefix = prefix_cache.A_prefix
    Q_cum = prefix_cache.Q_cum
    Y_cum = prefix_cache.Y_cum
    T = prefix_cache.T
    N = prefix_cache.N

    if b is None:
        b = T

    if a < 0 or b > T or b - a < 2 or N < 2:
        return np.zeros((N, N), dtype=float)

    L = b - a
    X_a = X_cum[a]
    X_b = X_cum[b]
    dX = X_b - X_a

    # 1. Chen's Concatenation Identity on raw skew area
    A_a = A_prefix[a]
    A_b = A_prefix[b]
    cross_X = 0.5 * (np.outer(X_a, X_b) - np.outer(X_b, X_a))
    A_raw = A_b - A_a - cross_X

    if not standardize:
        return A_raw / float(L)

    # 2. Abel Summation Centering Correction
    mu = dX / float(L)
    sum_R2 = Q_cum[b] - Q_cum[a]
    var = (sum_R2 - float(L) * (mu**2)) / float(L - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(sd < 1e-8, 1.0, sd)

    T_vec = (Y_cum[b] - Y_cum[a]) - float(a) * dX
    A_centered = A_raw + (np.outer(T_vec, mu) - np.outer(mu, T_vec))

    # 3. Standardization and 1/L scaling
    sd_outer = np.outer(sd, sd)
    return (A_centered / sd_outer) / float(L)


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


__all__ = [
    "LevyPrefixCache",
    "build_matrices_batch",
    "build_matrix",
    "compute_levy_matrix_vectorized",
    "compute_levy_prefix_chen_abel",
    "prepare_levy_prefix_cache",
]
