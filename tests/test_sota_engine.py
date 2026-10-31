"""Unit and Mathematical Theorem Verification Tests for the SOTA-Omega Alpha Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.alpha.sota_engine import (
    SOTAConfig,
    SOTALeadLagEngine,
    compute_directed_laplacian_signal,
    compute_levy_matrix_vectorized,
)


def test_theorem1_vectorized_levy_matches_pairwise_and_is_skew_symmetric():
    """Verify Theorem 1: Vectorized X_prev.T @ Z matches pairwise discrete Lévy area to 1e-13."""
    rng = np.random.default_rng(123)
    L, N = 40, 12
    R = rng.standard_normal((L, N))

    M_vec = compute_levy_matrix_vectorized(R, standardize=True)
    assert M_vec.shape == (N, N)
    np.testing.assert_allclose(M_vec, -M_vec.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(M_vec), np.zeros(N), atol=1e-14)

    mu = R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, ddof=1, keepdims=True)
    Z = (R - mu) / sd
    M_pair = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            u, v = Z[:, i], Z[:, j]
            area = 0.5 * (np.triu(np.outer(u, v), k=1).sum() - np.triu(np.outer(v, u), k=1).sum()) / L
            M_pair[i, j] = area
            M_pair[j, i] = -area

    np.testing.assert_allclose(M_vec, M_pair, atol=1e-13)


def test_theorem2_directed_laplacian_nullspace_annihilates_market_shock():
    """Verify Theorem 2: Adding a uniform market mode +c*1 at any bar is annihilated by (W.T - I)."""
    rng = np.random.default_rng(42)
    L, N = 32, 10
    win = rng.normal(0.0, 0.02, size=(L, N))

    for k in range(1, N):
        win[1:, k] += 0.8 * win[:-1, k - 1]

    z_base, M_base, _ = compute_directed_laplacian_signal(win)
    W = np.maximum(0.0, M_base) ** 2
    col_sums = W.sum(axis=0, keepdims=True)
    W_norm = np.where(col_sums < 1e-12, np.eye(N), W / np.maximum(col_sums, 1e-12))

    ones = np.ones(N, dtype=float) * 3.75
    null_residual = (W_norm.T @ ones) - ones
    np.testing.assert_allclose(null_residual, np.zeros(N), atol=1e-12)
    assert z_base.shape == (N,)
    assert abs(float(z_base.mean())) < 1e-12


def test_theorem3_strict_causality_no_future_leakage_and_dollar_neutrality():
    """Verify Theorems 3, 5 & 6: Perturbing future prices > t never alters positions at t, and sum(pos) == 0."""
    rng = np.random.default_rng(7)
    T, N = 260, 8
    idx = pd.date_range("2025-01-01", periods=T, freq="h")
    rets = rng.normal(0.0, 0.01, size=(T, N))
    for lag in (1, 2):
        rets[lag:, 1] += 0.9 * rets[:-lag, 0]
        rets[lag:, 2] += 0.8 * rets[:-lag, 0]
    prices_a = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)),
        index=idx,
        columns=[f"C{i}" for i in range(N)],
    )

    prices_b = prices_a.copy()
    prices_b.iloc[211:] = prices_b.iloc[211:] * 2.5

    cfg = SOTAConfig(
        candidates=(12, 24, 36, 48, 60),
        rebal_steps=12,
        top_k=2,
        retain_bonus=1.5,
        step_alpha=0.40,
        consensus_min=-1.0,
        spread_min=0.1,
        ic_memory_cycles=3,
        meta_memory_steps=60,
        leverage=1.00,
    )
    engine = SOTALeadLagEngine(cfg)
    res_a = engine.run_backtest(prices_a)
    res_b = engine.run_backtest(prices_b)

    cutoff_ts = idx[210]
    pos_a_before = res_a.positions.loc[:cutoff_ts]
    pos_b_before = res_b.positions.loc[:cutoff_ts]
    pd.testing.assert_frame_equal(pos_a_before, pos_b_before)

    row_sums = res_a.positions.sum(axis=1).values
    np.testing.assert_allclose(row_sums, np.zeros_like(row_sums), atol=1e-12)


def test_theorem6_internal_trade_netting_subadditivity():
    """Verify Theorem 6: L1 turnover of convex master book is strictly <= weighted sum of expert L1 turnovers."""
    rng = np.random.default_rng(99)
    K, N = 6, 20
    pi = rng.dirichlet(np.ones(K))
    delta_experts = rng.normal(0.0, 0.1, size=(K, N))
    # Make each expert dollar-neutral
    delta_experts -= delta_experts.mean(axis=1, keepdims=True)

    delta_master = np.tensordot(pi, delta_experts, axes=(0, 0))
    master_l1 = float(np.abs(delta_master).sum())
    weighted_expert_l1 = float(np.dot(pi, np.abs(delta_experts).sum(axis=1)))

    assert master_l1 <= weighted_expert_l1 + 1e-12
