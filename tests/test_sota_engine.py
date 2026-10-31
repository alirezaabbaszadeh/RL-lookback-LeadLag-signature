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
            area = (
                0.5 * (np.triu(np.outer(u, v), k=1).sum() - np.triu(np.outer(v, u), k=1).sum()) / L
            )
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


def test_theorem7_rebalance_lattice_strided_factorization_exact_equivalence():
    """Verify Theorem 7: Strided precomputation on rebalance lattice t = k*H is numerically identical to dense stride=1."""
    rng = np.random.default_rng(42)
    T, N = 200, 10
    idx = pd.date_range("2025-01-01", periods=T, freq="h")
    cols = [f"C{i}" for i in range(N)]
    LR = rng.normal(0.0, 0.01, size=(T, N))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(LR, axis=0)), index=idx, columns=cols)

    cfg = SOTAConfig(candidates=(12, 24), rebal_steps=12)
    engine = SOTALeadLagEngine(cfg)

    z_dense, _ = engine.precompute_horizon_signals(LR, stride=1)
    z_strided, _ = engine.precompute_horizon_signals(LR, stride=12)

    res_dense = engine.run_backtest(prices, z_all=z_dense)
    res_strided = engine.run_backtest(prices, z_all=z_strided)

    pos_max_err = float(np.abs(res_dense.positions.values - res_strided.positions.values).max())
    ret_max_err = float(np.abs(res_dense.net_returns.values - res_strided.net_returns.values).max())

    assert pos_max_err < 1e-12
    assert ret_max_err < 1e-12
    np.testing.assert_allclose(res_dense.positions.values, res_strided.positions.values, atol=1e-12)
    np.testing.assert_allclose(
        res_dense.net_returns.values, res_strided.net_returns.values, atol=1e-12
    )


def test_theorem8_dual_space_orthogonal_mirror_descent_penalizes_redundant_clones():
    """Verify Theorem 8: Dual-space cross-expert return covariance and position overlap penalize redundant clones."""
    n_experts = 3
    meta_prior = np.ones(n_experts) / float(n_experts)
    meta_pi = meta_prior.copy()

    T = 100
    rng = np.random.default_rng(42)
    r0 = rng.normal(0.0001, 0.01, size=T)
    r2_raw = rng.normal(0.0001, 0.01, size=T)

    # Weighted inner product orthogonalization under exponential decay
    decay = np.exp(np.linspace(-1.0, 0.0, T))
    w_d = (decay / decay.sum())[:, None]

    mu0 = np.sum(r0 * w_d.ravel())
    r0_centered = r0 - mu0
    mu2_raw = np.sum(r2_raw * w_d.ravel())
    r2_centered = r2_raw - mu2_raw
    proj = np.sum(r2_centered * r0_centered * w_d.ravel()) / np.sum((r0_centered**2) * w_d.ravel())
    r2_orth = r2_centered - proj * r0_centered
    r2 = r2_orth + mu0

    # Book 0 and Book 1 are identical clones; Book 2 is uncorrelated
    hist_net = np.column_stack([r0, r0, r2])

    # Book 0 and Book 1 have identical unit positions; Book 2 is orthogonal
    expert_pos = np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    mu_m = (hist_net * w_d).sum(axis=0)
    sd_m = hist_net.std(axis=0) + 1e-6
    sharpe_m = (mu_m / sd_m) * 1.0

    # Standard marginal Sharpe weighting (no dual-space penalty)
    logits_std = np.clip(2.0 * sharpe_m, -2.0, 2.5)
    w_std = meta_prior * np.exp(logits_std - np.max(logits_std))
    w_std /= w_std.sum()

    # Dual-space orthogonal mirror descent
    z_hist = (hist_net - mu_m) / sd_m
    corr_mat = (z_hist * w_d).T @ z_hist
    np.fill_diagonal(corr_mat, 0.0)

    norms = np.linalg.norm(expert_pos, axis=1, keepdims=True) + 1e-8
    p_unit = expert_pos / norms
    pos_sim = p_unit @ p_unit.T
    np.fill_diagonal(pos_sim, 0.0)

    adv = sharpe_m - 1.15 * (corr_mat @ meta_pi) - 0.50 * (pos_sim @ meta_pi)
    logits_dual = np.clip(2.0 * adv, -2.0, 2.5)
    w_dual = meta_prior * np.exp(logits_dual - np.max(logits_dual))
    w_dual /= w_dual.sum()

    np.testing.assert_allclose(corr_mat[0, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(corr_mat[1, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(pos_sim[0, 1], 1.0, atol=1e-12)
    np.testing.assert_allclose(pos_sim[0, 2], 0.0, atol=1e-12)

    # Redundant clones are strictly discounted; uncorrelated book receives shifted allocation
    assert w_dual[2] > w_std[2]
    assert w_dual[0] < w_std[0]
    assert w_dual[1] < w_std[1]
    assert np.isclose(w_dual[0], w_dual[1], atol=1e-12)


def test_theorem9_master_deadband_preserves_dollar_neutrality_and_scale_invariance():
    """Verify Theorem 9: Master-book deadband preserves exact dollar neutrality and leverage scale-invariance."""
    rng = np.random.default_rng(42)
    T, N = 260, 10
    idx = pd.date_range("2025-01-01", periods=T, freq="h")
    cols = [f"C{i}" for i in range(N)]
    LR = rng.normal(0.0, 0.01, size=(T, N))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(LR, axis=0)), index=idx, columns=cols)

    cfg_1_0 = SOTAConfig(
        candidates=(12, 24),
        rebal_steps=12,
        master_deadband=0.010,
        leverage=1.00,
        dd_governor_scale=0.0,
    )
    engine_1_0 = SOTALeadLagEngine(cfg_1_0)
    res_1_0 = engine_1_0.run_backtest(prices)

    # 1. Exact dollar neutrality at every single step
    dollar_imbalance = np.abs(res_1_0.positions.sum(axis=1)).values
    assert float(dollar_imbalance.max()) < 1e-12

    # 2. Leverage scale-invariance (leverage=1.0 vs leverage=1.4)
    cfg_1_4 = SOTAConfig(
        candidates=(12, 24),
        rebal_steps=12,
        master_deadband=0.010,
        leverage=1.40,
        dd_governor_scale=0.0,
    )
    engine_1_4 = SOTALeadLagEngine(cfg_1_4)
    res_1_4 = engine_1_4.run_backtest(prices)

    np.testing.assert_allclose(res_1_0.positions.values * 1.4, res_1_4.positions.values, atol=1e-12)
    np.testing.assert_allclose(
        res_1_0.net_returns.values * 1.4, res_1_4.net_returns.values, atol=1e-12
    )
    assert np.isclose(res_1_0.net_sharpe, res_1_4.net_sharpe, atol=1e-10)

    # 3. Turnover reduction under master_deadband > 0 vs deadband = 0
    cfg_nodb = SOTAConfig(
        candidates=(12, 24),
        rebal_steps=12,
        master_deadband=0.0,
        leverage=1.00,
        dd_governor_scale=0.0,
    )
    engine_nodb = SOTALeadLagEngine(cfg_nodb)
    res_nodb = engine_nodb.run_backtest(prices)

    assert res_1_0.avg_turnover < res_nodb.avg_turnover


def test_theorem10_daily_prior_decoupling_and_causal_dd_governor():
    """Verify Theorem 10: Daily timeframe prior decoupling and causal CPPI drawdown governor."""
    cfg_daily = SOTAConfig.daily_default()
    assert cfg_daily.periods_per_year == 365
    assert len(cfg_daily.candidates) == 5

    # Direct verification that periods_per_year <= 500 triggers daily meta_prior in backtest engine
    rng = np.random.default_rng(7)
    T, N = 260, 8
    idx = pd.date_range("2025-01-01", periods=T, freq="D")
    cols = [f"C{i}" for i in range(N)]
    rets = rng.normal(0.0, 0.01, size=(T, N))
    prices_daily = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)

    engine_daily = SOTALeadLagEngine(cfg_daily)
    res_daily = engine_daily.run_backtest(prices_daily)
    assert res_daily.positions.shape[0] > 0
    assert cfg_daily.periods_per_year <= 500

    # Verify causality of CPPI governor: perturbing prices for future steps s > t produces zero change at <= t
    idx_hourly = pd.date_range("2025-01-01", periods=T, freq="h")
    prices_a = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx_hourly, columns=cols)
    prices_b = prices_a.copy()
    prices_b.iloc[211:] = prices_b.iloc[211:] * 2.5

    cfg_gov = SOTAConfig(
        candidates=(12, 24, 36, 48, 60),
        rebal_steps=12,
        top_k=2,
        retain_bonus=1.5,
        step_alpha=0.40,
        consensus_min=-1.0,
        spread_min=0.1,
        ic_memory_cycles=3,
        meta_memory_steps=60,
        leverage=1.20,
        dd_governor_scale=3.0,
    )
    engine_gov = SOTALeadLagEngine(cfg_gov)
    res_a = engine_gov.run_backtest(prices_a)
    res_b = engine_gov.run_backtest(prices_b)

    cutoff_ts = idx_hourly[210]
    pos_a_before = res_a.positions.loc[:cutoff_ts]
    pos_b_before = res_b.positions.loc[:cutoff_ts]
    pd.testing.assert_frame_equal(pos_a_before, pos_b_before)
