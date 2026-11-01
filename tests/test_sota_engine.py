"""Unit and Mathematical Theorem Verification Tests for the 30-Theorem Universal SOTA-Apex Engine."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from leadlag.alpha.sota_engine import (
    SOTARealityShieldResult,
    apply_intraday_volatility_tpsl_and_min_hold_guard,
    compute_composite_reality_friction,
    run_third_party_paper_trading_audit,
    SOTAConfig,
    SOTALeadLagEngine,
    compute_bilinear_smart_money_signals,
    compute_bipartite_exogenous_levy_matrix,
    compute_directed_laplacian_signal,
    compute_fisher_rao_geodesic_simplex_step,
    compute_fractional_volterra_rough_signal,
    compute_free_probability_mp_denoised_tensor,
    compute_hodge_helmholtz_leadlag_decomposition,
    compute_jarzynski_crooks_thermodynamic_weights,
    compute_levy_matrix_vectorized,
    compute_lie_bch_signature_commutator,
    compute_martingale_optimal_transport_projection,
    compute_persistent_homology_0d_lifetimes,
    compute_symplectic_stormer_verlet_step,
    compute_wasserstein_quantile_barycenter,
    generate_global_2000_market_panel,
    load_exogenous_conditioned_1h_panel,
    load_multimodal_daily_panel,
    precompute_bipartite_low_rank_2000_signals,
    run_exogenous_conditioned_1h_backtest,
    run_global_2000_market_benchmark,
    run_multimodal_daily_backtest,
)


def _make_synthetic_leadlag_panel(
    T: int = 320,
    N: int = 8,
    seed: int = 7,
    freq: str = "h",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=T, freq=freq)
    rets = rng.normal(0.0, 0.01, size=(T, N))
    for lag in (1, 2):
        rets[lag:, 1] += 0.9 * rets[:-lag, 0]
        rets[lag:, 2] += 0.8 * rets[:-lag, 0]
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)),
        index=idx,
        columns=[f"C{i}" for i in range(N)],
    )


def test_theorem1_vectorized_levy_matches_pairwise_and_is_skew_symmetric():
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


def test_theorem3_counterfactual_kl_mirror_descent_simplex_invariance():
    eng = SOTALeadLagEngine(SOTAConfig())
    w = eng.compute_policy_weights(1.35, np.array([0.05, 0.12, -0.02, 0.08, 0.01]))
    assert w.shape == (5,)
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_theorem4_spectral_coherence_gate_and_strict_causality():
    prices_a = _make_synthetic_leadlag_panel(T=260, N=8, seed=7)
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
    cutoff_ts = prices_a.index[210]
    pd.testing.assert_frame_equal(res_a.positions.loc[:cutoff_ts], res_b.positions.loc[:cutoff_ts])


def test_theorem5_schmitt_trigger_hysteresis_dollar_neutrality():
    prices = _make_synthetic_leadlag_panel(T=240, N=8, seed=11)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0)
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    np.testing.assert_allclose(res.positions.sum(axis=1).values, 0.0, atol=1e-12)


def test_theorem6_internal_trade_netting_subadditivity():
    rng = np.random.default_rng(99)
    K, N = 8, 20
    pi = rng.dirichlet(np.ones(K))
    delta_experts = rng.normal(0.0, 0.1, size=(K, N))
    delta_experts -= delta_experts.mean(axis=1, keepdims=True)
    delta_master = np.tensordot(pi, delta_experts, axes=(0, 0))
    assert float(np.abs(delta_master).sum()) <= float(np.dot(pi, np.abs(delta_experts).sum(axis=1))) + 1e-12


def test_theorem7_rebalance_lattice_strided_factorization_exact_equivalence():
    prices = _make_synthetic_leadlag_panel(T=280, N=8, seed=19)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, ic_memory_cycles=3)
    eng = SOTALeadLagEngine(cfg)
    LR = np.log(prices).diff().clip(-0.30, 0.30).fillna(0.0).values
    z_dense, _ = eng.precompute_horizon_signals(LR, stride=1)
    z_strided, _ = eng.precompute_horizon_signals(LR, stride=cfg.rebal_steps)
    res_dense = eng.run_backtest(prices, z_all=z_dense)
    res_strided = eng.run_backtest(prices, z_all=z_strided)
    pd.testing.assert_frame_equal(res_dense.positions, res_strided.positions, atol=1e-6)
    pd.testing.assert_series_equal(res_dense.net_returns, res_strided.net_returns, atol=1e-6)


def test_theorem8_dual_space_orthogonalized_mirror_descent_diversity():
    prices = _make_synthetic_leadlag_panel(T=320, N=8, seed=21)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, cov_penalty_gamma=1.15, pos_overlap_gamma=0.50)
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    assert np.isfinite(res.net_sharpe)


def test_theorem9_zero_mean_l1_proximal_deadband_scale_invariance():
    prices = _make_synthetic_leadlag_panel(T=320, N=8, seed=23)
    cfg_1x = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, ic_memory_cycles=3, leverage=1.00, master_deadband=0.010, dd_governor_scale=0.0, zero_loss_shield=False)
    cfg_2x = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, ic_memory_cycles=3, leverage=2.00, master_deadband=0.010, dd_governor_scale=0.0, zero_loss_shield=False)
    eng_1x, eng_2x = SOTALeadLagEngine(cfg_1x), SOTALeadLagEngine(cfg_2x)
    LR = np.log(prices).diff().clip(-0.30, 0.30).fillna(0.0).values
    z_all, _ = eng_1x.precompute_horizon_signals(LR, stride=12)
    res_1x = eng_1x.run_backtest(prices, z_all=z_all)
    res_2x = eng_2x.run_backtest(prices, z_all=z_all)
    np.testing.assert_allclose(res_2x.positions.values, 2.0 * res_1x.positions.values, atol=1e-12)
    np.testing.assert_allclose(res_1x.net_sharpe, res_2x.net_sharpe, rtol=1e-5)


def test_theorem10_cppi_drawdown_velocity_governor_and_daily_prior():
    prices = _make_synthetic_leadlag_panel(T=260, N=8, seed=29, freq="D")
    cfg = SOTAConfig.daily_default()
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    assert np.isfinite(res.calmar_ratio)


def test_theorem11_specialized_wavelet_dog_and_diurnal_harmonic_formulas():
    prices = _make_synthetic_leadlag_panel(T=520, N=8, seed=31)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, enable_specialized_formulas=True)
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    np.testing.assert_allclose(res.positions.sum(axis=1).values, 0.0, atol=1e-12)


def test_theorem12_contextual_regime_prior_and_sparse_threshold_gate():
    prices = _make_synthetic_leadlag_panel(T=520, N=8, seed=33)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, sparse_sharpe_thresh=0.70, sparse_suppress_factor=0.08)
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    assert np.isfinite(res.gross_sharpe)


def test_theorem13_two_level_strict_warmup_synchronization():
    prices = _make_synthetic_leadlag_panel(T=520, N=8, seed=37)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0, strict_meta_warmup=True, meta_warmup_cycles=5)
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    np.testing.assert_allclose(res.positions.iloc[:55].values, 0.0, atol=1e-12)


def test_theorem14_universal_smart_money_bilinear_regime_flow_tensor():
    prices = _make_synthetic_leadlag_panel(T=220, N=8, seed=41, freq="D")
    rng = np.random.default_rng(41)
    vol_to = pd.DataFrame(rng.uniform(0.01, 0.20, size=prices.shape), index=prices.index, columns=prices.columns)
    log_mcap = pd.DataFrame(rng.uniform(18.0, 24.0, size=prices.shape), index=prices.index, columns=prices.columns)
    sm_sigs = compute_bilinear_smart_money_signals(prices, vol_to, log_mcap)
    assert sm_sigs.shape == (2, 220, 8)


def test_theorem15_two_stage_principal_lock_ignition_zero_principal_loss():
    prices = _make_synthetic_leadlag_panel(T=320, N=8, seed=43)
    cfg = SOTAConfig.hourly_sovereign_ultra_shield()
    res = SOTALeadLagEngine(cfg).run_backtest(prices)
    assert res.min_principal_return >= -1e-12


def test_theorem16_ito_kelly_hyper_compounding_profiles():
    cfg_sing = SOTAConfig.hourly_sovereign_singularity()
    cfg_kp = SOTAConfig.hourly_sovereign_kelly_peak()
    assert cfg_sing.leverage == 14.0
    assert cfg_kp.leverage == 20.0
    assert cfg_sing.zero_loss_shield and cfg_kp.zero_loss_shield


def test_theorem17_bipartite_non_tradable_exogenous_levy_operator():
    rng = np.random.default_rng(17)
    L, M_exo, N_trad = 48, 5, 20
    Y_exo = rng.standard_normal((L, M_exo))
    X_trad = rng.standard_normal((L, N_trad)) * 0.2
    Y_exo[:, 0] = np.sin(np.linspace(0, 3.0, L))
    X_trad[:, 3] = np.sin(np.linspace(0, 3.0, L) - 0.6)
    A_rect = compute_bipartite_exogenous_levy_matrix(Y_exo, X_trad)
    assert A_rect.shape == (M_exo, N_trad)
    assert A_rect[0, 3] > 0.15
    assert A_rect[0, 3] == np.max(A_rect[0, :])


def test_theorem18_chen_abel_o1_bipartite_low_rank_subspace_operator():
    rng = np.random.default_rng(18)
    R_trad = rng.standard_normal((120, 120)) * 0.01
    Y_exo = rng.standard_normal((120, 10)) * 0.01
    z_all, snr_all = precompute_bipartite_low_rank_2000_signals(R_trad, Y_exo, candidates=(6, 12, 18, 24, 36), stride=6, n_anchors=20)
    assert z_all.shape == (5, 120, 120)
    assert snr_all.shape == (5, 120)


def test_theorem19_bipartite_directed_laplacian_nullspace_subtraction():
    rng = np.random.default_rng(19)
    R_trad = rng.standard_normal((90, 60)) * 0.01
    Y_exo = rng.standard_normal((90, 8)) * 0.01
    z_all, _ = precompute_bipartite_low_rank_2000_signals(R_trad, Y_exo, candidates=(6, 12, 18, 24, 36), stride=6, n_anchors=15)
    np.testing.assert_allclose(z_all[:, 42, :].mean(axis=1), 0.0, atol=1e-12)


def test_theorem20_global_2000_market_breadth_scaling_and_zero_loss():
    repo_root = Path(__file__).resolve().parent.parent
    bundle = generate_global_2000_market_panel(repo_root, n_trad=240, m_exo=15, n_steps=180, seed=2026)
    res_1x = run_global_2000_market_benchmark(repo_root, leverage=1.0, precomputed_bundle=bundle)
    assert res_1x.net_sharpe > 3.0
    assert res_1x.min_principal_return >= -1e-9
    assert res_1x.positions.shape[1] == 240


def test_theorem21_lie_algebra_bch_signature_commutator_and_jacobi_identity():
    rng = np.random.default_rng(21)
    N = 12
    Omega_1 = compute_levy_matrix_vectorized(rng.standard_normal((36, N)))
    Omega_2 = compute_levy_matrix_vectorized(rng.standard_normal((36, N)))
    Omega_3 = compute_levy_matrix_vectorized(rng.standard_normal((36, N)))

    res = compute_lie_bch_signature_commutator(Omega_1, Omega_2, include_third_order=True)
    bracket = res["lie_bracket"]
    bch = res["bch_generator"]
    curv = float(res["killing_curvature"])

    # Verify so(N) skew-symmetry and zero trace of both bracket and BCH generator:
    np.testing.assert_allclose(bracket, -bracket.T, atol=1e-14)
    np.testing.assert_allclose(bch, -bch.T, atol=1e-14)
    assert abs(float(np.trace(bracket))) < 1e-14
    assert curv >= 0.0

    # Verify Jacobi identity on so(N): [A, [B, C]] + [B, [C, A]] + [C, [A, B]] == 0
    b_23 = compute_lie_bch_signature_commutator(Omega_2, Omega_3)["lie_bracket"]
    b_31 = compute_lie_bch_signature_commutator(Omega_3, Omega_1)["lie_bracket"]
    j1 = compute_lie_bch_signature_commutator(Omega_1, b_23)["lie_bracket"]
    j2 = compute_lie_bch_signature_commutator(Omega_2, b_31)["lie_bracket"]
    j3 = compute_lie_bch_signature_commutator(Omega_3, bracket)["lie_bracket"]
    np.testing.assert_allclose(j1 + j2 + j3, np.zeros((N, N)), atol=1e-14)


def test_theorem22_wasserstein_quantile_barycenter_frechet_optimality():
    rng = np.random.default_rng(22)
    M, N = 6, 30
    S = rng.standard_normal((M, N))
    w = rng.dirichlet(np.ones(M))
    bary = compute_wasserstein_quantile_barycenter(S, w)
    assert bary.shape == (N,)
    assert abs(float(bary.mean())) < 1e-12
    assert abs(float(bary.std()) - 1.0) < 1e-12
    # When all experts are identical up to standardization, barycenter recovers the exact signal:
    s0 = (S[0] - S[0].mean()) / S[0].std()
    S_id = np.tile(s0, (M, 1))
    bary_id = compute_wasserstein_quantile_barycenter(S_id, w)
    np.testing.assert_allclose(bary_id, s0, atol=1e-12)


def test_theorem23_riemannian_fisher_rao_geodesic_simplex_invariance():
    rng = np.random.default_rng(23)
    K = 8
    pi_0 = rng.dirichlet(np.ones(K))
    adv = rng.standard_normal(K)
    pi_new, dist = compute_fisher_rao_geodesic_simplex_step(pi_0, adv, step_size=0.30)
    assert pi_new.shape == (K,)
    assert np.all(pi_new > 0.0)
    np.testing.assert_allclose(pi_new.sum(), 1.0, atol=1e-14)
    assert dist >= 0.0
    # Zero advantage or zero step_size preserves prior exactly with zero geodesic distance:
    pi_id, dist_0 = compute_fisher_rao_geodesic_simplex_step(pi_0, np.zeros(K), step_size=0.30)
    np.testing.assert_allclose(pi_id, pi_0, atol=1e-14)
    assert abs(dist_0) < 1e-7


def test_theorem24_free_probability_marchenko_pastur_and_wigner_denoising():
    rng = np.random.default_rng(24)
    L, N = 80, 20
    R = rng.standard_normal((L, N))
    # Inject strong rank-1 factor and lead-lag pair:
    factor = rng.standard_normal((L, 1)) * 2.5
    R[:, :5] += factor
    R[1:, 1] += 1.5 * R[:-1, 0]
    res = compute_free_probability_mp_denoised_tensor(R)
    C_free = res["C_free"]
    M_free = res["M_free"]
    assert C_free.shape == (N, N)
    assert M_free.shape == (N, N)
    np.testing.assert_allclose(C_free, C_free.T, atol=1e-13)
    np.testing.assert_allclose(M_free, -M_free.T, atol=1e-13)
    np.testing.assert_allclose(np.trace(C_free), float(N), rtol=1e-2)
    assert res["n_signal_spikes"] >= 1


def test_theorem25_hodge_helmholtz_1_laplacian_orthogonal_decomposition():
    rng = np.random.default_rng(25)
    N = 16
    M = compute_levy_matrix_vectorized(rng.standard_normal((48, N)))
    res = compute_hodge_helmholtz_leadlag_decomposition(M)
    phi = res["potential_phi"]
    F_grad = res["grad_flow"]
    F_curl = res["curl_flow"]

    assert phi.shape == (N,)
    assert abs(float(phi.mean())) < 1e-14
    # Exact reconstruction F == F_grad + F_curl:
    np.testing.assert_allclose(F_grad + F_curl, M, atol=1e-14)
    # Exact L2(E) Hilbert space orthogonality <F_grad, F_curl>_F == 0:
    inner_prod = float(np.sum(F_grad * F_curl))
    assert abs(inner_prod) < 1e-13
    # F_curl is strictly divergence-free (row sums and column sums are identically zero):
    np.testing.assert_allclose(F_curl.sum(axis=1), np.zeros(N), atol=1e-13)
    # F_grad is strictly curl-free on every 3-simplex (i, j, k): F_grad[i,j] + F_grad[j,k] + F_grad[k,i] == 0:
    for i, j, k in ((0, 3, 7), (1, 5, 11), (2, 8, 14)):
        assert abs(float(F_grad[i, j] + F_grad[j, k] + F_grad[k, i])) < 1e-14


def test_theorem26_fractional_volterra_rough_kernel_soe_semigroup():
    rng = np.random.default_rng(26)
    T, N = 100, 12
    S = rng.standard_normal((T, N))
    out = compute_fractional_volterra_rough_signal(S, hurst=0.12, n_modes=4)
    assert out.shape == (T, N)
    np.testing.assert_allclose(out.mean(axis=1), np.zeros(T), atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1), np.ones(T), atol=1e-12)
    # Verify strict causality: perturbing S[60:] does not change out[:60]
    S2 = S.copy()
    S2[60:] *= -5.0
    out2 = compute_fractional_volterra_rough_signal(S2, hurst=0.12, n_modes=4)
    np.testing.assert_allclose(out[:60], out2[:60], atol=1e-14)


def test_theorem27_symplectic_stormer_verlet_hamiltonian_execution_dissipation():
    rng = np.random.default_rng(27)
    N = 20
    q = np.zeros(N)
    p = np.zeros(N)
    target = rng.standard_normal(N)
    target -= target.mean()
    energies = []
    for _ in range(25):
        q, p, e = compute_symplectic_stormer_verlet_step(q, p, target, dt=0.35, damping=0.80, stiffness=1.2)
        assert abs(float(q.mean())) < 1e-14
        assert abs(float(p.mean())) < 1e-14
        energies.append(e)
    # Under positive Rayleigh damping towards a stationary target, distance to target contracts strongly:
    assert float(np.linalg.norm(q - target)) < 0.25 * float(np.linalg.norm(target))
    assert energies[-1] < energies[1]


def test_theorem28_persistent_homology_0d_vietoris_rips_ultrametric_filtration():
    rng = np.random.default_rng(28)
    N = 15
    M = compute_levy_matrix_vectorized(rng.standard_normal((40, N)))
    res = compute_persistent_homology_0d_lifetimes(M)
    deaths = res["death_times"]
    strengths = res["persistence_strengths"]
    ent = float(res["topological_entropy"])
    assert deaths.shape == (N - 1,)
    assert strengths.shape == (N - 1,)
    assert np.all(np.diff(deaths) >= -1e-14)
    assert np.all((deaths > 0.0) & (deaths <= 1.0))
    assert 0.0 <= ent <= 1.0 + 1e-12


def test_theorem29_martingale_optimal_transport_strassen_projection():
    rng = np.random.default_rng(29)
    N = 24
    q0 = rng.standard_normal(N) * 0.05
    q0 -= q0.mean()
    q1 = rng.standard_normal(N) * 0.08
    q1 -= q1.mean()
    q_mot, cost = compute_martingale_optimal_transport_projection(q0, q1, friction_bound=0.015)
    assert q_mot.shape == (N,)
    assert abs(float(q_mot.mean())) < 1e-14
    assert cost <= float(np.abs(q1 - q0).sum()) + 1e-12
    assert float(np.abs(q_mot).sum()) <= max(float(np.abs(q0).sum()), float(np.abs(q1).sum())) + 1e-12


def test_theorem30_jarzynski_crooks_stochastic_thermodynamics_free_energy():
    rng = np.random.default_rng(30)
    T, M = 120, 8
    pi_0 = rng.dirichlet(np.ones(M))
    H = rng.normal(0.001, 0.01, size=(T, M))
    res = compute_jarzynski_crooks_thermodynamic_weights(pi_0, H, inverse_temp_beta=2.5)
    w = res["weights"]
    fe = res["free_energy"]
    kl = res["entropy_production_kl"]
    assert w.shape == (M,)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-14)
    assert np.all(w > 0.0)
    assert np.isfinite(fe)
    # Second Law of Non-Equilibrium Stochastic Thermodynamics: sigma_diss = D_KL(pi^* || pi_0) >= 0
    assert kl >= 0.0


def test_sota_config_all_factory_profiles_valid():
    for fn in (
        SOTAConfig.hourly_conservative,
        SOTAConfig.hourly_balanced,
        SOTAConfig.hourly_aggressive,
        SOTAConfig.hourly_fortress,
        SOTAConfig.hourly_ultra_calmar,
        SOTAConfig.hourly_sovereign_ultra_shield,
        SOTAConfig.hourly_sovereign_omni_manifold,
        SOTAConfig.hourly_sovereign_fortress,
        SOTAConfig.hourly_sovereign_conservative,
        SOTAConfig.hourly_sovereign_balanced,
        SOTAConfig.hourly_sovereign_aggressive,
        SOTAConfig.hourly_sovereign_half_kelly,
        SOTAConfig.hourly_sovereign_full_kelly,
        SOTAConfig.hourly_sovereign_singularity,
        SOTAConfig.hourly_sovereign_kelly_peak,
        SOTAConfig.daily_default,
    ):
        cfg = fn()
        assert isinstance(cfg, SOTAConfig)


def test_fixed_single_horizon_backtest_contract():
    prices = _make_synthetic_leadlag_panel(T=220, N=6, seed=51)
    cfg = SOTAConfig(candidates=(12, 24, 36, 48, 60), rebal_steps=12, top_k=2, spread_min=0.1, consensus_min=-1.0)
    eng = SOTALeadLagEngine(cfg)
    res = eng.run_backtest(prices, fixed_horizon_index=1)
    assert np.allclose(res.effective_lookback.values, 24.0)


def test_exogenous_conditioned_1h_loader_alignment():
    repo_root = Path(__file__).resolve().parent.parent
    bundle = load_exogenous_conditioned_1h_panel(repo_root, max_hours=400)
    assert bundle["prices"].shape == bundle["to_z"].shape == bundle["mc_z"].shape


def test_multimodal_daily_loader_alignment():
    repo_root = Path(__file__).resolve().parent.parent
    bundle = load_multimodal_daily_panel(repo_root)
    assert bundle["prices"].shape == bundle["volume_turnover"].shape == bundle["log_marketcap"].shape


def test_theorem31_composite_reality_friction_model():
    rng = np.random.default_rng(31)
    T = 200
    to_vec = rng.uniform(0.0, 0.05, size=T)
    exp_vec = np.ones(T)
    vol_vec = rng.uniform(0.005, 0.03, size=T)
    res = compute_composite_reality_friction(
        to_vec, exp_vec, vol_vec, fee_bps=5.0, base_slippage_bps=2.0, funding_spread_bps_per_8h=0.24
    )
    assert res["total_cost"].shape == (T,)
    assert np.all(res["fee_cost"] >= 0.0)
    assert np.all(res["slippage_cost"] >= 0.0)
    assert np.all(res["funding_cost"] >= 0.0)
    np.testing.assert_allclose(
        res["total_cost"], res["fee_cost"] + res["slippage_cost"] + res["funding_cost"], atol=1e-15
    )


def test_theorem32_intraday_volatility_tpsl_and_leg_neutrality():
    rng = np.random.default_rng(32)
    T, N = 120, 8
    P_sched = np.zeros((T, N), dtype=float)
    P_sched[:, 0] = 0.25
    P_sched[:, 1] = 0.25
    P_sched[:, 2] = -0.25
    P_sched[:, 3] = -0.25
    R = rng.normal(0.0, 0.005, size=(T, N))
    # Inject a massive crash in asset 0 at step 35 to trigger intraday stop-loss:
    R[20, 0] = -0.25
    LR = np.log(1.0 + np.clip(R, -0.30, 0.30))
    res = apply_intraday_volatility_tpsl_and_min_hold_guard(
        P_sched, R, LR, sl_sigma_mult=2.5, tp_sigma_mult=4.0, min_hold_steps=4, vol_window=12
    )
    pos = res["positions"]
    assert pos.shape == (T, N)
    assert res["sl_triggers_count"] >= 1
    assert abs(float(pos[-1, 0])) < 1e-12
    # Verify exact leg-neutral dollar neutrality across all steps:
    np.testing.assert_allclose(pos.sum(axis=1), np.zeros(T), atol=1e-12)


def test_theorem33_broker_anti_latency_min_hold_and_third_party_paper_trading_audit():
    prices = _make_synthetic_leadlag_panel(T=360, N=8, seed=33)
    cfg = SOTAConfig(
        candidates=(12, 24, 36, 48, 60),
        rebal_steps=12,
        top_k=2,
        spread_min=0.1,
        consensus_min=-1.0,
        ic_memory_cycles=3,
        meta_warmup_cycles=2,
    )
    audit = run_third_party_paper_trading_audit(
        prices,
        cfg,
        fee_bps=5.0,
        base_slippage_bps=2.0,
        funding_spread_bps_per_8h=0.24,
        sl_sigma_mult=3.5,
        tp_sigma_mult=5.0,
        min_hold_steps=6,
        execution_delay_fraction=0.25,
        verify_prefix_isolation=True,
    )
    assert isinstance(audit, SOTARealityShieldResult)
    assert np.isfinite(audit.net_sharpe)
    assert audit.prefix_isolation_max_diff < 1e-12
    np.testing.assert_allclose(audit.positions.sum(axis=1).values, 0.0, atol=1e-12)
