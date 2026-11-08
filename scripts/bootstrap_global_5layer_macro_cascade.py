#!/usr/bin/env python3
"""Global 5-Layer Macro-to-Micro Smart-Money Cascade & Hybrid Regime-Tilt Audit (Theorems 56-60).

Demonstrates the complete 5-Layer Global Liquidity Cascade on real historical market & macro data:
  - Layer 1: Central Bank Plumbing (US T-Bill / SOFR proxy, Net Liquidity & G4 Monetary Tensor - Theorem 59)
  - Layer 2: Sovereign Debt & Credit Spreads
  - Layer 3: Global Reserve FX & Carry (DXY / USDJPY / Cross-Asset Basis proxies)
  - Layer 4: Physical & Strategic Commodities / Bellwether Majors
  - Layer 5: Broad Tradable Risk Universe (Equities & High-Beta Markets)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.alpha.sota_engine import (
    compute_continuous_time_rough_signature_kernel,
    compute_five_layer_macro_liquidity_cascade_operator,
    compute_g4_central_bank_monetary_impulse_tensor,
    compute_macro_regime_gated_alpha_tilt_allocation,
    compute_multi_horizon_daily_hourly_1m_state_space_fusion,
    load_real_multimodal_exogenous_2000_panel,
)


def main() -> None:
    t0 = time.time()
    bundle = load_real_multimodal_exogenous_2000_panel(ROOT, max_hours=3000)
    R_trad = bundle["R_trad"]
    Y_exo = bundle["Y_exo"]
    T, N = R_trad.shape

    # Partition real exogenous & macro channels into the 5 structural layers
    L1_cb = Y_exo[:, -2:]  # Macro T-Bill & BGCI index channels
    L2_rates = Y_exo[:, :10]
    L3_fx = Y_exo[:, 10:20]
    L4_comm = R_trad[:, :8]
    L5_risk = R_trad

    cascade = compute_five_layer_macro_liquidity_cascade_operator(L1_cb, L2_rates, L3_fx, L4_comm, L5_risk)
    g4 = compute_g4_central_bank_monetary_impulse_tensor(L1_cb, L2_rates[:, :3], L3_fx[:, :3], L4_comm[:, :3])

    t_grid = np.arange(200, dtype=float)
    ct_kernel = compute_continuous_time_rough_signature_kernel(
        t_grid, np.cumsum(L1_cb[:200, 0]), t_grid + 0.5, np.cumsum(L5_risk[:200, 0])
    )

    fusion = compute_multi_horizon_daily_hourly_1m_state_space_fusion(
        Y_exo[:, :N],
        cascade["layer5_cascade_signals"],
        Y_exo[:, N : 2 * N],
        R_trad,
    )

    w_neutral = np.tanh(fusion["fused_multi_horizon_signal"])
    w_neutral -= w_neutral.mean(axis=1, keepdims=True)
    w_neutral /= np.maximum(np.abs(w_neutral).sum(axis=1, keepdims=True), 1e-8)

    w_macro_dir = np.sign(g4["global_net_liquidity_impulse"])[:, None] * np.full((T, N), 1.0 / float(N))
    hybrid = compute_macro_regime_gated_alpha_tilt_allocation(
        w_neutral, w_macro_dir, g4["global_net_liquidity_impulse"], max_directional_tilt=0.25, impulse_threshold=0.35
    )

    report = {
        "suite": "Level-14 Global 5-Layer Macro-to-Micro Cascade (Theorems 56-60)",
        "hours_evaluated": int(T),
        "tradable_markets": int(N),
        "exogenous_channels": int(Y_exo.shape[1]),
        "inter_layer_levy_flow": cascade["inter_layer_levy_flow"],
        "continuous_time_signature_kernel": ct_kernel,
        "kalman_horizon_weights_1d_1h_1m": fusion["terminal_weights_1d_1h_1m"],
        "g4_central_bank_weights": g4["regional_weights"],
        "hybrid_regime_allocation": {
            "pure_neutral_steps": hybrid["pure_neutral_steps"],
            "macro_tilt_active_steps": hybrid["macro_tilt_active_steps"],
            "mean_abs_net_exposure": round(hybrid["mean_abs_net_exposure"], 4),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_file = ROOT / "results" / "level14_global_5layer_macro_cascade_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
