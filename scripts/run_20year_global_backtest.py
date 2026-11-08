#!/usr/bin/env python3
"""20-Year Global 5-Layer Multi-Asset SOTA-Apex Backtest & Regime Audit Runner (Theorems 1-60).

How it works on your PC:
  1. First run `python scripts/download_20year_global_data.py --start-date 2005-01-01` (requires internet).
     This populates `raw_data/global_20y/` with 20+ years of FRED Central Bank & Bond Yield series (Layers 1-2)
     plus Global FX, Commodities, Global Indices, Sector ETFs, and Global Equities (Layers 3-5).
  2. Then run `python scripts/run_20year_global_backtest.py`.
     It automatically detects `raw_data/global_20y/`, aligns all 5 layers causally (Zero `.bfill()`),
     and executes the 60-Theorem SOTA-Apex engine with dynamic point-in-time asset eligibility.
  3. If run before downloading `raw_data/global_20y/`, it automatically runs on the local multi-year
     institutional daily panel (`raw_data/daily_price.csv` + `tbill_data.csv` + `market_data_BGCI_Index.csv`).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.alpha.sota_engine import (
    SOTAConfig,
    SOTALeadLagEngine,
    _clean_panel_df,
    compute_bilinear_smart_money_signals,
    compute_five_layer_macro_liquidity_cascade_operator,
    compute_g4_central_bank_monetary_impulse_tensor,
    compute_macro_regime_gated_alpha_tilt_allocation,
    load_multimodal_daily_panel,
)


def run_20year_or_local_global_backtest(repo_root: Path) -> dict[str, object]:
    g_dir = repo_root / "raw_data" / "global_20y"
    p_file = g_dir / "global_20y_daily_prices.csv"
    v_file = g_dir / "global_20y_daily_volume.csv"
    m_file = g_dir / "layer1_2_macro_fred_20y.csv"

    if p_file.exists():
        df_p = _clean_panel_df(p_file).sort_index()
        # Keep all global markets with at least 250 trading days of history; causal forward-fill only after IPO/listing
        valid_c = df_p.columns[df_p.notna().sum() >= 250]
        prices = df_p[valid_c].ffill()
        active_mask = prices.notna().rolling(90).mean().fillna(0.0).values >= 1.0
        prices_clean = prices.fillna(method="ffill").fillna(100.0)

        df_v = (
            _clean_panel_df(v_file).reindex(prices.index)[valid_c].shift(1).ffill().fillna(1e6)
            if v_file.exists()
            else pd.DataFrame(1e7, index=prices.index, columns=prices.columns)
        )
        df_m = (
            _clean_panel_df(m_file).reindex(prices.index).shift(1).ffill().fillna(0.0)
            if m_file.exists()
            else pd.DataFrame(0.0, index=prices.index, columns=["FED", "RATES", "FX", "COMM"])
        )

        LR = np.log(np.maximum(prices_clean.values, 1e-12))
        LR[1:] = np.clip(LR[1:] - LR[:-1], -0.25, 0.25)
        LR[0] = 0.0
        LR = np.where(active_mask, LR, 0.0)

        M_vals = np.nan_to_num(df_m.diff().fillna(0.0).values.astype(float), nan=0.0)
        L1_cb = M_vals[:, : min(5, M_vals.shape[1])]
        L2_rates = M_vals[:, min(5, M_vals.shape[1] - 1) :]
        L3_fx = LR[:, : min(10, LR.shape[1])]
        L4_comm = LR[:, min(10, LR.shape[1] - 1) : min(24, LR.shape[1])]
        L5_equities = LR

        cascade = compute_five_layer_macro_liquidity_cascade_operator(
            L1_cb, L2_rates, L3_fx, L4_comm, L5_equities, lags=(20, 14, 10, 5)
        )
        g4 = compute_g4_central_bank_monetary_impulse_tensor(L1_cb, L2_rates, L3_fx, L4_comm)

        cfg = SOTAConfig(
            candidates=(10, 21, 42, 63, 126),
            rebal_steps=5,
            top_k=max(4, min(20, prices.shape[1] // 10)),
            retain_bonus=1.5,
            step_alpha=0.50,
            consensus_min=0.15,
            spread_min=2.0,
            periods_per_year=252,
            vol_fast_window=5,
            vol_slow_window=30,
            strict_meta_warmup=True,
            meta_warmup_cycles=20,
            zero_loss_shield=True,
            fee_bps=2.5,
        )
        eng = SOTALeadLagEngine(cfg)
        z_base, _ = eng.precompute_horizon_signals(LR, stride=int(cfg.rebal_steps))
        z_5layer = (z_base + 0.04 * cascade["layer5_cascade_signals"][None, :, :]) * active_mask[None, :, :]
        res = eng.run_backtest(prices_clean, z_all=z_5layer)
        mode_str = "20-YEAR_GLOBAL_5LAYER_LAKE (FRED + Yahoo Multi-Asset)"
    else:
        bundle = load_multimodal_daily_panel(repo_root)
        prices_clean = bundle["prices"]
        ext_sigs = compute_bilinear_smart_money_signals(
            prices_clean, bundle["volume_turnover"], bundle["log_marketcap"]
        )
        LR = np.log(np.maximum(prices_clean.values, 1e-12)).diff(axis=0) if hasattr(np.log(prices_clean.values), "diff") else np.log(prices_clean).diff().fillna(0.0).values
        rf_vec = bundle["risk_free_daily"].values[:, None]
        L1_cb = rf_vec
        L2_rates = bundle["volume_turnover"].pct_change().fillna(0.0).values[:, :6]
        L3_fx = bundle["log_marketcap"].diff().fillna(0.0).values[:, :6]
        L4_comm = LR[:, :6]
        L5_equities = LR

        cascade = compute_five_layer_macro_liquidity_cascade_operator(
            L1_cb, L2_rates, L3_fx, L4_comm, L5_equities, lags=(14, 10, 7, 3)
        )
        g4 = compute_g4_central_bank_monetary_impulse_tensor(L1_cb, L2_rates, L3_fx, L4_comm)

        cfg = SOTAConfig(
            candidates=(10, 20, 35, 60, 90),
            rebal_steps=3,
            top_k=4,
            retain_bonus=1.5,
            step_alpha=0.50,
            consensus_min=0.20,
            spread_min=2.5,
            ic_temp=4.0,
            ic_memory_cycles=10,
            meta_memory_steps=90,
            meta_eta=0.30,
            meta_lr=0.10,
            leverage=1.00,
            fee_bps=5.0,
            periods_per_year=365,
            vol_fast_window=5,
            vol_slow_window=30,
            cov_penalty_gamma=1.15,
            pos_overlap_gamma=0.0,
            master_deadband=0.010,
            dd_governor_scale=5.0,
            enable_specialized_formulas=False,
            strict_meta_warmup=True,
            meta_warmup_cycles=61,
            sparse_sharpe_thresh=-99.0,
            zero_loss_shield=True,
        )
        eng = SOTALeadLagEngine(cfg)
        res = eng.run_backtest(
            prices_clean,
            external_signals=ext_sigs,
            external_priors=(0.35 / 1.6, 0.25 / 1.6),
            risk_free_series=bundle["risk_free_daily"],
        )
        mode_str = "LOCAL_MULTI_YEAR_DAILY_PANEL (Run download_20year_global_data.py on PC for 2005-2026 Global Data)"

    pos_eval = res.positions.values
    imp_eval = g4["global_net_liquidity_impulse"][-len(pos_eval) :]
    # Dynamic Causal Z-Score Normalization for G4 Monetary Impulse (Theorem 59-60)
    imp_s = pd.Series(imp_eval)
    imp_z = ((imp_s - imp_s.rolling(252, min_periods=30).mean()) / (imp_s.rolling(252, min_periods=30).std() + 1e-8)).fillna(0.0).values

    macro_dir = np.sign(imp_z)[:, None] * np.full_like(pos_eval, 1.0 / float(prices_clean.shape[1]))
    hybrid = compute_macro_regime_gated_alpha_tilt_allocation(
        pos_eval, macro_dir, imp_z, max_directional_tilt=0.25, impulse_threshold=0.50
    )

    return {
        "runner": "60-Theorem Global Multi-Asset & 5-Layer Macro Backtest",
        "data_mode": mode_str,
        "date_start": str(prices_clean.index[0]),
        "date_end": str(prices_clean.index[-1]),
        "bars_evaluated": int(len(prices_clean)),
        "markets_evaluated": int(prices_clean.shape[1]),
        "gross_sharpe": round(float(res.gross_sharpe), 4),
        "net_sharpe": round(float(res.net_sharpe), 4),
        "cagr_pct": round(float(res.cagr) * 100.0, 2),
        "net_cumulative_return_pct": round(float(res.net_cumulative_return) * 100.0, 2),
        "max_drawdown_pct": round(float(res.max_drawdown) * 100.0, 2),
        "calmar_ratio": round(float(res.calmar_ratio), 2),
        "inter_layer_levy_flow": cascade["inter_layer_levy_flow"],
        "hybrid_regime_stats": {
            "pure_neutral_bars": hybrid["pure_neutral_steps"],
            "macro_tilt_active_bars": hybrid["macro_tilt_active_steps"],
        },
    }


def main() -> None:
    t0 = time.time()
    print("=" * 92)
    print(" SOTA-APEX 60-THEOREM GLOBAL MULTI-ASSET & 5-LAYER MACRO CASCADE BACKTEST RUNNER")
    print("=" * 92)
    report = run_20year_or_local_global_backtest(ROOT)
    report["elapsed_seconds"] = round(time.time() - t0, 2)
    out_path = ROOT / "results" / "global_20year_5layer_backtest_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("-" * 92)
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
