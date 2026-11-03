#!/usr/bin/env python3
"""Third-Party Platform Verification & Export Suite (VectorBT / Backtrader / QuantStats / Independent Audit).

Runs SOTA-Apex (Theorems 1-34) on the 100% real zero-leakage 15,000-hour crypto panel and exports:
  1. docs/third_party_audit/trade_blotter.csv       — Timestamped trade orders & fees for Backtrader / LEAN replay
  2. docs/third_party_audit/target_weights.csv      — Hourly target weights for VectorBT (`vbt.Portfolio.from_orders`)
  3. docs/third_party_audit/hourly_equity_curve.csv — Hourly net & gross returns and equity curve
  4. docs/third_party_audit/third_party_audit_summary.json — Complete multi-delay benchmark scorecard
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.alpha.sota_engine import (
    SOTAConfig,
    SOTALeadLagEngine,
    _clean_panel_df,
    run_third_party_paper_trading_audit,
)


def main() -> None:
    out_dir = ROOT / "docs" / "third_party_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = ROOT / "raw_data"
    df_p = _clean_panel_df(raw_dir / "1H_prices_20250811.csv")
    df_tail = df_p.iloc[-15000:]
    valid_t0 = df_tail.columns[df_tail.iloc[:200].notna().mean() == 1.0]
    prices_68 = df_tail[valid_t0].ffill()

    # Strictly causal exogenous alignment on full index before slicing [-15000:] (zero bfill!)
    vol_d = _clean_panel_df(raw_dir / "volume_data.csv")[valid_t0]
    to_d = _clean_panel_df(raw_dir / "Market_Turnover.csv")[valid_t0]
    vol_1h = np.log(vol_d.shift(1).reindex(df_p.index).ffill().iloc[-15000:].clip(lower=1.0)).values
    to_1h = np.log(to_d.shift(1).reindex(df_p.index).ffill().iloc[-15000:].clip(lower=1e-8)).values
    mcap_1h = vol_1h - to_1h
    to_z = (to_1h - to_1h.mean(axis=1, keepdims=True)) / (to_1h.std(axis=1, keepdims=True) + 1e-8)
    mc_z = (mcap_1h - mcap_1h.mean(axis=1, keepdims=True)) / (mcap_1h.std(axis=1, keepdims=True) + 1e-8)

    cfg_pure = SOTAConfig.hourly_conservative()
    cfg_exo = replace(SOTAConfig.hourly_sovereign_conservative(), meta_warmup_cycles=16)

    eng_exo = SOTALeadLagEngine(cfg_exo)
    LR_68 = np.log(prices_68).diff().clip(-0.30, 0.30).fillna(0.0).values
    z_base_68, _ = eng_exo.precompute_horizon_signals(LR_68, stride=int(cfg_exo.rebal_steps))
    z_exo_68 = z_base_68 + 0.020 * to_z[None, :, :] + 0.020 * mc_z[None, :, :]

    summary = {}

    for label, cfg_in, z_in in [
        ("Pure_Price_Only_68Coins_1.0x", cfg_pure, None),
        ("Exogenous_Sovereign_68Coins_1.0x", cfg_exo, z_exo_68),
    ]:
        summary[label] = {}
        for delay_name, delay_frac in [
            ("0m_Instant", 0.0),
            ("3m_Broker_Latency", 0.05),
            ("15m_TWAP_Execution", 0.25),
            ("60m_Full_Next_Candle", 1.00),
        ]:
            res = run_third_party_paper_trading_audit(
                prices_68,
                cfg_in,
                z_all=z_in,
                fee_bps=5.0,
                base_slippage_bps=2.0,
                funding_spread_bps_per_8h=0.24,
                sl_sigma_mult=3.5,
                tp_sigma_mult=3.8,
                trailing_pullback_sigma=1.6,
                redistribute_to_winners=True,
                min_hold_steps=6,
                execution_delay_fraction=delay_frac,
                verify_prefix_isolation=(delay_frac == 0.0 and z_in is None),
            )
            summary[label][delay_name] = {
                "gross_sharpe": round(res.gross_sharpe, 4),
                "net_sharpe": round(res.net_sharpe, 4),
                "cagr_pct": round(res.cagr * 100.0, 2),
                "net_cum_return_pct": round(res.net_cumulative_return * 100.0, 2),
                "max_drawdown_pct": round(res.max_drawdown * 100.0, 2),
                "calmar_ratio": round(res.calmar_ratio, 2),
                "min_principal_return_pct": round(res.min_principal_return * 100.0, 2),
                "sl_triggers": res.sl_triggers_count,
                "tp_triggers": res.tp_triggers_count,
                "min_hold_blocks": res.min_hold_blocks_count,
                "prefix_isolation_max_diff": res.prefix_isolation_max_diff,
            }

            if label == "Exogenous_Sovereign_68Coins_1.0x" and delay_name == "15m_TWAP_Execution":
                res.positions.to_csv(out_dir / "target_weights.csv")
                eq_df = pd.DataFrame(
                    {
                        "gross_return": res.gross_returns,
                        "net_return": res.net_returns,
                        "turnover": res.turnover,
                        "net_equity": (1.0 + res.net_returns).cumprod(),
                    }
                )
                eq_df.to_csv(out_dir / "hourly_equity_curve.csv")

                pos_diff = res.positions.diff().fillna(res.positions.iloc[0])
                records = []
                for ts, row in pos_diff.iterrows():
                    changed = row[row.abs() > 1e-5]
                    for sym, dw in changed.items():
                        records.append(
                            {
                                "timestamp": ts,
                                "symbol": sym,
                                "side": "BUY" if dw > 0 else "SELL",
                                "weight_delta": round(float(dw), 6),
                                "new_target_weight": round(float(res.positions.loc[ts, sym]), 6),
                                "estimated_cost_bps": 7.0,
                            }
                        )
                pd.DataFrame(records).to_csv(out_dir / "trade_blotter.csv", index=False)

    (out_dir / "third_party_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print("Exported third-party audit files to:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
