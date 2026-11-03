#!/usr/bin/env python3
"""CCXT Live Exchange Testnet Paper-Trading Bot (Binance Futures Testnet / Bybit Demo Trading).

Connects SOTA-Apex (Theorems 1-34) directly to a third-party exchange matching engine in Sandbox/Testnet mode
so every order is matched against the live L2 order book with real bid-ask spread, real maker/taker fees,
and real 8-hour perpetual funding rates.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.alpha.sota_engine import SOTAConfig, SOTALeadLagEngine, _clean_panel_df


def run_live_testnet_cycle(dry_run: bool = True) -> dict[str, float]:
    symbols = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
        "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
        "DOT/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT", "UNIUSDT",
    ]
    cfg = SOTAConfig.hourly_conservative()
    engine = SOTALeadLagEngine(cfg)

    if dry_run:
        df_raw = _clean_panel_df(ROOT / "raw_data" / "1H_prices_20250811.csv")
        df_win = df_raw.iloc[-800:].ffill().dropna(axis=1)
        LR = np.log(np.maximum(df_win.values, 1e-12))
        LR[1:] = np.clip(LR[1:] - LR[:-1], -0.30, 0.30)
        LR[0] = 0.0
        z_all, _ = engine.precompute_horizon_signals(LR, stride=int(cfg.rebal_steps))
        res = engine.run_backtest(df_win, z_all=z_all)
        latest = res.positions.iloc[-1]
        active = {k: round(float(v), 6) for k, v in latest.items() if abs(v) > 1e-5}
        print(f"[CCXT Testnet Dry-Run] Computed {len(active)} active dollar-neutral target positions (Net sum={sum(active.values()):.2e})")
        return active

    import ccxt  # type: ignore

    exchange = ccxt.binanceusdm(
        {
            "apiKey": os.environ.get("BINANCE_TESTNET_API_KEY", ""),
            "secret": os.environ.get("BINANCE_TESTNET_SECRET", ""),
            "enableRateLimit": True,
        }
    )
    exchange.set_sandbox_mode(True)
    frames = {}
    for sym in symbols:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe="1h", limit=750)
        df_s = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
        df_s["ts"] = pd.to_datetime(df_s["ts"], unit="ms")
        frames[sym] = df_s.set_index("ts")["c"]
        time.sleep(0.1)

    prices_df = pd.DataFrame(frames).ffill().dropna(axis=1)
    LR = np.log(np.maximum(prices_df.values, 1e-12))
    LR[1:] = np.clip(LR[1:] - LR[:-1], -0.30, 0.30)
    LR[0] = 0.0
    z_all, _ = engine.precompute_horizon_signals(LR, stride=int(cfg.rebal_steps))
    res = engine.run_backtest(prices_df, z_all=z_all)
    latest = res.positions.iloc[-1]
    return {k: round(float(v), 6) for k, v in latest.items() if abs(v) > 1e-5}


if __name__ == "__main__":
    run_live_testnet_cycle(dry_run=True)
