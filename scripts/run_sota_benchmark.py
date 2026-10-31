#!/usr/bin/env python3
"""Empirical SOTA Benchmark Runner for RL-lookback-LeadLag-signature.

Evaluates the SOTA Rough-Path Directed-Laplacian & Counterfactual Mirror-Descent Alpha Engine
on the last 15,000 hours of liquid cryptocurrency price history (70 coins) and daily price history.
Computes and reproduces the Pareto Frontier:
  - 5 Static Baselines: L in (12h, 24h, 48h, 72h, 120h)
  - 3 SOTA Profiles: Conservative, Balanced, Aggressive
  - Daily SOTA Profile
Saves tables and equity curves into results/sota_benchmark/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from leadlag.alpha.sota_engine import SOTAConfig, SOTALeadLagEngine


def load_hourly_prices(path: str = "raw_data/1H_prices_20250811.csv", hours: int = 15000) -> pd.DataFrame:
    """Load and prepare hourly price series."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Hourly price dataset not found at {path}")

    print(f"[DataLoader] Reading {path}...")
    df = pd.read_csv(csv_path)

    # Clean date / datetime columns
    if "datetime" in df.columns:
        df.index = pd.to_datetime(df["datetime"])
        df = df.drop(columns=["datetime"])
    elif "date" in df.columns:
        df.index = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Slice the requested recent hours
    df = df.iloc[-hours:]

    # Select coins with < 5% missing data in the target window
    valid_mask = df.isna().mean() < 0.05
    valid_coins = df.columns[valid_mask].tolist()
    print(f"[DataLoader] Selected {len(valid_coins)} liquid coins (<5% NaNs) across {len(df)} hours.")

    df_clean = df[valid_coins].ffill().bfill()
    return df_clean


def load_daily_prices(path: str = "raw_data/daily_price.csv") -> pd.DataFrame:
    """Load and prepare daily price series."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Daily price dataset not found at {path}")

    print(f"[DataLoader] Reading {path}...")
    df = pd.read_csv(csv_path)

    # Clean ticker / date columns
    first_col = df.columns[0]
    df.index = pd.to_datetime(df[first_col])
    df = df.drop(columns=[first_col])

    valid_mask = df.isna().mean() < 0.05
    valid_coins = df.columns[valid_mask].tolist()
    print(f"[DataLoader] Selected {len(valid_coins)} daily coins (<5% NaNs) across {len(df)} days.")

    df_clean = df[valid_coins].ffill().bfill()
    return df_clean


def run_benchmark():
    output_dir = Path("results/sota_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" SOTA ROUGH-PATH DIRECTED-LAPLACIAN & MIRROR-DESCENT ALPHA BENCHMARK ")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # 1. Hourly Benchmark (15,000 Hours)
    # ---------------------------------------------------------
    hourly_prices = load_hourly_prices()
    clip_lim = 0.30
    LR_hourly = np.log(hourly_prices).diff().clip(-clip_lim, clip_lim).fillna(0.0).values

    print("\n[Hourly Benchmark] Precomputing 5-horizon signals across 15,000 bars...")
    t0 = time.time()
    base_engine = SOTALeadLagEngine(SOTAConfig.hourly_balanced())
    z_all_hourly, _ = base_engine.precompute_horizon_signals(LR_hourly)
    print(f"[Hourly Benchmark] Precomputation finished in {time.time() - t0:.2f}s (Shape: {z_all_hourly.shape})")

    hourly_configs = [
        ("Static L = 12h", SOTAConfig.hourly_balanced(), 0),
        ("Static L = 24h", SOTAConfig.hourly_balanced(), 1),
        ("Static L = 48h", SOTAConfig.hourly_balanced(), 2),
        ("Static L = 72h", SOTAConfig.hourly_balanced(), 3),
        ("Static L = 120h", SOTAConfig.hourly_balanced(), 4),
        ("SOTA Conservative (sa=0.35, cm=0.38)", SOTAConfig.hourly_conservative(), None),
        ("SOTA Balanced (sa=0.40, cm=0.38)", SOTAConfig.hourly_balanced(), None),
        ("SOTA Aggressive (sa=0.50, cm=0.38)", SOTAConfig.hourly_aggressive(), None),
    ]

    hourly_results = []
    equity_curves = {}

    for name, cfg, fixed_idx in hourly_configs:
        engine = SOTALeadLagEngine(cfg)
        res = engine.run_backtest(hourly_prices, z_all=z_all_hourly, fixed_horizon_index=fixed_idx)
        cum_ret = (1.0 + res.net_returns).cumprod()
        equity_curves[name] = cum_ret

        row = {
            "Profile": name,
            "Gross Sharpe": round(res.gross_sharpe, 2),
            "Net Sharpe (5bps)": round(res.net_sharpe, 2),
            "Net Cumulative Profit": f"{res.net_cumulative_return * 100.0:+.1f}%",
            "Max Drawdown": f"{res.max_drawdown * 100.0:.1f}%",
            "Calmar Ratio": round(res.calmar_ratio, 2),
            "Avg Turnover (L1)": round(res.avg_turnover, 4),
        }
        hourly_results.append(row)

    df_hourly = pd.DataFrame(hourly_results)
    df_hourly.to_csv(output_dir / "hourly_pareto_table.csv", index=False)
    print("\n" + df_hourly.to_string(index=False) + "\n")

    # ---------------------------------------------------------
    # 2. Daily Benchmark
    # ---------------------------------------------------------
    daily_prices = load_daily_prices()
    daily_cfg = SOTAConfig.daily_default()
    daily_engine = SOTALeadLagEngine(daily_cfg)
    daily_res = daily_engine.run_backtest(daily_prices)
    equity_curves["Daily SOTA Profile"] = (1.0 + daily_res.net_returns).cumprod()

    daily_row = {
        "Profile": "Daily SOTA Profile",
        "Gross Sharpe": round(daily_res.gross_sharpe, 2),
        "Net Sharpe (5bps)": round(daily_res.net_sharpe, 2),
        "Net Cumulative Profit": f"{daily_res.net_cumulative_return * 100.0:+.1f}%",
        "Max Drawdown": f"{daily_res.max_drawdown * 100.0:.1f}%",
        "Calmar Ratio": round(daily_res.calmar_ratio, 2),
        "Avg Turnover (L1)": round(daily_res.avg_turnover, 4),
    }
    df_daily = pd.DataFrame([daily_row])
    df_daily.to_csv(output_dir / "daily_table.csv", index=False)
    print("[Daily Benchmark]")
    print(df_daily.to_string(index=False) + "\n")

    # Save equity curves
    df_equity = pd.DataFrame(equity_curves).ffill()
    df_equity.to_csv(output_dir / "sota_equity_curves.csv", index=True)

    print(f"[Done] All benchmark tables and curves successfully saved to {output_dir}/")


if __name__ == "__main__":
    run_benchmark()
