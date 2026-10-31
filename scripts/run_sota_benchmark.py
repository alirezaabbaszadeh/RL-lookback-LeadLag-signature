#!/usr/bin/env python3
"""Fast, zero-error SOTA-Zenith benchmark runner on hourly and daily panels.

Mathematical Features Evaluated:
1. Hourly panel (15,000h liquid crypto from raw_data/1H_prices_20250811.csv):
   - 5 static lookback horizons: 12h, 24h, 48h, 72h, 120h
   - SOTA-Zenith Conservative (lev=1.00)
   - SOTA-Zenith Balanced (lev=1.20)
   - SOTA-Zenith Aggressive (lev=1.40)
   - SOTA-Zenith Fortress (lev=1.00, cppi=3.0)
   - SOTA-Zenith Ultra-Calmar (lev=1.40, cppi=3.0)
2. Daily panel (1,277 days liquid crypto from raw_data/daily_price.csv):
   - 5 static lookback horizons: 10d, 20d, 35d, 60d, 90d
   - SOTA-Zenith Daily Default (lev=1.00)

Produces:
- results/sota_benchmark/hourly_pareto_table.csv
- results/sota_benchmark/hourly_pareto_summary.json
- results/sota_benchmark/daily_table.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from leadlag.alpha.sota_engine import SOTAConfig, SOTALeadLagEngine


def load_clean_1h_panel(csv_path: Path, max_hours: int = 15000) -> pd.DataFrame:
    cache_pkl = csv_path.parent.parent / "results" / "sota_benchmark" / ".cache_clean_1h.pkl"
    if cache_pkl.exists() and cache_pkl.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_pickle(cache_pkl)
    df = pd.read_csv(csv_path, index_col=0)
    for col in ["date", "datetime", "timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df_tail = df.iloc[-max_hours:]
    valid_cols = df_tail.columns[df_tail.notna().mean() > 0.99]
    clean = df_tail[valid_cols].ffill().bfill()
    cache_pkl.parent.mkdir(parents=True, exist_ok=True)
    clean.to_pickle(cache_pkl)
    return clean


def load_clean_daily_panel(csv_path: Path) -> pd.DataFrame:
    cache_pkl = csv_path.parent.parent / "results" / "sota_benchmark" / ".cache_clean_daily.pkl"
    if cache_pkl.exists() and cache_pkl.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_pickle(cache_pkl)
    df = pd.read_csv(csv_path, index_col=0)
    for col in ["date", "datetime", "timestamp", "ticker"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    valid_cols = df.columns[df.notna().mean() > 0.90]
    clean = df[valid_cols].ffill().bfill()
    cache_pkl.parent.mkdir(parents=True, exist_ok=True)
    clean.to_pickle(cache_pkl)
    return clean


def run_hourly_benchmark(out_dir: Path) -> pd.DataFrame:
    csv_path = REPO_ROOT / "raw_data" / "1H_prices_20250811.csv"
    panel = load_clean_1h_panel(csv_path, max_hours=15000)
    print(f"\n================================================================================")
    print(f"Loaded 1H panel: {panel.shape[0]} hours x {panel.shape[1]} coins ({panel.index.min()} to {panel.index.max()})")
    print(f"================================================================================")

    base_engine = SOTALeadLagEngine(SOTAConfig.hourly_balanced())
    LR = np.log(panel).diff().clip(-0.30, 0.30).fillna(0.0).values

    cache_path = out_dir / ".cache_z_all_hourly.npz"
    rebal_h = int(base_engine.config.rebal_steps)
    if cache_path.exists():
        z_all = np.load(cache_path)["z_all"]
        if z_all.shape != (len(base_engine.config.candidates), panel.shape[0], panel.shape[1]):
            z_all, _ = base_engine.precompute_horizon_signals(LR, stride=rebal_h)
            np.savez_compressed(cache_path, z_all=z_all)
    else:
        z_all, _ = base_engine.precompute_horizon_signals(LR, stride=rebal_h)
        np.savez_compressed(cache_path, z_all=z_all)

    rows = []
    # 1. 5 static lookback horizons
    for k, h in enumerate(base_engine.config.candidates):
        res = base_engine.run_backtest(panel, z_all=z_all, fixed_horizon_index=k)
        rows.append({
            "Strategy": f"Fixed L = {h}h",
            "gross_sharpe": round(res.gross_sharpe, 2),
            "net_sharpe": round(res.net_sharpe, 2),
            "cum_return_pct": round(res.net_cumulative_return * 100.0, 1),
            "max_dd_pct": round(res.max_drawdown * 100.0, 1),
            "calmar": round(res.calmar_ratio, 2),
            "turnover": round(res.avg_turnover, 4),
        })

    # 2. 5 SOTA-Zenith dynamic profiles (sharing precomputed counterfactual IC)
    R_1h = panel.pct_change().clip(-0.30, 0.30).fillna(0.0).values
    oos_ic = base_engine.compute_causal_counterfactual_ic(z_all, R_1h)
    profiles = [
        ("SOTA-Zenith Conservative (lev=1.00)", SOTAConfig.hourly_conservative()),
        ("SOTA-Zenith Balanced (lev=1.20)", SOTAConfig.hourly_balanced()),
        ("SOTA-Zenith Aggressive (lev=1.40)", SOTAConfig.hourly_aggressive()),
        ("SOTA-Zenith Fortress (lev=1.00, cppi=3.0)", SOTAConfig.hourly_fortress()),
        ("SOTA-Zenith Ultra-Calmar (lev=1.40, cppi=3.0)", SOTAConfig.hourly_ultra_calmar()),
    ]
    for name, cfg in profiles:
        eng = SOTALeadLagEngine(cfg)
        res = eng.run_backtest(panel, z_all=z_all, oos_ic=oos_ic)
        rows.append({
            "Strategy": name,
            "gross_sharpe": round(res.gross_sharpe, 2),
            "net_sharpe": round(res.net_sharpe, 2),
            "cum_return_pct": round(res.net_cumulative_return * 100.0, 1),
            "max_dd_pct": round(res.max_drawdown * 100.0, 1),
            "calmar": round(res.calmar_ratio, 2),
            "turnover": round(res.avg_turnover, 4),
        })

    df_hourly = pd.DataFrame(rows)
    df_hourly.to_csv(out_dir / "hourly_pareto_table.csv", index=False)
    with open(out_dir / "hourly_pareto_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\n--- HOURLY BENCHMARK PARETO RESULTS (15,000 Hours, 5 bps fee) ---")
    print(df_hourly.to_string(index=False))
    return df_hourly


def run_daily_benchmark(out_dir: Path) -> pd.DataFrame:
    csv_path = REPO_ROOT / "raw_data" / "daily_price.csv"
    panel = load_clean_daily_panel(csv_path)
    print(f"\n================================================================================")
    print(f"Loaded Daily panel: {panel.shape[0]} days x {panel.shape[1]} coins ({panel.index.min()} to {panel.index.max()})")
    print(f"================================================================================")

    daily_cfg = SOTAConfig.daily_default()
    daily_engine = SOTALeadLagEngine(daily_cfg)
    LR = np.log(panel).diff().clip(-0.30, 0.30).fillna(0.0).values

    cache_path = out_dir / ".cache_z_all_daily.npz"
    rebal_d = int(daily_cfg.rebal_steps)
    if cache_path.exists():
        z_all = np.load(cache_path)["z_all"]
        if z_all.shape != (len(daily_cfg.candidates), panel.shape[0], panel.shape[1]):
            z_all, _ = daily_engine.precompute_horizon_signals(LR, stride=rebal_d)
            np.savez_compressed(cache_path, z_all=z_all)
    else:
        z_all, _ = daily_engine.precompute_horizon_signals(LR, stride=rebal_d)
        np.savez_compressed(cache_path, z_all=z_all)

    rows = []
    # 1. 5 static lookback horizons for daily (10d, 20d, 35d, 60d, 90d)
    for k, h in enumerate(daily_cfg.candidates):
        res = daily_engine.run_backtest(panel, z_all=z_all, fixed_horizon_index=k)
        rows.append({
            "Strategy": f"Fixed L = {h}d",
            "gross_sharpe": round(res.gross_sharpe, 2),
            "net_sharpe": round(res.net_sharpe, 2),
            "cum_return_pct": round(res.net_cumulative_return * 100.0, 1),
            "max_dd_pct": round(res.max_drawdown * 100.0, 1),
            "calmar": round(res.calmar_ratio, 2),
            "turnover": round(res.avg_turnover, 4),
        })

    # 2. SOTA-Zenith Daily Default
    res_zenith = daily_engine.run_backtest(panel, z_all=z_all)
    rows.append({
        "Strategy": "SOTA-Zenith Daily Default (lev=1.00)",
        "gross_sharpe": round(res_zenith.gross_sharpe, 2),
        "net_sharpe": round(res_zenith.net_sharpe, 2),
        "cum_return_pct": round(res_zenith.net_cumulative_return * 100.0, 1),
        "max_dd_pct": round(res_zenith.max_drawdown * 100.0, 1),
        "calmar": round(res_zenith.calmar_ratio, 2),
        "turnover": round(res_zenith.avg_turnover, 4),
    })

    df_daily = pd.DataFrame(rows)
    df_daily.to_csv(out_dir / "daily_table.csv", index=False)

    print("\n--- DAILY BENCHMARK RESULTS (1,277 Days, 5 bps fee) ---")
    print(df_daily.to_string(index=False))
    return df_daily


def main() -> None:
    t_start = time.time()
    out_dir = REPO_ROOT / "results" / "sota_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_hourly_benchmark(out_dir)
    run_daily_benchmark(out_dir)

    t_total = time.time() - t_start
    print(f"\n================================================================================")
    print(f"All benchmarks finished successfully in {t_total:.2f} seconds.")
    print(f"Artifacts generated:")
    print(f"  - {out_dir / 'hourly_pareto_table.csv'}")
    print(f"  - {out_dir / 'hourly_pareto_summary.json'}")
    print(f"  - {out_dir / 'daily_table.csv'}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
