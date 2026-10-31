#!/usr/bin/env python3
"""Fast, zero-error SOTA-Omega benchmark runner on raw_data/1H_prices_20250811.csv.

Key error-elimination & speed features:
1. Automatically injects repository `src/` into `sys.path` (never fails with ModuleNotFoundError).
2. Explicitly drops the trailing string column `date` in `1H_prices_20250811.csv` and enforces
   `pd.to_numeric(..., errors="coerce")` before filtering (eliminating Exit Code 1 on 1H coin filtering).
3. Precomputes the K=5 multi-scale directed Laplacian signals ONCE and shares `z_all` across
   all 5 static baselines and all 3 SOTA-Omega profiles (~8x faster execution).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from leadlag.alpha.sota_engine import SOTAConfig, SOTALeadLagEngine


def load_clean_1h_panel(csv_path: Path, max_hours: int = 15000) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    for col in ["date", "datetime", "timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df_tail = df.iloc[-max_hours:]
    valid_cols = df_tail.columns[df_tail.notna().mean() > 0.99]
    return df_tail[valid_cols].ffill().bfill()


def main() -> None:
    csv_path = REPO_ROOT / "raw_data" / "1H_prices_20250811.csv"
    out_dir = REPO_ROOT / "results" / "sota_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_clean_1h_panel(csv_path, max_hours=15000)
    print(f"Loaded 1H panel: {panel.shape[0]} hours x {panel.shape[1]} coins")

    base_engine = SOTALeadLagEngine(SOTAConfig.hourly_balanced())
    LR = np.log(panel).diff().clip(-0.30, 0.30).fillna(0.0).values

    cache_path = Path("/tmp/sota_z_all.npz")
    if cache_path.exists():
        z_all = np.load(cache_path)["z_all"]
        if z_all.shape != (len(base_engine.config.candidates), panel.shape[0], panel.shape[1]):
            z_all, _ = base_engine.precompute_horizon_signals(LR)
            np.savez_compressed(cache_path, z_all=z_all)
    else:
        z_all, _ = base_engine.precompute_horizon_signals(LR)
        np.savez_compressed(cache_path, z_all=z_all)

    rows = []
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

    profiles = [
        ("SOTA-Omega Conservative (lev=1.00)", SOTAConfig.hourly_conservative()),
        ("SOTA-Omega Balanced (lev=1.20)", SOTAConfig.hourly_balanced()),
        ("SOTA-Omega Aggressive (lev=1.40)", SOTAConfig.hourly_aggressive()),
    ]
    for name, cfg in profiles:
        eng = SOTALeadLagEngine(cfg)
        res = eng.run_backtest(panel, z_all=z_all)
        rows.append({
            "Strategy": name,
            "gross_sharpe": round(res.gross_sharpe, 2),
            "net_sharpe": round(res.net_sharpe, 2),
            "cum_return_pct": round(res.net_cumulative_return * 100.0, 1),
            "max_dd_pct": round(res.max_drawdown * 100.0, 1),
            "calmar": round(res.calmar_ratio, 2),
            "turnover": round(res.avg_turnover, 4),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "hourly_pareto_table.csv", index=False)
    with open(out_dir / "hourly_pareto_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
