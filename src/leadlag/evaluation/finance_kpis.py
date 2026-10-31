from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

CANDIDATE_RETURN_COLUMNS = [
    "portfolio_return",
    "strategy_return",
    "reward",
    "reward_step_mean",
    "pnl",
]
FALLBACK_COLUMN = "mean_abs_matrix"
ANNUALIZATION_FACTOR = 252  # assume daily frequency


def _detect_periods_per_year(returns: pd.Series) -> int:
    """Detect periods per year from index frequency or delta, defaulting to ANNUALIZATION_FACTOR (252)."""
    # 1. Check Series attrs if present
    if hasattr(returns, "attrs") and isinstance(returns.attrs, dict):
        freq_attr = str(
            returns.attrs.get("freq", "")
            or returns.attrs.get("frequency", "")
            or returns.attrs.get("period", "")
        ).lower()
        if any(h in freq_attr for h in ("h", "hour", "8760")):
            return 8760

    # 2. Check Series name
    if returns.name:
        name_str = str(returns.name).lower()
        if any(h in name_str for h in ("1h", "hourly", "hour")):
            return 8760

    idx = returns.index
    # 3. Check DatetimeIndex freq or inferred_freq
    if isinstance(idx, pd.DatetimeIndex):
        freq_str = getattr(idx, "freqstr", None) or (
            idx.freq.freqstr if idx.freq is not None else None
        )
        if freq_str and any(h in freq_str.lower() for h in ("h", "hour")):
            return 8760
        inferred = getattr(idx, "inferred_freq", None)
        if inferred and any(h in inferred.lower() for h in ("h", "hour")):
            return 8760
        # Check median delta between consecutive timestamps
        if len(idx) >= 2:
            try:
                deltas = idx[1:] - idx[:-1]
                median_sec = deltas.median().total_seconds()
                # Hourly is ~3600 seconds; allow range [1800, 7199]
                if 1800 <= median_sec <= 7199:
                    return 8760
            except Exception:
                pass

    # 4. Check if object/string index can be converted to timestamps with hourly interval
    elif len(idx) >= 2 and isinstance(idx[0], (str, pd.Timestamp)):
        try:
            dt_idx = pd.to_datetime(idx[:20], errors="coerce")
            if not dt_idx.isna().any():
                deltas = dt_idx[1:] - dt_idx[:-1]
                median_sec = deltas.median().total_seconds()
                if 1800 <= median_sec <= 7199:
                    return 8760
        except Exception:
            pass

    return ANNUALIZATION_FACTOR


def _derive_returns(df: pd.DataFrame) -> Optional[Tuple[pd.Series, str]]:
    for col in CANDIDATE_RETURN_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 2:
                return series, col
    if FALLBACK_COLUMN in df.columns:
        fallback = pd.to_numeric(df[FALLBACK_COLUMN], errors="coerce").dropna()
        if len(fallback) > 5:
            returns = fallback.pct_change().dropna()
            if len(returns) > 2:
                return returns, f"{FALLBACK_COLUMN}_pct_change"
    return None


def compute_kpis(
    returns: pd.Series,
    *args: object,
    periods_per_year: Optional[int] = None,
) -> dict:
    if args and periods_per_year is None:
        periods_per_year = int(args[0])

    if periods_per_year is not None:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        factor = int(periods_per_year)
    else:
        factor = _detect_periods_per_year(returns)

    n_samples = len(returns)
    if n_samples == 0:
        return {
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "annualized_volatility": float("nan"),
            "max_drawdown": float("nan"),
        }

    cum_prod = float((1 + returns).prod())
    if cum_prod > 0:
        ann_return = cum_prod ** (factor / n_samples) - 1.0
    elif cum_prod == 0:
        ann_return = -1.0
    else:
        ann_return = float("nan")

    std = float(returns.std(ddof=1)) if n_samples > 1 else float("nan")
    ann_vol = float(std * sqrt(factor)) if n_samples > 1 and not pd.isna(std) else float("nan")

    sharpe = float("nan")
    if not pd.isna(std) and std > 0:
        sharpe = float(sqrt(factor) * returns.mean() / std)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = float((cumulative / running_max - 1).min())

    return {
        "annualized_return": float(ann_return),
        "sharpe_ratio": float(sharpe),
        "annualized_volatility": float(ann_vol),
        "max_drawdown": float(drawdown),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute financial KPIs for run directories.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing run outputs (each with metrics_timeseries.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/finance_kpis.csv"),
        help="Path to write the aggregated KPI table.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=None,
        help="Periods per year for annualisation (default: auto-detect or 252).",
    )
    args = parser.parse_args(argv)

    runs_root = args.results_root
    rows: List[dict] = []
    for run_dir in runs_root.rglob("*_*"):
        if not run_dir.is_dir() or run_dir.name.endswith("_aggregate"):
            continue
        metrics_path = run_dir / "metrics_timeseries.csv"
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if "date" in df.columns:
            try:
                df = df.sort_values("date")
                df.index = pd.to_datetime(df["date"])
            except Exception:
                pass
        elif "timestamp" in df.columns:
            try:
                df = df.sort_values("timestamp")
                df.index = pd.to_datetime(df["timestamp"])
            except Exception:
                pass
        derived = _derive_returns(df)
        if derived is None:
            continue
        returns, column_used = derived
        if returns.empty:
            continue
        kpis = compute_kpis(returns, periods_per_year=args.periods_per_year)
        row = {
            "run_dir": run_dir.name,
            "returns_column": column_used,
            **kpis,
            "num_points": len(returns),
        }
        rows.append(row)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_path, index=False)
        print(f"Wrote finance KPIs for {len(rows)} run(s) to {output_path}")
    else:
        print("No suitable runs found for finance KPIs (missing returns data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
