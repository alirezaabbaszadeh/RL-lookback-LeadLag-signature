from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

CANDIDATE_RETURN_COLUMNS = [
    'portfolio_return',
    'strategy_return',
    'reward',
    'reward_step_mean',
    'pnl',
]
FALLBACK_COLUMN = 'mean_abs_matrix'
ANNUALIZATION_FACTOR = 252  # assume daily frequency


def _derive_returns(df: pd.DataFrame) -> Optional[Tuple[pd.Series, str]]:
    for col in CANDIDATE_RETURN_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) > 2:
                return series, col
    if FALLBACK_COLUMN in df.columns:
        fallback = pd.to_numeric(df[FALLBACK_COLUMN], errors='coerce').dropna()
        if len(fallback) > 5:
            returns = fallback.pct_change().dropna()
            if len(returns) > 2:
                return returns, f'{FALLBACK_COLUMN}_pct_change'
    return None


def compute_kpis(returns: pd.Series) -> dict:
    # Simple metrics on daily returns
    ann_return = (1 + returns).prod() ** (ANNUALIZATION_FACTOR / len(returns)) - 1
    sharpe = float('nan')
    if returns.std(ddof=1) > 0:
        sharpe = sqrt(ANNUALIZATION_FACTOR) * returns.mean() / returns.std(ddof=1)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1).min()
    return {
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': drawdown,
    }


def main() -> int:
    runs_root = Path('results')
    rows: List[dict] = []
    for run_dir in runs_root.glob('*_*'):
        if not run_dir.is_dir() or run_dir.name.endswith('_aggregate'):
            continue
        metrics_path = run_dir / 'metrics_timeseries.csv'
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if 'date' in df.columns:
            try:
                df = df.sort_values('date')
            except Exception:
                pass
        derived = _derive_returns(df)
        if derived is None:
            continue
        returns, column_used = derived
        if returns.empty:
            continue
        kpis = compute_kpis(returns)
        row = {
            'run_dir': run_dir.name,
            'returns_column': column_used,
            **kpis,
            'num_points': len(returns),
        }
        rows.append(row)

    out_dir = Path('evaluation')
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_dir / 'finance_kpis.csv', index=False)
        print(f'Wrote finance KPIs for {len(rows)} run(s) to {out_dir / "finance_kpis.csv"}')
    else:
        print('No suitable runs found for finance KPIs (missing returns data).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

