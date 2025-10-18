from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
from scipy import stats


def _collect_summaries(results_root: Path) -> pd.DataFrame:
    rows = []
    for scenario_dir in results_root.glob('*'):
        if not scenario_dir.is_dir():
            continue
        summary = scenario_dir / 'summary.csv'
        meta = scenario_dir / 'run_metadata.json'
        if summary.exists():
            df = pd.read_csv(summary)
            # pivot summary to wide
            wide = df.set_index('metric').stack().reset_index()
            wide.columns = ['metric', 'stat', 'value']
            w = wide.pivot(index='stat', columns='metric', values='value')
            w = w.reset_index()
            # attach scenario name
            row = {'scenario_dir': scenario_dir.name}
            # add top-level stats (means)
            for col in w.columns:
                if col == 'index':
                    continue
                # prefer 'mean' row where available
                try:
                    if 'mean' in w['stat'].values:
                        row[col] = w.loc[w['stat'] == 'mean', col].values[0]
                except Exception:
                    pass
            # attach metadata
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    row['git_commit'] = m.get('git', {}).get('git_commit')
                except Exception:
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


def to_latex_table(df: pd.DataFrame, path: Path):
    # choose a few key columns if available
    cols = [
        'scenario_dir',
        'mean_abs_matrix',
        'max_abs_matrix',
        'row_sum_range',
        'stability_matrix_corr',
        'stability_rowsum_corr',
    ]
    use = [c for c in cols if c in df.columns]
    try:
        tex = df[use].to_latex(index=False, float_format=lambda x: f"{x:0.4f}")
        path.write_text(tex, encoding='utf-8')
    except Exception:
        # fallback: write CSV when LaTeX renderer deps are missing
        path.with_suffix('.csv').write_text(df[use].to_csv(index=False), encoding='utf-8')


def _bh_correction(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg FDR correction.

    Returns list of q-values in the original order.
    """
    m = len(pvals)
    if m == 0:
        return []
    # Pair each p with index and sort by p
    order = sorted(range(m), key=lambda i: (float('inf') if pd.isna(pvals[i]) else pvals[i]))
    qvals = [float('nan')] * m
    prev = float('inf')
    for rank, i in enumerate(reversed(order), start=1):
        # reverse iterate to enforce monotonicity
        j = order[-rank]
        p = pvals[j]
        if pd.isna(p):
            q = float('nan')
        else:
            q = (p * m) / (m - rank + 1)
        prev = min(prev, q)
        qvals[j] = prev
    return qvals


def _cohens_d(xa: np.ndarray, xb: np.ndarray) -> float:
    xa = xa.astype(float)
    xb = xb.astype(float)
    xa = xa[~np.isnan(xa)]
    xb = xb[~np.isnan(xb)]
    if xa.size < 2 or xb.size < 2:
        return float('nan')
    mean_a, mean_b = xa.mean(), xb.mean()
    var_a, var_b = xa.var(ddof=1), xb.var(ddof=1)
    n_a, n_b = xa.size, xb.size
    # pooled std
    denom = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if denom == 0 or np.isnan(denom):
        return float('nan')
    return float((mean_a - mean_b) / denom)


def aggregate(results_root: str) -> Path:
    root = Path(results_root)
    out_dir = root / 'aggregate'
    out_dir.mkdir(exist_ok=True, parents=True)
    df = _collect_summaries(root)
    if not df.empty:
        df.to_csv(out_dir / 'comparison_summary.csv', index=False)
        to_latex_table(df, out_dir / 'comparison_table.tex')

    # Pairwise Welch t-tests on a key metric across scenarios
    # Load metric time series per scenario
    metrics = {}
    for scenario_dir in root.glob('*'):
        if not scenario_dir.is_dir():
            continue
        mt = scenario_dir / 'metrics_timeseries.csv'
        if mt.exists():
            try:
                mdf = pd.read_csv(mt, parse_dates=['date']).set_index('date')
                metrics[scenario_dir.name] = mdf
            except Exception:
                pass
    if metrics:
        def pairwise_test(column: str) -> pd.DataFrame:
            names = list(metrics.keys())
            mat = np.full((len(names), len(names)), np.nan)
            # also build long-form pair list for BH and effect sizes
            pairs: List[Dict[str, object]] = []
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    if i >= j:
                        continue
                    xa = metrics[a][column].dropna().values
                    xb = metrics[b][column].dropna().values
                    if len(xa) > 5 and len(xb) > 5:
                        t, p = stats.ttest_ind(xa, xb, equal_var=False)
                        mat[i, j] = p
                        mat[j, i] = p
                        pairs.append({
                            'metric': column,
                            'scenario_a': a,
                            'scenario_b': b,
                            'n_a': int(len(xa)),
                            'n_b': int(len(xb)),
                            'p_value': float(p),
                            'cohens_d': _cohens_d(xa, xb),
                        })
            # BH correction in long form
            if pairs:
                pvals = [float(r['p_value']) for r in pairs]
                qvals = _bh_correction(pvals)
                for r, q in zip(pairs, qvals):
                    r['q_value'] = q
                pairs_df = pd.DataFrame(pairs)
                pairs_df.to_csv(out_dir / f'significance_{column}_pairs.csv', index=False)
            return pd.DataFrame(mat, index=names, columns=names)

        for col in ['mean_abs_matrix', 'stability_matrix_corr']:
            try:
                pmat = pairwise_test(col)
                pmat.to_csv(out_dir / f'significance_{col}.csv')
            except Exception:
                pass
    return out_dir


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='results')
    args = ap.parse_args()
    out = aggregate(args.root)
    print(f"Aggregated into: {out}")
