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
    tex = df[use].to_latex(index=False, float_format=lambda x: f"{x:0.4f}")
    path.write_text(tex, encoding='utf-8')


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
