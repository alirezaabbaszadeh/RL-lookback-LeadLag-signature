from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


def _maybe_skip_plot(func):  # pragma: no cover
    if MATPLOTLIB_AVAILABLE:
        return func

    def wrapper(*args, **kwargs):
        return None

    return wrapper


def _flatten_offdiag(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]


def compute_metrics_timeseries(rolling: pd.Series) -> pd.DataFrame:
    rows = []
    prev_mat = None
    prev_rowsum = None
    for dt, df in rolling.items():
        mat = df.values.astype(float)
        n = mat.shape[0]
        # mask diagonal for magnitude metrics
        off = _flatten_offdiag(mat)
        mean_abs = np.nanmean(np.abs(off)) if off.size else np.nan
        max_abs = np.nanmax(np.abs(off)) if off.size else np.nan
        # row sums
        row_sums = np.nansum(mat, axis=1)
        row_range = np.nanmax(row_sums) - np.nanmin(row_sums) if n > 0 else np.nan
        row_std = np.nanstd(row_sums) if n > 0 else np.nan
        # stability
        stab_mat = np.nan
        stab_rows = np.nan
        if prev_mat is not None:
            a = _flatten_offdiag(mat)
            b = _flatten_offdiag(prev_mat)
            if a.size and b.size and not (np.all(np.isnan(a)) or np.all(np.isnan(b))):
                # safe corr
                try:
                    stab_mat = np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1]
                except Exception:
                    stab_mat = np.nan
        if prev_rowsum is not None:
            try:
                stab_rows = np.corrcoef(np.nan_to_num(row_sums), np.nan_to_num(prev_rowsum))[0, 1]
            except Exception:
                stab_rows = np.nan

        rows.append({
            'date': pd.Timestamp(dt),
            'mean_abs_matrix': mean_abs,
            'max_abs_matrix': max_abs,
            'row_sum_range': row_range,
            'row_sum_std': row_std,
            'stability_matrix_corr': stab_mat,
            'stability_rowsum_corr': stab_rows,
            'n_assets': n,
        })
        prev_mat = mat
        prev_rowsum = row_sums

    out = pd.DataFrame(rows).set_index('date').sort_index()
    return out


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        'mean_abs_matrix': ['mean', 'median', 'std', 'max'],
        'max_abs_matrix': ['mean', 'median', 'std', 'max'],
        'row_sum_range': ['mean', 'median', 'std', 'max'],
        'row_sum_std': ['mean', 'median', 'std', 'max'],
        'stability_matrix_corr': ['mean', 'median'],
        'stability_rowsum_corr': ['mean', 'median'],
        'n_assets': ['mean', 'min', 'max']
    }
    out = df.agg(agg)
    out = out.T.reset_index().rename(columns={'index': 'metric'})
    return out


@_maybe_skip_plot
def plot_signal_strength(df: pd.DataFrame, path: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(df.index, df['mean_abs_matrix'], label='mean|M|')
    plt.plot(df.index, df['max_abs_matrix'], label='max|M|', alpha=0.6)
    plt.plot(df.index, df['row_sum_range'], label='row-sum range', alpha=0.6)
    plt.legend()
    plt.title('Signal Strength over Time')
    plt.xlabel('date')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@_maybe_skip_plot
def plot_stability(df: pd.DataFrame, path: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(df.index, df['stability_matrix_corr'], label='corr(M_t, M_{t-1})')
    plt.plot(df.index, df['stability_rowsum_corr'], label='corr(rowsum_t, rowsum_{t-1})', alpha=0.6)
    plt.legend()
    plt.title('Stability over Time')
    plt.xlabel('date')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
