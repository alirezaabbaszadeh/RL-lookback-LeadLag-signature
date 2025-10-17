from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False

try:  # optional for Welch test p-values
    from scipy import stats as _scipy_stats  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _scipy_stats = None  # type: ignore
    SCIPY_AVAILABLE = False

from training.run_scenario import run_scenario
from training.run_dynamic_baselines import run_dynamic
import logging
from reporting.logging_utils import setup_logging


def _select_runner(config: Dict[str, Any]):
    runner_type = config.get('runner', 'scenario')
    if runner_type == 'dynamic':
        return run_dynamic
    if runner_type == 'rl':
        try:
            from training.run_rl import run_rl  # type: ignore

            return run_rl
        except ImportError:  # pragma: no cover
            print("[WARN] stable-baselines3 not installed; falling back to scenario runner.")
            return run_scenario
    return run_scenario


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05, resamples: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng()
    samples = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(resamples)
    ])
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _aggregate_summaries(summaries: List[pd.DataFrame]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    if not summaries:
        return pd.DataFrame()

    combined = pd.concat(summaries, ignore_index=True)
    group_cols = [col for col in ['scenario', 'metric'] if col in combined.columns]
    grouped = combined.groupby(group_cols)

    for keys, frame in grouped:
        cols_to_drop = [col for col in ['metric', 'seed', 'scenario'] if col in frame.columns]
        numeric = frame.drop(columns=cols_to_drop, errors='ignore')
        numeric = numeric.select_dtypes(include='number')
        if numeric.empty:
            continue
        count = len(frame)
        means = numeric.mean()
        stds = numeric.std(ddof=0)
        ci = 1.96 * stds / np.sqrt(count)

        record: Dict[str, Any] = {}
        if isinstance(keys, tuple):
            for name, value in zip(group_cols, keys):
                record[name] = value
        else:
            record[group_cols[0]] = keys

        record['count'] = count
        for col in numeric.columns:
            values = numeric[col].to_numpy()
            boot_low, boot_high = _bootstrap_ci(values)
            record[f'{col}_mean'] = means[col]
            record[f'{col}_std'] = stds[col]
            record[f'{col}_ci95'] = ci[col]
            record[f'{col}_boot_low'] = boot_low
            record[f'{col}_boot_high'] = boot_high
        records.append(record)

    return pd.DataFrame(records)


def _welch_ttest(
    a: np.ndarray, b: np.ndarray
) -> Tuple[float, float, float]:
    """Compute Welch's t-test between two independent samples.

    Returns (t_stat, df, p_value). If SciPy is not available, p_value is NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return float('nan'), float('nan'), float('nan')

    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = a.size, b.size
    # Welch t-statistic
    denom = np.sqrt(var_a / n_a + var_b / n_b)
    if denom == 0 or np.isnan(denom):
        return float('nan'), float('nan'), float('nan')
    t_stat = (mean_a - mean_b) / denom
    # Welch–Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a ** 2) / (n_a ** 2 * (n_a - 1)) + (var_b ** 2) / (n_b ** 2 * (n_b - 1))
    df = df_num / df_den if df_den != 0 else float('nan')
    # two-sided p-value if SciPy is available
    if SCIPY_AVAILABLE and np.isfinite(df):  # pragma: no cover - depends on optional SciPy
        p = 2 * (1 - _scipy_stats.t.cdf(abs(t_stat), df))  # type: ignore[attr-defined]
    else:
        p = float('nan')
    return float(t_stat), float(df), float(p)


def _welch_by_metric(summaries: List[pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise Welch t-tests across scenarios for available numeric columns.

    Expects each summary to include 'scenario', 'metric', and one or more numeric columns
    (e.g., 'mean', 'median', 'std', 'max').
    """
    if not summaries:
        return pd.DataFrame()

    combined = pd.concat(summaries, ignore_index=True)
    needed = {'scenario', 'metric'}
    if not needed.issubset(set(combined.columns)):
        return pd.DataFrame()

    # Identify numeric columns to compare
    drop_cols = {'seed', 'scenario', 'metric'}
    numeric_cols = combined.select_dtypes(include='number').columns.difference(drop_cols)
    if len(numeric_cols) == 0:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    for metric, df_metric in combined.groupby('metric'):
        scenarios = sorted(df_metric['scenario'].unique().tolist())
        if len(scenarios) < 2:
            continue
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                s_a, s_b = scenarios[i], scenarios[j]
                df_a = df_metric[df_metric['scenario'] == s_a]
                df_b = df_metric[df_metric['scenario'] == s_b]
                for col in numeric_cols:
                    a_vals = df_a[col].to_numpy(dtype=float)
                    b_vals = df_b[col].to_numpy(dtype=float)
                    t, df_w, p = _welch_ttest(a_vals, b_vals)
                    results.append({
                        'metric': metric,
                        'column': col,
                        'scenario_a': s_a,
                        'scenario_b': s_b,
                        'mean_a': float(np.nanmean(a_vals)) if a_vals.size else float('nan'),
                        'mean_b': float(np.nanmean(b_vals)) if b_vals.size else float('nan'),
                        'n_a': int(a_vals.size),
                        'n_b': int(b_vals.size),
                        't_stat': t,
                        'df': df_w,
                        'p_value': p,
                    })
    return pd.DataFrame(results)


def run_multiseed(
    scenario_cfg: Dict[str, Any],
    seeds: Iterable[int],
    output_root: str,
) -> Path:
    scenario_path = Path(scenario_cfg['path']).resolve()
    runner = _select_runner(scenario_cfg)
    scenario_name = scenario_cfg.get('name', scenario_path.stem)

    aggregate_dir = Path(output_root) / f"{scenario_name}_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    # initialize logging at aggregator level
    try:
        setup_logging(aggregate_dir / "aggregate.log", level="INFO", config_path=Path("logging_config.yaml"))
    except Exception:
        pass
    logger = logging.getLogger("runner_multiseed")
    logger.info("Starting multi-seed aggregation for scenario: %s", scenario_name)

    summaries = []
    run_metadata: List[Dict[str, Any]] = []

    for seed in seeds:
        overrides = {
            'run': {
                'seed': seed,
                'run_name': f"{scenario_name}_seed{seed}"
            }
        }
        if 'raw_config' in scenario_cfg:
            overrides['_raw_config'] = scenario_cfg['raw_config']
        logger.info("Running seed %s", seed)
        out_dir = runner(str(scenario_path), output_root, overrides)

        summary_path = out_dir / 'summary.csv'
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary['seed'] = seed
            summary['scenario'] = scenario_name
            summaries.append(summary)

        run_metadata.append({
            'seed': seed,
            'output_dir': str(out_dir.resolve()),
        })

    if summaries:
        stats_df = _aggregate_summaries(summaries)
        stats_path = aggregate_dir / 'stats.csv'
        stats_df.to_csv(stats_path, index=False)

        significant_rows: List[Dict[str, Any]] = []
        for _, row in stats_df.iterrows():
            record = {k: row[k] for k in row.index if k in {'scenario', 'metric'}}
            for col in row.index:
                if col.endswith('_boot_low'):
                    base = col[:-10]
                    record[f'{base}_boot_low'] = row[col]
                if col.endswith('_boot_high'):
                    base = col[:-11]
                    record[f'{base}_boot_high'] = row[col]
            record['note'] = 'Bootstrap CI at 95%'
            significant_rows.append(record)
        significance = pd.DataFrame(significant_rows)
        significance.to_csv(aggregate_dir / 'significance.csv', index=False)

        # Pairwise Welch tests across scenarios when multiple scenarios are present
        if 'scenario' in stats_df.columns and len(set(stats_df['scenario'])) > 1:
            welch_df = _welch_by_metric(summaries)
            if not welch_df.empty:
                welch_df.to_csv(aggregate_dir / 'welch.csv', index=False)
                logger.info("Welch test results saved: %s", aggregate_dir / 'welch.csv')

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"{scenario_name}_aggregate", nested=True):
                for _, row in stats_df.iterrows():
                    metric = row.get('metric', '')
                    for col, value in row.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            mlflow.log_metric(f"{metric}_{col}", float(value))

    if MLFLOW_AVAILABLE and summaries:
        with mlflow.start_run(run_name=f"{scenario_name}_runs_summary", nested=True):
            mlflow.log_artifact(str(aggregate_dir / 'stats.csv'))
            mlflow.log_artifact(str(aggregate_dir / 'significance.csv'))
            welch_p = aggregate_dir / 'welch.csv'
            if welch_p.exists():
                mlflow.log_artifact(str(welch_p))

    with open(aggregate_dir / 'runs.json', 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, indent=2)

    logger.info("Aggregation completed: %s", aggregate_dir)
    return aggregate_dir
