from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False

from training.run_scenario import run_scenario
from training.run_dynamic_baselines import run_dynamic


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

    with open(aggregate_dir / 'runs.json', 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, indent=2)

    return aggregate_dir
