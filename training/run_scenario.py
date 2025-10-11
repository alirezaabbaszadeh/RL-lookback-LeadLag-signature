import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

from models.LeadLag_main import LeadLagConfig, LeadLagAnalyzer
from evaluation.metrics import (
    compute_metrics_timeseries,
    summarize_metrics,
    plot_signal_strength,
    plot_stability,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs. Install with: pip install pyyaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _merge_extends(cfg_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    if 'extends' in cfg and cfg['extends']:
        base_path = (cfg_path.parent / cfg['extends']).resolve()
        base = _load_yaml(base_path)
        # shallow merge: base <- cfg
        def deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    a[k] = deep_update(a[k], v)
                else:
                    a[k] = v
            return a
        merged = deep_update(base, {k: v for k, v in cfg.items() if k != 'extends'})
        return merged
    return cfg


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _detect_git() -> Dict[str, Any]:
    import subprocess
    meta = {}
    try:
        meta['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        meta['git_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode()
        meta['git_dirty'] = len(status.strip()) > 0
    except Exception:
        meta['git_commit'] = None
        meta['git_branch'] = None
        meta['git_dirty'] = None
    return meta


def _env_info() -> Dict[str, Any]:
    import sys, platform
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
    }
    # capture key libs if available
    def v(pkg):
        try:
            mod = __import__(pkg)
            return getattr(mod, '__version__', 'unknown')
        except Exception:
            return None
    for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'tqdm', 'iisignature', 'dcor']:
        info[f'{pkg}_version'] = v(pkg)
    return info


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def _read_prices(cfg: Dict[str, Any]) -> pd.DataFrame:
    price_path = Path(cfg['data'].get('price_csv', 'raw_data/daily_price.csv'))
    if not price_path.exists():
        # try to autodetect a daily_prices_* file
        candidates = list(Path('raw_data').glob('daily_prices_*.csv'))
        if candidates:
            price_path = candidates[0]
    if not price_path.exists():
        raise FileNotFoundError(f"Price CSV not found: {price_path}")
    df = pd.read_csv(price_path)
    # best-effort standardization: expect a first column timestamp and rest as assets
    # try common names
    if 'date' in df.columns:
        idx = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    elif 'Date' in df.columns:
        idx = pd.to_datetime(df['Date'])
        df = df.drop(columns=['Date'])
    else:
        # assume first column is datetime
        idx = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]
    df.index = idx
    df = df.sort_index()
    return df


def _config_to_leadlag(cfg: Dict[str, Any]) -> LeadLagConfig:
    # Flatten config into the dict expected by LeadLagConfig.from_dict
    a = cfg['analysis']
    merged = dict(a)
    # carry method-specific block under the same key name
    if a.get('method') and a.get(a['method']):
        merged[a['method']] = a[a['method']]
    return LeadLagConfig.from_dict(merged)


def run_scenario(config_path: str, out_root: str | None = None) -> Path:
    cfg_path = Path(config_path)
    cfg = _merge_extends(cfg_path)

    # seeds and output dir
    _set_seed(int(cfg['run'].get('seed', 42)))
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_name = cfg['run'].get('run_name', 'auto')
    out_root = out_root or cfg['run'].get('output_root', 'results')
    out_dir = Path(out_root) / f"{run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # snapshot configs and metadata
    with open(out_dir / 'config_merged.yaml', 'w', encoding='utf-8') as f:
        if yaml is None:
            f.write(json.dumps(cfg, indent=2))
        else:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    meta = {
        'config_path': str(cfg_path.resolve()),
        'data_price_path': cfg['data'].get('price_csv', ''),
        'created_at': ts,
        'git': _detect_git(),
        'env': _env_info(),
    }
    with open(out_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # load data
    prices = _read_prices(cfg)
    # optional universe is not wired due to format variability; pass None

    # build analyzer
    ll_cfg = _config_to_leadlag(cfg)
    analyzer = LeadLagAnalyzer(ll_cfg)

    # compute rolling matrices
    rolling = analyzer.analyze(prices, return_rolling=True)
    # compute metrics
    metrics_df = compute_metrics_timeseries(rolling)
    metrics_df.to_csv(out_dir / 'metrics_timeseries.csv', index=True)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(out_dir / 'summary.csv', index=False)

    # plots
    if 'metrics' in cfg and 'plots' in cfg['metrics']:
        if 'signal_strength' in cfg['metrics']['plots']:
            plot_signal_strength(metrics_df, out_dir / 'fig_signal_strength.png')
        if 'stability' in cfg['metrics']['plots']:
            plot_stability(metrics_df, out_dir / 'fig_stability.png')

    # save a small sample matrix for inspection
    if len(rolling) > 0:
        first_date = rolling.index[0]
        last_date = rolling.index[-1]
        rolling[first_date].to_csv(out_dir / f"matrix_{first_date.date()}.csv")
        rolling[last_date].to_csv(out_dir / f"matrix_{last_date.date()}.csv")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to scenario YAML')
    ap.add_argument('--out', type=str, default=None, help='Output root directory')
    args = ap.parse_args()
    out_dir = run_scenario(args.config, args.out)
    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()

