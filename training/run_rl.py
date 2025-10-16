from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from models.LeadLag_main import LeadLagConfig
from envs.leadlag_env import LeadLagEnv
from training.run_scenario import (
    _merge_extends,
    _read_prices,
    _config_to_leadlag,
    _detect_git,
    _env_info,
)
from evaluation.metrics import compute_metrics_timeseries, summarize_metrics, plot_signal_strength, plot_stability


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_config(cfg: Dict[str, Any], out_dir: Path):
    if yaml is not None:
        with open(out_dir / 'config_merged.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    else:
        with open(out_dir / 'config_merged.yaml', 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)


def _instantiate_env(prices: pd.DataFrame, cfg: Dict[str, Any]) -> LeadLagEnv:
    ll_cfg = _config_to_leadlag(cfg)
    rl_cfg = cfg.get('rl', {})
    env = LeadLagEnv(
        price_df=prices,
        leadlag_config=ll_cfg,
        min_lookback=int(rl_cfg.get('min_lookback', 10)),
        max_lookback=int(rl_cfg.get('max_lookback', 120)),
        discrete_actions=bool(rl_cfg.get('discrete_actions', True)),
        reward_weights=rl_cfg.get('reward_weights', None),
        penalty_same=float(rl_cfg.get('penalty_same', 0.05)),
        penalty_step=int(rl_cfg.get('penalty_step', 10)),
    )
    return env


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run_rl(cfg_path: str, out_root: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Path:
    overrides = dict(overrides or {})
    raw_cfg = overrides.pop('_raw_config', None)

    if raw_cfg is not None:
        cfg = raw_cfg
        cfg_path = Path(cfg_path)
        if overrides:
            cfg = _deep_update(cfg, overrides)
    else:
        cfg_path = Path(cfg_path)
        cfg = _merge_extends(cfg_path)
        if overrides:
            cfg = _deep_update(cfg, overrides)
    rl_cfg = cfg.get('rl', {})
    seed = int(cfg['run'].get('seed', 42))
    set_random_seed(seed)

    # prepare output
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_name = cfg['run'].get('run_name', 'rl_ppo')
    out_root = out_root or cfg['run'].get('output_root', 'results')
    out_dir = Path(out_root) / f"{run_name}_{ts}"
    _ensure_dir(out_dir)

    _save_config(cfg, out_dir)
    metadata = {
        'config_path': str(cfg_path.resolve()),
        'created_at': ts,
        'git': _detect_git(),
        'env': _env_info(),
    }
    with open(out_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    prices = _read_prices(cfg)

    env = _instantiate_env(prices, cfg)

    policy = rl_cfg.get('policy', 'MlpPolicy')
    total_timesteps = int(rl_cfg.get('total_timesteps', 50000))

    model = PPO(
        policy,
        env,
        learning_rate=rl_cfg.get('learning_rate', 3e-4),
        n_steps=int(rl_cfg.get('n_steps', 512)),
        batch_size=int(rl_cfg.get('batch_size', 256)),
        gamma=float(rl_cfg.get('gamma', 0.99)),
        ent_coef=float(rl_cfg.get('ent_coef', 0.0)),
        verbose=1 if rl_cfg.get('verbose', False) else 0,
        seed=seed,
    )

    # optional evaluation callback (self-play, so reuse env)
    eval_freq = int(rl_cfg.get('eval_freq', 0))
    callbacks = []
    if eval_freq > 0:
        eval_env = _instantiate_env(prices, cfg)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(out_dir / 'eval'),
                log_path=str(out_dir / 'eval_logs'),
                eval_freq=eval_freq,
                deterministic=True,
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)

    model.save(str(out_dir / 'model.zip'))

    # Evaluate deterministic policy over the dataset
    eval_env = _instantiate_env(prices, cfg)
    obs = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

    # collect history matrices for metrics
    rolling = eval_env.get_history_matrices()
    metrics_df = compute_metrics_timeseries(rolling)
    # enrich with lookback decisions
    decisions = eval_env.get_history_dataframe()
    if not decisions.empty:
        metrics_df = metrics_df.join(decisions[['lookback']], how='left')

    metrics_df.to_csv(out_dir / 'metrics_timeseries.csv', index=True)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(out_dir / 'summary.csv', index=False)

    plot_signal_strength(metrics_df, out_dir / 'fig_signal_strength.png')
    plot_stability(metrics_df, out_dir / 'fig_stability.png')

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to RL scenario YAML')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    out_dir = run_rl(args.config, args.out)
    print(f"Saved RL results to: {out_dir}")


if __name__ == '__main__':
    main()
