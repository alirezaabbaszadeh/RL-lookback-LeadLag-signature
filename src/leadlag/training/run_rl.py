from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.utils import set_random_seed

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    EvalCallback = None  # type: ignore

    def set_random_seed(seed: int) -> None:  # type: ignore
        np.random.seed(seed)

    SB3_AVAILABLE = False

from leadlag.envs.leadlag_env import LeadLagEnv
from leadlag.evaluation.metrics import (
    compute_metrics_timeseries,
    plot_signal_strength,
    plot_stability,
    summarize_metrics,
)
from leadlag.training.policy_factory import make_algorithm_spec
from leadlag.training.run_scenario import _config_to_leadlag
from leadlag.training.scenario_config import _merge_extends
from leadlag.training.run_support import prepare_run_environment
from leadlag.utils.config import deep_update
from leadlag.utils import update_run_manifest


_UNSET = object()


def _instantiate_env(
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    random_start: Optional[bool] = None,
    episode_length: object = _UNSET,
) -> LeadLagEnv:
    ll_cfg = _config_to_leadlag(cfg)
    rl_cfg = cfg.get("rl", {})
    resolved_episode_length = rl_cfg.get("episode_length") if episode_length is _UNSET else episode_length
    if resolved_episode_length is not None:
        resolved_episode_length = int(resolved_episode_length)
    resolved_random_start = (
        bool(rl_cfg.get("random_start", True)) if random_start is None else bool(random_start)
    )
    env = LeadLagEnv(
        price_df=prices,
        leadlag_config=ll_cfg,
        min_lookback=int(rl_cfg.get("min_lookback", 10)),
        max_lookback=int(rl_cfg.get("max_lookback", 120)),
        discrete_actions=bool(rl_cfg.get("discrete_actions", True)),
        reward_weights=rl_cfg.get("reward_weights", None),
        reward_template=rl_cfg.get("reward_template"),
        penalty_same=float(rl_cfg.get("penalty_same", 0.05)),
        penalty_step=int(rl_cfg.get("penalty_step", 10)),
        action_mode=rl_cfg.get("action_mode", "absolute"),
        relative_step=int(rl_cfg.get("relative_step", 5)),
        episode_length=resolved_episode_length,
        random_start=resolved_random_start,
        random_seed=rl_cfg.get("random_seed"),
        ema_alpha=rl_cfg.get("ema_alpha"),
    )
    return env


def _compute_backtest_series(
    prices: pd.DataFrame,
    history: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Convert env history into a simple position-based backtest over the price basket."""

    if history is None or history.empty:
        return None
    if prices is None or prices.empty:
        return None

    env_cfg = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    max_abs_position = float(env_cfg.get("max_abs_position", 1.0)) if env_cfg else 1.0
    if max_abs_position <= 0.0:
        max_abs_position = 1.0
    allow_short = bool(env_cfg.get("allow_short", True)) if env_cfg else True
    min_position = -max_abs_position if allow_short else 0.0
    initial_position = float(env_cfg.get("initial_position", 0.0)) if env_cfg else 0.0
    initial_position = float(np.clip(initial_position, min_position, max_abs_position))

    if "position" in history.columns:
        signals = pd.to_numeric(history["position"], errors="coerce")
        kind = "position"
    elif "trading_signal" in history.columns:
        signals = pd.to_numeric(history["trading_signal"], errors="coerce")
        kind = "trading_signal"
    elif "lookback_normalized" in history.columns:
        signals = pd.to_numeric(history["lookback_normalized"], errors="coerce")
        kind = "lookback_normalized"
    else:
        return None

    signals = signals.reindex(history.index).astype(float)

    if kind == "position":
        positions = signals
    elif kind == "trading_signal":
        if allow_short:
            positions = signals.clip(-1.0, 1.0) * max_abs_position
        else:
            positions = ((signals + 1.0) / 2.0).clip(0.0, 1.0) * max_abs_position
    else:  # lookback_normalized
        normalized = signals.clip(0.0, 1.0)
        if allow_short:
            positions = (normalized * 2.0 - 1.0) * max_abs_position
        else:
            positions = normalized * max_abs_position

    positions = positions.ffill().fillna(initial_position)
    positions = positions.clip(lower=min_position, upper=max_abs_position).astype(float)

    base_returns = prices.sort_index().pct_change().mean(axis=1).astype(float)
    base_returns = base_returns.reindex(history.index).ffill().fillna(0.0).astype(float)

    trades = positions.diff().fillna(positions.iloc[0] - initial_position).astype(float)

    costs_cfg = cfg.get("costs") if isinstance(cfg.get("costs"), dict) else {}
    slippage_cfg = cfg.get("slippage") if isinstance(cfg.get("slippage"), dict) else {}
    fee_bps = float(costs_cfg.get("fee_bps", 0.0)) if costs_cfg else 0.0
    slippage_bps = float(slippage_cfg.get("bps", 0.0)) if slippage_cfg else 0.0
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = trades.abs() * cost_rate
    costs = costs.astype(float)

    executed_positions = positions.shift(1).fillna(initial_position).astype(float)
    portfolio_returns = executed_positions * base_returns - costs
    portfolio_returns = portfolio_returns.astype(float)

    return portfolio_returns, positions, trades, costs


def run_rl(
    cfg_path: str, out_root: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Path:
    overrides = dict(overrides or {})
    raw_cfg = overrides.pop("_raw_config", None)

    if raw_cfg is not None:
        cfg = raw_cfg
        cfg_path = Path(cfg_path)
        if overrides:
            cfg = deep_update(cfg, overrides)
    else:
        cfg_path = Path(cfg_path)
        cfg = _merge_extends(cfg_path)
        if overrides:
            cfg = deep_update(cfg, overrides)
    rl_cfg = cfg.get("rl", {})
    policy_cfg = rl_cfg.get("policy", "")
    normalized_policy = str(policy_cfg).lower() if isinstance(policy_cfg, str) else ""
    use_random_policy = normalized_policy == "random" or bool(rl_cfg.get("random_policy", False))
    if not SB3_AVAILABLE and not use_random_policy:
        raise ImportError("stable-baselines3 is required for RL policies other than 'random'.")
    run_name = cfg["run"].get("run_name", "rl_ppo")
    out_root = out_root or cfg["run"].get("output_root", "results")
    requested_env_steps = int(rl_cfg.get("total_timesteps", 0)) if rl_cfg.get("total_timesteps") else None

    preparation = prepare_run_environment(
        cfg,
        cfg_path=cfg_path,
        module="rl",
        logger_name="run_rl",
        out_root=out_root,
        run_name=run_name,
        extra_logging_context={"scenario": cfg_path.stem},
        extra_metadata={"scenario": cfg_path.stem, "use_random_policy": use_random_policy},
        requested_env_steps=requested_env_steps,
    )
    out_dir = preparation.out_dir
    logger = preparation.logger
    logger.info("Starting RL run", context={"use_random_policy": use_random_policy})
    logger.info("Dataset manifest captured", context={"manifest": str(preparation.manifest_path)})
    set_random_seed(preparation.seed)

    rl_manifest = {
        "use_random_policy": use_random_policy,
        "episode_length": rl_cfg.get("episode_length"),
        "random_start": rl_cfg.get("random_start", True),
        "ema_alpha": rl_cfg.get("ema_alpha"),
    }
    if requested_env_steps is not None:
        rl_manifest["requested_timesteps"] = requested_env_steps
    update_run_manifest(
        preparation.run_manifest_path,
        {
            "rl": rl_manifest,
        },
    )

    prices = preparation.prices
    env = _instantiate_env(prices, cfg)

    step_count = 0

    def reset_env(env):
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            return reset_out
        return reset_out, {}

    def step_env(env, action):
        step_out = env.step(action)
        if isinstance(step_out, tuple):
            if len(step_out) == 5:
                return step_out
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                return obs, reward, bool(done), False, info
        raise RuntimeError("Environment step returned unexpected structure.")

    if use_random_policy:
        logger.info("Executing random policy baseline")
        # Random policy baseline (negative control)
        eval_env = _instantiate_env(prices, cfg, random_start=False)
        obs, _ = reset_env(eval_env)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, info = step_env(eval_env, action)
            step_count += 1
    else:
        algo_spec = make_algorithm_spec(rl_cfg)
        total_timesteps = int(rl_cfg.get("total_timesteps", 50000))

        algo_kwargs = {
            "learning_rate": rl_cfg.get("learning_rate", 3e-4),
            "n_steps": int(rl_cfg.get("n_steps", 512)),
            "batch_size": int(rl_cfg.get("batch_size", 256)),
            "gamma": float(rl_cfg.get("gamma", 0.99)),
            "ent_coef": float(rl_cfg.get("ent_coef", 0.0)),
            "verbose": 1 if rl_cfg.get("verbose", False) else 0,
            "seed": preparation.seed,
        }
        # Allow device override ("auto", "cuda", or "cpu") if provided in config
        device = rl_cfg.get("device")
        if device:
            algo_kwargs["device"] = str(device)
        if algo_spec.policy_kwargs:
            algo_kwargs["policy_kwargs"] = algo_spec.policy_kwargs

        logger.info(
            "Initializing RL algorithm",
            context={
                "algo": algo_spec.algo_cls.__name__,
                "policy": str(algo_spec.policy),
                "total_timesteps": total_timesteps,
            },
        )
        model = algo_spec.algo_cls(algo_spec.policy, env, **algo_kwargs)

        update_run_manifest(
            preparation.run_manifest_path,
            {
                "rl": {
                    "algo": algo_spec.algo_cls.__name__,
                    "policy": str(algo_spec.policy),
                    "n_steps": algo_kwargs["n_steps"],
                    "batch_size": algo_kwargs["batch_size"],
                    "learning_rate": algo_kwargs["learning_rate"],
                    "gamma": algo_kwargs["gamma"],
                    "ent_coef": algo_kwargs["ent_coef"],
                    "eval_freq": int(rl_cfg.get("eval_freq", 0)),
                },
            },
        )

        # optional evaluation callback (self-play, so reuse env)
        eval_freq = int(rl_cfg.get("eval_freq", 0))
        callbacks = []
        if eval_freq > 0:
            eval_env = _instantiate_env(prices, cfg, random_start=False)
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(out_dir / "eval"),
                    log_path=str(out_dir / "eval_logs"),
                    eval_freq=eval_freq,
                    deterministic=True,
                )
            )

        logger.info("Starting training loop")
        model.learn(total_timesteps=total_timesteps, callback=callbacks if callbacks else None)
        logger.info("Training loop completed")

        model.save(str(out_dir / "model.zip"))

        # Evaluate deterministic policy over the dataset
        eval_env = _instantiate_env(prices, cfg, random_start=False)
        obs, _ = reset_env(eval_env)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = step_env(eval_env, action)
            step_count += 1

    # collect history matrices for metrics
    rolling = eval_env.get_history_matrices()
    metrics_df = compute_metrics_timeseries(rolling)
    # enrich with lookback decisions
    decisions = eval_env.get_history_dataframe()
    if not decisions.empty:
        backtest = _compute_backtest_series(prices, decisions, cfg)
        if backtest is not None:
            portfolio_returns, positions, trades, costs = backtest
            decisions = decisions.copy()
            decisions["portfolio_return"] = portfolio_returns
            decisions["position"] = positions
            decisions["trade"] = trades
            decisions["cost"] = costs

        extra_cols = [
            col
            for col in (
                "lookback",
                "lookback_normalized",
                "trading_signal",
                "reward",
                "portfolio_return",
                "position",
                "trade",
                "cost",
            )
            if col in decisions.columns
        ]
        if extra_cols:
            metrics_df = metrics_df.join(decisions[extra_cols], how="left")

    metrics_df.to_csv(out_dir / "metrics_timeseries.csv", index=True)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    plot_signal_strength(metrics_df, out_dir / "fig_signal_strength.png")
    plot_stability(metrics_df, out_dir / "fig_stability.png")

    model_path = out_dir / "model.zip"
    logger.info(
        "RL run completed",
        context={
            "steps": step_count,
            "summary": str(out_dir / "summary.csv"),
            "model_path": str(model_path) if model_path.exists() else "random_policy",
        },
    )

    update_run_manifest(
        preparation.run_manifest_path,
        {
            "actual_env_steps": int(step_count),
            "rl": {
                "evaluated_policy": "deterministic",
                "vectorised_envs": int(rl_cfg.get("n_envs", 1)),
            },
            "artifacts": {
                "metrics_timeseries": str(out_dir / "metrics_timeseries.csv"),
                "summary": str(out_dir / "summary.csv"),
                "model": str(model_path) if model_path.exists() else None,
            },
        },
    )

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to RL scenario YAML")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_dir = run_rl(args.config, args.out)
    print(f"Saved RL results to: {out_dir}")


if __name__ == "__main__":
    main()
