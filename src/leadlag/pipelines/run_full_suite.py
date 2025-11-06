"""Hydra-driven pipeline entry point for the lead-lag project."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

try:  # Optional RL dependency
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except Exception:  # pragma: no cover - library optional in CI
    A2C = DQN = PPO = SAC = TD3 = None  # type: ignore[assignment]
    DummyVecEnv = Monitor = None  # type: ignore[assignment]
    SB3_AVAILABLE = False

from leadlag.cv.purged import walk_forward_purged
from leadlag.env.trading_env import TradeMetrics
from leadlag.envs.leadlag_env import LeadLagEnv
from leadlag.features.cache import FeatureCacheKey, load_feature_stack, save_feature_stack
from leadlag.features.leadlag import compute_lead_lag
from leadlag.features.signature import compute_signature_features
from leadlag.governance import dataset as dataset_mod
from leadlag.eval import stats as stats_mod
from leadlag.eval.stats_cli import run_workflow as run_stats_workflow
from leadlag.models.config import LeadLagConfig, SIGNATURE_AVAILABLE
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row
from leadlag.utils import (
    collect_determinism_settings,
    select_device,
    set_all_seeds,
    write_run_manifest,
)


logger = logging.getLogger(__name__)


@dataclass
class RealizedTradingPath:
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    costs: pd.Series
    metrics: TradeMetrics


def _collect_config_sources(cfg: DictConfig) -> List[str]:
    sources: List[str] = []

    metadata = getattr(cfg, "_metadata", None)
    meta_sources = getattr(metadata, "sources", None) if metadata else None
    if meta_sources:
        for source in meta_sources:
            path = getattr(source, "path", None)
            provider = getattr(source, "provider", None)
            if path and provider:
                sources.append(f"{provider}:{path}")
            elif path:
                sources.append(str(path))
            elif provider:
                sources.append(str(provider))
            else:
                sources.append(str(source))

    if not sources:
        runtime_sources = OmegaConf.select(cfg, "hydra.runtime.config_sources")
        if runtime_sources is None:
            try:
                from hydra.core.hydra_config import HydraConfig

                if HydraConfig.initialized():
                    runtime_sources = OmegaConf.select(
                        HydraConfig.get().cfg, "hydra.runtime.config_sources"
                    )
            except Exception:  # pragma: no cover - defensive fallback
                runtime_sources = None

        if runtime_sources:
            for entry in runtime_sources:
                if isinstance(entry, dict):
                    path = entry.get("path")
                    provider = entry.get("provider")
                    if path and provider:
                        sources.append(f"{provider}:{path}")
                    elif path:
                        sources.append(str(path))
                    elif provider:
                        sources.append(str(provider))
                    else:
                        sources.append(str(entry))
                else:
                    sources.append(str(entry))

    if not sources:
        try:
            from hydra.utils import get_original_cwd

            original_cwd = get_original_cwd()
        except Exception:  # pragma: no cover - Hydra not initialised
            original_cwd = None

        if original_cwd:
            sources.append(original_cwd)

    return sources


def _log_config_sources(cfg: DictConfig) -> List[str]:
    sources = _collect_config_sources(cfg)
    if sources:
        logger.info(
            "Hydra config sources (load order): %s",
            " -> ".join(sources),
        )
    else:
        logger.info("Hydra config sources (load order): <unavailable>")
    return sources



def _format_run_id(base: str, seed: int, window_idx: int, append_suffix: bool) -> str:
    if append_suffix:
        return f"{base}-s{seed:02d}-w{window_idx:02d}"
    return base


def _prepare_directories(results_root: Path, run_id: str) -> Path:
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_price_data(cfg: DictConfig, seed: int) -> tuple[pd.DataFrame, Optional[Path]]:
    data_cfg = cfg.get("data") or {}
    dataset_dir = Path(data_cfg.get("dataset_dir", "")) if data_cfg else None
    if dataset_dir and dataset_dir.exists():
        candidates: List[Path] = []
        for name in ("prices.parquet", "prices.feather", "prices.csv"):
            candidate = dataset_dir / name
            if candidate.exists():
                candidates.append(candidate)
        if not candidates:
            candidates.extend(sorted(dataset_dir.glob("*.parquet")))
            candidates.extend(sorted(dataset_dir.glob("*.feather")))
            candidates.extend(sorted(dataset_dir.glob("*.csv")))
        for path in candidates:
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                elif path.suffix == ".feather":
                    df = pd.read_feather(path)
                else:
                    df = pd.read_csv(path)
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df.pop("date")))
                elif "Date" in df.columns:
                    df = df.set_index(pd.to_datetime(df.pop("Date")))
                elif not isinstance(df.index, pd.DatetimeIndex):
                    first = df.columns[0]
                    df = df.set_index(pd.to_datetime(df.pop(first)))
                df = df.sort_index()
                if df.shape[1] >= 2:
                    return df, path
            except Exception:  # pragma: no cover - corrupted file fallback
                continue

    # Fallback: synthesize a lightweight dataset
    rng = np.random.default_rng(seed)
    periods = int(cfg.window.lookback) + int(cfg.training.total_env_steps) + 32
    periods = max(periods, 128)
    index = pd.date_range("2020-01-01", periods=periods, freq="D")
    levels = 100 + np.cumsum(rng.normal(scale=0.01, size=(periods, 3)), axis=0)
    df = pd.DataFrame(levels, index=index, columns=["AssetA", "AssetB", "AssetC"])
    return df, None


def _build_feature_stack(
    prices: pd.DataFrame,
    features_cfg: DictConfig,
    *,
    universe: Optional[str],
    timeframe: Optional[str],
    lookback: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    cache_cfg = features_cfg.get("cache") if features_cfg else None
    cache_enabled = bool(cache_cfg.get("enabled")) if cache_cfg else False
    cache_dir = None
    signature_cfg = features_cfg.get("signature") if features_cfg else None
    signature_enabled = bool(signature_cfg and signature_cfg.get("enabled"))
    signature_depth = int(signature_cfg.get("depth", 0)) if signature_cfg else 0
    leadlag_cfg = features_cfg.get("leadlag") if features_cfg else None
    leadlag_enabled = bool(leadlag_cfg and leadlag_cfg.get("enabled"))
    time_channel_enabled = bool(features_cfg.get("time_channel")) if features_cfg else False

    if cache_enabled:
        cache_dir = Path(cache_cfg.get("dir", ".cache/features")).expanduser()
        key = FeatureCacheKey(
            universe=universe,
            timeframe=timeframe,
            lookback=int(lookback),
            signature_depth=signature_depth,
            seed=int(seed),
            signature_enabled=signature_enabled,
            leadlag_enabled=leadlag_enabled,
            time_channel=time_channel_enabled,
        )
        cached = load_feature_stack(cache_dir, key)
        if cached is not None:
            logger.info("Loaded feature stack from cache: %s", cache_dir / key.filename())
            return cached

    stack: Dict[str, np.ndarray] = {}
    returns = prices.pct_change().dropna()
    stack["returns"] = returns.to_numpy(dtype=float)

    if signature_enabled:
        depth = int(signature_cfg.get("depth", 2))
        flattened = returns.to_numpy(dtype=float).ravel()
        stack["signature"] = compute_signature_features(flattened, depth)

    if leadlag_enabled:
        reference_series = returns.mean(axis=1).to_numpy(dtype=float)
        stack["leadlag"] = compute_lead_lag(reference_series)

    if time_channel_enabled:
        stack["time_channel"] = np.linspace(0.0, 1.0, num=len(returns), dtype=float)

    if cache_enabled and cache_dir is not None:
        save_feature_stack(cache_dir, key, stack)
        logger.info("Saved feature stack to cache: %s", cache_dir / key.filename())

    return stack


def _leadlag_config(features_cfg: DictConfig, lookback: int) -> LeadLagConfig:
    signature_cfg = features_cfg.get("signature") if features_cfg else None
    wants_signature = bool(signature_cfg and signature_cfg.get("enabled"))
    method = "signature" if wants_signature and SIGNATURE_AVAILABLE else "ccf_at_lag"
    if method == "signature" and not SIGNATURE_AVAILABLE:
        wants_signature = False
    params: Dict[str, object] = {
        "method": method,
        "lookback": int(lookback),
        "update_freq": 1,
        "use_parallel": False,
        "num_cpus": 1,
        "show_progress": False,
        "Scaling_Method": "mean-centering",
    }
    if method == "ccf_at_lag":
        params["lag"] = 1
        params["correlation_method"] = "pearson"
        params["quantiles"] = 4
    else:
        sig_method = signature_cfg.get("method", "custom") if signature_cfg else "custom"
        params["sig_method"] = sig_method
        params["correlation_method"] = "pearson"
        params["quantiles"] = 4
    return LeadLagConfig.from_dict(params)


def _make_leadlag_env(
    cfg: DictConfig,
    prices: pd.DataFrame,
    *,
    seed: int,
) -> LeadLagEnv:
    features_cfg = cfg.get("features") or {}
    env_cfg = cfg.get("env") or {}
    lookback = int(cfg.window.lookback)
    ll_config = _leadlag_config(features_cfg, lookback)

    action_space = str(env_cfg.get("action_space", "discrete3")).lower()
    if action_space == "discrete3":
        discrete_actions = True
        action_mode = "relative"
        relative_step = 5
    else:
        discrete_actions = True
        action_mode = "absolute"
        relative_step = 1

    min_lookback = max(5, lookback // 2)
    max_lookback = max(min_lookback + 1, lookback)
    episode_length = max(lookback + 1, min(len(prices) - lookback, int(cfg.training.total_env_steps)))

    env = LeadLagEnv(
        price_df=prices,
        leadlag_config=ll_config,
        min_lookback=min_lookback,
        max_lookback=max_lookback,
        discrete_actions=discrete_actions,
        action_mode=action_mode,
        relative_step=relative_step,
        episode_length=episode_length,
        random_start=True,
        random_seed=seed,
        ema_alpha=None,
    )
    return env


def _summarize_trade_history(
    cfg: DictConfig,
    history: pd.DataFrame,
    returns: pd.Series,
) -> TradeMetrics:
    """Convert environment history into :class:`TradeMetrics`.

    The environment records the lookback applied at each step and, when available,
    the normalised change between consecutive steps (``delta_norm``). These values
    allow us to recover a turnover profile and apply configured commission and
    slippage rates instead of relying on reward magnitudes.
    """

    pnl = float(returns.sum())

    if history.empty:
        env_steps = int(returns.size)
        return TradeMetrics(
            pnl=pnl,
            turnover=0.0,
            exposure=0.0,
            env_steps=env_steps,
            costs=0.0,
        )

    turnover_series: pd.Series
    if "delta_norm" in history.columns:
        turnover_series = history["delta_norm"].astype(float).abs()
    elif "lookback" in history.columns:
        turnover_series = history["lookback"].astype(float).diff().fillna(0.0).abs()
    else:
        turnover_series = pd.Series(0.0, index=history.index, dtype=float)

    env_steps = int(turnover_series.size) if not turnover_series.empty else int(returns.size)
    turnover = float(turnover_series.mean()) if not turnover_series.empty else 0.0
    cumulative_turnover = float(turnover_series.sum()) if not turnover_series.empty else 0.0

    if "lookback" in history.columns and not history["lookback"].empty:
        lookback_series = history["lookback"].astype(float)
        lb_min = float(lookback_series.min())
        lb_max = float(lookback_series.max())
        denominator = max(lb_max - lb_min, 1.0)
        normalised_positions = (lookback_series - lb_min) / denominator
        exposure = float(normalised_positions.abs().mean())
    else:
        exposure = 0.0

    costs_cfg = cfg.get("costs") or {}
    slippage_cfg = cfg.get("slippage") or {}
    fee_bps = float(costs_cfg.get("fee_bps", 0.0))
    slippage_bps = float(slippage_cfg.get("bps", 0.0))
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    total_costs = cumulative_turnover * cost_rate

    return TradeMetrics(
        pnl=pnl,
        turnover=turnover,
        exposure=exposure,
        env_steps=env_steps,
        costs=float(total_costs),
    )


def _replay_trading_path(
    cfg: DictConfig,
    prices: pd.DataFrame,
    history: pd.DataFrame,
) -> Optional[RealizedTradingPath]:
    if history is None or history.empty or "lookback" not in history.columns:
        return None

    lookbacks = history["lookback"].astype(float).dropna()
    if lookbacks.empty:
        return None

    window_cfg = cfg.get("window")
    base_lookback = int(getattr(window_cfg, "lookback", 0) or 0) if window_cfg else 0
    if base_lookback > 0:
        min_lb = max(5, base_lookback // 2)
        max_lb = max(min_lb + 1, base_lookback)
    else:
        min_lb = int(np.floor(lookbacks.min()))
        max_lb = int(np.ceil(lookbacks.max()))
        if min_lb == max_lb:
            max_lb += 1

    denom = max(float(max_lb - min_lb), 1.0)
    normalized = (lookbacks - min_lb) / denom
    normalized = normalized.clip(0.0, 1.0)

    env_cfg = cfg.get("env") or {}
    max_abs_position = float(env_cfg.get("max_abs_position", 1.0))
    allow_short = bool(env_cfg.get("allow_short", True))
    initial_position = float(env_cfg.get("initial_position", 0.0))
    min_position = -max_abs_position if allow_short else 0.0
    max_position = max_abs_position
    initial_position = float(np.clip(initial_position, min_position, max_position))

    if allow_short:
        scaled = normalized * 2.0 - 1.0
    else:
        scaled = normalized

    positions = pd.Series(scaled * max_abs_position, index=lookbacks.index, dtype=float)
    positions = positions.clip(lower=min_position, upper=max_position)
    positions = positions.sort_index()

    price_returns = prices.sort_index().pct_change().reindex(positions.index)
    if price_returns is None or price_returns.empty:
        return None

    base_returns = price_returns.mean(axis=1).fillna(0.0).astype(float)
    if base_returns.empty:
        return None

    positions = positions.reindex(base_returns.index).ffill().fillna(initial_position).astype(float)
    prev_positions = positions.shift(fill_value=initial_position)
    trades = (positions - prev_positions).astype(float)

    costs_cfg = cfg.get("costs") or {}
    slippage_cfg = cfg.get("slippage") or {}
    fee_bps = float(costs_cfg.get("fee_bps", 0.0))
    slippage_bps = float(slippage_cfg.get("bps", 0.0))
    cost_rate = (fee_bps + slippage_bps) / 10000.0

    costs = trades.abs() * cost_rate
    realized_returns = positions * base_returns - costs
    realized_returns = realized_returns.astype(float)
    costs = costs.astype(float)

    if realized_returns.empty:
        return None

    turnover = float(trades.abs().mean()) if not trades.empty else 0.0
    exposure = float(positions.abs().mean()) if not positions.empty else 0.0
    metrics = TradeMetrics(
        pnl=float(realized_returns.sum()),
        turnover=turnover,
        exposure=exposure,
        env_steps=int(realized_returns.size),
        costs=float(costs.sum()),
    )

    return RealizedTradingPath(
        returns=realized_returns,
        positions=positions,
        trades=trades,
        costs=costs,
        metrics=metrics,
    )


def _random_rollout(
    cfg: DictConfig,
    env: LeadLagEnv,
    total_steps: int,
    seed: int,
) -> tuple[pd.Series, TradeMetrics, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    rewards: List[float] = []
    while steps < total_steps and not (terminated or truncated):
        if hasattr(env.action_space, "sample"):
            action = env.action_space.sample()
        else:  # pragma: no cover - defensive
            action = rng.integers(0, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        steps += 1
        if terminated or truncated:
            terminated = False
            truncated = False
            obs, _ = env.reset()
    history = env.get_history_dataframe()
    if "reward" in history.columns and not history["reward"].empty:
        returns = history["reward"].astype(float)
    else:
        index = pd.RangeIndex(len(rewards))
        returns = pd.Series(rewards, index=index, dtype=float)
    trade_metrics = _summarize_trade_history(cfg, history, returns)
    return returns, trade_metrics, history


def _train_sb3_agent(
    cfg: DictConfig,
    prices: pd.DataFrame,
    total_steps: int,
    *,
    seed: int,
) -> tuple[pd.Series, TradeMetrics, Dict[str, object], pd.DataFrame]:
    if not SB3_AVAILABLE or cfg.agent.library != "sb3":
        env = _make_leadlag_env(cfg, prices, seed=seed)
        returns, trade_metrics, history = _random_rollout(cfg, env, total_steps, seed)
        return (
            returns,
            trade_metrics,
            {"name": "random", "reason": "sb3_unavailable"},
            history,
        )

    algo_name = str(cfg.agent.name).lower()
    algo_map = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN,
        "sac": SAC,
        "td3": TD3,
    }
    algo_cls = algo_map.get(algo_name)
    if algo_cls is None:
        env = _make_leadlag_env(cfg, prices, seed=seed)
        returns, trade_metrics, history = _random_rollout(cfg, env, total_steps, seed)
        return (
            returns,
            trade_metrics,
            {"name": "random", "reason": f"unsupported_agent:{algo_name}"},
            history,
        )

    n_envs = max(1, int(cfg.hardware.n_envs))

    def _make_env_fn(offset: int) -> callable:
        def _init() -> LeadLagEnv:
            env_seed = seed * 1_000 + offset
            env = _make_leadlag_env(cfg, prices, seed=env_seed)
            return Monitor(env) if Monitor is not None else env

        return _init

    if DummyVecEnv is None:  # pragma: no cover - sb3 import mismatch
        vec_env = _make_env_fn(0)()
    else:
        vec_env = DummyVecEnv([_make_env_fn(idx) for idx in range(n_envs)])

    hyperparams = dict(cfg.agent.get("hyperparams", {}))
    device = cfg.hardware.get("device", "cpu")
    hyperparams.setdefault("seed", seed)
    hyperparams.setdefault("device", device)

    policy = cfg.agent.get("policy", "MlpPolicy")
    model = algo_cls(policy, vec_env, **hyperparams)
    model.learn(total_timesteps=total_steps)

    eval_env = _make_leadlag_env(cfg, prices, seed=seed + 99)
    obs, _ = eval_env.reset()
    terminated = False
    truncated = False
    rewards: List[float] = []
    steps = 0
    while steps < total_steps and not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        rewards.append(float(reward))
        steps += 1
    history = eval_env.get_history_dataframe()
    if "reward" in history.columns:
        returns = history["reward"].astype(float)
    else:
        index = pd.RangeIndex(len(rewards))
        returns = pd.Series(rewards, index=index, dtype=float)
    trade_metrics = _summarize_trade_history(cfg, history, returns)
    return (
        returns,
        trade_metrics,
        {"name": algo_name, "library": cfg.agent.library, "trained_steps": steps},
        history,
    )


def _simulate_episode(
    cfg: DictConfig,
    seed: int,
    window_idx: int,
) -> Dict[str, object]:
    training_cfg = cfg.training
    total_steps = max(1, int(training_cfg.total_env_steps))
    effective_seed = seed * 10_000 + window_idx

    prices, dataset_path = _load_price_data(cfg, effective_seed)
    features_cfg = cfg.get("features") or {}
    data_cfg = cfg.get("data")
    window_cfg = cfg.get("window")
    feature_stack = _build_feature_stack(
        prices,
        features_cfg,
        universe=getattr(data_cfg, "universe", None) if data_cfg else None,
        timeframe=getattr(data_cfg, "timeframe", None) if data_cfg else None,
        lookback=int(getattr(window_cfg, "lookback", len(prices))) if window_cfg else len(prices),
        seed=effective_seed,
    )

    reward_returns, reward_metrics, agent_meta, history = _train_sb3_agent(
        cfg, prices, total_steps, seed=effective_seed
    )
    realized_path = _replay_trading_path(cfg, prices, history)

    if realized_path is not None:
        returns_series = realized_path.returns
        trade_metrics = realized_path.metrics
        positions = realized_path.positions
        trades = realized_path.trades
        cost_series = realized_path.costs
    else:
        returns_series = reward_returns
        if returns_series.empty:
            returns_series = pd.Series([0.0], dtype=float)
        trade_metrics = _summarize_trade_history(cfg, history, returns_series)
        positions = None
        trades = None
        cost_series = None

    summary = stats_mod.summarize_performance(
        returns_series,
        periods_per_year=training_cfg.periods_per_year,
    )
    equity = stats_mod.compute_equity_curve(summary.returns)
    if trade_metrics.env_steps == 0:
        trade_metrics = _summarize_trade_history(cfg, history, summary.returns)
    actual_env_steps = trade_metrics.env_steps or int(summary.returns.size)

    return {
        "returns": summary.returns,
        "equity": equity,
        "summary": summary,
        "trade_metrics": trade_metrics,
        "positions": positions,
        "trades": trades,
        "cost_series": cost_series,
        "reward_returns": reward_returns,
        "reward_trade_metrics": reward_metrics,
        "history": history,
        "env_steps": actual_env_steps,
        "requested_env_steps": total_steps,
        "n_envs": int(cfg.hardware.n_envs),
        "prices": prices,
        "dataset_path": dataset_path,
        "feature_stack": feature_stack,
        "agent_info": agent_meta,
        "dataset_length": len(prices),
    }


def _write_artifacts(
    run_dir: Path,
    cfg: DictConfig,
    seed: int,
    window_idx: int,
    simulation: Dict[str, object],
    metrics_writer: MetricsWriter,
) -> None:
    run_id = run_dir.name
    returns_series = simulation["returns"]
    equity_series = simulation["equity"]
    summary = simulation["summary"]
    trade_metrics = simulation["trade_metrics"]
    env_steps = int(simulation.get("env_steps", len(returns_series)))
    prices = simulation.pop("prices", None)
    dataset_path = simulation.get("dataset_path")
    feature_stack = simulation.get("feature_stack", {})
    agent_info = simulation.get("agent_info")

    stats_mod.export_returns(run_dir / "returns.csv", returns_series)
    stats_mod.export_equity(run_dir / "equity.csv", equity_series)

    metrics_row = build_metadata_row(
        run_id,
        OmegaConf.to_container(cfg, resolve=True),
        {
            "sharpe": summary.sharpe,
            "sortino": summary.sortino,
            "max_drawdown": summary.max_drawdown,
            "pnl": summary.pnl,
        },
        seed=seed,
        window_idx=window_idx,
        turnover=trade_metrics.turnover,
        exposure=trade_metrics.exposure,
        costs=trade_metrics.costs,
        env_steps=env_steps,
    )
    metrics_writer.write_row(run_dir / "metrics.csv", metrics_row)

    _write_data_manifest(run_dir, cfg, prices=prices, dataset_path=dataset_path)

    manifest_payload = {
        "run_id": run_id,
        "seed": seed,
        "window_index": window_idx,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics_row,
        "requested_env_steps": int(simulation.get("requested_env_steps", env_steps)),
        "actual_env_steps": env_steps,
        "vectorised_envs": int(simulation.get("n_envs", 1)),
        "agent": agent_info,
        "feature_stack": {
            key: {"shape": list(np.shape(value))}
            for key, value in feature_stack.items()
        },
    }
    manifest_payload["presets"] = {
        "training": OmegaConf.select(cfg, "training.preset_name"),
        "hardware": OmegaConf.select(cfg, "hardware.preset_name"),
    }
    manifest_payload["determinism"] = collect_determinism_settings(int(seed))
    write_run_manifest(run_dir / "run_manifest.json", manifest_payload)


def _resolve_embargo_fraction(split_cfg: DictConfig) -> float:
    if split_cfg is None:
        return 0.0

    explicit_frac = OmegaConf.select(split_cfg, "embargo_frac")
    if explicit_frac is not None:
        return float(explicit_frac)

    nested_frac = OmegaConf.select(split_cfg, "embargo.frac")
    if nested_frac is not None:
        return float(nested_frac)

    return 0.0


def _materialize_walk_forward(cfg: DictConfig, total_samples: int) -> Dict[str, object]:
    embargo_frac = _resolve_embargo_fraction(cfg.split)
    splits: List[Dict[str, object]] = []
    for idx, split in enumerate(
        walk_forward_purged(
            total_samples=total_samples,
            n_splits=int(cfg.split.n_splits),
            embargo_frac=embargo_frac,
        )
    ):
        test_start = split.test_start
        test_end = split.test_end
        splits.append(
            {
                "split": idx,
                "train": split.train_indices.tolist(),
                "test": split.test_indices.tolist(),
                "test_window": {
                    "start": test_start,
                    "end": test_end,
                },
                "embargo": split.embargo,
            }
        )

    return {
        "scheme": str(cfg.split.scheme),
        "total_samples": int(total_samples),
        "parameters": {
            "n_splits": int(cfg.split.n_splits),
            "embargo_frac": embargo_frac,
        },
        "nested": OmegaConf.to_container(cfg.split.get("nested_tuning", {}), resolve=True),
        "splits": splits,
    }


def _persist_split_manifest(
    payload: Dict[str, object],
    split_cfg: DictConfig,
    destination: Path,
    *,
    run_dirs: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    manifest_cfg = split_cfg.get("manifest") if split_cfg else None
    persist = True
    filename = "data_splits.json"
    output_dir: Path = destination

    if manifest_cfg is not None:
        persist = bool(manifest_cfg.get("persist", True))
        filename = str(manifest_cfg.get("filename", filename))
        target_root = manifest_cfg.get("output_dir")
        if target_root:
            output_dir = Path(target_root).expanduser().resolve()

    csv_columns = [
        "split",
        "train_indices",
        "test_indices",
        "test_start",
        "test_end",
        "embargo",
    ]
    csv_records: List[Dict[str, object]] = []
    for entry in payload.get("splits", []):
        test_window = entry.get("test_window") if isinstance(entry, dict) else {}
        if not isinstance(test_window, dict):
            test_window = {}
        csv_records.append(
            {
                "split": entry.get("split") if isinstance(entry, dict) else None,
                "train_indices": entry.get("train") if isinstance(entry, dict) else None,
                "test_indices": entry.get("test") if isinstance(entry, dict) else None,
                "test_start": test_window.get("start"),
                "test_end": test_window.get("end"),
                "embargo": entry.get("embargo") if isinstance(entry, dict) else None,
            }
        )

    splits_frame = pd.DataFrame(csv_records, columns=csv_columns)
    if run_dirs:
        for run_dir in run_dirs:
            run_dir.mkdir(parents=True, exist_ok=True)
            splits_frame.to_csv(run_dir / "splits.csv", index=False)

    if not persist:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / filename
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return manifest_path


def _write_data_manifest(
    run_dir: Path,
    cfg: DictConfig,
    *,
    prices: Optional[pd.DataFrame],
    dataset_path: Optional[Path],
) -> None:
    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    training_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True)
    split_cfg = OmegaConf.to_container(cfg.get("split", {}), resolve=True)

    extras = {
        "universe": data_cfg.get("universe"),
        "timeframe": data_cfg.get("timeframe"),
        "market": data_cfg.get("market"),
        "training": {
            "total_env_steps": training_cfg.get("total_env_steps"),
            "seeds": training_cfg.get("seeds"),
            "windows": training_cfg.get("windows"),
        },
        "split": split_cfg,
    }

    extras = {key: value for key, value in extras.items() if value is not None}

    manifest = dataset_mod.build_manifest(
        prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(),
        source_path=dataset_path,
        extras=extras,
    )
    dataset_dir = data_cfg.get("dataset_dir")
    if dataset_dir:
        manifest["dataset_dir"] = str(dataset_dir)

    dataset_mod.record_manifest(manifest, run_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _log_config_sources(cfg)

    device_info = select_device(dict(cfg.hardware))
    set_all_seeds(int(cfg.training.seeds[0]))

    results_root = Path(cfg.results_root).resolve()
    paper_root = Path(cfg.paper_outputs_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))
    base_run_id = cfg.logging.run_id

    dataset_length: Optional[int] = None
    run_dirs: List[Path] = []
    for window_idx in range(cfg.training.windows):
        for seed in cfg.training.seeds:
            set_all_seeds(int(seed))
            run_id = _format_run_id(base_run_id, int(seed), window_idx, bool(cfg.logging.append_seed_window))
            run_dir = _prepare_directories(results_root, run_id)
            run_dirs.append(run_dir)
            simulation = _simulate_episode(cfg, int(seed), window_idx)
            _write_artifacts(run_dir, cfg, int(seed), window_idx, simulation, metrics_writer)
            dataset_length = simulation.get("dataset_length", dataset_length)

    total_samples = int(dataset_length) if dataset_length else int(cfg.training.total_env_steps)
    split_manifest = _materialize_walk_forward(cfg, total_samples=total_samples)
    _persist_split_manifest(split_manifest, cfg.split, paper_root, run_dirs=run_dirs)

    reporting_cfg = cfg.get("reporting")
    reporting_enabled = True
    if reporting_cfg is not None and hasattr(reporting_cfg, "enabled"):
        reporting_enabled = bool(reporting_cfg.enabled)
    if reporting_enabled:
        periods = int(
            getattr(reporting_cfg, "periods_per_year", cfg.training.periods_per_year)
        )
        spa_iterations = int(getattr(reporting_cfg, "spa_iterations", 500))
        spa_seed = int(getattr(reporting_cfg, "seed", cfg.training.seeds[0]))
        run_stats_workflow(
            results_root,
            paper_root,
            periods=periods,
            spa_iterations=spa_iterations,
            seed=spa_seed,
        )

    (paper_root / "manifest.json").write_text(
        json.dumps(
            {
                "device": device_info.__dict__,
                "results_root": str(results_root),
                "paper_root": str(paper_root),
                "run_id": base_run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
