"""Hydra-driven pipeline entry point for the lead-lag project."""

from __future__ import annotations

import json
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
from leadlag.features.leadlag import compute_lead_lag
from leadlag.features.signature import compute_signature_features
from leadlag.governance import dataset as dataset_mod
from leadlag.eval import stats as stats_mod
from leadlag.models.config import LeadLagConfig, SIGNATURE_AVAILABLE
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row
from leadlag.utils import select_device, set_all_seeds, write_run_manifest



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


def _build_feature_stack(prices: pd.DataFrame, features_cfg: DictConfig) -> Dict[str, np.ndarray]:
    stack: Dict[str, np.ndarray] = {}
    returns = prices.pct_change().dropna()
    stack["returns"] = returns.to_numpy(dtype=float)

    signature_cfg = features_cfg.get("signature") if features_cfg else None
    if signature_cfg and signature_cfg.get("enabled"):
        depth = int(signature_cfg.get("depth", 2))
        flattened = returns.to_numpy(dtype=float).ravel()
        stack["signature"] = compute_signature_features(flattened, depth)

    leadlag_cfg = features_cfg.get("leadlag") if features_cfg else None
    if leadlag_cfg and leadlag_cfg.get("enabled"):
        reference_series = returns.mean(axis=1).to_numpy(dtype=float)
        stack["leadlag"] = compute_lead_lag(reference_series)

    if features_cfg.get("time_channel"):
        stack["time_channel"] = np.linspace(0.0, 1.0, num=len(returns), dtype=float)

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


def _compute_trade_metrics(returns: pd.Series) -> TradeMetrics:
    arr = returns.to_numpy(dtype=float)
    turnover = float(np.mean(np.abs(arr))) if arr.size else 0.0
    cumulative = (1.0 + arr).cumprod()
    exposure = float(np.mean(np.abs(cumulative - 1.0))) if cumulative.size else 0.0
    pnl = float(arr.sum())
    env_steps = int(arr.size)
    return TradeMetrics(pnl=pnl, turnover=turnover, exposure=exposure, env_steps=env_steps)


def _random_rollout(env: LeadLagEnv, total_steps: int, seed: int) -> pd.Series:
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
    if "reward" in history.columns:
        returns = history["reward"].astype(float)
    else:
        index = pd.RangeIndex(len(rewards))
        returns = pd.Series(rewards, index=index, dtype=float)
    return returns


def _train_sb3_agent(
    cfg: DictConfig,
    prices: pd.DataFrame,
    total_steps: int,
    *,
    seed: int,
) -> tuple[pd.Series, Dict[str, object]]:
    if not SB3_AVAILABLE or cfg.agent.library != "sb3":
        env = _make_leadlag_env(cfg, prices, seed=seed)
        returns = _random_rollout(env, total_steps, seed)
        return returns, {"name": "random", "reason": "sb3_unavailable"}

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
        returns = _random_rollout(env, total_steps, seed)
        return returns, {"name": "random", "reason": f"unsupported_agent:{algo_name}"}

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
    return returns, {"name": algo_name, "library": cfg.agent.library, "trained_steps": steps}


def _simulate_episode(
    cfg: DictConfig,
    seed: int,
    window_idx: int,
) -> Dict[str, object]:
    training_cfg = cfg.training
    total_steps = max(1, int(training_cfg.total_env_steps))
    effective_seed = seed * 10_000 + window_idx

    prices, dataset_path = _load_price_data(cfg, effective_seed)
    feature_stack = _build_feature_stack(prices, cfg.get("features") or {})

    returns, agent_meta = _train_sb3_agent(cfg, prices, total_steps, seed=effective_seed)
    if returns.empty:
        returns = pd.Series([0.0], dtype=float)

    summary = stats_mod.summarize_performance(
        returns,
        periods_per_year=training_cfg.periods_per_year,
    )
    equity = stats_mod.compute_equity_curve(summary.returns)
    trade_metrics = _compute_trade_metrics(summary.returns)
    actual_env_steps = trade_metrics.env_steps

    return {
        "returns": summary.returns,
        "equity": equity,
        "summary": summary,
        "trade_metrics": trade_metrics,
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
    write_run_manifest(run_dir / "run_manifest.json", manifest_payload)


def _materialize_walk_forward(cfg: DictConfig, total_samples: int) -> List[Dict[str, Iterable[int]]]:
    splits = []
    for idx, split in enumerate(
        walk_forward_purged(
            total_samples=total_samples,
            n_splits=cfg.split.n_splits,
            embargo_frac=cfg.split.embargo_frac,
        )
    ):
        splits.append(
            {
                "split": idx,
                "train": split.train_indices.tolist(),
                "test": split.test_indices.tolist(),
            }
        )
    return splits


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
    device_info = select_device(dict(cfg.hardware))
    set_all_seeds(int(cfg.training.seeds[0]))

    results_root = Path(cfg.results_root).resolve()
    paper_root = Path(cfg.paper_outputs_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))
    base_run_id = cfg.logging.run_id

    dataset_length: Optional[int] = None
    for window_idx in range(cfg.training.windows):
        for seed in cfg.training.seeds:
            set_all_seeds(int(seed))
            run_id = _format_run_id(base_run_id, int(seed), window_idx, bool(cfg.logging.append_seed_window))
            run_dir = _prepare_directories(results_root, run_id)
            simulation = _simulate_episode(cfg, int(seed), window_idx)
            _write_artifacts(run_dir, cfg, int(seed), window_idx, simulation, metrics_writer)
            dataset_length = simulation.get("dataset_length", dataset_length)

    total_samples = int(dataset_length) if dataset_length else int(cfg.training.total_env_steps)
    split_payload = _materialize_walk_forward(cfg, total_samples=total_samples)
    with (paper_root / "data_splits.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)

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
