from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from envs.leadlag_env import LeadLagEnv
from governance.dataset import build_manifest, record_manifest, run_quality_checks
from training.run_scenario import (
    _config_to_leadlag,
    _merge_extends,
    _read_prices,
    _set_seed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline trajectories for RL training.")
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenarios/rl_ppo.yaml"))
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("results/offline/offline_dataset.h5"))
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Optional path to a saved SB3 policy (`model.zip`). Falls back to random actions when not provided.",
    )
    return parser


def _instantiate_env(cfg: dict, prices: pd.DataFrame) -> LeadLagEnv:
    ll_cfg = _config_to_leadlag(cfg)
    rl_cfg = cfg.get("rl", {})
    return LeadLagEnv(
        price_df=prices,
        leadlag_config=ll_cfg,
        min_lookback=int(rl_cfg.get("min_lookback", 10)),
        max_lookback=int(rl_cfg.get("max_lookback", 120)),
        discrete_actions=bool(rl_cfg.get("discrete_actions", True)),
        reward_weights=rl_cfg.get("reward_weights", None),
        penalty_same=float(rl_cfg.get("penalty_same", 0.05)),
        penalty_step=int(rl_cfg.get("penalty_step", 10)),
        action_mode=rl_cfg.get("action_mode", "absolute"),
        relative_step=int(rl_cfg.get("relative_step", 5)),
        episode_length=int(rl_cfg.get("episode_length", 252)),
        random_start=rl_cfg.get("random_start", True),
        random_seed=rl_cfg.get("random_seed"),
        ema_alpha=rl_cfg.get("ema_alpha"),
    )


def _load_policy(policy_path: Optional[Path]):
    if policy_path is None:
        return None
    try:
        from stable_baselines3 import PPO  # type: ignore
    except ImportError:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required to load a PPO policy.")
    model = PPO.load(str(policy_path))
    return model


def _run_episode(env: LeadLagEnv, model) -> List[dict]:
    obs = env.reset()
    done = False
    records: List[dict] = []
    step = 0
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        records.append(
            {
                "step": step,
                "observation": np.asarray(obs, dtype=float).tolist(),
                "action": int(action) if np.isscalar(action) else np.asarray(action).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "info": info,
                "lookback": info.get("lookback"),
            }
        )
        obs = next_obs
        step += 1
    return records


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = _merge_extends(args.scenario)
    cfg['run'] = cfg.get('run', {})
    cfg['run']['seed'] = args.seed

    _set_seed(args.seed)

    prices, price_path = _read_prices(cfg)
    manifest = build_manifest(prices, source_path=price_path, extras={"quality": run_quality_checks(prices)})

    env = _instantiate_env(cfg, prices)
    policy = _load_policy(args.policy)

    trajectories: List[dict] = []
    for episode in range(args.episodes):
        episode_records = _run_episode(env, policy)
        for record in episode_records:
            record["episode"] = episode
        trajectories.extend(episode_records)

    df = pd.DataFrame(trajectories)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(output_path, key="transitions", format="table")

    manifest_path = record_manifest(manifest, output_path.parent, "data_manifest.json")
    metadata = {
        "scenario": str(args.scenario),
        "episodes": args.episodes,
        "seed": args.seed,
        "policy": str(args.policy) if args.policy else "random",
        "dataset_path": str(output_path),
        "manifest_path": str(manifest_path),
    }
    with open(output_path.parent / "offline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Recorded {len(df)} transitions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
