"""Baseline Stable-Baselines3 stage compatible with Kaggle environments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO


def evaluate_model(model: PPO, env_id: str, episodes: int, seed: int) -> list[float]:
    """Run deterministic rollouts and collect episodic returns."""
    env = gym.make(env_id)
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        cumulative_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += float(reward)
        returns.append(cumulative_reward)
    env.close()
    return returns


def write_metadata(
    output_dir: Path,
    env_id: str,
    training_timesteps: int,
    returns: Sequence[float],
) -> None:
    """Persist key run metadata so downstream analysis can load it."""
    info = {
        "stage": "sb3_kaggle",
        "timestamp": time.time(),
        "gymnasium_version": gym.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "env_id": env_id,
        "training_timesteps": training_timesteps,
        "episodes_evaluated": len(returns),
        "returns": returns,
        "mean_return": float(sum(returns) / max(len(returns), 1)),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO agent on a lightweight environment and capture artifacts. "
            "Designed as a smoke test for the Kaggle-compatible SB3 stack."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id)
    env.reset(seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=args.seed)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(str(args.output_dir / "ppo_agent"))
    env.close()

    returns = evaluate_model(model, args.env_id, args.eval_episodes, args.seed + 1)
    write_metadata(args.output_dir, args.env_id, args.timesteps, returns)


if __name__ == "__main__":
    main()
