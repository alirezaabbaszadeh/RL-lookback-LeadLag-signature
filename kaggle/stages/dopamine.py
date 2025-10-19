"""Lightweight Dopamine stage targeting Gymnasium 1.x installations."""

from __future__ import annotations

import argparse
import json
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import gymnasium as gym
from dopamine.discrete_domains import iteration_stats
from numpy import floating


def random_policy_rollout(
    env_id: str, episodes: int, seed: int, max_steps: int
) -> list[dict[str, Any]]:
    """Collect quick random-policy rollouts to validate the stack."""
    env = gym.make(env_id)
    rollouts: list[dict[str, Any]] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        steps = 0
        cumulative_reward = 0.0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += float(reward)
            steps += 1
            if steps >= max_steps:
                truncated = True
        rollouts.append(
            {
                "episode": episode,
                "return": cumulative_reward,
                "length": steps,
            }
        )
    env.close()
    return rollouts


def infer_dopamine_version() -> str:
    try:
        return metadata.version("dopamine-rl")
    except metadata.PackageNotFoundError:
        return "unknown"


def serialise_iteration_stats(stats_obj: iteration_stats.IterationStatistics) -> dict[str, list[Any]]:
    """Convert Dopamine iteration statistics to JSON-friendly types."""
    output: dict[str, list[Any]] = {}
    for key, values in stats_obj.data_lists.items():
        serialised_values: list[Any] = []
        for value in values:
            if isinstance(value, floating):
                serialised_values.append(float(value))
            else:
                serialised_values.append(value)
        output[key] = serialised_values
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check Dopamine + Gymnasium 1.x installation via random rollouts."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = random_policy_rollout(
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    stats = iteration_stats.IterationStatistics()
    for item in rollouts:
        stats.append(
            {
                "episode": item["episode"],
                "return": item["return"],
                "length": item["length"],
            }
        )

    metadata_payload = {
        "stage": "dopamine",
        "timestamp": time.time(),
        "gymnasium_version": gym.__version__,
        "dopamine_version": infer_dopamine_version(),
        "env_id": args.env_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "returns": [item["return"] for item in rollouts],
        "mean_return": float(sum(item["return"] for item in rollouts) / max(len(rollouts), 1)),
    }

    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "iteration_stats.json").write_text(
        json.dumps(serialise_iteration_stats(stats), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
