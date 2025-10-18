from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from envs.leadlag_env import LeadLagEnv
from training.run_scenario import (
    _config_to_leadlag,
    _merge_extends,
    _read_prices,
    _set_seed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple offline RL policy via behavior cloning.")
    parser.add_argument("--dataset", type=Path, default=Path("results/offline/offline_dataset.h5"))
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenarios/rl_ppo.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/offline"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online-summary", type=Path, default=None, help="Optional path to online PPO summary.csv")
    return parser


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {path}")
    return pd.read_hdf(path, key="transitions")


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    obs_array = np.stack(df["observation"].apply(np.asarray))
    actions = df["action"].to_numpy(dtype=int)
    return obs_array, actions


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


class OfflinePolicy:
    def __init__(self, model: LogisticRegression):
        self.model = model

    def act(self, observation: np.ndarray) -> int:
        observation = observation.reshape(1, -1)
        return int(self.model.predict(observation)[0])


def _evaluate_policy(policy: OfflinePolicy, env: LeadLagEnv) -> dict:
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action = policy.act(np.asarray(obs))
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
    return {"total_reward": total_reward, "steps": steps}


def _load_online_summary(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = _load_dataset(args.dataset)
    X, y = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = LogisticRegression(max_iter=200, multi_class="auto")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    cfg = _merge_extends(args.scenario)
    cfg['run'] = cfg.get('run', {})
    cfg['run']['seed'] = args.seed
    _set_seed(args.seed)
    prices, _ = _read_prices(cfg)
    env = _instantiate_env(cfg, prices)
    policy = OfflinePolicy(model)
    eval_metrics = _evaluate_policy(policy, env)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results = {
        "accuracy": float(accuracy),
        "test_size": args.test_size,
        "total_reward": eval_metrics["total_reward"],
        "steps": eval_metrics["steps"],
    }
    with open(output_root / "offline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    comparison_rows = []
    online_summary = _load_online_summary(args.online_summary)
    if online_summary is not None:
        if "metric" in online_summary.columns and "mean" in online_summary.columns:
            reward_row = online_summary[online_summary["metric"] == "reward"].head(1)
            if not reward_row.empty:
                online_reward = float(reward_row["mean"].iloc[0])
                gap = online_reward - eval_metrics["total_reward"]
                comparison_rows.append(
                    {
                        "metric": "reward",
                        "online_mean": online_reward,
                        "offline_total_reward": eval_metrics["total_reward"],
                        "performance_gap": gap,
                    }
                )
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df.to_csv(output_root / "offline_vs_online.csv", index=False)

    # Always write a CSV summary for downstream tooling
    summary_df = pd.DataFrame([results])
    summary_path = output_root / "offline_results.csv"
    summary_df.to_csv(summary_path, index=False)

    global_summary_path = Path("results") / "offline_results.csv"
    global_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(global_summary_path, index=False)

    print("Offline RL training complete.")
    print(summary_df.to_string(index=False))
    if not comparison_df.empty:
        print("Comparison against online baseline:")
        print(comparison_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
