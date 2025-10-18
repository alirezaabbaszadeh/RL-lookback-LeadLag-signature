"""Factory helpers for selecting RL algorithms and policies via config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.policies import ActorCriticPolicy

    SB3_CORE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore

    class BaseAlgorithm:  # type: ignore
        ...

    class ActorCriticPolicy:  # type: ignore
        ...

    SB3_CORE_AVAILABLE = False

from training.policies import AttentionPolicy

try:  # Optional sb3-contrib integration for PPO-LSTM
    from sb3_contrib.ppo_recurrent import MlpLstmPolicy, RecurrentPPO

    SB3_CONTRIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    MlpLstmPolicy = None  # type: ignore[assignment]
    RecurrentPPO = None  # type: ignore[assignment]
    SB3_CONTRIB_AVAILABLE = False


@dataclass
class AlgorithmSpec:
    algo_cls: Type[BaseAlgorithm]
    policy: Any
    policy_kwargs: Dict[str, Any]


def _normalize_policy_name(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").lower()


def make_algorithm_spec(rl_cfg: Dict[str, Any]) -> AlgorithmSpec:
    if not SB3_CORE_AVAILABLE:
        raise ImportError("stable-baselines3 is required for RL algorithm training.")
    policy_cfg = rl_cfg.get("policy", "mlp")
    if isinstance(policy_cfg, dict):
        policy_name = policy_cfg.get("name", "mlp")
        policy_kwargs = dict(policy_cfg.get("kwargs", {}))
    else:
        policy_name = str(policy_cfg)
        policy_kwargs = dict(rl_cfg.get("policy_kwargs", {}))

    normalized = _normalize_policy_name(policy_name)

    if normalized in {"mlp", "mlp policy", "default"}:
        return AlgorithmSpec(algo_cls=PPO, policy="MlpPolicy", policy_kwargs=policy_kwargs)

    if normalized in {"mlp lstm", "ppo lstm", "recurrent"}:
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3-contrib is required for PPO-LSTM (RecurrentPPO)")
        algo_kwargs = dict(policy_kwargs)
        return AlgorithmSpec(algo_cls=RecurrentPPO, policy=MlpLstmPolicy, policy_kwargs=algo_kwargs)

    if normalized in {"attention", "attention policy"}:
        defaults = {"features_extractor_kwargs": {"features_dim": 128, "n_heads": 4}}
        merged = _merge_dicts(defaults, policy_kwargs)
        return AlgorithmSpec(algo_cls=PPO, policy=AttentionPolicy, policy_kwargs=merged)

    # Fallback: allow passing native SB3 policy string
    return AlgorithmSpec(algo_cls=PPO, policy=policy_name, policy_kwargs=policy_kwargs)


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
