from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _AttentionFeatureExtractor(BaseFeaturesExtractor):
    """Lightweight self-attention feature extractor for tabular observations."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, n_heads: int = 4):
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])
        if obs_dim <= 0:
            raise ValueError("Observation space must have positive dimension")

        self.obs_dim = obs_dim
        self.embed = nn.Linear(1, features_dim)
        self.attn = nn.MultiheadAttention(features_dim, num_heads=max(1, n_heads), batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Treat each scalar feature as a token
        tokens = observations.unsqueeze(-1)
        embedded = self.embed(tokens)
        attn_out, _ = self.attn(embedded, embedded, embedded)
        pooled = attn_out.mean(dim=1)
        return self.proj(pooled)


class AttentionPolicy(ActorCriticPolicy):
    """Actor-critic policy that routes observations through a self-attention block."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - SB3 handles runtime
        features_kwargs: Dict[str, Any] = kwargs.pop("features_extractor_kwargs", {}) or {}
        features_kwargs.setdefault("features_dim", 128)
        features_kwargs.setdefault("n_heads", 4)
        kwargs.setdefault("net_arch", [256, 256])
        kwargs.setdefault("activation_fn", nn.ReLU)
        super().__init__(
            *args,
            features_extractor_class=_AttentionFeatureExtractor,
            features_extractor_kwargs=features_kwargs,
            **kwargs,
        )
