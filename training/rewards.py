"""Reward template utilities for LeadLag reinforcement learning scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - yaml optional in prod
    yaml = None


RewardDict = Dict[str, float]


@dataclass
class RewardContext:
    lookback: int
    prev_lookback: Optional[int]
    min_lookback: int
    max_lookback: int


@dataclass
class RewardTemplate:
    weights: RewardDict
    params: Dict[str, Any]
    name: str = "custom"

    def compute(
        self,
        metrics: Mapping[str, float],
        extras: Mapping[str, Mapping[str, float]],
        context: RewardContext,
    ) -> tuple[float, RewardDict]:
        components = compute_components(metrics, extras, context, self.params)
        reward = 0.0
        for key, value in components.items():
            weight = self.weights.get(key, 0.0)
            reward += weight * value
        return reward, components


def load_reward_template(template: Optional[Any]) -> Optional[RewardTemplate]:
    """Load a template from dict or name. Returns None if template is falsy."""
    if template in (None, False):
        return None
    if isinstance(template, RewardTemplate):
        return template
    if isinstance(template, Mapping):
        return _template_from_dict(template, name="inline")
    if isinstance(template, str):
        path = _resolve_template_path(template)
        data = _load_yaml(path)
        if not isinstance(data, Mapping):
            raise ValueError(f"Reward template at {path} must be a mapping")
        name = data.get("name", Path(path).stem)
        return _template_from_dict(data, name=name)
    raise TypeError("template must be None, dict, RewardTemplate, or str path/name")


def compute_components(
    metrics: Mapping[str, float],
    extras: Mapping[str, Mapping[str, float]],
    context: RewardContext,
    params: Optional[Dict[str, Any]] = None,
) -> RewardDict:
    params = params or {}
    components: RewardDict = {}
    components["S"] = _component_strength(metrics)
    components["C"] = _component_consistency(metrics)
    components["E"] = _component_edge_penalty(metrics, context, params.get("edge", {}))
    components["Sharpe"] = _component_sharpe(extras.get("regime") or {}, params.get("sharpe", {}))
    components["LookbackPenalty"] = _component_lookback_penalty(context, params.get("lookback", {}))
    return components


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _component_strength(metrics: Mapping[str, float]) -> float:
    mean_abs = float(metrics.get("mean_abs", 0.0))
    row_range = float(metrics.get("row_range", 0.0))
    return 0.5 * mean_abs + 0.5 * row_range


def _component_consistency(metrics: Mapping[str, float]) -> float:
    stab_matrix = float(metrics.get("stability_matrix", 0.0))
    stab_rows = float(metrics.get("stability_rows", 0.0))
    return 0.5 * stab_matrix + 0.5 * stab_rows


def _component_edge_penalty(
    metrics: Mapping[str, float],
    context: RewardContext,
    params: Mapping[str, Any],
) -> float:
    edge_penalty = 0.0
    # penalty for hitting min/max
    if context.lookback in (context.min_lookback, context.max_lookback):
        edge_penalty += params.get("edge_boundary", 1.0)
    prev = context.prev_lookback
    if prev is not None:
        jump = abs(context.lookback - prev)
        threshold = max(1, int(params.get("jump_threshold", 10)))
        if jump > threshold:
            edge_penalty += jump / threshold
        if jump == 0:
            edge_penalty += params.get("penalty_same", 0.0)
    # incorporate volatility of signal to discourage high swings
    delta_mean = float(metrics.get("delta_mean", 0.0))
    edge_penalty += abs(delta_mean) * params.get("delta_weight", 0.0)
    return edge_penalty


def _component_sharpe(regime: Mapping[str, float], params: Mapping[str, Any]) -> float:
    mean = float(regime.get("returns_mean", 0.0))
    vol = float(regime.get("returns_vol", 0.0))
    min_vol = float(params.get("min_vol", 1e-6))
    denom = max(abs(vol), min_vol)
    return mean / denom


def _component_lookback_penalty(
    context: RewardContext,
    params: Mapping[str, Any],
) -> float:
    min_lb = context.min_lookback
    max_lb = context.max_lookback
    if max_lb <= min_lb:
        return 0.0
    target = params.get("target")
    if target is None:
        target = min_lb + 0.5 * (max_lb - min_lb)
    width = float(params.get("width", 0.5))
    width = max(width, 1e-6)
    normalized = abs(context.lookback - target) / (max_lb - min_lb)
    return normalized / width


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _template_from_dict(data: Mapping[str, Any], name: str) -> RewardTemplate:
    components = data.get("components", {})
    if not isinstance(components, Mapping):
        raise ValueError("Reward template must include a 'components' mapping")
    weights: RewardDict = {str(k): float(v) for k, v in components.items()}
    params = data.get("params", {})
    if params and not isinstance(params, Mapping):
        raise ValueError("Reward template 'params' must be a mapping")
    return RewardTemplate(weights=weights, params=dict(params or {}), name=name)


def _resolve_template_path(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    # Search under configs/rewards
    candidate = Path("configs") / "rewards" / f"{name}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Reward template '{name}' not found (looked for {path} and {candidate})")


def _load_yaml(path: Path) -> Any:
    if yaml is None:
        raise ImportError("PyYAML is required to load reward templates")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
