"""Feature caching utilities for precomputed feature stacks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import numpy as np

FeatureStack = MutableMapping[str, np.ndarray]


def _normalise_component(component: Optional[str]) -> str:
    if component is None:
        return "unknown"
    value = str(component)
    value = value.replace(os.sep, "_")
    value = value.replace(os.path.sep, "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "-", value)


@dataclass(frozen=True)
class FeatureCacheKey:
    """Unique key describing a cached feature stack.

    When introducing new feature toggles, extend this key with boolean flags so
    cache hits remain aligned with the enabled feature set. The filename embeds
    the toggle states for human readability while the digest guards against
    collisions if additional attributes are added later.
    """

    universe: Optional[str]
    timeframe: Optional[str]
    lookback: int
    signature_depth: int
    seed: int
    signature_enabled: bool
    leadlag_enabled: bool
    time_channel: bool

    def to_components(self) -> Dict[str, object]:
        return {
            "universe": self.universe,
            "timeframe": self.timeframe,
            "lookback": self.lookback,
            "signature_depth": self.signature_depth,
            "seed": self.seed,
            "signature_enabled": self.signature_enabled,
            "leadlag_enabled": self.leadlag_enabled,
            "time_channel": self.time_channel,
        }

    def filename(self) -> str:
        payload = json.dumps(self.to_components(), sort_keys=True)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        stem = "__".join(
            [
                _normalise_component(self.universe),
                _normalise_component(self.timeframe),
                str(self.lookback),
                str(self.signature_depth),
                str(self.seed),
                "signature-on" if self.signature_enabled else "signature-off",
                "leadlag-on" if self.leadlag_enabled else "leadlag-off",
                "timech-on" if self.time_channel else "timech-off",
            ]
        )
        return f"{stem}__{digest}.npz"


def load_feature_stack(cache_dir: Path, key: FeatureCacheKey) -> Optional[Dict[str, np.ndarray]]:
    path = cache_dir / key.filename()
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except Exception:
        return None


def save_feature_stack(cache_dir: Path, key: FeatureCacheKey, stack: Mapping[str, np.ndarray]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / key.filename()
    np.savez_compressed(path, **stack)
    return path


__all__ = [
    "FeatureCacheKey",
    "load_feature_stack",
    "save_feature_stack",
    "FeatureStack",
]
