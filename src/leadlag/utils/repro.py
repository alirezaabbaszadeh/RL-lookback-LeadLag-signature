"""Utility helpers for reproducible experiments."""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional during unit tests
    torch = None  # type: ignore


def set_all_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy and Torch if available."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


@dataclass
class DeviceInfo:
    """Metadata about the selected compute device."""

    device: str
    mixed_precision: str
    is_cuda: bool
    device_name: str | None


def select_device(config: Dict[str, Any]) -> DeviceInfo:
    """Select the compute device based on the hardware config."""

    requested = config.get("device", "auto")
    mixed_precision = config.get("mixed_precision", "off")

    if torch is None:
        return DeviceInfo("cpu", "off", False, None)

    if requested == "cuda":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return DeviceInfo("cuda", mixed_precision, True, device_name)
        return DeviceInfo("cpu", "off", False, None)

    if requested == "auto" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return DeviceInfo("cuda", mixed_precision, True, device_name)

    return DeviceInfo("cpu", "off", False, None)


def collect_environment_manifest() -> Dict[str, Any]:
    """Collect lightweight environment metadata for manifests."""

    manifest: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
    }
    if torch is not None:
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_device"] = torch.cuda.get_device_name(0)
    return manifest


def write_run_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a ``run_manifest.json`` file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("environment", collect_environment_manifest())
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
