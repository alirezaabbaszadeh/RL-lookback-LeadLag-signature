"""Utility helpers for reproducible experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from importlib import metadata
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
    git_commit = _git_commit()
    if git_commit:
        manifest["git_commit"] = git_commit
        manifest["git_dirty"] = _git_is_dirty()
        git_branch = _git_branch()
        if git_branch:
            manifest["git_branch"] = git_branch

    if torch is not None:
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_device"] = torch.cuda.get_device_name(0)

    packages = _package_versions(
        [
            "numpy",
            "pandas",
            "scipy",
            "torch",
            "stable-baselines3",
            "gymnasium",
            "hydra-core",
            "statsmodels",
        ]
    )
    if packages:
        manifest["packages"] = packages
    return manifest


def write_run_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a ``run_manifest.json`` file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("environment", collect_environment_manifest())
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def collect_determinism_settings(seed: int) -> Dict[str, Any]:
    """Capture deterministic settings applied for a run."""

    info: Dict[str, Any] = {"seed": int(seed)}
    if torch is not None:
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            info["torch_cudnn_deterministic"] = bool(getattr(cudnn, "deterministic", False))
            info["torch_cudnn_benchmark"] = bool(getattr(cudnn, "benchmark", False))
        info["torch_seeded"] = True
    else:
        info["torch_seeded"] = False
    return info


def update_run_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Merge payload data into an existing ``run_manifest.json``."""

    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - guard against manual edits
            existing = {}

    merged = _deep_merge(existing, payload)
    write_run_manifest(path, merged)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover - git optional
        return None


def _git_is_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(result.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover - git optional
        return False


def _git_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


def _package_versions(packages: list[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional deps
            continue
    return versions


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base
