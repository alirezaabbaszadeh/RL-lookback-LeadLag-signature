"""Compatibility shim for legacy ``envs`` imports."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_leadlag_importable() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path and (src_root / "leadlag").exists():
        sys.path.insert(0, str(src_root))


try:  # Prefer the installed package if available.
    from leadlag.envs import *  # type: ignore  # noqa: F401,F403
except ModuleNotFoundError:
    _ensure_leadlag_importable()
    from leadlag.envs import *  # type: ignore  # noqa: F401,F403
