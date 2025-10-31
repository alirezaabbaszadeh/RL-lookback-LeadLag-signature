"""Compatibility layer exposing lead-lag models via the legacy namespace."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_leadlag_importable() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path and (src_root / "leadlag").exists():
        sys.path.insert(0, str(src_root))


try:
    from leadlag import models as _package_models  # type: ignore
except ModuleNotFoundError:
    _ensure_leadlag_importable()
    from leadlag import models as _package_models  # type: ignore

globals().update({name: getattr(_package_models, name) for name in dir(_package_models)})
__all__ = getattr(_package_models, "__all__", [])
