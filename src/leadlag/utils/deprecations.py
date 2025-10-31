from __future__ import annotations

import warnings
from typing import Set

_EMITTED_KEYS: Set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Emit a deprecation warning once per process for the given key.

    Key should be a stable identifier for the deprecated feature (e.g., flag name
    or entrypoint). Subsequent calls with the same key will be ignored.
    """
    if key in _EMITTED_KEYS:
        return
    _EMITTED_KEYS.add(key)
    warnings.warn(message, category=DeprecationWarning, stacklevel=2)


def warn_flag_deprecated(flag: str, *, replacement: str, remove_in: str) -> None:
    """Standardized deprecation notice for a CLI flag.

    Example:
        warn_flag_deprecated("--json", replacement="--format json", remove_in="0.2.0")
    """
    msg = (
        f"Flag '{flag}' is deprecated and will be removed in {remove_in}. "
        f"Use '{replacement}' instead."
    )
    warn_once(f"flag:{flag}", msg)


def warn_entrypoint_deprecated(entry: str, *, replacement: str, remove_in: str) -> None:
    """Standardized deprecation notice for a legacy entrypoint/module."""
    msg = (
        f"Entrypoint '{entry}' is deprecated and will be removed in {remove_in}. "
        f"Use '{replacement}' instead."
    )
    warn_once(f"entry:{entry}", msg)

