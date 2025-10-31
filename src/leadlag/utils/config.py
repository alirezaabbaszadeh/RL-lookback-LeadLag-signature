"""Configuration-related helpers."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

__all__ = ["deep_update"]


def deep_update(
    base: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge ``overrides`` into ``base``.

    Nested mappings are merged while preserving other types (lists, scalars, etc.)
    by replacing the corresponding value in ``base``. The ``base`` mapping is
    updated in-place and returned to facilitate chaining.
    """

    for key, value in overrides.items():
        base_value = base.get(key)
        if isinstance(value, Mapping) and isinstance(base_value, MutableMapping):
            base[key] = deep_update(base_value, value)
        else:
            base[key] = value
    return base
