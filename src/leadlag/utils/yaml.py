"""YAML loading utilities for LeadLag."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Union

try:  # pragma: no cover - import guard only exercised when PyYAML is missing
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML optional dependency
    yaml = None  # type: ignore


def _copy_default(default: Any) -> Any:
    if default is None:
        return None
    return deepcopy(default)


def load_yaml(
    path: Union[str, Path],
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    """Safely load YAML content from ``path``.

    Parameters
    ----------
    path:
        File system path to the YAML file.
    required:
        When ``True`` (default), missing files or loader errors raise an exception.
        When ``False``, the function returns ``default`` instead.
    default:
        Value to return when ``required`` is ``False`` and the file cannot be
        loaded. If the file loads successfully but is empty, ``default`` is also
        returned.
    """

    resolved = Path(path)

    if yaml is None:
        if required:
            raise RuntimeError("PyYAML is required to load YAML files.")
        return _copy_default(default)

    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"YAML file not found: {resolved}")
        return _copy_default(default)

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:
        if required:
            raise
        return _copy_default(default)

    if data is None:
        return _copy_default(default)

    return data
