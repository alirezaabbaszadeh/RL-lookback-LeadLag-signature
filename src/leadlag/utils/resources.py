"""Helpers for accessing package resources with source-tree fallbacks."""

from __future__ import annotations

import importlib.resources as resources
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

_FALLBACK_PACKAGE_MAP: dict[str, Path] | None = None


def _project_root() -> Path | None:
    """Return the project root (with pyproject.toml) when running from source."""

    marker = Path(__file__).resolve()
    for parent in marker.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _fallback_map() -> dict[str, Path]:
    global _FALLBACK_PACKAGE_MAP
    if _FALLBACK_PACKAGE_MAP is None:
        root = _project_root()
        mapping: dict[str, Path] = {}
        if root is not None:
            mapping = {
                "leadlag": root / "src" / "leadlag",
            }
            src_configs = root / "src" / "leadlag" / "configs"
            if src_configs.exists():
                mapping["leadlag.configs"] = src_configs
            else:
                legacy = root / "configs"
                if legacy.exists():
                    mapping["leadlag.configs"] = legacy
        _FALLBACK_PACKAGE_MAP = mapping
    return _FALLBACK_PACKAGE_MAP


def resolve_path(package: str, resource: str) -> Path | None:
    """Return a filesystem path for ``resource`` within ``package`` if possible."""

    try:
        traversable = resources.files(package).joinpath(resource)
    except (ModuleNotFoundError, AttributeError):
        traversable = None

    if traversable is not None:
        try:
            with resources.as_file(traversable) as extracted:
                return extracted
        except (FileNotFoundError, IsADirectoryError):
            if traversable.is_dir():
                return Path(str(traversable))

    fallback_root = _fallback_map().get(package)
    if fallback_root is not None:
        candidate = fallback_root / resource
        if candidate.exists():
            return candidate
    return None


def resolve_text(package: str, resource: str, *, encoding: str = "utf-8") -> str | None:
    """Return the text contents of ``resource`` or ``None`` if unavailable."""

    path = resolve_path(package, resource)
    if path is None or path.is_dir():
        return None
    return path.read_text(encoding=encoding)


@contextmanager
def open_text(package: str, resource: str, *, encoding: str = "utf-8") -> Iterator[IO[str]]:
    """Yield a text handle for the given resource with source fallback."""

    try:
        traversable = resources.files(package).joinpath(resource)
        with resources.as_file(traversable) as extracted:
            with extracted.open(encoding=encoding) as handle:
                yield handle
            return
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        pass

    path = resolve_path(package, resource)
    if path is None:
        raise FileNotFoundError(f"Unable to locate resource '{resource}' in package '{package}'")
    with path.open(encoding=encoding) as handle:
        yield handle
