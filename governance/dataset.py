from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def hash_file(path: Path) -> Optional[str]:
    """Return the SHA-256 hash for `path`, or None if the file is missing."""
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    prices: pd.DataFrame,
    *,
    source_path: Optional[Path] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a manifest describing the dataset that powered a run."""
    manifest: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "row_count": int(len(prices)),
        "asset_count": int(prices.shape[1]),
        "columns": list(map(str, prices.columns)),
        "index_timezone": str(getattr(prices.index, "tz", None)),
    }

    if source_path is not None:
        manifest["source"] = str(source_path)
        file_hash = hash_file(source_path)
        if file_hash is not None:
            manifest["source_hash_sha256"] = file_hash
        manifest["source_exists"] = source_path.exists()
        manifest["source_size_bytes"] = source_path.stat().st_size if source_path.exists() else None

    if not prices.empty:
        manifest["start"] = prices.index.min().isoformat()
        manifest["end"] = prices.index.max().isoformat()
        manifest["inferred_frequency"] = getattr(prices.index, "inferred_freq", None)
        manifest["missing_values"] = int(prices.isna().sum().sum())
    else:
        manifest["start"] = None
        manifest["end"] = None
        manifest["inferred_frequency"] = None
        manifest["missing_values"] = 0

    if extras:
        manifest.update(extras)
    return manifest


def run_quality_checks(prices: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate dataset quality and emit structured findings."""
    findings: Dict[str, Any] = {}

    duplicate_index = bool(prices.index.duplicated().any())
    findings["duplicate_index"] = duplicate_index

    missing_ratio = float(prices.isna().sum().sum() / max(1, prices.size))
    findings["missing_ratio"] = missing_ratio

    zero_variance_assets: List[str] = []
    for col in prices.columns:
        series = prices[col].dropna()
        if series.empty:
            continue
        if float(series.std(ddof=0)) == 0.0:
            zero_variance_assets.append(str(col))
    findings["zero_variance_assets"] = zero_variance_assets

    if hasattr(prices.index, "is_monotonic_increasing"):
        findings["monotonic_index"] = bool(prices.index.is_monotonic_increasing)
    else:
        findings["monotonic_index"] = True

    return findings


def record_manifest(manifest: Dict[str, Any], out_dir: Path, filename: str = "data_manifest.json") -> Path:
    """Persist the manifest to `out_dir/filename`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / filename
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
