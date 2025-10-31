from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ScenarioAggregate:
    """Container for precomputed aggregate statistics for a scenario."""

    name: str
    aggregate_dir: Path
    stats: pd.DataFrame
    significance: pd.DataFrame
    welch: pd.DataFrame
    runs: List[Dict[str, object]]


def discover_aggregate_dirs(root: Path) -> List[Path]:
    """Return sorted aggregate directories below ``root``."""

    dirs = [path for path in root.rglob("*_aggregate") if path.is_dir()]
    return sorted(dirs)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from ``path`` if it exists, otherwise return empty frame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def extract_seed(name: str) -> Optional[int]:
    """Extract the integer seed from a directory name."""

    parts = name.split("_seed")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].split("_")[0])
    except ValueError:
        return None


def load_runs_metadata(aggregate_dir: Path, scenario_name: str) -> List[Dict[str, object]]:
    """Load per-run metadata for the given aggregate directory."""

    runs: List[Dict[str, object]] = []
    runs_manifest = aggregate_dir / "runs.json"
    if runs_manifest.exists():
        manifest_data = json.loads(runs_manifest.read_text(encoding="utf-8"))
    else:
        manifest_data = []
        parent = aggregate_dir.parent
        seed_dirs = sorted(parent.glob(f"{scenario_name}_seed*"))
        for seed_dir in seed_dirs:
            manifest_data.append({"seed": extract_seed(seed_dir.name), "output_dir": str(seed_dir)})

    for entry in manifest_data:
        run_dir = Path(entry["output_dir"])
        meta_path = run_dir / "run_metadata.json"
        summary_path = run_dir / "summary.csv"
        metadata: Dict[str, object] = {
            "scenario": scenario_name,
            "seed": entry.get("seed"),
            "run_path": str(run_dir),
            "config_path": None,
            "data_path": None,
            "created_at": None,
            "git_commit": None,
            "git_branch": None,
            "python_version": None,
            "platform": None,
            "summary": [],
        }
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["config_path"] = meta.get("config_path")
            metadata["data_path"] = meta.get("data_price_path")
            metadata["created_at"] = meta.get("created_at")
            git_data = meta.get("git", {})
            if isinstance(git_data, dict):
                metadata["git_commit"] = git_data.get("git_commit")
                metadata["git_branch"] = git_data.get("git_branch")
            env_data = meta.get("env", {})
            if isinstance(env_data, dict):
                metadata["python_version"] = env_data.get("python_version")
                metadata["platform"] = env_data.get("platform")
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            metadata["summary"] = summary_df.to_dict(orient="records")
        runs.append(metadata)
    return runs


def build_aggregate_bundle(path: Path) -> Optional[ScenarioAggregate]:
    """Create a :class:`ScenarioAggregate` from the aggregate directory."""

    stats = load_dataframe(path / "stats.csv")
    if stats.empty:
        return None
    scenario_name = str(stats["scenario"].iloc[0])
    significance = load_dataframe(path / "significance.csv")
    welch = load_dataframe(path / "welch.csv")
    runs = load_runs_metadata(path, scenario_name)
    return ScenarioAggregate(
        name=scenario_name,
        aggregate_dir=path,
        stats=stats,
        significance=significance,
        welch=welch,
        runs=runs,
    )


def load_aggregates(root: Path) -> List[ScenarioAggregate]:
    """Load all aggregates discovered under ``root``."""

    aggregates: List[ScenarioAggregate] = []
    for aggregate_dir in discover_aggregate_dirs(root):
        bundle = build_aggregate_bundle(aggregate_dir)
        if bundle is not None:
            aggregates.append(bundle)
    return aggregates


__all__ = [
    "ScenarioAggregate",
    "discover_aggregate_dirs",
    "load_dataframe",
    "extract_seed",
    "load_runs_metadata",
    "build_aggregate_bundle",
    "load_aggregates",
]

