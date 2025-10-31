"""Helpers for loading and validating scenario configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from leadlag.utils.config import deep_update
from leadlag.utils.yaml import load_yaml


def _merge_extends(cfg_path: Path) -> Dict[str, Any]:
    """Load *cfg_path* and merge the optional ``extends`` reference."""

    cfg = load_yaml(cfg_path)
    extends = cfg.get("extends")
    if extends:
        base_path = (cfg_path.parent / extends).resolve()
        base = load_yaml(base_path)
        merged = deep_update(base, {k: v for k, v in cfg.items() if k != "extends"})
        return merged
    return cfg


def _validate_scenario_schema(cfg: Dict[str, Any], *, scenario: str) -> None:
    """Ensure the merged scenario config contains required sections."""

    required_sections = ("run", "data", "analysis")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Scenario '{scenario}' missing sections: {', '.join(missing)}")

    run_section = cfg["run"]
    if not isinstance(run_section, dict):
        raise TypeError(f"Scenario '{scenario}' section 'run' must be a mapping")

    data_section = cfg["data"]
    if not isinstance(data_section, dict):
        raise TypeError(f"Scenario '{scenario}' section 'data' must be a mapping")

    price_csv = data_section.get("price_csv")
    if not isinstance(price_csv, str) or not price_csv:
        raise ValueError(f"Scenario '{scenario}' must define data.price_csv as a string path")

    analysis_section = cfg["analysis"]
    if not isinstance(analysis_section, dict):
        raise TypeError(f"Scenario '{scenario}' section 'analysis' must be a mapping")

    method = analysis_section.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"Scenario '{scenario}' must define analysis.method as a string")

    lookback = analysis_section.get("lookback")
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError(
            f"Scenario '{scenario}' must define analysis.lookback as a positive integer"
        )

    metrics_cfg = cfg.get("metrics")
    if metrics_cfg is not None and not isinstance(metrics_cfg, dict):
        raise TypeError(
            f"Scenario '{scenario}' section 'metrics' must be a mapping when provided"
        )


# Provide public aliases for callers that prefer non-private names.
merge_extends = _merge_extends
validate_scenario_schema = _validate_scenario_schema

