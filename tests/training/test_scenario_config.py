from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.training.scenario_config import _merge_extends, _validate_scenario_schema


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_merge_extends_loads_base_configuration(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    _write_yaml(
        base,
        """
run:
  run_name: base
analysis:
  method: signature
  lookback: 15
data:
  price_csv: base.csv
""".strip(),
    )

    _write_yaml(
        child,
        """
extends: base.yaml
analysis:
  lookback: 30
""".strip(),
    )

    merged = _merge_extends(child)
    assert merged["run"]["run_name"] == "base"
    assert merged["analysis"]["lookback"] == 30
    assert merged["analysis"]["method"] == "signature"


def test_validate_scenario_schema_requires_expected_sections() -> None:
    cfg = {
        "run": {"run_name": "example"},
        "data": {"price_csv": "prices.csv"},
        "analysis": {"method": "signature", "lookback": 25},
    }
    _validate_scenario_schema(cfg, scenario="example")

    cfg.pop("analysis")
    with pytest.raises(ValueError):
        _validate_scenario_schema(cfg, scenario="broken")
