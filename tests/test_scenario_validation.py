from __future__ import annotations

import pytest

try:  # pragma: no cover - environment guard
    import pandas  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"pandas import failed: {exc}", allow_module_level=True)

from leadlag.training.run_scenario import _validate_scenario_schema


def test_validate_scenario_schema_rejects_missing_sections():
    cfg = {
        "run": {"run_name": "broken"},
        "analysis": {"method": "signature", "lookback": 30},
    }
    with pytest.raises(ValueError, match="missing sections: data"):
        _validate_scenario_schema(cfg, scenario="broken")


def test_validate_scenario_schema_rejects_invalid_types():
    cfg = {
        "run": {},
        "data": {"price_csv": 123},
        "analysis": {"method": "", "lookback": -1},
    }
    with pytest.raises(ValueError):
        _validate_scenario_schema(cfg, scenario="invalid")


def test_validate_scenario_schema_accepts_minimal_valid_config():
    cfg = {
        "run": {"run_name": "valid"},
        "data": {"price_csv": "path/to/data.csv"},
        "analysis": {"method": "signature", "lookback": 30},
    }
    _validate_scenario_schema(cfg, scenario="valid")
