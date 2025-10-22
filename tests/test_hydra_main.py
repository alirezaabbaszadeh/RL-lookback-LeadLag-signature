from pathlib import Path

import pytest

from leadlag.hydra_main import (
    _load_scenario_cfg,
    get_available_scenarios,
    validate_scenario_cfg,
)

pytestmark = pytest.mark.e2e


def test_get_available_scenarios_contains_known_entries():
    names = get_available_scenarios()
    assert "fixed_30" in names
    assert "fast_smoke" in names  # preset-only


def test_load_scenario_cfg_and_validate_fixed_30():
    cfg = _load_scenario_cfg("fixed_30")
    assert isinstance(cfg, dict)
    assert cfg.get("name") == "fixed_30"
    # referenced path should exist
    path = Path(cfg["path"]).resolve()
    assert path.exists(), f"scenario path missing: {path}"
    # validate should not raise
    validate_scenario_cfg(cfg)


def test_load_scenario_cfg_and_validate_rl_ppo():
    cfg = _load_scenario_cfg("rl_ppo")
    assert isinstance(cfg, dict)
    assert cfg.get("name") == "rl_ppo"
    validate_scenario_cfg(cfg)
