import os
import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path for module imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_main import (
    get_available_scenarios,
    _load_scenario_cfg,
    validate_scenario_cfg,
)


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
