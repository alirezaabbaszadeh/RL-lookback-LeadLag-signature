from pathlib import Path

import pytest

pytest.importorskip("hydra")
from importlib import resources

from hydra import compose, initialize, initialize_config_module
from omegaconf import OmegaConf

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


def test_hydra_default_config_composes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    with initialize(version_base=None, config_path="../src/leadlag/configs"):
        cfg = compose(config_name="config")

    assert OmegaConf.select(cfg, "scenario.name") == "fixed_30"

    with initialize(version_base=None, config_path="../src/leadlag/configs"):
        cfg_override = compose(
            config_name="config",
            overrides=["scenario=fixed_90", "multi_seed.enabled=false"],
        )

    assert OmegaConf.select(cfg_override, "scenario.name") == "fixed_90"
    assert OmegaConf.select(cfg_override, "multi_seed.enabled") is False


def test_packaged_configs_are_canonical():
    repo_root = Path(__file__).resolve().parents[1]
    packaged = repo_root / "src" / "leadlag" / "configs"
    legacy_root = repo_root / "configs"

    tracked_files = [
        "config.yaml",
        "default.yaml",
        "base.yaml",
        "features/signature.yaml",
    ]

    assert not legacy_root.exists(), "legacy configs/ directory should be removed"

    for relative in tracked_files:
        packaged_path = packaged / relative
        assert packaged_path.exists(), f"missing packaged config: {relative}"

    packaged_scenarios = sorted(p.name for p in (packaged / "scenario").glob("*.yaml"))
    packaged_scenarios += sorted(p.name for p in (packaged / "scenarios").glob("*.yaml"))
    assert "rl_ppo.yaml" in packaged_scenarios


def test_packaged_configs_resolve_via_module_api():
    cfg_root = resources.files("leadlag").joinpath("configs")
    assert cfg_root.joinpath("config.yaml").is_file()

    with initialize_config_module(version_base=None, config_module="leadlag.configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "agent=ppo",
                "training=smoke",
                "hardware=gpu",
                "data=sp500_sector",
                "split=walk_forward_purged",
            ],
        )

    assert OmegaConf.select(cfg, "agent.policy") == "MlpPolicy"
    assert OmegaConf.select(cfg, "training.total_env_steps") > 0
