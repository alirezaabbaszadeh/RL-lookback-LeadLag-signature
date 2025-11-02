from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from leadlag.eval import stats as stats_mod
from leadlag.pipelines import run_full_suite
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row


def _write_mock_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    data = {
        "date": dates,
        "AssetA": 100 + pd.Series(range(len(dates))) * 0.1,
        "AssetB": 120 + pd.Series(range(len(dates))) * 0.05,
        "AssetC": 90 + pd.Series(range(len(dates))) * 0.08,
    }
    df = pd.DataFrame(data)
    df.to_csv(path / "prices.csv", index=False)


def _compose_config(tmp_path: Path):
    config_dir = Path(__file__).resolve().parents[1] / "src" / "leadlag" / "configs"
    _write_mock_dataset(tmp_path / "dataset")
    with initialize_config_dir(
        config_dir=str(config_dir), job_name="test-suite", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"results_root={tmp_path / 'results'}",
                f"paper_outputs_root={tmp_path / 'paper'}",
                "logging.run_id=test",
                "training.seeds=[0]",
                "training.windows=1",
                "hardware.n_envs=1",
                "training.total_env_steps=100",
                f"data.dataset_dir={tmp_path / 'dataset'}",
            ],
        )
    return cfg


def test_simulate_episode_produces_metrics(tmp_path):
    cfg = _compose_config(tmp_path)
    run_dir = run_full_suite._prepare_directories(Path(cfg.results_root), "test-s00-w00")
    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))

    seed = 0
    simulation = run_full_suite._simulate_episode(cfg, seed=seed, window_idx=0)
    run_full_suite._write_artifacts(run_dir, cfg, seed, 0, simulation, metrics_writer)

    metrics_path = run_dir / "metrics.csv"
    equity_path = run_dir / "equity.csv"
    returns_path = run_dir / "returns.csv"
    data_manifest_path = run_dir / "data_manifest.json"
    run_manifest_path = run_dir / "run_manifest.json"
    assert metrics_path.exists()
    assert equity_path.exists()
    assert returns_path.exists()
    assert data_manifest_path.exists()
    assert run_manifest_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    assert not metrics_df.empty
    assert "Sharpe" in metrics_df.columns
    assert "EnvSteps" in metrics_df.columns
    assert metrics_df.loc[0, "EnvSteps"] == simulation["env_steps"]

    equity_df = pd.read_csv(equity_path)
    returns_df = pd.read_csv(returns_path)
    assert len(equity_df) == len(returns_df)

    manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["config"]["data"]["dataset_dir"] == str(tmp_path / "dataset")
    environment = manifest_payload.get("environment", {})
    assert environment.get("python")
    assert "git_commit" in environment
    assert isinstance(environment.get("packages", {}), dict)
    presets = manifest_payload.get("presets", {})
    assert "training" in presets
    assert "hardware" in presets
    determinism = manifest_payload.get("determinism", {})
    assert determinism.get("seed") == seed
    assert manifest_payload.get("agent")
    feature_meta = manifest_payload.get("feature_stack", {})
    assert "returns" in feature_meta

    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    assert data_manifest.get("dataset_dir") == str(tmp_path / "dataset")
    training_meta = data_manifest.get("training", {})
    assert training_meta.get("total_env_steps") == cfg.training.total_env_steps
    assert data_manifest.get("row_count")


def test_build_metadata_row_matches_config(tmp_path):
    cfg = _compose_config(tmp_path)
    metrics = {"sharpe": 1.0, "sortino": 1.2, "max_drawdown": -0.1, "pnl": 0.05}
    env_steps = int(cfg.training.total_env_steps)
    row = build_metadata_row(
        "test-run",
        OmegaConf.to_container(cfg, resolve=True),
        metrics,
        seed=0,
        window_idx=0,
        turnover=0.0,
        exposure=0.0,
        costs=0.0,
        env_steps=env_steps,
    )
    assert row["experiment_id"] == "test-run"
    assert row["agent"] == cfg.agent.name
    assert row["Sharpe"] == metrics["sharpe"]


def test_hac_confidence_interval_returns_bounds(tmp_path):
    cfg = _compose_config(tmp_path)
    simulation = run_full_suite._simulate_episode(cfg, seed=0, window_idx=0)
    lower, upper = stats_mod.hac_confidence_interval(
        simulation["returns"], periods_per_year=cfg.training.periods_per_year
    )
    assert lower <= upper


def test_feature_toggle_signature_leadlag(tmp_path):
    cfg = _compose_config(tmp_path)
    cfg.features = OmegaConf.merge(cfg.features, OmegaConf.create({"name": "signature_leadlag"}))
    cfg.features.signature.enabled = True
    cfg.features.signature.depth = 3
    cfg.features.leadlag.enabled = True
    cfg.features.time_channel = True

    simulation = run_full_suite._simulate_episode(cfg, seed=0, window_idx=0)
    feature_stack = simulation["feature_stack"]
    assert "signature" in feature_stack
    assert feature_stack["signature"].shape[0] == cfg.features.signature.depth
    assert "leadlag" in feature_stack
    assert feature_stack["leadlag"].shape[0] == 2
    assert "time_channel" in feature_stack


def test_walk_forward_manifest_persistence(tmp_path):
    cfg = _compose_config(tmp_path)
    total_samples = 32
    manifest = run_full_suite._materialize_walk_forward(cfg, total_samples)
    output_dir = tmp_path / "paper_outputs"
    path = run_full_suite._persist_split_manifest(manifest, cfg.split, output_dir)

    assert path is not None
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scheme"] == cfg.split.scheme
    assert payload["parameters"]["n_splits"] == cfg.split.n_splits
    assert np.isclose(payload["parameters"]["embargo_frac"], cfg.split.embargo_frac)

    for split in payload["splits"]:
        assert set(split["train"]).isdisjoint(split["test"])
        if split["test"]:
            embargo = int(np.ceil(len(split["test"]) * payload["parameters"]["embargo_frac"]))
            assert split["embargo"] == embargo
