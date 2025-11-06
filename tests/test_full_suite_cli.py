from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from leadlag.env.trading_env import TradeMetrics
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
                "costs.fee_bps=25",
                "slippage.bps=25",
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
    assert simulation["trade_metrics"].costs > 0.0
    assert metrics_df.loc[0, "Costs"] == pytest.approx(simulation["trade_metrics"].costs)
    assert metrics_df.loc[0, "Costs"] > 0.0

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


def test_realized_metrics_follow_trading_path(tmp_path, monkeypatch):
    cfg = _compose_config(tmp_path)
    cfg.training.total_env_steps = 4
    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))

    dates = pd.date_range("2024-02-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 105, len(dates)),
            "AssetB": np.linspace(50, 53, len(dates)),
        },
        index=dates,
    )
    dataset_path = tmp_path / "dataset" / "prices.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    prices.reset_index().rename(columns={"index": "date"}).to_csv(dataset_path, index=False)

    def _fake_load_price_data(cfg_param, seed):
        return prices, dataset_path

    monkeypatch.setattr(run_full_suite, "_load_price_data", _fake_load_price_data)

    min_lb = max(5, int(cfg.window.lookback) // 2)
    max_lb = max(min_lb + 1, int(cfg.window.lookback))
    history_index = prices.index[1 : 1 + cfg.training.total_env_steps]

    base_returns = pd.Series([0.1] * cfg.training.total_env_steps, dtype=float)
    trade_metrics = TradeMetrics(
        pnl=float(base_returns.sum()),
        turnover=0.0,
        exposure=0.0,
        env_steps=len(base_returns),
        costs=0.0,
    )

    histories = []
    positive_history = pd.DataFrame(
        {
            "lookback": [float(max_lb)] * len(history_index),
            "reward": [0.0] * len(history_index),
            "delta_norm": [0.0] * len(history_index),
        },
        index=history_index,
    )
    negative_history = pd.DataFrame(
        {
            "lookback": [float(min_lb)] * len(history_index),
            "reward": [0.0] * len(history_index),
            "delta_norm": [0.0] * len(history_index),
        },
        index=history_index,
    )
    histories.extend([positive_history, negative_history])

    reward_returns = pd.Series(np.zeros(len(history_index), dtype=float))
    call_counter = {"count": 0}

    def _fake_train(cfg_param, prices_param, total_steps, seed):
        idx = call_counter["count"]
        call_counter["count"] += 1
        history = histories[idx]
        return reward_returns, trade_metrics, {"name": "patched"}, history

    monkeypatch.setattr(run_full_suite, "_train_sb3_agent", _fake_train)

    pnl_values = []
    sharpe_values = []
    for idx, run_id in enumerate(["path-pos", "path-neg"]):
        run_dir = run_full_suite._prepare_directories(Path(cfg.results_root), run_id)
        simulation = run_full_suite._simulate_episode(cfg, seed=0, window_idx=idx)
        run_full_suite._write_artifacts(run_dir, cfg, 0, idx, simulation, metrics_writer)
        metrics_df = pd.read_csv(run_dir / "metrics.csv")
        pnl_values.append(metrics_df.loc[0, "PnL"])
        sharpe_values.append(metrics_df.loc[0, "Sharpe"])

    assert pnl_values[0] > pnl_values[1]
    assert sharpe_values[0] > sharpe_values[1]


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
    run_dir = tmp_path / "results" / "test-run"
    path = run_full_suite._persist_split_manifest(
        manifest, cfg.split, output_dir, run_dirs=[run_dir]
    )

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

    csv_path = run_dir / "splits.csv"
    assert csv_path.exists()
    splits_df = pd.read_csv(csv_path)
    expected_columns = [
        "split",
        "train_indices",
        "test_indices",
        "test_start",
        "test_end",
        "embargo",
    ]
    assert splits_df.columns.tolist() == expected_columns
    assert len(splits_df) == cfg.split.n_splits
