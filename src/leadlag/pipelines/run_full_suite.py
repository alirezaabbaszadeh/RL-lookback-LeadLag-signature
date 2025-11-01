"""Hydra-driven pipeline entry point for the lead-lag project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from leadlag.cv.purged import walk_forward_purged
from leadlag.env.trading_env import SyntheticTradingEnvironment
from leadlag.governance import dataset as dataset_mod
from leadlag.eval import stats as stats_mod
from leadlag.reporting.metrics_writer import MetricsWriter, build_metadata_row
from leadlag.utils import select_device, set_all_seeds, write_run_manifest



def _format_run_id(base: str, seed: int, window_idx: int, append_suffix: bool) -> str:
    if append_suffix:
        return f"{base}-s{seed:02d}-w{window_idx:02d}"
    return base


def _prepare_directories(results_root: Path, run_id: str) -> Path:
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _simulate_episode(
    cfg: DictConfig,
    seed: int,
    window_idx: int,
) -> Dict[str, object]:
    training_cfg = cfg.training
    costs_cfg = cfg.costs
    slippage_cfg = cfg.slippage

    env = SyntheticTradingEnvironment(
        lookback=cfg.window.lookback,
        horizon=cfg.target.horizon,
        fee_bps=costs_cfg.fee_bps,
        slippage_bps=slippage_cfg.bps,
        seed=seed * 1_000 + window_idx,
    )
    total_steps = max(1, int(training_cfg.total_env_steps))
    n_envs = max(1, int(cfg.hardware.n_envs))
    returns = env.simulate_returns(total_steps, n_envs=n_envs)
    summary = stats_mod.summarize_performance(
        returns,
        periods_per_year=training_cfg.periods_per_year,
    )
    equity = stats_mod.compute_equity_curve(summary.returns)
    trade_metrics = env.summarize_trades(summary.returns)
    actual_env_steps = trade_metrics.env_steps
    return {
        "returns": summary.returns,
        "equity": equity,
        "summary": summary,
        "trade_metrics": trade_metrics,
        "env_steps": actual_env_steps,
        "requested_env_steps": total_steps,
        "n_envs": n_envs,
    }


def _write_artifacts(
    run_dir: Path,
    cfg: DictConfig,
    seed: int,
    window_idx: int,
    simulation: Dict[str, object],
    metrics_writer: MetricsWriter,
) -> None:
    run_id = run_dir.name
    returns_series = simulation["returns"]
    equity_series = simulation["equity"]
    summary = simulation["summary"]
    trade_metrics = simulation["trade_metrics"]
    env_steps = int(simulation.get("env_steps", len(returns_series)))

    stats_mod.export_returns(run_dir / "returns.csv", returns_series)
    stats_mod.export_equity(run_dir / "equity.csv", equity_series)

    metrics_row = build_metadata_row(
        run_id,
        OmegaConf.to_container(cfg, resolve=True),
        {
            "sharpe": summary.sharpe,
            "sortino": summary.sortino,
            "max_drawdown": summary.max_drawdown,
            "pnl": summary.pnl,
        },
        seed=seed,
        window_idx=window_idx,
        turnover=trade_metrics.turnover,
        exposure=trade_metrics.exposure,
        env_steps=env_steps,
    )
    metrics_writer.write_row(run_dir / "metrics.csv", metrics_row)

    _write_data_manifest(run_dir, cfg)

    manifest_payload = {
        "run_id": run_id,
        "seed": seed,
        "window_index": window_idx,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics_row,
        "requested_env_steps": int(simulation.get("requested_env_steps", env_steps)),
        "actual_env_steps": env_steps,
        "vectorised_envs": int(simulation.get("n_envs", 1)),
    }
    write_run_manifest(run_dir / "run_manifest.json", manifest_payload)


def _materialize_walk_forward(cfg: DictConfig, total_samples: int) -> List[Dict[str, Iterable[int]]]:
    splits = []
    for idx, split in enumerate(
        walk_forward_purged(
            total_samples=total_samples,
            n_splits=cfg.split.n_splits,
            embargo_frac=cfg.split.embargo_frac,
        )
    ):
        splits.append(
            {
                "split": idx,
                "train": split.train_indices.tolist(),
                "test": split.test_indices.tolist(),
            }
        )
    return splits


def _write_data_manifest(run_dir: Path, cfg: DictConfig) -> None:
    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    training_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True)
    split_cfg = OmegaConf.to_container(cfg.get("split", {}), resolve=True)

    extras = {
        "universe": data_cfg.get("universe"),
        "timeframe": data_cfg.get("timeframe"),
        "market": data_cfg.get("market"),
        "training": {
            "total_env_steps": training_cfg.get("total_env_steps"),
            "seeds": training_cfg.get("seeds"),
            "windows": training_cfg.get("windows"),
        },
        "split": split_cfg,
    }

    extras = {key: value for key, value in extras.items() if value is not None}

    manifest = dataset_mod.build_manifest(pd.DataFrame(), extras=extras)
    dataset_dir = data_cfg.get("dataset_dir")
    if dataset_dir:
        manifest["dataset_dir"] = str(dataset_dir)

    dataset_mod.record_manifest(manifest, run_dir)


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device_info = select_device(dict(cfg.hardware))
    set_all_seeds(int(cfg.training.seeds[0]))

    results_root = Path(cfg.results_root).resolve()
    paper_root = Path(cfg.paper_outputs_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    paper_root.mkdir(parents=True, exist_ok=True)

    metrics_writer = MetricsWriter(OmegaConf.to_container(cfg, resolve=True))
    base_run_id = cfg.logging.run_id

    for window_idx in range(cfg.training.windows):
        for seed in cfg.training.seeds:
            set_all_seeds(int(seed))
            run_id = _format_run_id(base_run_id, int(seed), window_idx, bool(cfg.logging.append_seed_window))
            run_dir = _prepare_directories(results_root, run_id)
            simulation = _simulate_episode(cfg, int(seed), window_idx)
            _write_artifacts(run_dir, cfg, int(seed), window_idx, simulation, metrics_writer)

    split_payload = _materialize_walk_forward(cfg, total_samples=cfg.training.total_env_steps)
    with (paper_root / "data_splits.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)

    (paper_root / "manifest.json").write_text(
        json.dumps(
            {
                "device": device_info.__dict__,
                "results_root": str(results_root),
                "paper_root": str(paper_root),
                "run_id": base_run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
