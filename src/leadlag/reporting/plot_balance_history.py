from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args

from leadlag.reporting.logging_utils import get_logger, setup_logging
from leadlag.utils.yaml import load_yaml

CANDIDATE_RETURN_COLUMNS = [
    "portfolio_return",
    "strategy_return",
    "reward",
    "reward_step_mean",
    "pnl",
]
FALLBACK_COLUMN = "mean_abs_matrix"


@dataclass
class RunInfo:
    run_dir: Path
    scenario: str
    method: str
    lookback_label: str
    seed_label: str
    label: str
    equity: pd.Series
def _derive_returns(df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[str]]:
    for col in CANDIDATE_RETURN_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 2:
                return series, col
    if FALLBACK_COLUMN in df.columns:
        base = pd.to_numeric(df[FALLBACK_COLUMN], errors="coerce").dropna()
        if len(base) > 5:
            returns = base.pct_change().dropna()
            if len(returns) > 2:
                return returns, f"{FALLBACK_COLUMN}_pct_change"
    return None, None


def _compute_equity(returns: pd.Series, start_balance: float) -> pd.Series:
    equity = (1 + returns).cumprod() * start_balance
    equity.index = range(len(equity))
    return equity


def _scenario_from_metadata(run_dir: Path) -> str:
    meta_p = run_dir / "run_metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            cfg_path = meta.get("config_path")
            if isinstance(cfg_path, str):
                return Path(cfg_path).stem
        except Exception:
            pass
    name = run_dir.name
    parts = name.split("_")
    if len(parts) > 2:
        return "_".join(parts[:-2])
    return name


def _extract_config_info(run_dir: Path, scenario: str) -> tuple[str, str, str, str]:
    cfg = load_yaml(run_dir / "config_merged.yaml", required=False, default={})
    analysis = cfg.get("analysis", {}) if isinstance(cfg, dict) else {}
    runner = cfg.get("runner", "scenario") if isinstance(cfg, dict) else "scenario"
    lookback = analysis.get("lookback") if isinstance(analysis, dict) else None
    method = analysis.get("method") if isinstance(analysis, dict) else None

    if method is None:
        if runner == "rl":
            method = "rl"
        elif "dynamic" in scenario:
            method = "dynamic"
        elif "ccf" in scenario:
            method = "ccf_at_lag"
        else:
            method = "signature"
    method = str(method)

    if lookback is None:
        if runner == "rl":
            lookback_label = "adaptive"
        else:
            lookback_label = "unknown"
    else:
        lookback_label = f"L={lookback}"

    seed_label = ""
    seed = cfg.get("run", {}).get("seed") if isinstance(cfg, dict) else None
    if seed is None:
        for tok in run_dir.name.split("_"):
            if tok.startswith("seed"):
                try:
                    seed = int(tok.replace("seed", ""))
                except Exception:
                    seed = None
    if seed is not None:
        seed_label = f"seed={seed}"

    label_parts = [scenario]
    if lookback_label:
        label_parts.append(lookback_label)
    if seed_label:
        label_parts.append(seed_label)
    label = " | ".join(label_parts)
    return method, lookback_label, seed_label, label


def collect_run_infos(results_root: Path, start_balance: float) -> List[RunInfo]:
    infos: List[RunInfo] = []
    for metrics_path in sorted(results_root.rglob("metrics_timeseries.csv")):
        run_dir = metrics_path.parent
        try:
            df = pd.read_csv(metrics_path)
            if "date" in df.columns:
                try:
                    df = df.sort_values("date")
                except Exception:
                    pass
            returns, source = _derive_returns(df)
            if returns is None or returns.empty:
                continue
            scenario = _scenario_from_metadata(run_dir)
            method, lookback_label, seed_label, label = _extract_config_info(run_dir, scenario)
            equity = _compute_equity(returns, start_balance)
            infos.append(
                RunInfo(
                    run_dir=run_dir,
                    scenario=scenario,
                    method=method,
                    lookback_label=lookback_label,
                    seed_label=seed_label,
                    label=label,
                    equity=equity,
                )
            )
        except Exception:
            continue
    return infos


def _plot_group(
    infos: List[RunInfo],
    title: str,
    out_path: Path,
    legend: bool = True,
) -> Optional[Path]:
    if not infos:
        return None
    plt.figure(figsize=(12, 6))
    for info in infos:
        plt.plot(info.equity.index, info.equity.values, label=info.label, linewidth=1)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Balance")
    if legend:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_all_charts(
    infos: List[RunInfo],
    out_dir: Path,
    max_lines: Optional[int] = None,
) -> List[Path]:
    generated: List[Path] = []
    if not infos:
        return generated

    if max_lines is not None:
        path = _plot_group(
            infos[:max_lines],
            "Portfolio Balance History - All Runs (Subset)",
            out_dir / "balance_all_runs_subset.png",
        )
        if path:
            generated.append(path)
    path = _plot_group(
        infos,
        "Portfolio Balance History - All Runs",
        out_dir / "balance_all_runs.png",
    )
    if path:
        generated.append(path)

    by_scenario: Dict[str, List[RunInfo]] = {}
    by_method: Dict[str, List[RunInfo]] = {}
    by_lookback: Dict[str, List[RunInfo]] = {}

    for info in infos:
        by_scenario.setdefault(info.scenario, []).append(info)
        by_method.setdefault(info.method, []).append(info)
        by_lookback.setdefault(info.lookback_label, []).append(info)

    for scenario, items in by_scenario.items():
        path = _plot_group(
            items,
            f"Portfolio Balance History - Scenario: {scenario}",
            out_dir / "scenario" / f"balance_{scenario}.png",
        )
        if path:
            generated.append(path)

    for method, items in by_method.items():
        path = _plot_group(
            items,
            f"Portfolio Balance History - Method: {method}",
            out_dir / "method" / f"balance_method_{method}.png",
        )
        if path:
            generated.append(path)

    for lookback, items in by_lookback.items():
        label = lookback or "unknown"
        path = _plot_group(
            items,
            f"Portfolio Balance History - Lookback: {label}",
            out_dir / "lookback" / f"balance_lookback_{label.replace('=', '')}.png",
        )
        if path:
            generated.append(path)

    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Plot equity curves for all runs, grouped by scenario/method/lookback.")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/plots/balance"))
    parser.add_argument("--start-balance", type=float, default=100_000.0)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List available runs and exit without generating plots.",
    )
    add_format_flags(parser, default="text")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-plot-balance"
    if argv:
        command = "leadlag-plot-balance " + " ".join(argv)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log_path or out_dir / "plot_balance.log"
    setup_logging(log_path, level=str(args.log_level).upper(), context={"module": "plot_balance"})
    logger = get_logger(
        "reporting.plot_balance_history",
        context={"results_root": args.results_root.resolve()},
    )

    infos = collect_run_infos(args.results_root, args.start_balance)
    logger.info("Collected runs", context={"count": len(infos)})
    runs_payload = [
        {
            "run_dir": str(info.run_dir),
            "scenario": info.scenario,
            "method": info.method,
            "label": info.label,
        }
        for info in infos
    ]
    base_data = {
        "results_root": str(args.results_root.resolve()),
        "output_dir": str(out_dir),
        "runs": runs_payload,
        "dry_run": bool(args.dry_run),
        "max_lines": args.max_lines,
    }

    if not infos:
        logger.warning("No qualifying runs found")
        emit_formatted_output(
            args,
            success=False,
            data=base_data,
            text="No qualifying runs found.",
            message="No qualifying runs found.",
            errors=[{"code": "no_runs", "message": "No qualifying runs found."}],
            pretty=True,
            command=command,
        )
        return 0

    if args.dry_run:
        for info in infos:
            logger.info(
                "[dry-run] run info",
                context={"run_dir": str(info.run_dir), "label": info.label},
            )
        dry_text_lines = ["[dry-run] Available runs:"] + [
            f"  - {entry['label']} ({entry['run_dir']})" for entry in runs_payload
        ]
        emit_formatted_output(
            args,
            data=base_data,
            text="\n".join(dry_text_lines),
            message="Balance plot dry-run completed.",
            artifacts=None,
            pretty=True,
            command=command,
        )
        return 0

    generated_paths = plot_all_charts(infos, out_dir, args.max_lines)
    artifacts = {"plots": [str(path) for path in generated_paths]}
    logger.info(
        "Plots generated",
        context={"output_dir": str(out_dir), "plots": len(generated_paths)},
    )

    text_lines = [
        f"Generated {len(generated_paths)} plot(s) in {out_dir}.",
    ]
    emit_formatted_output(
        args,
        data={**base_data, "generated": len(generated_paths)},
        text="\n".join(text_lines),
        message="Balance plots generated.",
        artifacts=artifacts or None,
        pretty=True,
        command=command,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
