from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.evaluation.finance_kpis import compute_kpis
from leadlag.reporting.logging_utils import get_logger, setup_logging


CANDIDATE_RETURN_COLUMNS = [
    "portfolio_return",
    "strategy_return",
    "reward",
    "reward_step_mean",
    "pnl",
]
FALLBACK_COLUMN = "mean_abs_matrix"
ANNUALIZATION_FACTOR = 252


@dataclass
class RunRecord:
    run_dir: Path
    scenario: str
    returns: pd.Series
    returns_column: str
    equity: pd.Series
    drawdown: pd.Series
    kpis: Dict[str, float]
    num_points: int
    manifest: Dict[str, object] = field(default_factory=dict)
    extra_plots: Dict[str, Path] = field(default_factory=dict)


def _derive_returns(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    for col in CANDIDATE_RETURN_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 2:
                return series.astype(float), col
    if FALLBACK_COLUMN in df.columns:
        base = pd.to_numeric(df[FALLBACK_COLUMN], errors="coerce").dropna()
        if len(base) > 5:
            returns = base.pct_change().dropna()
            if len(returns) > 2:
                return returns.astype(float), f"{FALLBACK_COLUMN}_pct_change"
    return None, None


def _compute_equity(returns: pd.Series, start_balance: float) -> pd.Series:
    equity = (1 + returns).cumprod() * start_balance
    equity.index = range(len(equity))
    return equity


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown


def _infer_scenario_from_dir(run_dir: Path) -> str:
    name = run_dir.name
    match = re.match(r"^(.*)_\d{8}_\d{6}$", name)
    if match:
        return match.group(1)
    return name


def _read_run_manifest(run_dir: Path) -> Dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _resolve_scenario(run_dir: Path) -> str:
    manifest = _read_run_manifest(run_dir)
    run_section = manifest.get("run", {}) if isinstance(manifest.get("run"), dict) else {}
    run_name = run_section.get("name")
    if isinstance(run_name, str) and run_name:
        return run_name

    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg_path = meta.get("config_path")
            if isinstance(cfg_path, str):
                return Path(cfg_path).stem
        except Exception:
            pass

    return _infer_scenario_from_dir(run_dir)


def _collect_extra_plots(run_dir: Path) -> Dict[str, Path]:
    plots = {}
    for name in ("fig_signal_strength.png", "fig_stability.png"):
        path = run_dir / name
        if path.exists():
            plots[name.replace(".png", "")] = path
    return plots


def collect_run_records(results_root: Path, start_balance: float) -> List[RunRecord]:
    records: List[RunRecord] = []
    for metrics_path in sorted(results_root.rglob("metrics_timeseries.csv")):
        run_dir = metrics_path.parent
        if run_dir.name.endswith("_aggregate"):
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if "date" in df.columns:
            try:
                df = df.sort_values("date")
            except Exception:
                pass
        returns, source = _derive_returns(df)
        if returns is None or returns.empty or source is None:
            continue
        equity = _compute_equity(returns, start_balance)
        drawdown = _compute_drawdown(equity)
        kpis = compute_kpis(returns)
        total_return = float((1 + returns).prod() - 1)
        volatility = (
            float(returns.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTOR))
            if len(returns) > 1
            else float("nan")
        )
        kpis.update(
            {
                "total_return": total_return,
                "volatility": volatility,
            }
        )
        scenario = _resolve_scenario(run_dir)
        manifest = _read_run_manifest(run_dir)
        extra_plots = _collect_extra_plots(run_dir)
        records.append(
            RunRecord(
                run_dir=run_dir,
                scenario=scenario,
                returns=returns,
                returns_column=source,
                equity=equity,
                drawdown=drawdown,
                kpis=kpis,
                num_points=len(returns),
                manifest=manifest,
                extra_plots=extra_plots,
            )
        )
    return records


def _plot_equity_curves(records: List[RunRecord], out_path: Path) -> Optional[Path]:
    if not records:
        return None
    plt.figure(figsize=(11, 5))
    for record in records:
        plt.plot(record.equity.index, record.equity.values, linewidth=1.8, label=record.scenario)
    plt.title("Equity Curves")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _plot_drawdown_curves(records: List[RunRecord], out_path: Path) -> Optional[Path]:
    if not records:
        return None
    plt.figure(figsize=(11, 4))
    for record in records:
        plt.plot(record.drawdown.index, record.drawdown.values, linewidth=1.5, label=record.scenario)
    plt.title("Drawdown Curves")
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _plot_kpi_bars(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    higher_is_better: bool = True,
) -> Optional[Path]:
    if summary_df.empty or metric not in summary_df.columns:
        return None
    data = summary_df[["scenario", metric]].dropna()
    if data.empty:
        return None
    data = data.sort_values(metric, ascending=not higher_is_better)
    plt.figure(figsize=(9, 4.5))
    plt.bar(data["scenario"], data[metric], color="#3b82f6")
    plt.title(f"{metric.replace('_', ' ').title()} by Scenario")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _plot_returns_hist(records: List[RunRecord], out_path: Path) -> Optional[Path]:
    if not records:
        return None
    plt.figure(figsize=(8, 4))
    for record in records:
        plt.hist(record.returns.values, bins=40, alpha=0.5, label=record.scenario)
    plt.title("Return Distribution")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _scenario_summary(records: List[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "scenario": record.scenario,
                "run_dir": str(record.run_dir),
                "returns_column": record.returns_column,
                "num_points": record.num_points,
                "annualized_return": record.kpis.get("annualized_return"),
                "sharpe_ratio": record.kpis.get("sharpe_ratio"),
                "max_drawdown": record.kpis.get("max_drawdown"),
                "total_return": record.kpis.get("total_return"),
                "volatility": record.kpis.get("volatility"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("scenario")


def _pick_best(summary_df: pd.DataFrame, metric: str, higher_is_better: bool) -> Optional[Dict[str, object]]:
    if summary_df.empty or metric not in summary_df.columns:
        return None
    data = summary_df.dropna(subset=[metric])
    if data.empty:
        return None
    data = data.sort_values(metric, ascending=not higher_is_better)
    row = data.iloc[0].to_dict()
    return row


def _build_html_report(
    *,
    summary_df: pd.DataFrame,
    chart_paths: Dict[str, Path],
    output_dir: Path,
    title: str,
    notes: Optional[str] = None,
) -> Path:
    table_html = summary_df.to_html(index=False, classes="table", float_format="%.4f")
    chart_tags = []
    for key, path in chart_paths.items():
        if not path.exists():
            continue
        chart_tags.append(
            f"""
            <div class="card">
              <div class="card-title">{key.replace('_', ' ').title()}</div>
              <img src="{_encode_image(path)}" alt="{key}">
            </div>
            """
        )

    best_sharpe = _pick_best(summary_df, "sharpe_ratio", True)
    best_return = _pick_best(summary_df, "total_return", True)
    best_drawdown = _pick_best(summary_df, "max_drawdown", True)

    def _card(label: str, row: Optional[Dict[str, object]]) -> str:
        if not row:
            return ""
        value = row.get(label)
        return f"""
        <div class="metric-card">
          <div class="metric-title">{label.replace('_', ' ').title()}</div>
          <div class="metric-value">{value:.4f}</div>
          <div class="metric-scenario">{row.get('scenario')}</div>
        </div>
        """

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        :root {{
          color-scheme: light;
          --bg: #f8fafc;
          --card: #ffffff;
          --text: #0f172a;
          --muted: #64748b;
          --accent: #2563eb;
          --border: #e2e8f0;
          font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }}
        body {{ margin: 0; background: var(--bg); color: var(--text); }}
        .container {{ max-width: 1200px; margin: 32px auto; padding: 0 24px; }}
        h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
        .metric-card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 16px;
          box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }}
        .metric-title {{ font-size: 12px; text-transform: uppercase; color: var(--muted); }}
        .metric-value {{ font-size: 22px; font-weight: 700; color: var(--accent); margin: 8px 0; }}
        .metric-scenario {{ font-size: 14px; color: var(--text); }}
        .section {{ margin-top: 28px; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
        .card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 16px;
          box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        }}
        .card-title {{ font-weight: 600; margin-bottom: 12px; }}
        img {{ max-width: 100%; border-radius: 10px; }}
        .table {{
          width: 100%;
          border-collapse: collapse;
          background: var(--card);
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        }}
        .table th, .table td {{
          padding: 10px 12px;
          border-bottom: 1px solid var(--border);
          text-align: left;
          font-size: 13px;
        }}
        .table th {{ background: #e2e8f0; font-weight: 600; }}
        .notes {{
          margin-top: 18px;
          padding: 12px 16px;
          background: #dbeafe;
          border-left: 4px solid var(--accent);
          border-radius: 10px;
          color: #1e3a8a;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">Supervisor-ready summary report</div>
        <div class="metrics">
          {_card("sharpe_ratio", best_sharpe)}
          {_card("total_return", best_return)}
          {_card("max_drawdown", best_drawdown)}
        </div>
        <div class="section">
          <h2>Key Metrics Table</h2>
          {table_html}
        </div>
        <div class="section">
          <h2>Charts</h2>
          <div class="cards">
            {''.join(chart_tags)}
          </div>
        </div>
        {f'<div class="notes">{notes}</div>' if notes else ''}
      </div>
    </body>
    </html>
    """
    out_path = output_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _build_charts(records: List[RunRecord], out_dir: Path) -> Dict[str, Path]:
    charts: Dict[str, Path] = {}
    style = "seaborn-v0_8"
    try:
        plt.style.use(style)
    except Exception:
        pass

    equity_path = _plot_equity_curves(records, out_dir / "equity_curves.png")
    if equity_path:
        charts["equity_curves"] = equity_path

    drawdown_path = _plot_drawdown_curves(records, out_dir / "drawdown_curves.png")
    if drawdown_path:
        charts["drawdown_curves"] = drawdown_path

    summary_df = _scenario_summary(records)
    sharpe_path = _plot_kpi_bars(summary_df, "sharpe_ratio", out_dir / "kpi_sharpe.png")
    if sharpe_path:
        charts["kpi_sharpe"] = sharpe_path
    total_return_path = _plot_kpi_bars(
        summary_df, "total_return", out_dir / "kpi_total_return.png"
    )
    if total_return_path:
        charts["kpi_total_return"] = total_return_path
    drawdown_bar = _plot_kpi_bars(
        summary_df, "max_drawdown", out_dir / "kpi_max_drawdown.png", higher_is_better=True
    )
    if drawdown_bar:
        charts["kpi_max_drawdown"] = drawdown_bar

    hist_path = _plot_returns_hist(records, out_dir / "returns_hist.png")
    if hist_path:
        charts["returns_hist"] = hist_path

    return charts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a modern supervisor-ready report.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/supervisor"))
    parser.add_argument("--title", type=str, default="Lead-Lag RL Supervisor Report")
    parser.add_argument("--start-balance", type=float, default=100_000.0)
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
    add_format_flags(parser, default="text")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-supervisor-report"
    if argv:
        command = "leadlag-supervisor-report " + " ".join(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or output_dir / "supervisor_report.log"
    setup_logging(log_path, level=str(args.log_level).upper(), context={"module": "supervisor_report"})
    logger = get_logger(
        "reporting.supervisor_report",
        context={"results_root": str(args.results_root.resolve())},
    )

    records = collect_run_records(args.results_root.resolve(), args.start_balance)
    logger.info("Collected runs", context={"count": len(records)})

    base_payload = {
        "results_root": str(args.results_root.resolve()),
        "output_dir": str(output_dir),
        "records": len(records),
    }

    if not records:
        emit_formatted_output(
            args,
            success=False,
            data=base_payload,
            text="No runs found with metrics_timeseries.csv.",
            message="No runs found.",
            errors=[{"code": "no_runs", "message": "No runs found for reporting."}],
            pretty=True,
            command=command,
        )
        return 0

    summary_df = _scenario_summary(records)
    summary_csv = output_dir / "summary_kpis.csv"
    summary_df.to_csv(summary_csv, index=False)

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = _build_charts(records, charts_dir)

    report_path = _build_html_report(
        summary_df=summary_df,
        chart_paths=chart_paths,
        output_dir=output_dir,
        title=args.title,
        notes="Report generated from local runs. Use for supervisor review and iteration.",
    )

    artifacts = {
        "summary_csv": str(summary_csv),
        "report_html": str(report_path),
        "charts": [str(path) for path in chart_paths.values()],
    }

    emit_formatted_output(
        args,
        data={**base_payload, "artifacts": artifacts},
        text=f"Report written to {report_path}",
        message="Supervisor report generated.",
        artifacts=artifacts,
        pretty=True,
        command=command,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
