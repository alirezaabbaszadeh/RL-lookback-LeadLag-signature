from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


@dataclass
class RunRecord:
    run_dir: Path
    run_name: str
    scenario: Optional[str]
    metrics: pd.DataFrame


def _load_run_metadata(run_dir: Path) -> Dict[str, object]:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_runs(root: Path) -> Sequence[RunRecord]:
    records: List[RunRecord] = []
    if not root.exists():
        return records

    for summary_path in root.glob("**/summary.csv"):
        run_dir = summary_path.parent
        if not run_dir.is_dir():
            continue
        try:
            metrics = pd.read_csv(summary_path)
        except Exception:
            continue
        if metrics.empty:
            continue
        meta = _load_run_metadata(run_dir)
        scenario = None
        if isinstance(meta.get("config_path"), str):
            scenario = Path(meta["config_path"]).stem
        elif isinstance(meta.get("scenario"), str):
            scenario = meta["scenario"]
        records.append(
            RunRecord(
                run_dir=run_dir,
                run_name=run_dir.name,
                scenario=scenario,
                metrics=metrics,
            )
        )
    return sorted(records, key=lambda rec: rec.run_dir.stat().st_mtime, reverse=True)


def _select_columns(frame: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    available = [col for col in metric_columns if col in frame.columns]
    if not available:
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            return frame
        available = list(numeric.columns)
    return frame[available]


def _format_record(record: RunRecord, metric_columns: Sequence[str]) -> str:
    cols = _select_columns(record.metrics, metric_columns)
    header = f"Run: {record.run_name}"
    if record.scenario:
        header += f" | Scenario: {record.scenario}"

    lines = [header]
    col_widths = {col: max(len(col), *(len(f"{val:.6g}") for val in cols[col])) for col in cols.columns}
    col_header = " | ".join(col.ljust(col_widths[col]) for col in cols.columns)
    lines.append(col_header)
    lines.append("-" * len(col_header))
    for _, row in cols.iterrows():
        parts = []
        for col in cols.columns:
            value = row[col]
            if isinstance(value, float):
                parts.append(f"{value:.6g}".ljust(col_widths[col]))
            else:
                parts.append(str(value).ljust(col_widths[col]))
        lines.append(" | ".join(parts))
    lines.append("")
    return "\n".join(lines)


def render_dashboard(records: Sequence[RunRecord], metric_columns: Sequence[str]) -> str:
    if not records:
        return "No runs discovered. Execute a scenario to populate results."
    return "\n".join(_format_record(record, metric_columns) for record in records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a lightweight monitoring view for experiment outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Root directory containing run outputs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["metric", "mean", "median", "max", "std"],
        help="Preferred metric columns to display (fallback to numeric columns if missing).",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Set to >0 for continuous monitoring.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    def _render_once() -> None:
        records = _discover_runs(args.root)
        output = render_dashboard(records, args.metrics)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

    if args.refresh <= 0:
        _render_once()
        return 0

    clear_sequence = "\033[2J\033[H" if sys.stdout.isatty() else "\n\n"

    try:
        while True:
            sys.stdout.write(clear_sequence)
            _render_once()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
