from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

KEY_METRICS: Sequence[str] = (
    "mean_abs_matrix",
    "row_sum_std",
    "row_sum_range",
    "stability_matrix_corr",
    "reward_episode_mean",
)


@dataclass
class ScenarioAggregate:
    name: str
    aggregate_dir: Path
    stats: pd.DataFrame
    significance: pd.DataFrame
    welch: pd.DataFrame
    runs: List[Dict[str, object]]


def discover_aggregate_dirs(root: Path) -> List[Path]:
    dirs = [p for p in root.rglob("*_aggregate") if p.is_dir()]
    return sorted(dirs)


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_runs_metadata(aggregate_dir: Path, scenario_name: str) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    runs_manifest = aggregate_dir / "runs.json"
    if runs_manifest.exists():
        manifest_data = json.loads(runs_manifest.read_text(encoding="utf-8"))
    else:
        manifest_data = []
        parent = aggregate_dir.parent
        seed_dirs = sorted(parent.glob(f"{scenario_name}_seed*"))
        for seed_dir in seed_dirs:
            manifest_data.append({"seed": extract_seed(seed_dir.name), "output_dir": str(seed_dir)})

    for entry in manifest_data:
        run_dir = Path(entry["output_dir"])
        meta_path = run_dir / "run_metadata.json"
        summary_path = run_dir / "summary.csv"
        metadata: Dict[str, object] = {
            "scenario": scenario_name,
            "seed": entry.get("seed"),
            "run_path": str(run_dir),
            "config_path": None,
            "data_path": None,
            "created_at": None,
            "git_commit": None,
            "git_branch": None,
            "python_version": None,
            "platform": None,
            "summary": [],
        }
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["config_path"] = meta.get("config_path")
            metadata["data_path"] = meta.get("data_price_path")
            metadata["created_at"] = meta.get("created_at")
            git_data = meta.get("git", {})
            if isinstance(git_data, dict):
                metadata["git_commit"] = git_data.get("git_commit")
                metadata["git_branch"] = git_data.get("git_branch")
            env_data = meta.get("env", {})
            if isinstance(env_data, dict):
                metadata["python_version"] = env_data.get("python_version")
                metadata["platform"] = env_data.get("platform")
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            metadata["summary"] = summary_df.to_dict(orient="records")
        runs.append(metadata)
    return runs


def extract_seed(name: str) -> Optional[int]:
    parts = name.split("_seed")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].split("_")[0])
    except ValueError:
        return None


def build_aggregate_bundle(path: Path) -> Optional[ScenarioAggregate]:
    stats = load_dataframe(path / "stats.csv")
    if stats.empty:
        return None
    scenario_name = str(stats["scenario"].iloc[0])
    significance = load_dataframe(path / "significance.csv")
    welch = load_dataframe(path / "welch.csv")
    runs = load_runs_metadata(path, scenario_name)
    return ScenarioAggregate(
        name=scenario_name,
        aggregate_dir=path,
        stats=stats,
        significance=significance,
        welch=welch,
        runs=runs,
    )


def summarise_metric_table(df: pd.DataFrame, metrics: Sequence[str]) -> List[Tuple[str, float, float]]:
    result: List[Tuple[str, float, float]] = []
    for metric in metrics:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue
        try:
            mean_val = float(metric_df["mean_mean"].iloc[0])
        except (KeyError, ValueError, TypeError):
            continue
        std_val = metric_df.get("mean_std", pd.Series([float("nan")])).iloc[0]
        try:
            std_val_f = float(std_val) if std_val == std_val else float("nan")
        except (ValueError, TypeError):
            std_val_f = float("nan")
        result.append((metric, mean_val, std_val_f))
    return result


def render_metric_table(rows: Sequence[Tuple[str, float, float]]) -> List[str]:
    if not rows:
        return ["(no key metrics available)"]
    header = "| Metric | Mean | Std Dev |"
    separator = "| --- | --- | --- |"
    body = []
    for metric, mean_val, std_val in rows:
        mean_fmt = f"{mean_val:.4f}"
        std_fmt = "n/a" if std_val != std_val else f"{std_val:.4f}"
        body.append(f"| {metric} | {mean_fmt} | {std_fmt} |")
    return [header, separator, *body]


def summarise_significance(df: pd.DataFrame, metrics: Sequence[str]) -> List[str]:
    lines: List[str] = []
    if df.empty:
        return lines
    for metric in metrics:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue
        row = metric_df.iloc[0].to_dict()
        low = row.get("mea_boot_low")
        high = row.get("mea_boot_high")
        if low in (None, "", float("nan")) or high in (None, "", float("nan")):
            continue
        lines.append(f"- {metric}: bootstrap 95% CI [{float(low):.4f}, {float(high):.4f}]")
    return lines


def summarise_welch(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    columns = [c for c in df.columns if c.lower().startswith("p_")]
    lines: List[str] = []
    for _, row in df.iterrows():
        scenario_a = row.get("scenario_a") or row.get("scenarioA") or row.get("scenario_1")
        scenario_b = row.get("scenario_b") or row.get("scenarioB") or row.get("scenario_2")
        for col in columns:
            try:
                p_val = float(row[col])
            except (TypeError, ValueError):
                continue
            label = col.replace("p_", "")
            lines.append(f"- Welch test ({label}) {scenario_a} vs {scenario_b}: p={p_val:.4f}")
    return lines


def build_introduction(now_str: str, aggregates: Sequence[ScenarioAggregate]) -> List[str]:
    scenario_names = ", ".join(sorted({agg.name for agg in aggregates}))
    intro = [
        "# Lead-Lag Signature RL Research Report",
        "",
        f"_Generated on {now_str}_",
        "",
        "This report compiles the latest reinforcement learning lookback experiments and baseline comparisons.",
        f"Scenarios covered: {scenario_names or 'n/a'}.",
    ]
    return intro


def build_methodology_section(aggregates: Sequence[ScenarioAggregate]) -> List[str]:
    lines = ["## Methodology", ""]
    lines.append(
        "Experiments were run via the ExperimentOrchestrator multi-seed pipeline with Hydra-driven configs."
    )
    lines.append(
        "Aggregates include mean/median statistics, bootstrap confidence intervals, and optional Welch tests."
    )
    lines.append("")
    for agg in aggregates:
        seeds = [str(run.get("seed")) for run in agg.runs if run.get("seed") is not None]
        seed_list = ", ".join(seeds) if seeds else "n/a"
        lines.append(f"- **{agg.name}**: aggregate directory `{agg.aggregate_dir}`; seeds: {seed_list}")
    lines.append("")
    return lines


def build_experiments_section(aggregates: Sequence[ScenarioAggregate]) -> List[str]:
    lines = ["## Experiments", ""]
    for agg in aggregates:
        lines.append(f"### Scenario: {agg.name}")
        lines.append("")
        metric_rows = summarise_metric_table(agg.stats, KEY_METRICS)
        lines.extend(render_metric_table(metric_rows))
        lines.append("")
        significance_lines = summarise_significance(agg.significance, KEY_METRICS)
        if significance_lines:
            lines.append("Bootstrap confidence intervals:")
            lines.extend(significance_lines)
            lines.append("")
        welch_lines = summarise_welch(agg.welch)
        if welch_lines:
            lines.append("Welch significance checks:")
            lines.extend(welch_lines)
            lines.append("")
        if agg.runs:
            lines.append("Per-run highlights:")
            for run in agg.runs:
                seed = run.get("seed", "n/a")
                created = run.get("created_at", "n/a")
                config = run.get("config_path", "n/a")
                lines.append(f"- Seed {seed}, created {created}, config `{config}`")
            lines.append("")
        else:
            lines.append("_No run-level metadata available._")
            lines.append("")
    return lines


def build_conclusion_section(aggregates: Sequence[ScenarioAggregate]) -> List[str]:
    lines = ["## Conclusion", ""]
    if not aggregates:
        lines.append("No aggregate data available to summarise.")
        return lines
    best_scenario = None
    best_score = float("-inf")
    for agg in aggregates:
        metric_rows = summarise_metric_table(agg.stats, ("mean_abs_matrix",))
        if not metric_rows:
            continue
        score = metric_rows[0][1]
        if score > best_score:
            best_score = score
            best_scenario = agg.name
    if best_scenario is not None and best_score > float("-inf"):
        lines.append(
            f"{best_scenario} achieved the strongest mean_abs_matrix signal ({best_score:.4f}), "
            "indicating the highest overall signal strength among evaluated runs."
        )
    lines.append(
        "Next actions: finalise the research narrative, attach visual artefacts from ER-01, and schedule peer review."
    )
    return lines


def build_appendix_text(aggregates: Sequence[ScenarioAggregate], now_str: str) -> str:
    lines = ["# Reproducibility Appendix", "", f"_Generated on {now_str}_", ""]
    lines.append("## Run Metadata")
    lines.append("")
    header = "| Scenario | Seed | Created At | Config | Data | Python | Platform |"
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    lines.append(header)
    lines.append(separator)
    for agg in aggregates:
        for run in agg.runs:
            row = [
                agg.name,
                str(run.get("seed") or "n/a"),
                str(run.get("created_at") or "n/a"),
                str(run.get("config_path") or "n/a"),
                str(run.get("data_path") or "n/a"),
                str(run.get("python_version") or "n/a"),
                str(run.get("platform") or "n/a"),
            ]
            lines.append("| " + " | ".join(row) + " |")
    if len(lines) == 4:  # header only
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    lines.append("## Artefact Inventory")
    lines.append("")
    for agg in aggregates:
        lines.append(f"- `{agg.aggregate_dir}` -> stats.csv, significance.csv, welch.csv (when available)")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Report generated via reporting/generate_report.py.")
    lines.append("- Ensure MLflow run links and figures from ER-01 are attached alongside this appendix.")
    return "\n".join(lines)


def wrap_paragraphs(lines: Iterable[str]) -> str:
    wrapped: List[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if line.startswith("| ") or line.startswith("- ") or line.startswith("#"):
            wrapped.append(line)
            continue
        wrapped.extend(textwrap.wrap(line, width=96) or [""])
    return "\n".join(wrapped)


def chunk_for_pdf(text: str, lines_per_page: int = 44) -> List[List[str]]:
    raw_lines = text.splitlines()
    pages: List[List[str]] = []
    for start in range(0, len(raw_lines), lines_per_page):
        chunk = raw_lines[start : start + lines_per_page]
        pages.append(chunk or [" "])
    if not pages:
        pages.append([" "])
    return pages


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_stream(lines: Sequence[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "14 TL", "50 780 Td"]
    first_line = True
    for raw in lines:
        line = escape_pdf_text(raw)
        if not first_line:
            commands.append("T*")
        if not line:
            commands.append("( ) Tj")
        else:
            commands.append(f"({line}) Tj")
        first_line = False
    commands.append("ET")
    stream = "\n".join(commands) + "\n"
    return stream.encode("utf-8")


def write_simple_pdf(pages: Sequence[Sequence[str]], output_path: Path) -> None:
    content_streams = [build_pdf_stream(page) for page in pages]
    num_pages = len(content_streams)
    font_obj_num = 3 + num_pages * 2
    pdf_parts: List[bytes] = [b"%PDF-1.4\n"]
    offsets: List[int] = []

    def append_object(obj: bytes) -> None:
        offset = sum(len(part) for part in pdf_parts)
        offsets.append(offset)
        pdf_parts.append(obj)

    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    append_object(catalog)

    kids_refs = " ".join(f"{3 + idx * 2} 0 R" for idx in range(num_pages))
    pages_obj = f"2 0 obj\n<< /Type /Pages /Count {num_pages} /Kids [ {kids_refs} ] >>\nendobj\n".encode(
        "utf-8"
    )
    append_object(pages_obj)

    for idx, stream in enumerate(content_streams):
        page_obj_num = 3 + idx * 2
        content_obj_num = page_obj_num + 1
        page_obj = (
            f"{page_obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {content_obj_num} 0 R >>\n"
            "endobj\n"
        ).encode("utf-8")
        append_object(page_obj)

        content_obj = (
            f"{content_obj_num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("utf-8")
            + stream
            + b"endstream\nendobj\n"
        )
        append_object(content_obj)

    font_obj = (
        f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("utf-8")
    )
    append_object(font_obj)

    xref_offset = sum(len(part) for part in pdf_parts)
    xref_header = f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("utf-8")
    pdf_parts.append(xref_header)
    for offset in offsets:
        pdf_parts.append(f"{offset:010} 00000 n \n".encode("utf-8"))

    trailer = (
        f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("utf-8")
    )
    pdf_parts.append(trailer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(pdf_parts))


def write_report_files(report_text: str, appendix_text: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "appendix.md").write_text(appendix_text, encoding="utf-8")
    pages = chunk_for_pdf(report_text)
    write_simple_pdf(pages, output_dir / "final_report.pdf")


def generate_report(results_root: Path, output_dir: Path) -> List[ScenarioAggregate]:
    aggregate_dirs = discover_aggregate_dirs(results_root)
    aggregates: List[ScenarioAggregate] = []
    for aggregate_dir in aggregate_dirs:
        bundle = build_aggregate_bundle(aggregate_dir)
        if bundle:
            aggregates.append(bundle)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    intro = build_introduction(now, aggregates)
    methodology = build_methodology_section(aggregates)
    experiments = build_experiments_section(aggregates)
    conclusion = build_conclusion_section(aggregates)

    report_lines = intro + [""] + methodology + [""] + experiments + [""] + conclusion
    report_text = wrap_paragraphs(report_lines)
    appendix_text = build_appendix_text(aggregates, now)

    write_report_files(report_text, appendix_text, output_dir)
    return aggregates


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate research report and appendix from aggregated runs.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Root directory containing aggregates.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory to write report artefacts.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    aggregates = generate_report(args.results_root, args.output_dir)
    if aggregates:
        print(f"Report generated for {len(aggregates)} scenario(s) in {args.output_dir}.")
    else:
        print("Warning: no aggregate directories found; generated empty report.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
