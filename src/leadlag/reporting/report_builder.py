from __future__ import annotations

import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from leadlag.reporting.data_access import ScenarioAggregate

KEY_METRICS: Sequence[str] = (
    "mean_abs_matrix",
    "row_sum_std",
    "row_sum_range",
    "stability_matrix_corr",
    "reward_episode_mean",
)


@dataclass
class ReportArtifacts:
    generated_at: str
    report_markdown: str
    appendix_markdown: str
    aggregate_names: List[str]
    metadata: Dict[str, object]


class ReportBuilder:
    """Build Markdown and structured summaries for reporting aggregates."""

    def __init__(
        self,
        key_metrics: Sequence[str] = KEY_METRICS,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.key_metrics = tuple(key_metrics)
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def build(self, aggregates: Sequence[ScenarioAggregate]) -> ReportArtifacts:
        timestamp = self._now_fn()
        now_str = timestamp.strftime("%Y-%m-%d %H:%M:%SZ")
        intro = self._build_introduction(now_str, aggregates)
        methodology = self._build_methodology_section(aggregates)
        experiments = self._build_experiments_section(aggregates)
        conclusion = self._build_conclusion_section(aggregates)

        report_lines = intro + [""] + methodology + [""] + experiments + [""] + conclusion
        report_text = self._wrap_paragraphs(report_lines)
        appendix_text = self._build_appendix_text(aggregates, now_str)

        aggregate_names = [agg.name for agg in aggregates]
        metadata: Dict[str, object] = {
            "generated_at": now_str,
            "scenarios": aggregate_names,
            "key_metrics": list(self.key_metrics),
        }

        return ReportArtifacts(
            generated_at=now_str,
            report_markdown=report_text,
            appendix_markdown=appendix_text,
            aggregate_names=aggregate_names,
            metadata=metadata,
        )

    def _summarise_metric_table(
        self, df: pd.DataFrame, metrics: Sequence[str]
    ) -> List[Tuple[str, float, float]]:
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

    def _render_metric_table(self, rows: Sequence[Tuple[str, float, float]]) -> List[str]:
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

    def _summarise_significance(self, df: pd.DataFrame, metrics: Sequence[str]) -> List[str]:
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

    def _summarise_welch(self, df: pd.DataFrame) -> List[str]:
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

    def _build_introduction(
        self, now_str: str, aggregates: Sequence[ScenarioAggregate]
    ) -> List[str]:
        scenario_names = ", ".join(sorted({agg.name for agg in aggregates}))
        intro = [
            "# Lead-Lag Signature RL Research Report",
            "",
            f"_Generated on {now_str}_",
            "",
            (
                "This report compiles the latest reinforcement learning lookback experiments "
                "and baseline comparisons."
            ),
            f"Scenarios covered: {scenario_names or 'n/a'}.",
        ]
        return intro

    def _build_methodology_section(
        self, aggregates: Sequence[ScenarioAggregate]
    ) -> List[str]:
        lines = ["## Methodology", ""]
        lines.append(
            "Experiments were run via the ExperimentOrchestrator multi-seed pipeline with "
            "Hydra-driven configs."
        )
        lines.append(
            "Aggregates include mean/median statistics, bootstrap confidence intervals, "
            "and optional Welch tests."
        )
        lines.append("")
        for agg in aggregates:
            seeds = [str(run.get("seed")) for run in agg.runs if run.get("seed") is not None]
            seed_list = ", ".join(seeds) if seeds else "n/a"
            lines.append(
                f"- **{agg.name}**: aggregate directory `{agg.aggregate_dir}`; seeds: {seed_list}"
            )
        lines.append("")
        return lines

    def _build_experiments_section(
        self, aggregates: Sequence[ScenarioAggregate]
    ) -> List[str]:
        lines = ["## Experiments", ""]
        for agg in aggregates:
            lines.append(f"### Scenario: {agg.name}")
            lines.append("")
            metric_rows = self._summarise_metric_table(agg.stats, self.key_metrics)
            lines.extend(self._render_metric_table(metric_rows))
            lines.append("")
            significance_lines = self._summarise_significance(agg.significance, self.key_metrics)
            if significance_lines:
                lines.append("Bootstrap confidence intervals:")
                lines.extend(significance_lines)
                lines.append("")
            welch_lines = self._summarise_welch(agg.welch)
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

    def _build_conclusion_section(
        self, aggregates: Sequence[ScenarioAggregate]
    ) -> List[str]:
        lines = ["## Conclusion", ""]
        if not aggregates:
            lines.append("No aggregate data available to summarise.")
            return lines
        best_scenario = None
        best_score = float("-inf")
        for agg in aggregates:
            metric_rows = self._summarise_metric_table(agg.stats, ("mean_abs_matrix",))
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
            "Next actions: finalise the research narrative, attach visual artefacts from "
            "ER-01, and schedule peer review."
        )
        return lines

    def _build_appendix_text(
        self, aggregates: Sequence[ScenarioAggregate], now_str: str
    ) -> str:
        lines = ["# Reproducibility Appendix", "", f"_Generated on {now_str}_", ""]
        lines.append("## Run Metadata")
        lines.append("")
        header = "| Scenario | Seed | Created At | Config | Data | Python | Platform |"
        separator = "| --- | --- | --- | --- | --- | --- | --- |"
        lines.append(header)
        lines.append(separator)
        has_run_rows = False
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
                has_run_rows = True
        if not has_run_rows:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append("")
        lines.append("## Artefact Inventory")
        lines.append("")
        for agg in aggregates:
            lines.append(
                f"- `{agg.aggregate_dir}` -> stats.csv, significance.csv, welch.csv (when available)"
            )
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- Report generated via reporting/generate_report.py.")
        lines.append(
            "- Ensure MLflow run links and figures from ER-01 are attached alongside this appendix."
        )
        return "\n".join(lines)

    def _wrap_paragraphs(self, lines: Iterable[str]) -> str:
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


__all__ = ["KEY_METRICS", "ReportArtifacts", "ReportBuilder"]
