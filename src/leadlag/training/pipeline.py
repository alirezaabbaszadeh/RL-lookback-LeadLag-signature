"""Reusable pipeline for orchestrating scenario runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

import pandas as pd

from leadlag.reporting.profiling import profile_to
from leadlag.training.run_support import RunPreparation


class Analyzer(Protocol):  # pragma: no cover - simple structural protocol
    """Minimal protocol for analyzer-like objects."""

    def analyze(self, prices: pd.DataFrame, *, return_rolling: bool = True) -> Any:
        ...


AnalysisRunner = Callable[[Analyzer, pd.DataFrame, dict[str, Any]], Any]
MetricsComputer = Callable[[Any], pd.DataFrame]
MetricsSummarizer = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class PipelineContext:
    """Context shared with hooks and artifact generators."""

    cfg: dict[str, Any]
    preparation: RunPreparation
    analyzer: Analyzer
    rolling: Any
    metrics: pd.DataFrame
    summary: pd.DataFrame
    metrics_path: Path
    summary_path: Path


Hook = Callable[[PipelineContext], None]


@dataclass(frozen=True)
class ScenarioPipelineHooks:
    """Optional integrations executed during the pipeline."""

    mlflow: Optional[Hook] = None
    plotting: Optional[Hook] = None
    extras: Iterable[Hook] = ()


@dataclass(frozen=True)
class ScenarioPipelineResult:
    """Result returned by :class:`ScenarioPipeline`."""

    context: PipelineContext

    @property
    def out_dir(self) -> Path:
        return self.context.preparation.out_dir


class ScenarioPipeline:
    """Coordinate the analysis, metrics, and artifact generation flow."""

    def __init__(
        self,
        analyzer_factory: Callable[[dict[str, Any]], Analyzer],
        analysis_runner: AnalysisRunner,
        metrics_computer: MetricsComputer,
        metrics_summarizer: MetricsSummarizer,
        *,
        artifact_generators: Iterable[Hook] = (),
        hooks: Optional[ScenarioPipelineHooks] = None,
    ) -> None:
        self._analyzer_factory = analyzer_factory
        self._analysis_runner = analysis_runner
        self._metrics_computer = metrics_computer
        self._metrics_summarizer = metrics_summarizer
        self._artifact_generators = tuple(artifact_generators)
        self._hooks = hooks or ScenarioPipelineHooks()

    def run(self, cfg: dict[str, Any], preparation: RunPreparation) -> ScenarioPipelineResult:
        analyzer = self._analyzer_factory(cfg)

        with profile_to(preparation.out_dir, label="analyze"):
            rolling = self._analysis_runner(analyzer, preparation.prices, cfg)

        with profile_to(preparation.out_dir, label="metrics"):
            metrics_df = self._metrics_computer(rolling)

        metrics_path = preparation.out_dir / "metrics_timeseries.csv"
        metrics_df.to_csv(metrics_path, index=True)

        summary_df = self._metrics_summarizer(metrics_df)
        summary_path = preparation.out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        context = PipelineContext(
            cfg=cfg,
            preparation=preparation,
            analyzer=analyzer,
            rolling=rolling,
            metrics=metrics_df,
            summary=summary_df,
            metrics_path=metrics_path,
            summary_path=summary_path,
        )

        for generator in self._artifact_generators:
            generator(context)

        if self._hooks.mlflow is not None:
            self._hooks.mlflow(context)
        if self._hooks.plotting is not None:
            self._hooks.plotting(context)
        for hook in tuple(self._hooks.extras):
            hook(context)

        return ScenarioPipelineResult(context=context)

