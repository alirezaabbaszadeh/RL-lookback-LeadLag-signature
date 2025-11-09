# Observability and logging overview

> **Metadata**
> - Last updated: 2025-02-16
> - Maintainer: Reliability Working Group
> - Status: Draft
> - Source of truth: `docs/observability.md`

This note summarises how runtime telemetry is assembled across the repository so that engineers can wire dashboards, inspect logs, and extend the instrumentation safely.

## Runtime log pipeline

The shared logging profile lives in [`logging_config.yaml`](../logging_config.yaml). It declares a `standard` formatter (`"%(asctime)s | %(levelname)s | %(name)s | %(message)s"`) that is bound to two handlers:

- **`console`** – a `StreamHandler` targeting `sys.stdout` for interactive work and local debugging.
- **`file`** – a `FileHandler` that writes the same payload to `run.log`. Scenario runners attach this handler through the named logger entries (`run_scenario`, `runner_multiseed`) so every execution persists a flat log artifact alongside results.

Any module can consume the profile via `logging.config.dictConfig`, but the CLI entry points already initialise it during bootstrap. Downstream code should prefer `logging.getLogger(__name__)` rather than constructing bespoke handlers so that this central policy continues to apply.

### Kaggle compatibility shim

Kaggle notebooks expect a top-level `log` module. The project supplies [`log.py`](../log.py), which re-exports the compatibility helpers from [`src/leadlag/log.py`](../src/leadlag/log.py). The shim:

- lazily initialises a stderr `StreamHandler` with the `standard` formatting string,
- exposes the `info`, `warning`, `error`, `debug`, `critical`, and `exception` helpers that Kaggle invokes,
- serialises payloads for `log.send` calls so notebook telemetry mirrors Kaggle's default behaviour, and
- degrades to no-op functions for unknown attributes so new Kaggle hooks do not break automation.

When porting notebooks or running tests in headless CI, import `log` exactly as the Kaggle runtime would—our shim prevents `ModuleNotFoundError` without interfering with standard Python logging.

### Observability dashboard entry point

Utilities within [`observability/dashboard.py`](../observability/dashboard.py) re-export `leadlag.observability.dashboard`. The module reads from the same logging pipeline, so dashboards can rely on `run.log` and stdout for live streams. When introducing dashboards or telemetry collectors, always emit structured data under a dedicated logger (for example, `logging.getLogger("observability.metrics")`) so the configuration file can opt-in to persistent storage later.

## Extending the logging setup

1. Update `logging_config.yaml` with any new handlers or formatters. Prefer rotating file handlers for long-running services.
2. Wire your module to the shared profile via `logging.config.dictConfig` or by reusing an existing CLI bootstrapper.
3. For notebook environments, confirm that the shimmed `log` module still fulfils Kaggle's expectations before introducing breaking changes.
4. Mirror any new log destinations in the runbook (see `docs/deployment/pipeline_runbook.md`) so operators know where to collect diagnostics.
