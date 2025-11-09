# Pipeline runbook

> **Metadata**
> - Last updated: 2025-02-16
> - Maintainer: Reliability Working Group
> - Status: Draft
> - Source of truth: `docs/deployment/pipeline_runbook.md`

This runbook captures the operational playbook for the LeadLag signature pipelines across local development, CI, and Kaggle backends.

## Monitoring

### Log surfaces

1. **Structured CLI output** – Execute pipeline CLIs with `--format json` (or `--status --format json`) to receive the canonical envelope described in [`docs/config_reference.md`](../config_reference.md#output-formatting-contract). Automation should capture these responses for dashboards.
2. **`run.log` artefact** – Scenario runners persist `run.log` in the working directory using the `run_scenario` and `runner_multiseed` loggers configured in [`logging_config.yaml`](../../logging_config.yaml). Ship this file to long-term storage or tail it live for regression triage.
3. **Kaggle `log` stream** – Notebook executions import the shim from [`log.py`](../../log.py). Messages flow to stderr and appear in the Kaggle UI; scrape them when diagnosing notebook runs.
4. **Dashboard probes** – The [`observability/dashboard.py`](../../observability/dashboard.py) entry point emits the same metrics that feed streamlit dashboards. When a panel degrades, run the module locally with `python -m observability.dashboard --format json` and inspect the envelope payload.

### Health checks

- **Smoke job**: `leadlag --list --format json` should return `success=true` and enumerate scenarios. Missing scenarios usually indicate a corrupt `results_root` or Hydra misconfiguration.
- **Results freshness**: `leadlag --status --format json --results-root <path>` must report recent timestamps. Stale entries imply stuck runners; check `run.log` for exceptions.
- **Telemetry completeness**: Ensure both console and file handlers are active by grepping `run.log` for `| INFO | run_scenario |`. Absence points to misapplied logging config.

## Troubleshooting

### Pipelines fail immediately

1. Inspect the JSON envelope `errors` field. Common codes include `resource_not_found` (missing dataset) and `missing_dependency` (optional extras not installed).
2. Review `run.log` for stack traces. When absent, re-run with `LEADLAG_DEBUG=1` to elevate log levels and include debug output.
3. For Kaggle notebooks, confirm the `log` shim initialised (`[log]` entries appear in the console). If not, re-import `log` before executing pipeline cells.

### Metrics drift or dashboard gaps

1. Validate the data directory by running `leadlag --status --format json --artifacts`. Missing expected artefacts indicates partial runs; rerun the impacted scenarios.
2. Execute `python -m observability.dashboard --format json --pretty 2` locally to mirror dashboard queries. Compare returned metrics with production panels.
3. If the CLI emits success but dashboards stay blank, inspect the collector service logs. Misconfigured logger names (anything other than `observability.*`) are ignored by default handlers—update the code to use `logging.getLogger("observability.metrics")`.

### Long-running jobs hang

1. Check `run.log` for repeated retries or warnings about rate limits. These usually stem from upstream data fetchers.
2. Confirm Hydra has not spawned nested runs by verifying the absence of multiple `output_root` directories with similar timestamps.
3. If running in CI, download artefacts and re-execute locally with `--format json --pretty 2` to reveal the last successful step. Attach the envelope to the incident ticket.

Keeping this runbook updated ensures on-call engineers have a single reference point when pipelines regress.
