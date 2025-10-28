# CLI Agent Output Contract

## Overview

All console entry points exposed by the `leadlag` package now share a common output contract driven by `src/leadlag/cli/formatters.py`.
Each CLI accepts `--format text|json` (with `--json` kept as a temporary alias) so that automation agents receive deterministic responses.

## Envelope Structure

When `--format json` is passed, the helper emits an envelope with the following top-level keys:

```json5
{
  "success": true,
  "command": "leadlag --status --format json",
  "args": { ... },          // parsed argparse namespace (paths normalised to strings)
  "format": "json",
  "message": "Human-friendly summary",
  "data": { ... },          // structured payload specific to the CLI
  "errors": [],             // list of {code, message, details?}
  "artifacts": { ... }      // optional file paths or derived outputs
}
```

- `success`: boolean result decided by the CLI exit path (non-zero exit codes still propagate via the process status).
- `command`: the exact invocation (helpful for auditing pipelines).
- `args`: a JSON-serialisable view of the parsed arguments; file system paths are rendered as strings.
- `message`: optional friendly summary; useful to show in logs even when `data` is consumed by tooling.
- `data`: operation-specific structure (selected scenarios, generated plots, etc.).
- `errors`: populated when failures occur. Helper utilities map common exceptions (`FileNotFoundError`, `ImportError`, etc.) to stable codes such as `resource_not_found` or `missing_dependency`.
- `artifacts`: optional dictionary listing generated files/directories that downstream automation can fetch.

## Text Mode

Plain `--format text` (the default) continues to print human-readable summaries. The same helper drives text output so the shape of CLI implementations remains consistent.

## Usage Guidelines

1. Prefer `--format json` in CI, orchestration workflows, or notebook automation.  
2. Consume `data` for command-specific results and treat the rest of the envelope as metadata.  
3. Check `success` and inspect `errors` to decide whether to halt or continue chained steps.  
4. `--json` remains an alias but will be removed after version `0.2.0`; migrate to `--format json`.  
5. If new CLIs are added, call `add_format_flags`, `finalize_format_args`, and `emit_formatted_output` to blend in with the standard contract.

## Example

```bash
leadlag --status --format json
```

```json
{
  "success": true,
  "command": "leadlag --status --format json",
  "args": {
    "status": true,
    "results_root": "results",
    "...": "..."
  },
  "format": "json",
  "message": "Run status summary.",
  "data": {
    "results_root": "/abs/path/results",
    "runs": [
      {"scenario": "alpha", "status": "success", "run_dir": "..."},
      {"scenario": "beta", "status": "incomplete", "run_dir": "..."}
    ]
  },
  "errors": [],
  "artifacts": null
}
```

Agents can now rely on identical envelopes whether they execute `leadlag`, `leadlag-full-suite`, `leadlag-plot-balance`, or any other companion utility.
