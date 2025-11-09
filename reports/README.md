# Reports

The active reports directory is intentionally empty. Previously generated campaign materials from 2025 now live under
`archive/2025-10-17-research-report/reports/`.

## Generating a fresh report

Use the packaged CLI (or module) once aggregated results are available under `results/`:

```bash
python -m leadlag.reporting.generate_report --results-root results --output-dir reports
# or equivalently
leadlag-report --results-root results --output-dir reports
```

The command writes:
- `final_report.md` – narrative summary assembled from scenario aggregates.
- `final_report.pdf` – lightweight PDF rendition of the markdown (no external LaTeX dependency).
- `appendix.md` – extended tables and references.
- `generate_report.log` – structured log of the build.

Pass `--dry-run` to inspect which aggregates would be consumed without writing files. Use `--format json` if the report needs to be orchestrated from automation scripts.

