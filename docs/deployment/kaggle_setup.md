Kaggle Setup and One‑Command Grand Run

This guide shows how to run the complete experiment suite on Kaggle with a single command, including baselines, ablations, RL, audits, and final reports. It also covers performance tips and optional overrides.

Prerequisites

- Kaggle Notebook settings:
  - Internet: ON (required for one‑time wheel downloads)
  - Accelerator: GPU recommended for RL; CPU works for non‑RL parts
- Attach your project code/data as a Kaggle Dataset or clone the repo in a cell.

One‑Command Grand Run

- From the notebook (after copying or cloning the repo into `/kaggle/working`):

```
python kaggle/run_all.py
```

- What it does:
  - Prefetches wheels (local wheelhouse) to speed up installations
  - Configures pip to use the local wheelhouse/cache
  - Executes the orchestrator with three stages in order:
    1) `full_suite` – runs `pipelines/run_full_suite.py` (baselines, ablations, meta‑RL, offline RL, audits, reports)
    2) `sb3_leadlag` – production SB3 training on LeadLag with device/timesteps overrides
    3) `dopamine` – Gymnasium 1.x + `dopamine-rl` validation
  - Bundles outputs to `/kaggle/working/multi_stage_artifacts.zip`

- If you prefer not to prefetch wheels (slower):

```
python kaggle/run_all.py --no-prefetch
```

Outputs

- Root: `/kaggle/working/multi_stage_artifacts/`
  - `full_suite/…` – core runs, ablations, robustness checks, reports, plots, audit logs
  - `sb3_leadlag/…` – production RL outputs (one folder per scenario)
  - `dopamine/…` – sanity run for Gymnasium 1.x + Dopamine
  - `summary.json` – stage statuses, durations, and log paths
- Bundle for download: `/kaggle/working/multi_stage_artifacts.zip`

Optional Overrides (advanced)

Use these environment variables before calling specific stages (only needed if you run `kaggle/run_multi_stage.py` directly or add `sb3_leadlag`):

- Hydra scenarios (stage `leadlag_hydra`):
  - `LEADLAG_SCENARIOS` – comma‑separated names (e.g., `fixed_30,rl_ppo`)
  - `LEADLAG_SEEDS` – comma‑separated ints (e.g., `42,52,62`)
  - `LEADLAG_MULTI_SEED` – `true/false/1/0`

- SB3 production training (stage `sb3_leadlag`):
  - `SB3_DEVICE` – `cuda`, `cpu`, or `auto`
  - `SB3_TIMESTEPS` – total timesteps (int)
  - `SB3_N_STEPS`, `SB3_BATCH_SIZE`, `SB3_LR`, `SB3_EVAL_FREQ`, `SB3_VERBOSE`, `SB3_SEED`

Notes

- The grand run intentionally avoids duplicate RL work by using the `full_suite` pipeline (which already includes RL ablations). If you want extra long RL training, run:

```
python kaggle/run_multi_stage.py --stage sb3_leadlag
```

Performance Tips

- Keep Internet ON for prefetch. If a wheel is missing for your platform, set `PIP_NO_INDEX=0` temporarily to allow PyPI fallback.
- Use GPU for RL with `SB3_DEVICE=cuda` (when invoking `sb3_leadlag`).
- Reduce RL timesteps if the Kaggle time budget is tight.
- Large outputs: download the ZIP bundle or selectively prune heavy logs after verifying results.

Troubleshooting

- If any stage fails, check `multi_stage_artifacts/<stage>/stderr.log` and the top‑level `summary.json` for quick pointers.
- If pip raises dependency issues, clear local caches (`/kaggle/working/.cache/pip`) and rerun.
