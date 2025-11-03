# Reactor v3 LeadLag Platform

GPU-first reinforcement-learning research stack for lead–lag signature trading.
Reactor v3 standardises the code layout, Hydra configuration surface, and Kaggle
notebook workflow so that a single pipeline produces reproducible, paper-grade
artefacts.

## Highlights

- **Single entry point** – `python -m leadlag.pipelines.run_full_suite` orchestrates
  training, evaluation, and reporting from the Kaggle notebook with Hydra overrides.
  Startup logs include the Hydra config source order so operators can verify which
  packages and directories populated the final configuration.
- **GPU aware** – Defaults to CUDA with AMP enabled; automatically falls back to
  CPU when unavailable.
- **Leakage-safe CV** – Purged/embargoed walk-forward splits, ready for nested
  tuning.
- **Standard outputs** – Canonical `metrics.csv`, equity curves, trade summaries,
  and JSON manifests under `/kaggle/working/results/<run_id>/`.
- **Paper-grade statistics** – HAC confidence intervals, PSR/DSR, SPA-lite, and a
  model confidence set automatically exported to `paper_outputs_root` after each
  run.
- **Kaggle-native** – One notebook path with Internet + GPU ON; all artefacts
  stored inside `/kaggle/working`.

See [`docs/reactor_v3_refactor_plan.md`](docs/reactor_v3_refactor_plan.md) for the
full roadmap and governance milestones.

## Repository Layout

```
src/leadlag/
  pipelines/run_full_suite.py   # Hydra launcher (only supported entry point)
  env/trading_env.py            # t→t+1 execution with cost modelling
  features/                     # signature & lead–lag transforms
  cv/purged.py                  # walk-forward purged/embargoed splits
  eval/{stats.py,stats_cli.py}  # sharpe/sortino/HAC/SPA/MCS utilities
  reporting/metrics_writer.py   # canonical metrics.csv writer
  utils/repro.py                # device selection, seed control, manifests
src/leadlag/configs/
  config.yaml                   # defaults (agent, data, features, hardware)
  agent/                        # PPO, DQN, A2C, SAC, TD3 presets
  features/                     # base/signature/leadlag toggles
  data/                         # universe + dataset directory profiles
  split/walk_forward_purged.yaml
  training/{smoke,base,paper}.yaml
  hardware/{gpu,auto}.yaml
```

Support files:
- `requirements*.txt` – dependency pins for Kaggle notebooks and supporting
  tooling.
- `tests/` – unit coverage for CV, stats, features, and reporting utilities.
- `notebooks/` – Jupyter notebooks mirroring the Kaggle flow for offline
  exploration.

## Quick Start (Kaggle Notebook)

Follow this single notebook path on Kaggle (GPU + Internet enabled):

1. **Verify the GPU runtime**
   ```python
   import torch

   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   !nvidia-smi -L
   ```
2. **Install the pinned commit and requirements**
   ```python
   !pip -q install "git+https://github.com/<owner>/<repo>@<COMMIT_SHA>"
   !pip -q install -r requirements-kaggle.txt
   ```
3. **Declare the Hydra overrides for your run**
   ```python
   BASE_OVERRIDES = {
       "agent": "ppo",
       "training": "smoke",
       "hardware": "gpu",
       "data.dataset_dir": "/kaggle/input/your-dataset",
       "split": "walk_forward_purged",
       "+logging.run_id": "smoke-ppo-0",
   }

   # Expand overrides into the CLI-friendly format expected by Hydra.
   overrides = " ".join(f"{k}={v}" for k, v in BASE_OVERRIDES.items())
   print("Overrides:", overrides)
   ```
4. **Run `leadlag.pipelines.run_full_suite` (canonical cell)**
   ```python
   !python -m leadlag.pipelines.run_full_suite {overrides}
   ```
5. **Inspect consolidated outputs under `/kaggle/working/`**
   ```python
   import glob, os, pandas as pd

   result_dirs = sorted(glob.glob('/kaggle/working/results/*'))
   metrics_frames = [
       pd.read_csv(os.path.join(run_dir, 'metrics.csv'))
       for run_dir in result_dirs
       if os.path.exists(os.path.join(run_dir, 'metrics.csv'))
   ]

   all_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
   all_metrics.to_csv('/kaggle/working/paper_outputs/all_metrics_raw.csv', index=False)
   all_metrics.head()
   ```
6. **Export paper-grade statistics (optional)**
   ```python
   !python -m leadlag.eval.stats_cli \
       --results /kaggle/working/results \
       --out /kaggle/working/paper_outputs \
       --spa-iterations 500 \
       --block-length 5
   ```

Every cell writes to `/kaggle/working/`, producing per-run directories under
`/kaggle/working/results/<run_id>/` and aggregated summaries inside
`/kaggle/working/paper_outputs/`.

## Standard Outputs

Per-run directory (`/kaggle/working/results/<run_id>/`):
- `metrics.csv` – canonical schema for aggregation (includes `EnvSteps` for
  auditing equal interaction budgets).
- `equity.csv` – equity curve per timestamp.
- `returns.csv` – per-step returns.
- `splits.csv` – walk-forward split audit trail (train/test indices, window
  bounds, and embargo for every fold).
- `run_manifest.json` – seeds, device info, package versions, and resolved config.

Aggregate directory (`/kaggle/working/paper_outputs/`):
- `all_metrics_raw.csv` – concatenated metrics for every run.
- `psr_dsr_pvalues.csv`, `hac_sharpe_confidence_intervals.csv`, SPA result tables
  (per-strategy and supremum p-values), forest/heatmap PNGs, and
  `paper_results.md`.

## Testing

Run the lightweight smoke suite:

```bash
pytest
```

For CI environments without CUDA, override `hardware=auto` and `training=smoke`
to keep runtimes bounded.
