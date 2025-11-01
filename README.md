# Reactor v3 LeadLag Platform

GPU-first reinforcement-learning research stack for lead–lag signature trading.
Reactor v3 standardises the code layout, Hydra configuration surface, and Kaggle
notebook workflow so that a single pipeline produces reproducible, paper-grade
artefacts.

## Highlights

- **Single entry point** – `python -m leadlag.pipelines.run_full_suite` orchestrates
  training, evaluation, and reporting with Hydra overrides.
- **GPU aware** – Defaults to CUDA with AMP enabled; automatically falls back to
  CPU when unavailable.
- **Leakage-safe CV** – Purged/embargoed walk-forward splits, ready for nested
  tuning.
- **Standard outputs** – Canonical `metrics.csv`, equity curves, trade summaries,
  and JSON manifests under `/kaggle/working/results/<run_id>/`.
- **Paper-grade statistics** – HAC confidence intervals, PSR/DSR, SPA-lite, and a
  model confidence set CLI automatically exported to `paper_outputs_root` after
  each run (the CLI remains available for ad-hoc aggregation).
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
- `requirements*.txt` – dependency pins for Kaggle and optional stacks.
- `tests/` – unit coverage for CV, stats, features, and reporting utilities.
- `notebooks/` – Jupyter notebooks mirroring the Kaggle flow (for local review).

## Quick Start (Local)

```bash
pip install "git+https://github.com/<owner>/<repo>@<COMMIT_SHA>"
python -m leadlag.pipelines.run_full_suite \
  agent=ppo training=smoke hardware=auto \
  data.dataset_dir=/path/to/dataset \
  +logging.run_id=local-ppo-smoke
```

Hydra overrides mirror the configuration tree under `leadlag/configs`. Example toggles:

```bash
python -m leadlag.pipelines.run_full_suite \
  agent=sac env.action_space=continuous \
  features=signature_leadlag features.signature.depth=3 \
  split=walk_forward_purged training=paper hardware=gpu \
  data.universe=crypto_top data.dataset_dir=/datasets/crypto \
  +logging.run_id=paper-sac-crypto
```

## Kaggle Notebook Workflow (Official Path)

1. **GPU check**
   ```python
   import torch, os
   print("CUDA:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   !nvidia-smi -L
   ```
2. **Install pinned commit**
   ```python
   !pip -q install "git+https://github.com/<owner>/<repo>@<COMMIT_SHA>"
   !pip -q install hydra-core stable-baselines3 gymnasium numpy pandas scipy matplotlib statsmodels
   ```
3. **Configure intensity & data** – choose `training=smoke` or `training=paper`,
   set `data.dataset_dir=/kaggle/input/<your-dataset>`, and enforce CUDA via
   `hardware=gpu`.
4. **Run experiments**
   ```python
   !python -m leadlag.pipelines.run_full_suite \
       agent=ppo env.action_space=discrete3 policy=mlp \
       features.signature=off features.leadlag=off features.time_channel=off \
       window.lookback=128 target.horizon=1 \
       data.universe=sp500_sector data.timeframe=5m \
       data.dataset_dir=/kaggle/input/your-data \
       split=walk_forward_purged training=smoke hardware=gpu \
       costs.fee_bps=1 slippage.bps=2 \
       +logging.run_id=smoke-ppo-0
   ```
5. **Aggregate metrics**
   ```python
   import glob, os, pandas as pd
   rows=[]
   for run_dir in sorted(glob.glob('/kaggle/working/results/*')):
       metrics_path=os.path.join(run_dir, 'metrics.csv')
       if os.path.exists(metrics_path):
           rows.append(pd.read_csv(metrics_path))
   all_metrics=pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
   all_metrics.to_csv('/kaggle/working/paper_outputs/all_metrics_raw.csv', index=False)
   all_metrics.head()
   ```
   _Note: `run_full_suite` already materialises these artefacts; rerun the CLI to
   customise bootstrap parameters or regenerate plots on demand._
6. **Paper-grade statistics**
   ```python
   !python -m leadlag.eval.stats_cli \
       --results /kaggle/working/results \
       --out /kaggle/working/paper_outputs
   ```

All notebook cells write outputs beneath `/kaggle/working/`. The consolidated
`paper_outputs` folder contains summary tables, confidence intervals, SPA/MCS
p-values, and plots ready for publication.

## Standard Outputs

Per-run directory (`/kaggle/working/results/<run_id>/`):
- `metrics.csv` – canonical schema for aggregation (includes `EnvSteps` for
  auditing equal interaction budgets).
- `equity.csv` – equity curve per timestamp.
- `returns.csv` – per-step returns.
- `run_manifest.json` – seeds, device info, package versions, and resolved config.

Aggregate directory (`/kaggle/working/paper_outputs/`):
- `all_metrics_raw.csv` – concatenated metrics for every run.
- `*_pvalues.csv`, `*_ci.csv`, forest/heatmap PNGs, and `paper_results.md`.

## Testing

Run the lightweight smoke suite:

```bash
pytest
```

For CI environments without CUDA, override `hardware=auto` and `training=smoke`
to keep runtimes bounded.
