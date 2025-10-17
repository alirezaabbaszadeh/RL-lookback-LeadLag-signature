Reproducibility Guide

This guide describes how to reproduce baseline and RL experiments for the LeadLag-signature RL project.

1) Environment

- Python 3.10 recommended
- Create Conda environment:

```
conda env create -f environment.yml
conda activate leadlag
```

Alternatively, use Docker (see Dockerfile below).

2) Quick Runs

- Single scenario (no Hydra installed):

```
python hydra_main.py --scenario fixed_30 --output_root results
```

- Multiple scenarios with multi-seed aggregation:

```
python hydra_main.py --scenarios fixed_30 fixed_90 --multi_seed_enabled --seeds 42 52 62 --output_root results
```

Artifacts will be written under `results/`.

3) Using Preset Scenarios

- `fixed_30`, `fixed_90`, `dynamic_adaptive`, `rl_ppo`, `fast_smoke` are available in code presets.
- To see all preset names programmatically:

```
from hydra_main import get_available_scenarios
print(get_available_scenarios())
```

4) Outputs

- Per-run: `config_merged.yaml`, `run_metadata.json`, `metrics_timeseries.csv`, `summary.csv`, `fig_*.png` (optional), `profiles/*.{pstats,json}`
- Aggregates (multi-seed): `stats.csv`, `significance.csv`, `welch.csv` (when multiple scenarios compared), `runs.json`

5) Optional MLflow

If `mlflow` is installed, the code logs metrics and artifacts automatically.
Set `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` as needed.

6) Docker

Build image:

```
docker build -t leadlag-rl:latest .
```

Run (mount your data directory if needed):

```
docker run --rm -it -v %CD%:/workspace leadlag-rl:latest \
  python hydra_main.py --scenario fixed_30 --output_root results
```

7) Troubleshooting

- If plotting fails (no display/matplotlib), artifacts still generate; plots are optional.
- Some advanced features (RL training) require extra dependencies like `stable-baselines3`.
- Tests:

```
pytest -q
```

8) Repro Steps (<= 10)

1. `conda env create -f environment.yml && conda activate leadlag`
2. `python hydra_main.py --scenario fixed_30 --output_root results`
3. Inspect `results/` for artifacts; repeat with `--multi_seed_enabled --seeds 42 52 62` for aggregates.
4. Optional: set MLflow env, rerun for experiment tracking.
