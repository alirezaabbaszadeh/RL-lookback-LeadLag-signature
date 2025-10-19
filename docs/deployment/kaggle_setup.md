Kaggle Setup and One-Command Grand Run
======================================

This guide shows how to execute every experiment (baselines, ablations, SB3 production training, Dopamine checks, audits, reports) on Kaggle with a single command. It also includes a ready-to-run notebook cell.

Prerequisites
-------------
- Kaggle notebook settings:
  - **Internet:** ON (wheel downloads)
  - **Accelerator:** GPU recommended for RL (CPU works for non-RL parts)
- Attach the project repository (or clone it) so that the files are available under `/kaggle/working/`.  
  If the repo lives inside a subfolder, you will switch into that directory before running the command.

One-Command Grand Run
---------------------
From the repository root (inside `/kaggle/working/...`):

```
!python kaggle/run_all.py
```

What happens:
1. Prefetches wheels into `/kaggle/working/wheelhouse` (for fast installs).
2. Configures pip to read from the wheelhouse/cache.
3. Runs the orchestrator stages in order:
   - `full_suite` – calls `pipelines/run_full_suite.py` (baselines, ablations, meta-RL, offline RL, dataset audits, aggregate reports, plots).
   - `sb3_leadlag` – production SB3 v2.1.0 + Gymnasium 0.29.1 training on the LeadLag environment (PPO, PPO-LSTM, attention policy). Defaults: `SB3_DEVICE=auto`, `SB3_TIMESTEPS=300000` (override via env vars if required).
   - `dopamine` – Gymnasium 1.x stack (`gymnasium==1.0.0`, `dopamine-rl==4.1.2`) sanity check.
4. Bundles outputs as `/kaggle/working/multi_stage_artifacts.zip`.

If you want to skip the prefetch step (slower installs):

```
!python kaggle/run_all.py --no-prefetch
```

Sample Notebook Cell (copy/paste)
---------------------------------
Use this cell when the repository is attached as a Dataset named `leadlag-signature` (adjust the dataset path if needed):

```
%%bash
set -e
WORK=/kaggle/working
cd "$WORK"

# Copy repo if it was attached as a dataset
if [ ! -f kaggle/run_all.py ]; then
  cp -r /kaggle/input/leadlag-signature/* "$WORK"/
fi

# If the repo lives inside a subdirectory, cd into it
if [ ! -f kaggle/run_all.py ]; then
  REPO=$(find "$WORK" -maxdepth 2 -type f -name run_all.py | head -n 1 || true)
  if [ -z "$REPO" ]; then
    echo "[error] run_all.py not found. Attach the repo dataset or clone it first." >&2
    exit 1
  fi
  cd "$(dirname "$(dirname "$REPO")")"
fi

python -m pip install --upgrade pip
mkdir -p wheelhouse .cache/pip

# Prefetch wheels for speed (requirements core + SB3 2.x + Gymnasium + Dopamine stack)
python -m pip download -d wheelhouse -r requirements-kaggle.txt
python -m pip download -d wheelhouse "gymnasium==0.29.1" "stable-baselines3==2.1.0" "sb3-contrib==2.1.0" "torch>=2.0"
python -m pip download -d wheelhouse "dopamine-rl==4.1.2" "gymnasium==1.0.0"

export PIP_CACHE_DIR=/kaggle/working/.cache/pip
export PIP_FIND_LINKS=/kaggle/working/wheelhouse
export PIP_NO_INDEX=1

export SB3_DEVICE=auto
export SB3_TIMESTEPS=300000

python kaggle/run_all.py
zip -r multi_stage_artifacts.zip multi_stage_artifacts
```

Outputs
-------
- `/kaggle/working/multi_stage_artifacts/`
  - `full_suite/…` – core runs, ablations, robustness checks, reports, audit logs.
  - `sb3_leadlag/…` – production RL outputs (per scenario).
  - `dopamine/…` – Gymnasium 1.x sanity stage.
  - `summary.json` – stage status, duration, log paths.
- `/kaggle/working/multi_stage_artifacts.zip` – ready for download.

Environment Overrides (advanced)
--------------------------------
Use these **before** invoking `run_all.py` (or when running `run_multi_stage.py` manually):

- `SB3_DEVICE` – `cuda`, `cpu`, or `auto`
- `SB3_TIMESTEPS` – total timesteps (e.g., `600000`)
- `SB3_N_STEPS`, `SB3_BATCH_SIZE`, `SB3_LR`, `SB3_EVAL_FREQ`, `SB3_VERBOSE`, `SB3_SEED`
- Hydra stage (`leadlag_hydra`): `LEADLAG_SCENARIOS`, `LEADLAG_SEEDS`, `LEADLAG_MULTI_SEED`

Performance & Troubleshooting
-----------------------------
- Keep Internet ON for the prefetch step; if a wheel is missing, temporarily set `PIP_NO_INDEX=0`.
- Monitor `/kaggle/working/multi_stage_artifacts/<stage>/stderr.log` and `summary.json` for quick diagnostics if a stage fails.
- If pip cache becomes stale, clear `/kaggle/working/.cache/pip` and rerun.
- GPU with `SB3_DEVICE=cuda` significantly speeds up production RL; adjust timesteps to fit Kaggle’s time budget.
