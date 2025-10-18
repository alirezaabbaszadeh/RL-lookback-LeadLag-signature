# Kaggle Deployment Checklist

Use this guide to package the project for Kaggle notebooks or competitions. The steps assume you are working from a clean git tree and have already generated results locally.

## 1. Curate Dependencies
- Install the lean stack defined in `requirements-kaggle.txt`:
  ```bash
  pip install -r requirements-kaggle.txt
  ```
- Optional extras (MLflow, Stable-Baselines3, PyTorch) are listed at the bottom of the same file; enable them only if the Kaggle runtime needs RL training or remote logging.

## 2. Stage Input Artifacts
- Copy required configuration files (`configs/`), scripts, and the minimal subset of `raw_data/` into a folder that will be uploaded as Kaggle dataset input.
- When private data cannot be shared, include a synthetic sample (e.g., generated via `research/meta_rl/run_meta_rl.py`) so that pipelines still execute.
- Ensure any saved models or checkpoints fit within Kaggle’s 20 GB dataset limit.

## 3. Prepare Execution Script
- The all-in-one helper `kaggle/starter.py` mirrors the typical workflow:
  ```bash
  python kaggle/starter.py --scenario fixed_30 --run-meta-rl --run-offline --output-root /kaggle/working
  ```
- If you prefer individual calls (e.g., inside notebook cells), invoke the specific entry points:
  ```bash
  python research/meta_rl/run_meta_rl.py --output-root /kaggle/working/meta_rl --samples 250
  python research/offline_rl/log_trajectories.py --episodes 3 --output /kaggle/working/offline/offline_dataset.h5
  python research/offline_rl/train_offline.py --dataset /kaggle/working/offline/offline_dataset.h5
  ```
- For ablation studies (signature baselines, dynamic, RL, and random controls) use the pipeline helper:
  ```bash
  python pipelines/run_ablation.py --output-root /kaggle/working/ablations
  ```
  Install optional dependencies (`stable-baselines3`, `torch`) when including RL-heavy presets, or pass `--skip-missing-deps` to skip them.
- To execute the entire suite (baseline+meta+offline, audits, ablations, reporting) in one shot:
  ```bash
  python pipelines/run_full_suite.py --output-root /kaggle/working/full_suite
  ```
  Use flags such as `--skip-ablation`, `--skip-meta-offline`, or `--skip-audit` to tailor the run. Add `--skip-optional-deps` if RL extras are unavailable.
- Persist outputs only under `/kaggle/working` so that Kaggle captures them as notebook results.

## 4. Validate Without Network
- Run the bundled smoke test to ensure all entry points succeed without internet access:
  ```bash
  python scripts/smoke_kaggle.py --output-root dist/kaggle_smoke --keep-meta-rl --keep-offline
  ```
- Execute governance checks prior to export:
  ```bash
  python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv
  pytest -q tests/smoke/
  ```

## 5. Notebook Template
- Create a Kaggle notebook that performs the following in order:
  1. `!pip install -r /kaggle/input/<dataset>/requirements-kaggle.txt`
  2. Copy configs or data into `/kaggle/working` as needed.
  3. Invoke the chosen scenario or research script.
  4. Display key artifacts (e.g., `summary.csv`, plots) to document success.
- Save the notebook with “Internet Disabled” to confirm portability.

## 6. Final Review
- Double-check `run_metadata.json` and `data_manifest.json` for each generated output directory before uploading to Kaggle.
- Update `README` or release notes with the exact commands used so peers can reproduce the Kaggle run.
- Tag the git commit used for export so future iterations can diff changes quickly.

Following this checklist ensures the repository is packaged with the minimal dependency footprint, reproducibility metadata, and clear run instructions required by Kaggle’s execution environment.
