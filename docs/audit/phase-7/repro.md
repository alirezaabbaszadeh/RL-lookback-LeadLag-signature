# Phase 7 - Reproducibility Metadata (Completed 2025-10-19)

## Summary
- Every runner now emits `data_manifest.json` with SHA-256 of the source dataset, row/asset counts, coverage window, and quality findings (`governance/dataset.py`).
- `run_metadata.json` captures Git commit, branch, dirty flag, and the path used to load prices (`training/run_scenario.py:202-245`, `training/run_dynamic_baselines.py:121-147`, `training/run_rl.py:139-157`).
- The governance audit CLI `scripts/audit/dataset_quality.py` verifies missing ratios, duplicate indices, and zero-variance series before release packaging (documented in `docs/deployment/kaggle_setup.md`).
- The Kaggle smoke pipeline (`scripts/smoke_kaggle.py`) and README quick start ensure deterministic seeds and output directories for automated reruns.

## Evidence
- Sample manifest: `results/wf_full_20251017_181105/data_manifest.json` (hash + coverage window).
- Metadata completeness checks baked into `scripts/audit/validate_artifacts.py` (warns if hashes or commits are absent).
- Reproducibility steps summarised in `docs/repro.md` and enforced in `CONTRIBUTING.md` (smoke + pytest + dataset audit).

## Outcome
Acceptance criteria met: reruns are deterministic (seeded scenarios), metadata captures dataset hash and commit, and end-of-life policy is documented for optional extras (README + deployment guide). Phase 7 is **completed**.
