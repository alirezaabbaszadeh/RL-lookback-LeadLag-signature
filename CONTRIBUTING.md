# Contributing Guide

Thanks for your interest in improving the LeadLag Signature RL Platform! This document outlines how to set up your environment, run validation checks, and submit high-quality changes.

## 1. Prerequisites
- Python 3.10+
- Git and a GitHub account
- (Optional) Conda for isolated environments

## 2. Environment Setup
Clone the repository and install dependencies. The lean stack used for Kaggle and CI is recommended for development:

```bash
git clone <repo-url>
cd RL-lookback-LeadLag-signature
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-kaggle.txt
```

If you need optional integrations (Stable-Baselines3, MLflow, PyTorch), install them from `requirements.txt` or append extras manually.

## 3. Coding Standards
- Follow existing code patterns; prefer explicit logging via `reporting/logging_utils.get_logger`.
- Keep configs declarative (YAML) and document new fields in `docs/config_reference.md` when applicable.
- Add targeted tests in `tests/` for new modules or behaviours.
- Maintain ASCII text unless a file already uses Unicode.

## 4. Quality Checks
Before submitting a pull request:

```bash
python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv  # adjust path if using custom data
python scripts/smoke_kaggle.py --output-root dist/kaggle_smoke            # add --keep-meta-rl/--keep-offline if relevant
pytest -q
```

Ensure generated artifacts (`results/`, `dist/`) are excluded from commits.

## 5. Documentation
- Update `README.md`, `CHANGELOG.md`, or relevant docs (`docs/`) when introducing new features.
- Describe new scenarios or datasets in the roadmap (`docs/future_roadmap.pseudo`) if they affect planning.

## 6. Submitting Changes
1. Create a feature branch: `git checkout -b feat/<short-description>`.
2. Commit logically separated changes with descriptive messages.
3. Rebase onto the latest main branch before opening a PR.
4. Provide a summary of changes, test results, and any remaining TODOs in the PR description.

## 7. Release Contributions
If you are cutting a release:
- Update `CHANGELOG.md` with the new version section.
- Run smoke tests/governance checks.
- Tag the release (`git tag -a vX.Y.Z`) and push tags after approval.

We appreciate your contributions—thank you for helping make the platform better!
