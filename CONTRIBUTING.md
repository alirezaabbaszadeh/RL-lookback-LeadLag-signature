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
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-kaggle.txt
python -m pip install -r requirements-dev.txt
```

`pyproject.toml` defines optional extras for reinforcement learning, signature
features, and MLflow exporters. Install them on-demand:

```bash
python -m pip install -e '.[rl,signature]'
python -m pip install -e '.[mlflow]'
```

The requirements files in the root directory (`requirements.txt`,
`requirements-rl.txt`) mirror the combinations used by CI pipelines and Kaggle
notebooks.

### CLI smoke check

After installing, verify the primary entry points to ensure the environment is
wired correctly:

```bash
leadlag --list --format json
python main.py --status --results-root results --format text
python hydra_main.py --scenario fixed_30 --output_root results
leadlag-full-suite results_root=results/full_suite training=smoke
```

See [`docs/config_reference.md`](docs/config_reference.md) and
[`reporting/`](reporting) for deeper usage patterns, including ablation helpers
and reporting CLIs.

## 3. Coding Standards
- Follow existing code patterns; prefer explicit logging via `reporting/logging_utils.get_logger`.
- Keep configs declarative (YAML) and document new fields in `docs/config_reference.md` when applicable.
- Add targeted tests in `tests/` for new modules or behaviours.
- For CLI changes that touch formatted output, regenerate or add coverage in `tests/test_cli_formatter_outputs.py` and validate JSON envelopes with `python scripts/validate_cli_payload.py <payload.json>`.
- Maintain ASCII text unless a file already uses Unicode.

## 4. Quality Checks
Before submitting a pull request:

```bash
pre-commit install
pre-commit run -a
python scripts/audit/dataset_quality.py --path raw_data/daily_price.csv  # adjust path if using custom data
python scripts/smoke_kaggle.py --output-root dist/kaggle_smoke            # add --keep-meta-rl/--keep-offline if relevant
pytest -q
```

Ensure generated artifacts (`results/`, `dist/`) are excluded from commits.

## 5. Documentation
- Update `README.md`, `CHANGELOG.md`, or relevant docs (`docs/`) when introducing new features.
- Coordinate roadmap updates (see
  `archive/2025-10-19-roadmap/docs/future_roadmap.pseudo` for the last
  snapshot) if new scenarios or datasets affect planning.

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
