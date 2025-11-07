# Anonymous Reproduction Instructions

This archive contains the minimal, anonymised snapshot required to reproduce the
results reported in the accompanying manuscript. All organisation- or
contributor-identifying metadata has been removed.

## Contents

- `src/leadlag/`: Lead-lag/signature training and evaluation package
- `scripts/reproduce_all.sh`: One-click pipeline for Kaggle/CI
- `pyproject.toml` + `requirements*.txt`: Build and dependency metadata
- `README_ANON.md`: This document (safe to include in anonymous submissions)

## Kaggle Notebook Workflow

1. Upload the generated `artifact_anonymous.zip` as a Kaggle Dataset.
2. Start a new Kaggle Notebook with **GPU** and **Internet** enabled.
3. In the first cell, unpack the archive and install it in editable mode:

   ```bash
   !unzip /kaggle/input/<dataset-name>/artifact_anonymous.zip -d /kaggle/working/artifact
   !pip install -e /kaggle/working/artifact
   ```

4. Run the one-click pipeline (writes to `/kaggle/working/paper_outputs/`):

   ```bash
   !bash /kaggle/working/artifact/scripts/reproduce_all.sh
   ```

5. Download the contents of `/kaggle/working/paper_outputs/` for submission.

## Notes

- The wheel packaging path is fully self-contained; no external credentials or
  proprietary datasets are required.
- Random seeds and device manifests are recorded automatically in each run
  directory to facilitate auditability.
- When publishing the Kaggle Dataset, ensure its description does not include
  author names, affiliations, or non-anonymised repository URLs.
