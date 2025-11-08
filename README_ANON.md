# Anonymous Reproduction Instructions

This archive contains the minimal, anonymised snapshot required to reproduce the
results reported in the accompanying manuscript. All organisation- or
contributor-identifying metadata has been removed.

## Contents

- `src/leadlag/`: Lead-lag/signature training and evaluation package
- `scripts/reproduce_all.sh`: One-click pipeline for Kaggle/CI
- `pyproject.toml` + `requirements*.txt`: Build and dependency metadata
- `docs/DATA_CARD.md`: Dataset provenance, licensing, and split policy
- `docs/MODEL_CARD.md`: Model assumptions, limitations, and compute profile
- `README_ANON.md`: This document (safe to include in anonymous submissions)

## Kaggle Notebook Workflow

The repository ships `docs/kaggle_camera_ready.ipynb` with the exact cells that
reviewers should execute. Follow this sequence on Kaggle (GPU + Internet ON):

1. Upload the generated `artifact_anonymous.zip` as a Kaggle Dataset and attach
   it to the notebook.
2. Open `docs/kaggle_camera_ready.ipynb` from the archive (either upload it as a
   Kaggle notebook file or copy the cells into a new notebook).
3. Run each cell in order:
   - **Cell 1 – Unpack `artifact_anonymous.zip`**: automatically discovers the
     uploaded dataset and extracts the repository into
     `/kaggle/working/artifact`.
   - **Cell 2 – Install the package**: executes `pip install -e
     /kaggle/working/artifact`.
   - **Cell 3 – Capture environment versions**: prints Python, platform,
     `leadlag-signature-rl`, and Torch diagnostics for the run log.
    - **Cell 4 – Run the camera-ready pipeline**: invokes
      `scripts/reproduce_all.sh` with its defaults (`RES=/kaggle/working/results`
      and `OUT=/kaggle/working/paper_outputs`).
    - **Cell 5 – Capture a status snapshot**: runs
      `leadlag --status --results-root /kaggle/working/results --format json` (or
      redirects it to `/kaggle/working/run_status.json`) to record the envelope
      in the run log. The structure mirrors the CI validation contract.
    - **Cell 6 – Preview paper outputs**: lists
      `/kaggle/working/paper_outputs` and prints `paper_status.txt`.
4. Download the contents of `/kaggle/working/paper_outputs/` for submission.

## Notes

- The wheel packaging path is fully self-contained; no external credentials or
  proprietary datasets are required.
- Random seeds and device manifests are recorded automatically in each run
  directory to facilitate auditability.
- Refer to the data and model cards (`docs/DATA_CARD.md`, `docs/MODEL_CARD.md`) when summarising resources in
  supplementary materials or reviewer questionnaires.
- When publishing the Kaggle Dataset, ensure its description does not include
  author names, affiliations, or non-anonymised repository URLs.
- Use `make clean` or `make distclean` before creating a fresh archive to strip
  caches, temporary JSON summaries, and local experiment directories from the
  bundle.
