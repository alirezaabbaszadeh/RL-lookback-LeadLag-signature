# Next Iteration Plan

This planning stub records the immediate follow-up tasks for the lead-lag/signature reinforcement learning refactor. Any work performed within this directory should keep the checklist below current and mark items complete as they land in main.

## Action Tasks

1. **Hydra Packaging Sweep**  
   - Finalise the packaged config tree under `src/leadlag/configs/` per the roadmap.  
   - Ensure wheel packaging includes all YAML artifacts and run the compose smoke tests.
2. **Purged Cross-Validation Module**  
   - Implement the purged walk-forward splitter and backfill the Hydra profile metadata.  
   - Add overlap guards in `tests/test_purged_cv.py`.
3. **Trading Realism Enhancements**  
   - Enforce next-bar execution, add configurable costs/slippage, and log turnover/exposure in metrics.  
   - Create synthetic regression tests that validate the slippage impact.
4. **Reporting & Reproducibility**  
   - Ship the canonical `metrics.csv` writer, stats/calibration CLIs, and manifest utilities.  
   - Update CI to build wheels, execute entry points, and validate schema.
5. **Artifact Automation**  
   - Script `reproduce_all.sh` and the anonymous packaging workflow for Kaggle/OpenReview delivery.  
   - Document the anonymised run instructions in `README_ANON.md`.

Keep updates concise: when completing a task, add a dated bullet noting the PR/commit that satisfied the requirement.
