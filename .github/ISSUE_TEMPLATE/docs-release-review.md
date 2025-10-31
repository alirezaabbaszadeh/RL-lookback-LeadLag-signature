---
name: "Docs release review"
about: "Run the recurring documentation quality checks before cutting a release."
title: "Docs review for release <version>"
labels:
  - documentation
  - release
assignees: []
---

## Preparation
- [ ] Confirm the targeted release tag and milestone are set.
- [ ] Open the latest rendered docs (site, README, notebooks) and note any stale screenshots or instructions.

## Linting and automation
- [ ] Run `markdownlint-cli2 "**/*.md"` locally and fix any remaining warnings.
- [ ] Run `lychee --config .lychee.toml` and resolve broken links or add justifications to the config.
- [ ] Ensure the CI "docs-quality" job is green on the release branch.

## Content freshness
- [ ] Review `docs/MAINTENANCE.md` and follow the grooming checklist.
- [ ] Verify CHANGELOG and release notes reference the latest docs changes.
- [ ] Confirm tutorials and notebooks mention the current default model checkpoints and data sources.

## Wrap-up
- [ ] File issues for any deferred clean-up work and link them here.
- [ ] Post a short summary in the release tracking discussion.
