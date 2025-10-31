# Documentation maintenance workflow

This guide explains how to keep the documentation set healthy between releases and how to prepare it for a tagged build.

## Weekly hygiene tasks

1. **Triage open documentation issues.** Confirm that each ticket has an owner and a release milestone.
2. **Review merged PRs.** Scan recent commits for doc-impacting changes (new CLIs, renamed configs) and ensure they are captured in the docs backlog.
3. **Check automation.** Verify the "docs-quality" workflow ran on the default branch. Fix or triage any markdownlint or link-check failures promptly.
4. **Groom examples and notebooks.** Open the most frequently referenced notebooks to confirm that cells still execute with the pinned environment.

## Pre-release checklist

Follow this checklist alongside the [Docs release review template](../.github/ISSUE_TEMPLATE/docs-release-review.md) when preparing a release:

1. Run `markdownlint-cli2 "**/*.md"` locally to catch formatting issues early.
2. Run `lychee --config .lychee.toml` to check for broken or redirected links.
3. Audit `README.md`, `CONTRIBUTING.md`, and `docs/` content for outdated configuration flags or dataset references.
4. Ensure screenshots and plots reflect the version being released.
5. Confirm CHANGELOG entries link to valid resources (blog posts, model cards, etc.).
6. Update the release issue with any deferred documentation work.

## Tooling references

- **Markdown lint:** uses [`markdownlint-cli2`](https://github.com/DavidAnson/markdownlint-cli2) configured via `.markdownlint.yaml`.
- **Link checking:** handled by [`lychee`](https://github.com/lycheeverse/lychee) with options defined in `.lychee.toml`.
- **CI enforcement:** the `docs-quality` job in `ci.yml` runs on every push and pull request targeting `main` or `master`.

Keeping this workflow in sync with automation helps contributors spot documentation regressions before they ship.
