# TMLR Submission Readiness Roadmap (2025–2027)

This document consolidates the latest publicly available guidance from **Transactions on Machine Learning Research (TMLR)** and outlines a preparation plan for the LeadLag project to meet and anticipate journal expectations through 2027.

## 1. Snapshot of Official 2025 Requirements

### 1.1 Scope and Eligible Contributions
- TMLR focuses on advances that deepen the understanding of computational and mathematical principles behind learning in biological or artificial systems, including new algorithms, theoretical analyses, reproducibility studies, and insightful surveys.
- Submissions must present original work; expanded versions of conference publications or text reused from archival venues are not allowed, though non-archival workshop or preprint overlap is acceptable.

### 1.2 Reviewing and Editorial Process
- Manuscripts are handled on OpenReview under a double-blind, open-review model with public visibility of the paper, reviews, and discussion once the reviewer set is complete.
- The review pipeline proceeds through submission, reviewer assignment, rebuttal/discussion, and a decision of *accept as is*, *accept with minor revision*, or *reject*; rejected papers may be resubmitted as new submissions with explicit change logs.

### 1.3 Acceptance Criteria
- Acceptance hinges on two questions: (1) Are the submission's claims supported by accurate, convincing, and clear evidence? and (2) Would at least a subset of the TMLR audience be interested in the findings?
- Reviewers are instructed not to rely on novelty or benchmark state-of-the-art claims; thorough, transparent studies with actionable insights are welcomed even when contributions are incremental.

### 1.4 Author Responsibilities and Formatting
- Authors must anonymize submissions, disclose conflicts, provide IRB and funding statements, and use the mandatory TMLR LaTeX template. Appendices and up to 100 MB of anonymized supplementary material are permitted at reviewer discretion.
- Broader impact statements are required when work carries significant risk of harm, and all submissions are licensed under CC BY 4.0 from initial submission through publication.

## 2. Anticipated Evolution Toward 2026–2027

The following trends extrapolate from 2025 guidance and broader community shifts; they should be treated as working hypotheses that inform our roadmap.

1. **Stricter Reproducibility and Compute Accounting.** Expect mandatory release of training/evaluation scripts, deterministic seeds, and compute cost summaries (e.g., carbon footprint, GPU hours) as reproducibility certifications become mainstream.
2. **Expanded Ethical and Societal Impact Reporting.** Broader impact statements may become compulsory for all submissions, with structured templates covering misuse risks, mitigation plans, and human subjects safeguards.
3. **Transparent Data Governance.** Provenance tracking, licensing audits, and documentation of data curation pipelines are likely to be enforced to address data ethics and privacy expectations.
4. **Sustained OpenReview Engagement.** Continuous interaction during rebuttal phases (public clarifications, community Q&A) will increasingly influence certification awards and perceived impact.
5. **Holistic Evaluation Metrics.** Beyond accuracy, submissions will be pushed to report fairness, calibration, financial risk (for trading domains), and uncertainty quantification metrics to capture real-world reliability.

## 3. LeadLag Project Preparation Blueprint

### 3.1 Immediate Actions (Q4 2025)
- **Documentation Audit:** Align README, CONTRIBUTING, and pipeline docs with TMLR scope/acceptance criteria, highlighting how claims are evidence-backed and why the financial RL community should care.
- **Artifact Packaging:** Finalize wheel-building and Hydra config distribution so reviewers can reproduce experiments directly from releases, satisfying the open-review expectation for accessible artifacts.
- **Ethics Checklist:** Add a living checklist covering data sourcing, potential trading harms, and mitigations to streamline broader impact drafting.

### 3.2 Mid-Term Refactors (H1 2026)
- **Reproducibility Pipeline:** Automate run manifests (seeds, env steps, hardware), compute usage logs, and deterministic evaluation harnesses for each experiment window.
- **Data Lineage Registry:** Version datasets, feature transforms, and preprocessing scripts with checksums and licenses to future-proof provenance reporting.
- **Risk & Calibration Metrics:** Integrate turnover, exposure, drawdown, and calibration outputs into canonical `metrics.csv` for richer evidence packages.

### 3.3 Long-Term Research Directions (H2 2026–2027)
- **Claim Strengthening:** Target claims around lead-lag signal interpretability, cross-market generalization, and sample-efficiency gains validated across multi-year, multi-asset datasets.
- **Benchmark Transparency:** Contribute standardized financial RL benchmarks (data splits, evaluation code) under permissive licenses to lead community adoption and bolster interest criteria.
- **Societal Impact Program:** Establish internal review of ethical risks (market manipulation, unequal access) with mitigation protocols and public documentation to support stronger impact statements.

### 3.4 Governance & Communication
- Schedule quarterly TMLR-readiness reviews with maintainers to update this roadmap, track policy changes, and ensure the CLI/automation tooling remains aligned with agent instructions.
- Maintain a shared knowledge base (meeting notes, reviewer feedback, certification targets) within the `research/` folder to shorten future submission cycles.

## 4. 2025 Q4 Compliance Update

TMLR's October 2025 bulletin emphasises verifiable research artefacts, public metadata, and responsible compute disclosures. The LeadLag project has adopted the following measures to stay aligned:

### 4.1 Metadata & Citation Enhancements
- `CITATION.cff` now publishes the canonical title, maintainer affiliations, OpenReview URL, and the minted arXiv DOI so reviewers can attribute the software artefact directly.
- Release notes must link to the citation entry and surface the submission date to satisfy the "public metadata within two weeks" requirement introduced in the bulletin.

### 4.2 Reproducibility & Compute Accountability
- Reporting pipelines are being expanded to attach GPU hour summaries and carbon estimates alongside the existing HAC/SPA statistics; this will become a hard requirement for the next tagged release.
- Hydra experiment manifests must include dataset licences and checksum attestations before scenarios are considered submission-ready.

### 4.3 Responsible Disclosure Checklist
- Governance documents now capture structural changes affecting compliance-critical modules (reporting, dataset governance) so that quarterly readiness reviews can track regressions.
- Broader impact drafts must reference the updated risk taxonomy (market manipulation, unequal access, systemic bias) mandated in the October bulletin.

---

**Action Item:** Keep this document synchronized with official TMLR pages and evolving community practices; when policies update, capture diffs in `CHANGELOG.md` and alert the engineering team via `agent.md`.
