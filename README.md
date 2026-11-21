# GradientGuard

Research implementation for early model-level prioritisation of subgroup-failure audits.

**Research question:** Can information about which layers a seemingly healthy model depends on help us choose better models to inspect for hidden subgroup failures?

"Depends on" denotes the measured allocation of local connection sensitivity, not a causal explanation of image-feature reliance. This repository preserves the completed feasibility studies, exploratory outcome proof of concept and post-hoc diagnostics. It does not claim established novelty or confirmed predictive superiority.

## Studies and evidence

| Stage | Implementation | Results and interpretation |
| --- | --- | --- |
| First measurement-feasibility study | [Harness](Feasibility/src), [configuration](Feasibility/configs/pilot.yaml), [tests](Feasibility/tests) | [Report](Feasibility/reports/20261116_01/REPORT.md), [decision](Feasibility/reports/20261116_01/decision.json) |
| Targeted checkpoint/reliability follow-up | [Package and protocol](Feasibility/followup/README.md) | [Report](Feasibility/followup/outputs/20261116_03/reports/REPORT.md), [decision](Feasibility/followup/outputs/20261116_03/reports/decision.json) |
| Exploratory outcome PoC | [Study](Feasibility/poc/README.md) | [Report](Feasibility/poc/outputs/20261116_poc01/reports/REPORT.md), [decision](Feasibility/poc/outputs/20261116_poc01/reports/decision.json) |
| Post-hoc capacity, slope and robustness diagnostics | [Diagnostic attempt](Feasibility/diagnostics/20261118_01/README.md) | [Report](Feasibility/diagnostics/20261118_01/reports/REPORT.md), [decision](Feasibility/diagnostics/20261118_01/reports/decision.json) |

The PoC completed 96 classifier trajectories. Its historical source-normal held-out pool had 36 models and 12 failures. At the original nine-audit budget E, B, L6 and L8 each found eight failures. At the post-hoc twelve-audit view E/B found eight and L6/L8 ten. Family uncertainty, an influential-family reversal and unresolved structural interpretation prevent treating that observation as a confirmed advantage. Read the study reports for the complete qualifications.

## Reproduce and inspect

- [Environment settings](Feasibility/instructions/environment.ps1)
- [First-study instructions](Feasibility/instructions/STEP_BY_STEP.txt)
- [Follow-up instructions](Feasibility/instructions/STEP_BY_STEP_FOLLOWUP.txt)
- [PoC instructions](Feasibility/instructions/STEP_BY_STEP_POC.txt)
- [Diagnostic instructions](Feasibility/instructions/STEP_BY_STEP_DIAGNOSTICS.txt)
- [Repository workflow and artifact policy](docs/REPOSITORY_WORKFLOW.md)

Exact dependency versions, hardware/provenance and frozen source snapshots are retained with each attempt. Historical paths and hashes intentionally describe the original Windows workspace. A clone does not contain its virtual environment, downloaded benchmark images, generated numerical dataset arrays, training weights or scalar SQLite caches; those remain preserved in the original workspace. Compact ID manifests and diagnostic metadata are included. Follow the corresponding preparation instructions and verify the pinned environment before rerunning; never execute legacy commands against protected attempts merely to refresh their reports.

The original implementation was imported in logical commits on 21 November 2026. These are archival import commits, not fabricated historical development timestamps. Proposal documents live outside this repository and are not part of this import.
