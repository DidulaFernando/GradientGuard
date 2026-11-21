# Author and supervisor review notes

## Main positive observations retained

- In the historical source-normal held-out pool, L6/L8 average precision is approximately 0.934 versus E's 0.775; AUROC is approximately 0.962 versus 0.816.
- At twelve audits, L6/L8 find ten failures versus E/B's eight. The same 10-versus-8 pattern occurs with the frozen A predictor on panel B and the previously fitted B pipelines.
- L6/L8 recover all twelve observed failures by seventeen inspections; E requires twenty-five and B twenty-eight. These are retrospective full-recall points, not known deployment thresholds.
- The continuous secondary models show MAE 0.1588 for L6 and 0.1473 for L8 versus E's 0.1650. These models are distinct from the binary audit forecasters.
- Correctness checks, frozen predictor/outcome boundaries, numerical reproduction and complete 60-epoch records support engineering credibility.

## Evidence that remains visible

- The original L8 primary endpoint was nine audits, where E/B/L6/L8 all found eight failures. L6 and twelve audits cannot be retroactively labelled the preregistered primary finding.
- The family interval includes zero, only three eligible held-out families contain failures, and one family removal reverses the larger-capacity comparison.
- The full cohort has budgets favouring conventional prediction. Existing nonlinear defaults weaken claims against a strong nonlinear baseline.
- Added slope features were retained but did not change the primary discovery curve; they have not demonstrated a directional temporal contribution.
- No matched rich structural-control or external-benchmark result exists. The proposal's future plans are not new evidence.

## Before submission

1. Confirm exact author names/order, institutional affiliations, corresponding-author email, authorship eligibility and both authors' approval. Maninda Edirisooriya was explicitly requested as coauthor; no contribution details were invented.
2. Review the complete manuscript and references against the archived code/results. In particular, verify version-sensitive close prior work and refresh the literature search.
3. Retain the acknowledgment of AI-generated drafting/formatting assistance, adjusting its scope accurately after author revision. Also review and disclose the extent of AI assistance in the original experimental software where applicable; the current acknowledgment describes assistance in this manuscript task. IEEE Access's current author guidance requires disclosure of generated content.
4. Replace/verify the accessible legacy template mirror against the current official IEEE Access template. Do not invent DOI, publication history, grant support or membership labels.
5. Decide with the supervisor whether the present exploratory contribution is suitable for submission or should be strengthened with the separately designed validation study. No score or acceptance probability is asserted.
6. Update any eventual availability statement if full dataset arrays or model binaries are hosted elsewhere. The current repository explicitly does not contain those local-only artifacts.

No journal submission, upload to a manuscript portal or public GitHub push is performed by this drafting task.
