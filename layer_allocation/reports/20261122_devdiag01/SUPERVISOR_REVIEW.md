# Supervisor-style review of the bounded diagnosis

22 November 2026 — advisory assessment for discussion with the actual supervisor.

**Decision: retain the project and its reproducible comparative study; pause the claim that the current six sensitivity features provide distinctive failure-prediction information. Do not launch a larger unchanged experiment next.**

## What the diagnosis establishes

The source-normal development cohort contains 107 models from 29 setting families. Fixed auxiliary regressions were trained on some development families to reconstruct the six sensitivity features in other development families. They did not predict subgroup-failure labels. All preprocessing remained inside the training folds.

The conventional/global bank B reconstructed these features poorly or moderately: panel-A R² ranged from -0.452 to 0.563. Adding norm-product allocation in N raised all six values to 0.929–0.958. Rich structural M achieved 0.861–0.972, with median 0.931; panel-B M values were 0.871–0.972. This is substantial evidence that much of the observed sensitivity-feature variation can be recovered from ordinary layer statistics in this development sample. It is not proof of exact mathematical redundancy or zero conditional outcome information. M need not reconstruct every feature better than N in a small fixed-regularisation regression, and better reconstruction is not equivalent to better failure prediction.

This helps explain the earlier pattern: adding layer summaries can help a predictor that lacks layer information, while offering little after a richer layer-statistics control is present. It is a plausible explanation, not a uniquely identified cause of the previous held-out result. Norm products and connection sensitivities use the same weights and mean gradients, making information overlap plausible even though their formulas are different.

Between-setting differences account for approximately 68%–92% of each feature's panel-A sample variation. The remaining 8%–32% is within setting families. Between-family information is legitimate for ranking a heterogeneous model pool; it is not inherently leakage or a disqualifying artifact. It does limit any stronger claim about distinguishing training seeds within the same setting.

## What remains uncertain

Only four source-normal development families contain both outcomes. Each has one failing model and three non-failing models: 16 models generate 12 dependent failure/non-failure pairs, of which four compare initialisations of the same source construction. The effective support is four mixed families, not 107 independent within-setting comparisons.

There are raw directional clues: the shallow endpoint orders 9 of 12 pairs in the same direction on each panel. But no raw or residual feature passes the bounded consistency check combining both panels, family omissions and the joint conditional-permutation adjustment. For example, residual deep endpoint concordance is about 0.917 on A but 0.583 on B. A favourable panel-A point is therefore insufficient. Exact enumeration used all 256 label assignments preserving the four family failure counts; full-cohort secondary enumeration used 4,096 assignments across six mixed families. These are post-hoc conditional diagnostics, not fresh confirmation.

The existing panel-A development OOF risk scores do not rescue the claim: equal-family within-setting concordance is 0.333 for both M and MZ. They were already used in development selection and are not independent validation. The small, coarse support prevents interpreting this as precise evidence that the models are systematically worse than random within settings. Likewise, the absence of an adjusted association does not prove that no residual relationship exists.

All previous scientific artifacts were preserved. Seventeen checks passed after a pre-freeze shape-handling repair; its failed test record is retained. The recorded dependency lock matched, 12,728 prior artifact hashes were verified unchanged, and the semantic-read audit recorded no evaluation/test outcome access, binary pipeline loading, classifier training or failure-forecaster fitting. There were 60 fixed auxiliary reconstruction fits, with no tuning grid. The distinction between implementation correctness and scientific usefulness remains essential.

## My recommendation

1. **Do not defend the current representation as an established sensitivity-specific methodological advance.** The combination of the previous held-out primary tie and this reconstruction diagnosis makes that claim difficult to justify. A reconstruction result is not itself a causal explanation or a novelty verdict.
2. **Keep a credible FYP contribution.** A reproducible system, analytically checked measurement, controlled comparison and carefully bounded negative result are substantial work. Position them as an investigation of when early layer summaries add information and when stronger structural monitoring absorbs their apparent benefit.
3. **Discuss a narrower research objective before more implementation.** One possible question is: “Can a compact layer-allocation representation provide a practical alternative to richer structural monitoring for subgroup-audit prioritisation, and where does it fail?” This is a proposed engineering/comparative direction, not a demonstrated result or an automatic novelty claim. Six features are compact; they are not automatically cheaper, equally accurate or more reliable. Any cost, non-inferiority or robustness claim needs its own justified measurements and independently evaluated criteria.
4. **If the project requires a novel superiority claim, require a conceptual change first.** Identify information that the present structural bank genuinely lacks and explain why it should predict subgroup failures. Do not simply search new feature combinations against the consumed evaluation outcomes. Any resulting experiment needs a separate protocol and fresh evaluation; no additional training is authorised by this diagnosis.

I would support completing the current comparative study and audit application while reconsidering the sensitivity-specific claim. I would not recommend another large SpuCoMNIST batch or a Waterbirds replication of the unchanged claim merely in the hope of a positive score. A materially different method can be considered with the actual supervisor if its rationale, independent evidence requirements and project timeframe are convincing.

The detailed numerical report is [REPORT.md](REPORT.md). No proposal, manuscript or previous study decision was modified. This assessment is not institutional approval, a journal acceptance forecast, or a claim that every use of layer sensitivity has been disproven.

An independent output audit can be repeated from the repository root with `& .\Feasibility\.venv\Scripts\python.exe .\layer_allocation\development_diagnostic_audit.py`. It recomputes all 72 reconstruction scores and 48 within-family summaries from saved scalar predictions/pairs, checks original family folds and exact permutation counts, verifies protected inputs and confirms the development-only boundary. It does not fit or load models.
