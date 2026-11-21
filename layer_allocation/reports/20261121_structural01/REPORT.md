# GradientGuard: first layer-allocation structural-control implementation

21 November 2026. Attempt `20261121_structural01`. **Post-hoc exploratory engineering and conditional-addition study.**

## Scope and execution
No new image classifier was trained. All 96 original completed classifiers were remeasured at six frozen early checkpoints on both original 256/class panels (1,152 measurements). Both 5% and 10% prefixes were exported. Original data, outcomes, predictor freezes and reports remain read-only. New forecasters were fitted only on original development families; their evaluation nevertheless reuses already consumed outcomes. This is not prospective confirmation.

## Implementation correctness and environment
The broken venv base-interpreter path was backed up and repaired to the available matching Python 3.12.14. All 80 package records match the saved lock; torch 2.7.1+cu128, CUDA 12.8, GPU NVIDIA GeForce RTX 5060 Ti. No package installation was needed. Isolation and microbatch checks are separate from predictive evidence. Original frozen predictions and original nine-audit counts reproduced. Maximum checkpoint scalar parity error: 7.10542736e-15. Full test output, source snapshot and preservation hashes accompany this attempt.

## Frozen exploratory comparison
E inputs are fixed to the original development-selected bank. B adds total sensitivity. L2 and L6 reproduce the nested endpoint/six-share representations. N adds a six-summary norm-product proxy. M adds per-layer weight/mean-gradient L2 norms and marginal statistics plus shared bias diagnostics. MZ adds exactly the six L6 shares to M; NZ similarly adds them to N. Last, mean and population SD use the original schedule; no slope enters Z. The structural bank includes head diagnostics on the same footing as other layers.

Every matched pair uses the same data, family folds, preprocessing, learner and grid allowance. Regularisation may differ after development selection. Logistic and bounded histogram controls use C=0.1/1/10 or depth=1/2/3. Histogram min-leaf=5 was prospectively specified for this *new exploratory* comparison; historical boosting defaults/results are untouched. Selection uses development discovery at one-third capacity, AP tie-break, smaller grid on exact ties. One-third is pilot-motivated, not a newly independent primary result. Fold-local preprocessing never uses evaluation data.

Original source-normal evaluation has 36 models, 12 failures and nine eligible families. Full evaluation has 40 models. Eligibility and failure threshold remain fixed. M inputs: 174; MZ inputs: 180. This dimensionality is large relative to 48 eligible development runs and is a limitation even with regularisation.

## Descriptive evaluation results: source-normal, 10%, panel A

| Predictor | D9 | D12 | AP | AUROC | Brier |
|---|---:|---:|---:|---:|---:|
| histogram_B | 9 | 10 | 0.9629 | 0.9792 | 0.1231 |
| histogram_E | 9 | 10 | 0.9569 | 0.9722 | 0.1235 |
| histogram_L2 | 9 | 10 | 0.9629 | 0.9809 | 0.1252 |
| histogram_L6 | 9 | 11 | 0.9646 | 0.9809 | 0.1292 |
| histogram_M | 8 | 10 | 0.9248 | 0.9583 | 0.1310 |
| histogram_MZ | 8 | 10 | 0.9248 | 0.9583 | 0.1310 |
| histogram_N | 9 | 9 | 0.9340 | 0.9583 | 0.1589 |
| histogram_NZ | 9 | 11 | 0.9624 | 0.9740 | 0.1657 |
| logistic_B | 8 | 8 | 0.7612 | 0.7847 | 0.2063 |
| logistic_E | 8 | 8 | 0.7751 | 0.8160 | 0.2036 |
| logistic_L2 | 8 | 8 | 0.7599 | 0.7812 | 0.2061 |
| logistic_L6 | 8 | 10 | 0.9336 | 0.9618 | 0.1648 |
| logistic_M | 9 | 9 | 0.9351 | 0.9583 | 0.1396 |
| logistic_MZ | 9 | 10 | 0.9405 | 0.9618 | 0.1402 |
| logistic_N | 8 | 10 | 0.9336 | 0.9618 | 0.1636 |
| logistic_NZ | 8 | 10 | 0.9304 | 0.9583 | 0.1654 |

Original published E/B/L6/L8 values are reproduced separately in `original_reproduction.json`; these new E/B fits can differ because the new exploratory selection rule differs. They must not replace the original primary result.

## Panel and cohort sensitivity: matched MZ versus M at one-third

| Cohort | Panel variant | Learner | MZ discovered | M discovered | Difference |
|---|---|---|---:|---:|---:|
| eligible | A | logistic | 10 | 9 | 1 |
| eligible | A | histogram | 10 | 10 | 0 |
| eligible | A_on_B | logistic | 10 | 10 | 0 |
| eligible | A_on_B | histogram | 10 | 10 | 0 |
| eligible | B_refit | logistic | 10 | 10 | 0 |
| eligible | B_refit | histogram | 11 | 11 | 0 |
| full | A | logistic | 10 | 10 | 0 |
| full | A | histogram | 9 | 9 | 0 |
| full | A_on_B | logistic | 10 | 10 | 0 |
| full | A_on_B | histogram | 9 | 9 | 0 |
| full | B_refit | logistic | 10 | 10 | 0 |
| full | B_refit | histogram | 9 | 9 | 0 |

Panel A is primary. A_on_B applies the new frozen A model to B inputs. B_refit uses a development B fit with the A-selected hyperparameters, not B-based tuning. Full curves and both 25%/one-third comparisons are exported, including losses and ties.

## Family uncertainty and influence
Eligible panel-A logistic MZ-M at one-third: 10 versus 9 discoveries; Delta20=0.555556. The paired-family pointwise 95% descriptive interval is [0.000000, 0.625000], with 1942 valid and 58 invalid draws of 2,000, seed 210212. Empty and single-label draws are excluded and counted. All ten original evaluation families participate in sampling, including the one with no eligible models. Nested models remain together. These intervals condition on selected forecasters, exclude selection uncertainty and are not simultaneous curve intervals or a significance claim. Leave-one-family-out differences at recomputed one-third capacity range from 0 to 1.

## Cost and artifacts
The sum of synchronized checkpoint-panel gradient/statistic timings is 77.999 seconds. Extraction invocation wall time: 103.768 seconds (resume invocation cost is separate from cumulative measurement cost). Development fitting cost: 23.712 seconds. GPU allocated peak/incremental counters, data reconstruction overhead, complete stage wall times and output bytes are recorded separately. These are research validation costs, not isolated production L6 monitoring cost. Dataset reconstruction uses the original compact identities/background assignments and cached raw MNIST; no new dataset or label generation occurs.

## Qualified conclusion
**Exploratory conditional added value observed; independent confirmation required.** Proceed to separately frozen validation only if the narrower question remains worthwhile; do not claim sensitivity-specific novelty from this reused evaluation.

Positive differences are candidates for fresh validation, not proof that coordinate-product information is unique or causal. Ties/losses may motivate narrowing the representation, but cannot show that all layer information is useless. Family concentration, panel variation, control-bank dimensionality, baseline headroom and learner limitations bound interpretation. No changes to the manuscript's completed evidence are implied. A future untouched outcome study and separately frozen second benchmark remain necessary before a publication-level novelty claim.

## Reproduction
Use `instructions/STEP_BY_STEP.txt` and `run.ps1`. `all` safely resumes scalar extraction and skips already frozen fits; `report` regenerates only this attempt's report. Never invoke legacy fitting/training/report commands for this stage. Prior artifact verification is mandatory at completion.

## Completed-output validation and resource audit
All 16 engineering tests pass; output audit passed. All 3088 protected prior files are unchanged. Stage order, frozen pipelines, strictly early exports, nested input schemas, complete curves, matched panels and recomputed omission budgets were checked. End-to-end invocation wall time: 191.207 s. Maximum measured allocated GPU memory: 157.43 MiB; maximum incremental probe allocation: 111.27 MiB. CPU memory is only an audit-time snapshot. See `validation_audit.json` and `costs.json`.

## Scientific synthesis for the next implementation
The positive result is narrow: logistic MZ adds 1 discovery over M at twelve audits on panel A. Both panel-B variants give MZ-M differences [0, 0], and histogram MZ/M tie on panel A. The conditional Delta20 interval includes zero (0.000000 to 0.625000). Thus the sensitivity-specific increment is not panel/learner-stable in this reused evaluation.

The simpler logistic norm-product allocation proxy N finds 10 failures at twelve audits, matching logistic L6's 10; this shows that this fixed-budget result alone does not identify coordinate-product specificity. The stronger bounded histogram E finds 9 at nine and 10 at twelve, so the historical weaker nonlinear baseline must not be treated as the strongest available control. The development-selected strongest non-sensitivity structural control is `logistic_B` (selected without evaluation labels); it finds 8/8 at nine/twelve. These are descriptive comparisons, not new selection based on test performance.

Recommendation: **revise toward the narrower layer-allocation hypothesis and retain N/M as serious controls in a separately frozen fresh validation.** The extension is working and there is a small encouraging conditional increment, but the present result does not establish a robust sensitivity-specific contribution, justify journal readiness, or support dropping stronger conventional controls. No further training was launched.
