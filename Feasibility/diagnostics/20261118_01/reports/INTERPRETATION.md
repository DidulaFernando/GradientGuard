# What the diagnostic findings mean

The encouraging finding is **two additional failures discovered with twelve audits**
on the original 36-model source-normal pool: L6 and L8 find 10, while E and B find 8.
At that budget, audit precision rises from 66.7% to 83.3%. These are observed
post-hoc discovery counts, not a confirmed generalisation improvement.

There is a broader pattern worth investigating. On the original primary panel,
L8 never discovers fewer failures than E across the full 0–36 budget curve. It
ties at nine audits and improves at several smaller and larger budgets. L6 has the
same discovery curve. L6/L8 recover all twelve failures by seventeen inspections;
E needs twenty-five and B twenty-eight. Those earliest full-recall budgets are
retrospective descriptions, not deployable thresholds known in advance.

## What appears to explain the gain

The gain is compatible with useful **layer-level endpoint/mean/spread information**
beyond the conventional and global-sensitivity predictors. It is not evidence that
the two proposed slopes add discovery value. L6, which omits those slopes, matches
L8 at every primary budget.

The slopes were present, variable, scaled and passed to the final estimator. They
were not accidentally omitted by preprocessing. Their standardized coefficients and
log-odds contributions are small in this fitted model. L6 and L8 probabilities differ
by up to about 0.295 percentage points, and four rank positions differ. The swaps
preserve the failure/non-failure sequence, explaining the equal AP, AUROC and full
discovery curves despite nonidentical probabilities and rankings.

Replacing both slopes by saved development-training reference means changes risk
by up to about 0.683 percentage points, but changes no primary discovery counts at
any budget. This frozen-input diagnostic is not causal and is not a substitute for
the separately fitted L6 model.

## What is stable, and what is not

The twelve-audit 10-versus-8 pattern persists with the frozen A predictor on B and
with the already fitted B pipelines. This is encouraging probe consistency. It is
not independent replication: these variants use the same trained classifiers and
underlying image benchmark.

Setting-family influence remains substantial. At the recomputed one-third capacity,
leaving out each held-out family gives four positive differences, five ties and one
negative difference for the original A predictors. One positive omission is the
family containing no eligible candidates, so it is effectively the original result.
Removing family 12 changes the comparison to 7 versus 8 discoveries (L8 versus E).
Removing families 19 or 21 produces larger positive differences. Remaining pool
sizes and audit budgets change as required, so these are influence diagnostics rather
than ten replications of the original pool.

The pointwise paired-family interval for Delta20 at one-third capacity is approximately
[-1.11, 4.44], including zero. Fifty-eight of 2,000 draws were invalid under the
original single-class/empty-pool rule. Only three eligible held-out families contain
failures, and none contains both failure labels. This limits precision and evidence
about distinguishing successful/failing seeds inside the same setting.

The full 40-model cohort remains a separate secondary analysis. Its curves include
budgets where L8 performs worse than the conventional or selected nonlinear control;
the source-normal gain is not a uniform advantage across all candidates and models.

## Recommended next step

**Revise the contribution toward order-free layer summaries, then assess novelty and
design a separately frozen independent validation if that narrower question is worth
pursuing.** This retains the positive signal without claiming slope-specific evidence.

Keep the original nine-audit result unchanged: all primary methods found eight.
The larger-capacity improvement cannot turn that original result into a pass. Do not
choose a favourable budget as a new primary endpoint on these same outcomes. Before
further training, agree the revised question, comparisons, audit capacities, family
allocation and precision/headroom requirements with the supervisor. The original
measurement and nonlinear/depth-comparator limitations remain relevant.

No new classifier was trained, no forecaster was fitted or tuned, and no original
artifact or proposal was changed. All sixteen diagnostic tests passed. Twenty-seven
saved-prediction comparisons reproduced to a maximum absolute error of about
1.11e-16; original family-bootstrap and influence summaries were reproduced too.

See REPORT.md for the complete methods, tables, figures and interpretation limits.
