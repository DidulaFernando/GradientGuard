# Bounded development-only diagnosis

Attempt `20261122_devdiag01` examines the already completed `20261122_fresh02` development evidence. It is post-hoc explanation, not independent validation or a new failure-predictor experiment. No old source or result is edited.

The fixed analysis covers six Z features, both original panels, source-normal primary and full secondary cohorts, sample family variance decomposition, matched-proxy correlations, fixed alpha=1 family-held-out Ridge reconstruction from B/N/M, within-family raw/residual feature concordance, exact/bounded conditional label reassignment, and existing development OOF risk concordance. The only new fits reconstruct feature values without failure labels. All risk predictors remain frozen.

Run `run.ps1 tests`, then `run.ps1 all`. The protocol/source/input/preservation freeze precedes new numerical results. Protected held-out files are hashed but never semantically loaded into this diagnosis. No binary model pipeline is deserialised. All semantic reads pass an explicit development-only allowlist.

Interpretation cautions: high reconstruction R2 does not prove conditional irrelevance; low R2 does not establish unique predictive value. Only mixed-outcome families can contribute within-setting comparisons. Pair counts and Monte Carlo/exact support are explicit. Retrospective p values and OOF diagnostics do not overturn the failed held-out progression gate or authorise new training/manuscript updates.

See `instructions/STEP_BY_STEP.txt` for commands and output locations. Automated repository preferences require logical local commits. This task does not separately request another remote publication.
