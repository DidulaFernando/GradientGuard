# GradientGuard diagnostic follow-up

Attempt 20261118_01 is a post-hoc read-only analysis of the completed 96-classifier
PoC. It uses frozen risk predictions and existing fitted pipelines. It fits no
classifier or forecaster and changes no original scientific artifact.

The original nine-audit result remains primary. Twelve audits and the whole capacity
curve are exploratory. The analysis distinguishes added layer information from the
extra contribution of slopes and retains panel/family sensitivity.

See config/analysis.json and config/frozen_plan.json for definitions/hashes. Code is
under src, synthetic/read-only checks under tests, and outputs under tables, figures,
reports and logs. Stage markers allow safe resume. Reports highlight small gains
without presenting post-hoc findings as confirmed temporal benefits or novelty.

Exact commands are in the project instructions/STEP_BY_STEP_DIAGNOSTICS.txt.
The original environment configuration backup is in environment/pyvenv_before.cfg.
