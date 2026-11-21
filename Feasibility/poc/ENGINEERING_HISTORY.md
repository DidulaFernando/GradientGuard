# Engineering history

Protocol and all 24 family/96 run identities frozen before classifier outcomes.
Initial nine synthetic/read-only checks passed; prior studies were hashed and preserved.
Before training, a module-name collision with legacy src/experiment.py was fixed by
namespacing the new implementation as study.py. Source revisions/snapshot retained.
Before forecaster fitting, exported early tables were restricted to early
fields: final source-monitoring fields and outcome labels remain separate. Predictor
inputs always used explicit whitelists. Predictor freeze records actual selector source
hashes separately from the earlier classifier-source snapshot. No scientific setting,
seed, threshold, family split or feature formula was changed. Development classifier
training/outcome collection had started during these export-only and boundary-check
improvements; no terminal outcome values were inspected to choose them. Source
snapshots distinguish initial classifier code from final selector/report code.

During held-out training, two additional separately labelled synthetic checks were
added: parity of the new observer with the legacy mean-gradient definition, and an
intentional interruption/resume of the unchanged training runner. The first resume
fixture invocation omitted deterministic() (the real command invokes it). Exact
weights differed under those default CUDA settings. The failed log is retained as
execution_tests_initial_failure.log. The fixture was corrected to invoke the same
production determinism settings, and both checks then passed with exact final
weights and checkpoint observations. No scientific training code, tolerance,
forecaster selection or frozen predictor was changed.

Before held-out unsealing, reporting-only audit handling was made fail-closed for
nonfinite/missing or nonbinary labels, with an explicit empty-pool status. No
scientific observation is filtered or relabelled. Actual valid-pool metric formulas
are unchanged. Additional synthetic family-resampling/invalid-input checks were added.
The command guard was also tightened before unsealing: an already-frozen attempt
cannot rerun freeze, and predictions cannot be regenerated after test outcome access.
Read-only verify.py is the safe existing-attempt preflight. Predictor/classifier
mathematics and frozen fitted pipelines are unaffected.

After one-time held-out unsealing, report generation failed when the Markdown JSON
renderer encountered a NumPy boolean. The metric CSVs and outcomes had been produced;
no model selection or training was repeated. The failed console log and first JSON
decision are retained. Reporting now converts NumPy scalars to native JSON scalars
before writing both JSON and Markdown. Numerical metrics and decision thresholds
are unchanged. The full 21-check integration suite passed after training completed.

Post-unseal presentation/validation is isolated in finalize.py. It fits no model,
selects no new feature and changes no cutoff. It retains the initial automatic
report/decision, adds explicit panel/influence qualification, independent completion
audits, exactly-eight-feature exports, structural diagnostics, plots and cost
limitations. Actual classifier and predictor hashes remain frozen separately.
