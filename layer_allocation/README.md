# GradientGuard: layer-allocation study

This namespaced implementation follows manuscript v1.2. First stage: reproduce old measurements and compare a rich structural bank with versus without exactly six L6 shares. It reuses 96 completed PoC classifiers and already consumed outcomes, so all predictive findings remain post-hoc exploratory. It does not launch new CNN training or Waterbirds.

Use `instructions/STEP_BY_STEP.txt`. The authoritative plan is `configs/structural.json`; `freeze` hashes source, tests, configuration and old inputs before measurement/fitting. Recovery caches have complete run/checkpoint/panel/protocol identities. Original code is imported only through read-only model/measurement helpers, never through legacy command modules. Dependencies stay in `../Feasibility/.venv`.

The standing auto-commit preference applies to completed, verified work; no public push or manuscript-result replacement occurs automatically.
