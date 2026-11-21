# GradientGuard: first measurement-feasibility experiment

Completed attempt: **20261116_01**, six real CNN A prefixes on official SpuCoMNIST.

Start with [the final report](reports/20261116_01/REPORT.md), [machine-readable decision](reports/20261116_01/decision.json), and [PowerShell instructions](instructions/STEP_BY_STEP.txt).

All 13 correctness checks and the separate smoke test passed. All six runs reached update 652 of a frozen 6,520-update horizon. Three correlated-background runs meet source health; all three independent-background runs miss 80% accuracy and remain in the analysis.

The result is qualified: strong level variation, but only deep slope at 256/class narrowly clears the provisional ratio screen (3.05). Shallow slope fails; 64/128-class slopes fail. Architectural aggregation and unequal learning progress require follow-up. No final WGA, forecaster, audit gain or novelty claim is made.

Training/configuration source is preserved under `runs/20261116_01/source_snapshot`; exact dependencies and hardware are in that run's lock/provenance files. The postanalysis finalizer is separate from the immutable training snapshot and independently verifies feature arithmetic.

Python 3.12 is a documented fallback: native modules in the project-local Python 3.11 distribution were blocked by Windows Application Control, and automatic approval review rejected the python.org installer command. No security controls or global package environment were changed. Failed setup logs/environment are retained.

No execution steps remain pending. Research follow-up is described in the report. The reference proposal was not edited.
