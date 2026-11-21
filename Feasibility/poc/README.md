# GradientGuard exploratory outcome PoC

Attempt: `outputs/20261116_poc01`. The user authorised this experiment despite the
unresolved measurement, aggregation and source-health gates in the earlier studies.
Those gates remain unchanged. The 60-epoch horizon is an exploratory amendment.

The frozen design contains 24 setting families and 96 CNN A classifiers. Four runs
within each family remain together: 14 development families (56 runs), ten held-out
families (40 runs). The immutable scientific settings and all seeds are in
`outputs/20261116_poc01/protocol.json`; the run manifest is beside it.

Use the existing project `.venv` and `instructions/environment.ps1`. Exact commands
and interpretation are in `instructions/STEP_BY_STEP_POC.txt` at the project root.
No legacy source, frozen follow-up code or prior scientific record is overwritten.

- `run.py`: frozen protocol, preservation and orchestration.
- `study.py`: compact official-generator constructions, CNN training, early
  observations and separately gated terminal outcomes.
- `prediction.py`: family-grouped development selection, frozen pipelines and
  held-out early predictions.
- `launch.py`: fail-closed command checks.
- `report.py`: predefined held-out metrics, family uncertainty and audit selections.
- `finalize.py`: read-only validation, reporting and figures after unsealing.
- `test_poc.py`, `tests/`: separately labelled synthetic/read-only checks.

Successful execution is distinct from adequate outcome diversity, useful audit
gain and temporal specificity. Panel A is primary; B is secondary. No forecaster
refit, cutoff selection, family replacement or favourable-panel selection is allowed
after test outcome access. Prediction depth is not implemented.
