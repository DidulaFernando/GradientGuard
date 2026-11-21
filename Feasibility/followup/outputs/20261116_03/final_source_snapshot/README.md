# GradientGuard targeted follow-up

Current scientific attempt: **20261116_03**. Failed engineering attempt **20261116_02** is preserved. Original **20261116_01** remains unchanged.

- [Report](outputs/20261116_03/reports/REPORT.md)
- [Decision](outputs/20261116_03/reports/decision.json)
- [PowerShell instructions](../instructions/STEP_BY_STEP_FOLLOWUP.txt)
- [Frozen protocol](outputs/20261116_03/protocol_frozen.json)
- [Final output audit](outputs/20261116_03/reports/validation_audit.json)

Original deep-slope ratio 3.050141 reproduced. The predefined target passes 15/20 alternative partitions, below the 16/20 screen. Conv3/head/common-deep slope counts are 16/14/15; conv3 convolution-only count is 10. Net A/B error cross terms reinforce rather than cancel, and a head artifact is not established.

All three calibration trajectories completed six actual epochs. Seed 101 remains below 80% health at all candidate points, so no candidate horizon qualifies. Fresh-realisation stage is correctly unlaunched and its source-health coverage is unassessed. No horizon, threshold, seed or learning rate was changed to obtain a better result.

The implementation tests and reproduction pass. Levels remain reliable within the original cohort; temporal direction and source-health coverage require further development. No completed-training WGA outcome pilot or forecasting model was run.

Use the existing project `.venv`; no dependencies were changed. All extension code/config/tests/results are namespaced here. The frozen numerical-source snapshot and final reporting-source snapshot are separate, with hashes recorded for each.
