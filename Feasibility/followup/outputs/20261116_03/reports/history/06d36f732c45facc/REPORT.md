# GradientGuard second targeted feasibility study - 20261116_03

## Findings and limits

- Original temporal screen is partition-sensitive
- Aggregation interpretation remains unresolved
- Source-health coverage unresolved
- Levels reliable; temporal direction requires revision

Original-result reproduction passed=True; saved-partition target stable=False; selected calibration horizon=None; ready for an outcome pilot=False. These are independent engineering/scientific flags, not a single success label. Structural review: Aggregation algebra and MSE decomposition pass. Conv3 common-denominator slope passes 16/20, head 14/20 and combined deep 15/20. Conv3 convolution-only slope passes 10/20, so denominator choice materially affects the diagnosis. The uncentred cross term is positive in every partition: some individual opposite-signed errors occur, but net cancellation does not explain combined reliability. No head-only artifact or causal depth-specific interpretation is established. Component and denominator dependence leave aggregation interpretation unresolved. The conv3 finding is diagnostic and does not replace the predefined deep_slope target..

The level-reliability conclusion applies to the original six prefixes and their held-out pool. With fresh validation gated, it is not a confirmed replication across newly seeded constructions. The recommended next work is a separately preregistered source-health/training-dynamics diagnostic and reconsideration of temporal/denominator sensitivity; no automatic extra epochs, learning-rate search or substitute feature is authorized by this report.

No forecasting model, terminal WGA, AUROC, audit gain, confidence interval or novelty claim was produced. The research proposal and the entire first experiment were left unchanged. Checkpoint-only computations construct no optimizer and perform zero training updates.

## Preservation, environment and frozen protocol

Pre/post SHA-256 audit covers 294 original files (legacy configs/src/tests, snapshot, data/MNIST archives, checkpoints, measurements, reports/decisions and original instructions) plus the reference proposal. Legacy artifacts unchanged=True. New implementation/tests/configurations are entirely under followup; no legacy resume guard is bypassed.

Actual Python=3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]; torch=2.7.1+cu128; torchvision=0.22.1+cu128; GPU=NVIDIA GeForce RTX 5060 Ti; CUDA=12.8; cuDNN=90701. The complete pip freeze --all lock matches the saved lock. Old provenance metadata included vendored packages absent from the current enumeration; this discrepancy and its adjudication are preserved in preflight/environment*. Numerical versions, generator source and hardware match; no package/security change or Python 3.11 retry occurred.

Protocol frozen before new gradient partitions or calibration: protocol_frozen.json, SHA-256 `ff533d48a1003c0e0606f79bbf65e8dd12a020d9ddfb21f4b3feb763fca74d79`. Training/measurement source hashes are guarded on resume. Reporting/CLI maintenance has separately recorded hashes. Failed engineering attempt 20261116_02 (Windows path separator hash-key bug) is retained with its source snapshot and failing test log. Current checks: 21 passed, including reused analytical/gradcheck/isolation tests redirected to the new namespace, original arithmetic, partition identities/nesting, recomputed gradients, additive summaries, independent SD, variable cohorts/floors/invalid cases, CPU/CUDA new-run resume, health-only calibration and source-train-only dataset construction. No tolerance was loosened.

## Original result reproduction

Exact retained formulas from legacy analysis.py were verified: mean-loss gradients accumulated signed and sample-weighted before absolute sensitivity; biases excluded; shallow conv1, middle conv2, deep conv3+head; epsilon=1e-12; original stabilizer rule. Last, fixed-checkpoint mean, population SD and OLS slope use actual fractions and exclude initialization. Steps are **65,130,196,326,489,652**, never extra recovery checkpoints. Prefixes remain 5%/10%.

Recalculated deep_slope ratio at 10%,256/class: **3.0501409990602664**; remeasured original A/B result: **3.0501409990602655**. Maximum formula discrepancy=0.0; remeasured primary-share discrepancy=8.326672684688674e-17; maximum feature discrepancy=1.4210854715202004e-14. All original tolerances pass. Reproduction measurement rows=432 (three A repeats/one B measurement for six runs, six monitoring steps, three budgets). Full parity errors are saved under reproduction/.

## Stage A: complementary probe-partition robustness

Within each class the original A+B union contains 512 unique IDs, disjoint from training and source health. Seeds 41000-41019 define 20 alternative complementary 256/256 panels with nested 64/128 subsets. All exact IDs/ranks were frozen before measurement, paired across conditions/seeds/checkpoints. Memberships use class and random seed only; no background balancing or image regeneration. Each new panel has its mean gradient recomputed; scalar cache keys include checkpoint/data/manifest/protocol/numerical-code/environment hashes and every run/step/panel/budget/repeat identity.

The unchanged reliability statistic is noise=RMS(A-B)/sqrt(2), spread=sample SD of runwise (A+B)/2 (ddof=1). Numerical floor=1e-5 for primary non-slope features; slope floor=2e-5/(last-first observed full-horizon fraction); larger measured repeat RMS/sqrt(2) replaces it. Zero raw noise gives an undefined raw ratio, with a floor-protected screen ratio; flat/invalid/incomplete cohorts are not silently repaired. Ratios>=3 pass; the primary follow-up target, deep_slope at 10%,256/class, must pass at least 16/20 partitions. It is never replaced by a better feature.

| feature | passing_partitions | pass_fraction | screen_ratio_min | screen_ratio_median | screen_ratio_max | noise_median | spread_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep_last | 19 | 0.95 | 2.619 | 4.7979 | 9.1037 | 0.031084 | 0.15069 |
| deep_mean | 20 | 1 | 6.9255 | 14.759 | 39.442 | 0.010889 | 0.1596 |
| deep_slope | 15 | 0.75 | 2.0864 | 3.8329 | 6.8864 | 0.36155 | 1.4196 |
| deep_std | 12 | 0.6 | 1.9155 | 3.2164 | 9.098 | 0.0097787 | 0.032689 |
| shallow_last | 16 | 0.8 | 2.0644 | 3.9876 | 7.72 | 0.02701 | 0.10929 |
| shallow_mean | 20 | 1 | 7.7601 | 13.692 | 40.113 | 0.0087667 | 0.11881 |
| shallow_slope | 11 | 0.55 | 1.7401 | 3.1398 | 7.6831 | 0.30449 | 0.94904 |
| shallow_std | 20 | 1 | 3.5409 | 4.9037 | 8.8518 | 0.0076782 | 0.037284 |

These 20 partitions overlap one finite fixed pool. Their pass fractions, ranges and 5/25/50/75/95% quantiles describe dependence on partition choice; they are not confidence intervals or independent data replications. Complementary A/B panels are dependent, so the retained RMS metric is a comparability diagnostic rather than an independent-panel variance estimate. Spread and noise distributions are both exported. Every feature, budget, prefix, invalid and flat result is retained.

![All primary feature distributions](figures/saved_partition_distributions.png)

## Stage B: classifier-head and aggregation interpretation

Common denominator T=conv1+conv2+conv3+head+epsilon. Conv3/T and head/T sum to the original deep share. Last/mean/slope additivity is checked at each panel/prefix; SD is computed separately and is **not additive**. Error_deep=error_conv3+error_head for additive summaries; MSE decomposition uses the **uncentred** term 2*mean(error_conv3*error_head), not an assumed covariance. A negative cross term can make the combined block less sampling-sensitive. This can be useful aggregation; it does not automatically establish an artifact or head mechanism.

| budget | head_flags | bias_flags | observations |
| --- | --- | --- | --- |
| 64 | 4 | 0 | 12 |
| 128 | 1 | 0 | 12 |
| 256 | 0 | 0 | 12 |

These are **original A/B endpoint** flags, separately by budget and across both panels. Alternative-partition flags at every checkpoint/panel/budget are in the full CSV. A 64/class warning is not a selected-budget 256/class warning.

Diagnostic reliability tables retain individual conv1/conv2/conv3/head shares under the common denominator, plus separately labelled convolution-only renormalized shares. Per-weight means use diagnostic total tolerance/n_weight as floor; bias diagnostics use 1e-6, with the same slope-span multiplier and repeated-measurement protection. These diagnostic-unit floors do not replace the original primary share floor or enter the eight-feature export. Head>=50% of deep and bias>=10% of weight-total flags are reported at every budget/panel/checkpoint.

![Conv3/head and combined reliability and error cross term](figures/saved_head_conv3_error.png)
![Separate learned layers](figures/saved_common_denominator_layers.png)
![Convolution-only denominator](figures/saved_convolution_only.png)
![Per-weight means](figures/saved_per_weight.png)
![Bias/head profiles](figures/saved_bias_head.png)
![Budget-specific original-panel flags](figures/saved_budget_flags.png)

Final structural assessment: Aggregation algebra and MSE decomposition pass. Conv3 common-denominator slope passes 16/20, head 14/20 and combined deep 15/20. Conv3 convolution-only slope passes 10/20, so denominator choice materially affects the diagnosis. The uncentred cross term is positive in every partition: some individual opposite-signed errors occur, but net cancellation does not explain combined reliability. No head-only artifact or causal depth-specific interpretation is established. Component and denominator dependence leave aggregation interpretation unresolved. The conv3 finding is diagnostic and does not replace the predefined deep_slope target.. See the separate review JSON for the quantitative evidence and scope of this interpretation.

## Stage C1: bounded source-health calibration

Fresh independent construction data seed=3701,reservation seed=4903; training seeds=101,102,103. Architecture/Adam/.001/batch128/no augmentation/palette/MAGNITUDE_LARGE/no noise are fixed. Each trajectory trains at most six actual epochs, with health at epochs 2,4,6 corresponding to candidate full horizons 20,40,60. These are points on three calibration trajectories, not nine independent experiments or completed 20/40/60-epoch training.

Choose the smallest full horizon whose 10% point has accuracy>=.80, CE<=ln(10), finite weights/loss/required telemetry for **all three** seeds. Selection uses only the health rule; gradient reliability/recolouring never enters. No candidate qualifies means stop fresh validation, preserve calibration, do not lower thresholds or expand the grid. Selected full horizon=None.

| seed | actual_epochs | health_accuracy | health_loss | finite_weights | required_telemetry_finite | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 2 | 0.61953 | 1.1443 | True | True | False |
| 101 | 4 | 0.76328 | 0.68998 | True | True | False |
| 101 | 6 | 0.78125 | 0.62731 | True | True | False |
| 102 | 2 | 0.67812 | 0.87182 | True | True | False |
| 102 | 4 | 0.8125 | 0.55933 | True | True | True |
| 102 | 6 | 0.86719 | 0.39674 | True | True | True |
| 103 | 2 | 0.45937 | 1.4514 | True | True | False |
| 103 | 4 | 0.80312 | 0.61454 | True | True | True |
| 103 | 6 | 0.85469 | 0.4557 | True | True | True |

![Calibration trajectories](figures/calibration_health.png)

## Stage C2: separately frozen fresh-realisation prefixes

Fresh-realisation stage was **not launched**, because no candidate horizon satisfied all three calibration seeds. No new horizon was adopted; the original 20-epoch/652-update design remains unchanged. Fresh-realisation source-health coverage is **not assessed**, rather than a measured 0% pass rate. The planned 18 prefixes and six coverage cells have explicit not-launched/not-assessed status tables. This is a predefined gate, not a software failure. No threshold, learning rate or horizon-grid expansion was made. The new-realisation runner is implemented and its update/checkpoint/resume primitives pass CPU/CUDA tests, but the full 18-prefix pipeline was not exercised on real fresh data because the gate prevented it.

## Cost, memory and storage

| stage | run | endpoint | training_seconds | primary_panel_monitor_seconds | health_seconds | recolour_seconds | checkpoint_seconds | warmup_seconds | max_gpu_allocated | max_gpu_reserved | training_loop_wall_seconds_this_invocation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | cal_s101 | 1956 | 6.7026 | 0 | 0.028468 | 0 | 0.088263 | 0.29569 | 118774272 | 144703488 | 16.402 |
| calibration | cal_s102 | 1956 | 6.6979 | 0 | 0.027871 | 0 | 0.089789 | 0.01038 | 118774272 | 144703488 | 16.269 |
| calibration | cal_s103 | 1956 | 6.8498 | 0 | 0.030685 | 0 | 0.092935 | 0.010328 | 118774272 | 144703488 | 16.952 |

Primary online-panel measurements are costed separately by 64/128/256 budget. The sum across three diagnostic budgets is not the price of one selected production panel. Repeated A triplicates, complementary B and all 20 partitions are **research validation costs**. Checkpoint-only cost, calibration, fresh training, source health, recolouring, data construction, checkpoint IO and end-to-end command wall time are separated in tables/stage_wall_costs.csv, training_costs.csv, online_primary_cost_by_budget.csv and checkpoint_measurement_costs.csv. Nested command/component times overlap and must not be summed as if disjoint. Progress logs record wall time and per-row throughput for safe resume.

Because fresh training was gated, its online monitoring and recolouring costs are not observed; empty/status-labelled outputs mean not applicable, not measured zero-cost performance. Calibration performs health-only selection and no recolouring. Training update component time excludes CPU orchestration/configuration reads and IO; these can be substantial and are included in loop/command wall time. reports/execution_timing.json records the interval from the first instrumented attempt command to the report snapshot, including engineering/idle gaps, with its startup/end-point limitations stated.

| folder | files | bytes |
| --- | --- | --- |
| data | 11 | 492814963 |
| checkpoints | 72 | 21474584 |
| saved | 27 | 53350495 |
| fresh_validation | 0 | 0 |
| fresh_original | 0 | 0 |
| reports | 26 | 8222863 |

GPU timings synchronize operations. Component probe timers exclude caller indexing; including-indexing timers are also retained. CUDA allocated/reserved/incremental peaks use PyTorch allocator counters, not device-wide memory; process peak working set is cumulative and cannot isolate CPU probe peaks. Dataset materialization and source-health/recolour CPU setup can add overhead beyond component timers. Other desktop GPU use and warm-up can affect timing. All assets/cache/temp files remain project-local; the existing base interpreter is external and read-only.

## Decision and next work

- Original temporal screen is partition-sensitive
- Aggregation interpretation remains unresolved
- Source-health coverage unresolved
- Levels reliable; temporal direction requires revision

Readiness for a separately designed completed-training outcome pilot=False. This judgment requires correctness/reproduction, partition stability, source-health coverage and an explicit structural review together. Passing engineering screens does not establish incremental predictive value over ordinary source loss/accuracy. A different promising feature is exploratory and would need a separately frozen follow-up. No completed-training outcome pilot is executed here.

Exact commands, stage completion indicators, safe resume and inconclusive-result handling are in ../../../../instructions/STEP_BY_STEP_FOLLOWUP.txt. Full scalar measurements remain in stage-specific CSV/SQLite files; all feature, reliability, structural, coverage and cost tables are in reports/tables. Original and new planned horizons remain separate.
