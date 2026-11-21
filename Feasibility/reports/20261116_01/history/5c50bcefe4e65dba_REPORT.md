# GradientGuard first feasibility test - attempt 20261116_01

## Result and scope

**Qualified outcome:** levels show clear variation, but the only passing 10% slope is deep slope at 256/class (ratio 3.05). This narrowly clears the provisional screen; it is not robust temporal validation. Classifier-head aggregation and unequal source health require follow-up.

**Decision: `reliable_level_and_slope_variation_targeted_structural_control_required`.** Completed 6/6 real prefixes. Correctness: True. Identical-probe numerical stability: True (maximum summary difference 4.440892098500626e-16). At 10%, at least one retained level passes the frozen ratio >=3 screen: True; at least one slope: True. Probe budgets measured: [64, 128, 256]; final diagnostic budget: 256 examples/class/panel.

The question is whether early layer-wise summaries vary reproducibly between these runs beyond numerical and probe sampling variation. These are **two fixed controlled settings with three paired seeds**, sharing benchmark identities, not six independent setting families. A passing screen is engineering evidence on these settings. It establishes no final WGA prediction, audit discovery gain, generalisation significance, methodological novelty or causal learning mechanism. No forecasting model was trained and no terminal validation/test evaluation was performed.

## Frozen design and data verification

Empty project inspected before modification; no applicable AGENTS.md or existing intended mapping/horizon. The unchanged proposal supplied CNN A's structure and left mapping/horizon open. User defaults: ten separate digits, fixed HSV palette, MAGNITUDE_LARGE, no label/core noise, Adam lr=.001, betas=(.9,.999), weight_decay=0, batch=128, no augmentation, 20 planned epochs, no dropped final batch. Configuration hash: `0338b266a1ebc892a798c54ad4f29f3a15b5410d3dd3ffca307dbd3cc45020af`. The resolved configuration and rationale were written before pilot training in `../../runs/20261116_01/frozen_config.json`.

Training examples: 41604; updates/epoch: 326; planned horizon: 6520 updates; actual endpoint: 652. Checkpoint updates: [65, 130, 196, 326, 489, 652]; fractions: [0.009969325153374233, 0.019938650306748466, 0.030061349693251534, 0.05, 0.075, 0.1]. Initialization is diagnostic and excluded from summaries. Full horizon was never adjusted from observed curves.

SpuCo official commit `91842463a577b0a90b4f829fec367308f3052e9f`, installed version `2.0.3`. The inspected generator assigns its class-associated background to the configured fraction; independence is 1/K=.1. Official source-training allocation excludes its fixed validation IDs. Reserve, by ordinary class only, 256/class in each of A and B and 128/class for health. Nested ranks fix 64,128,256 probes. Training IDs, reservations and digit order are identical across conditions. One initialization per seed is saved and shared across conditions. Data seed=1701, reservation seed=2903; initialization/training seeds=11,22,33; epoch permutation Generator seed=training_seed*100000+epoch. No DataLoader workers or stochastic transforms are used.

Verification passed: True. Source identity disjointness, official validation exclusion, labels, background palette, finite RGB values, and **every nonzero foreground pixel** were checked. Pixel errors are zero. Realised alignments and all role/class contingency tables are in the data directory. The torchvision downloader fetched its standard raw archives; test images/outcomes were not used. Background annotations remain in `data/20261116_01/offline_diagnostics` and are absent from forecasting feature files.

![Matched identities](../../data/20261116_01/matched_images.png)

## Implementation correctness

CNN A: 3→16→32→64 convolutions, 3x3/pad1 with trainable biases and ReLU; max pool after first two stages; global spatial average; 64→10 biased head. No BN/dropout. Primary shallow=conv1 weights, middle=conv2 weights, deep=conv3+head weights. Biases excluded. Per-layer signed/absolute totals, means per weight, convolution-only shares, head fraction and bias norms/sensitivities are separate diagnostics.

At frozen weights, gradients of each microbatch mean CE are accumulated with microbatch_size/total_probe_size; only then compute |weight*gradient| and aggregate. autograd.grad avoids .grad writes. Evaluation mode, module training flags and Python/NumPy/CPU/CUDA RNG states are restored. Optimizer is not passed to measurement. Gradients are discarded after summarization. Mean-loss gradient accumulation uses model dtype (float32 training; float64 analytical checks). epsilon=1e-12; denominator and relative epsilon logged; zero/nonfinite or >=1% stabilizer contribution invalidates shares.

Analytical state A normalized sensitivities: [1.0, 1.0, 1.0, 1.0]; state B: [1.0, 1.5000000000000004, 0.7500000000000001, 0.7499999999999999]. These are correctness fixtures, not feasibility evidence. Full actual errors, gradcheck settings, isolation checks and neutrality comparisons are in `../../runs/20261116_01/correctness_metrics.json`. pytest log/JUnit histories preserve attempted checks. Analytical tolerance: atol=1e-9,rtol=1e-7; float32 gradients/totals atol=1e-6,rtol=1e-4, with a separate 1e-5 block-share screen. Gradcheck eps=1e-6,atol=1e-5,rtol=.001. Tolerances were not relaxed after failures.

Checks cover embedded analytical states, selected multiplicative gates, the common-gate signed/absolute distinction, unequal-final microbatch invariance, positive channel+bias rescaling with outgoing compensation, repeated frozen probes, unchanged parameters/optimizer/RNG/gradients/modes, and deterministic training with/without probes on CPU and CUDA. A separate 12-update smoke run is labelled SMOKE ONLY and excluded from all feasibility tables.

## All six real prefixes and source health

Keep every run, including ineligible runs. Eligibility is accuracy>=.80, mean CE<=ln(10), finite loss and telemetry at 10%; eligibility never filters the reliability calculation.

| run | health_accuracy | health_loss | eligible |
| --- | --- | --- | --- |
| independent_s11 | 0.70625 | 1.00195 | False |
| independent_s22 | 0.684375 | 1.0443 | False |
| independent_s33 | 0.515625 | 1.32771 | False |
| correlated_s11 | 0.95625 | 0.267272 | True |
| correlated_s22 | 0.95625 | 0.255171 | True |
| correlated_s33 | 0.957031 | 0.260415 | True |

![Source health](figures/source_health.png)

## Reliability of all eight retained features

At 5% and 10%, each A/B panel separately gives shallow/deep last, checkpoint mean, population SD and OLS slope versus actual full-horizon fraction. Initialization excluded; no interpolation; any missing/invalid checkpoint invalidates the complete feature vector. Noise=RMS(A-B)/sqrt(2); spread=sample SD across six runwise (A+B)/2 values (ddof=1). Raw ratios and zero/flat/invalid statuses are retained in CSV.

The screen uses spread/max(noise,numerical_floor). A conservative prospective level/mean/SD floor is 1e-5 share units; slope floor is 2e-5 divided by the prefix's last-minus-first observed fraction. A measured repeated-feature RMS/sqrt(2) larger than this replaces the floor. Exact zero noise gives undefined raw ratio, never an artificial finite/infinite success; screen uses the floor. A feature below the floor is flat. No confidence intervals or significance tests are computed.

| probe_per_class | prefix | feature | noise | spread | numerical_floor | screen_ratio | passes_screen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.05 | shallow_last | 0.0236064 | 0.176455 | 1e-05 | 7.47489 | True |
| 64 | 0.05 | shallow_mean | 0.00718447 | 0.131504 | 1e-05 | 18.3039 | True |
| 64 | 0.05 | shallow_std | 0.00667576 | 0.0412991 | 1e-05 | 6.18642 | True |
| 64 | 0.05 | shallow_slope | 0.6719 | 3.18073 | 0.000499617 | 4.73393 | True |
| 64 | 0.05 | deep_last | 0.0399689 | 0.21547 | 1e-05 | 5.39093 | True |
| 64 | 0.05 | deep_mean | 0.0134805 | 0.172151 | 1e-05 | 12.7703 | True |
| 64 | 0.05 | deep_std | 0.013759 | 0.0297884 | 1e-05 | 2.16501 | False |
| 64 | 0.05 | deep_slope | 1.35371 | 3.54487 | 0.000499617 | 2.61863 | False |
| 64 | 0.1 | shallow_last | 0.0546486 | 0.113362 | 1e-05 | 2.07438 | False |
| 64 | 0.1 | shallow_mean | 0.0109297 | 0.122659 | 1e-05 | 11.2226 | True |
| 64 | 0.1 | shallow_std | 0.0133578 | 0.0338764 | 1e-05 | 2.53608 | False |
| 64 | 0.1 | shallow_slope | 0.543561 | 0.931703 | 0.000222147 | 1.71407 | False |
| 64 | 0.1 | deep_last | 0.0511585 | 0.159907 | 1e-05 | 3.12571 | True |
| 64 | 0.1 | deep_mean | 0.011878 | 0.164309 | 1e-05 | 13.8331 | True |
| 64 | 0.1 | deep_std | 0.0161301 | 0.0270376 | 1e-05 | 1.67622 | False |
| 64 | 0.1 | deep_slope | 0.605149 | 1.38763 | 0.000222147 | 2.29304 | False |
| 128 | 0.05 | shallow_last | 0.0124491 | 0.170241 | 1e-05 | 13.6749 | True |
| 128 | 0.05 | shallow_mean | 0.0155846 | 0.125155 | 1e-05 | 8.03072 | True |
| 128 | 0.05 | shallow_std | 0.00863653 | 0.0487111 | 1e-05 | 5.64013 | True |
| 128 | 0.05 | shallow_slope | 0.644191 | 3.15713 | 0.000499617 | 4.90092 | True |
| 128 | 0.05 | deep_last | 0.0143444 | 0.203378 | 1e-05 | 14.1782 | True |
| 128 | 0.05 | deep_mean | 0.0254399 | 0.163603 | 1e-05 | 6.43095 | True |
| 128 | 0.05 | deep_std | 0.0123291 | 0.0370112 | 1e-05 | 3.00195 | True |
| 128 | 0.05 | deep_slope | 1.21536 | 3.31337 | 0.000499617 | 2.72624 | False |
| 128 | 0.1 | shallow_last | 0.0525712 | 0.116823 | 1e-05 | 2.22219 | False |
| 128 | 0.1 | shallow_mean | 0.0150412 | 0.118584 | 1e-05 | 7.88393 | True |
| 128 | 0.1 | shallow_std | 0.00953014 | 0.0376385 | 1e-05 | 3.94941 | True |
| 128 | 0.1 | shallow_slope | 0.546237 | 0.948805 | 0.000222147 | 1.73698 | False |
| 128 | 0.1 | deep_last | 0.0638965 | 0.152024 | 1e-05 | 2.37922 | False |
| 128 | 0.1 | deep_mean | 0.0226208 | 0.157752 | 1e-05 | 6.97376 | True |
| 128 | 0.1 | deep_std | 0.0139334 | 0.0280332 | 1e-05 | 2.01194 | False |
| 128 | 0.1 | deep_slope | 0.784646 | 1.3325 | 0.000222147 | 1.69822 | False |
| 256 | 0.05 | shallow_last | 0.0118405 | 0.182367 | 1e-05 | 15.402 | True |
| 256 | 0.05 | shallow_mean | 0.00523755 | 0.125913 | 1e-05 | 24.0404 | True |
| 256 | 0.05 | shallow_std | 0.00527716 | 0.0441706 | 1e-05 | 8.37016 | True |
| 256 | 0.05 | shallow_slope | 0.494112 | 3.45286 | 0.000499617 | 6.98801 | True |
| 256 | 0.05 | deep_last | 0.0239134 | 0.22394 | 1e-05 | 9.36463 | True |
| 256 | 0.05 | deep_mean | 0.00554772 | 0.165787 | 1e-05 | 29.8839 | True |
| 256 | 0.05 | deep_std | 0.0139675 | 0.0328175 | 1e-05 | 2.34956 | False |
| 256 | 0.05 | deep_slope | 1.01305 | 4.01539 | 0.000499617 | 3.96368 | True |
| 256 | 0.1 | shallow_last | 0.0479864 | 0.109547 | 1e-05 | 2.28287 | False |
| 256 | 0.1 | shallow_mean | 0.0103377 | 0.120001 | 1e-05 | 11.6081 | True |
| 256 | 0.1 | shallow_std | 0.00965562 | 0.037396 | 1e-05 | 3.87298 | True |
| 256 | 0.1 | shallow_slope | 0.46267 | 0.948509 | 0.000222147 | 2.05008 | False |
| 256 | 0.1 | deep_last | 0.0446407 | 0.151086 | 1e-05 | 3.38448 | True |
| 256 | 0.1 | deep_mean | 0.0107776 | 0.161906 | 1e-05 | 15.0225 | True |
| 256 | 0.1 | deep_std | 0.0144077 | 0.031139 | 1e-05 | 2.16128 | False |
| 256 | 0.1 | deep_slope | 0.451292 | 1.3765 | 0.000222147 | 3.05014 | True |

Follow-up is checkpoint-only. If any initial feature failed the screen at either prefix, the same checkpoints were remeasured with the fixed nested 128 and 256/class subsets. No retraining or changed horizon/seed/definition. Panels share fixed source-distribution realizations and include class-balanced reservations by design; background groups were not used to select them. Independent A/B differences quantify this one split's sampling sensitivity, not all possible probe pools.

![All feature reliability](figures/reliability_all_features.png)
![Eight features at 5%](figures/eight_features_0.05.png)
![Eight features at 10%](figures/eight_features_0.1.png)

## Architectural, classifier and bias diagnostics

| run | conv1_abs | conv2_abs | conv3_abs | head_abs | bias_norm | bias_sensitivity | total | head_fraction_of_deep | bias_sensitivity_over_weight_total | head_aggregation_flag | bias_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent_s11 | 2.47916 | 0.660342 | 1.20154 | 1.28243 | 0.817864 | 0.114734 | 5.62347 | 0.516281 | 0.0204027 | True | False |
| independent_s22 | 4.58734 | 5.48772 | 4.4142 | 2.57671 | 0.648074 | 0.206814 | 17.066 | 0.36858 | 0.0121185 | False | False |
| independent_s33 | 20.8626 | 16.918 | 10.7866 | 4.11956 | 0.775231 | 1.54402 | 52.6868 | 0.276366 | 0.0293055 | False | False |
| correlated_s11 | 0.342023 | 0.503535 | 0.749073 | 0.764355 | 0.669977 | 0.0249728 | 2.35899 | 0.505049 | 0.0105862 | True | False |
| correlated_s22 | 0.357922 | 0.316953 | 0.526971 | 0.5549 | 0.663554 | 0.0224267 | 1.75675 | 0.512908 | 0.0127661 | True | False |
| correlated_s33 | 0.191652 | 0.372539 | 0.708262 | 0.765806 | 0.721297 | 0.0172661 | 2.03826 | 0.519519 | 0.008471 | True | False |

Head/bias diagnostic flag: True. The prospectively recorded descriptive screens are head>=50% of deep sensitivity or bias sensitivity>=10% of weight total. Such a flag motivates a targeted architectural/bias control; it does not prove dominance or confounding. The deep block includes two learned layers and unequal parameter counts. Six runs cannot rule out these explanations. Convolution-only profiles and means per weight are provided to inspect aggregation; no bias-free retraining/control was performed here.

![Primary blocks, all runs and both panels](figures/primary_shares.png)
![Four separate layers](figures/four_layer_totals.png)
![Signed separate layers](figures/signed_layer_totals.png)
![Convolution-only profiles](figures/convolution_profiles.png)
![Means per weight](figures/per_weight_means.png)
![Bias and head](figures/bias_diagnostics.png)

## Offline background manipulation check

Use the first 32 nested reserved health examples/class (320 total), alternative background=(original background+1) modulo K. Preserve original foreground pixels and labels exactly; only original zero pixels recoloured. Outcomes stay outside feature exports. Correlated-minus-independent mean prediction-flip rate: 0.8031250064571698. A difference <=.05 is descriptively flagged weak/absent, not tested statistically; inspect probability changes too.

| run | n | flip_rate | mean_true_class_probability_change | original_accuracy | alternative_accuracy | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| independent_s11 | 320 | 0.11875 | -0.00218374 | 0.715625 | 0.715625 | 0.0077232 |
| independent_s22 | 320 | 0.140625 | -0.00427858 | 0.7125 | 0.69375 | 0.0075508 |
| independent_s33 | 320 | 0.15 | -0.000168903 | 0.53125 | 0.525 | 0.0064011 |
| correlated_s11 | 320 | 0.925 | -0.855307 | 0.959375 | 0.071875 | 0.0066464 |
| correlated_s22 | 320 | 0.925 | -0.863762 | 0.959375 | 0.071875 | 0.0062511 |
| correlated_s33 | 320 | 0.96875 | -0.877262 | 0.959375 | 0.03125 | 0.0070245 |

## Timing, memory and storage

Dataset preparation: 45.83s. Total training: 13.51s; primary A monitoring: 0.90s; additional B monitoring: 0.90s. Additional checkpoint reliability measurements: 19.12s. Warm-up, source health, checkpoint save time and recolouring inference are separate. Report generation: 3.98s. Summary CSV storage: 566383 bytes. Checkpoint and dataset storage are separate reproducibility costs. GPU operations are synchronized for timed regions. Training time includes batch transfer/forward/backward/update and excludes monitors; CPU index permutation setup and data materialization outside measure() are not comprehensively timed. Probe timers include microbatch transfer/gradients/aggregation but exclude caller's indexing of CPU tensors. Therefore wall-clock recurring overhead may be higher than these component timers.

| training_seconds | primary_A_seconds | validation_B_seconds | health_seconds | warmup_seconds | gpu_training_peak_allocated | gpu_peak_reserved | checkpoint_save_seconds | run | A_monitor_training_ratio | AB_monitor_training_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.29635 | 0.152788 | 0.151035 | 0.0720444 | 0.304164 | 118774272 | 144703488 | 0.208325 | independent_s11 | 0.0665349 | 0.132307 |
| 2.24107 | 0.15041 | 0.143784 | 0.0746943 | 0.0213762 | 118774272 | 144703488 | 0.204989 | independent_s22 | 0.0671155 | 0.131274 |
| 2.23999 | 0.149424 | 0.148879 | 0.0706705 | 0.0200825 | 118774272 | 144703488 | 0.213037 | independent_s33 | 0.0667076 | 0.133172 |
| 2.28713 | 0.153851 | 0.154034 | 0.0719804 | 0.0203298 | 118774272 | 144703488 | 0.202437 | correlated_s11 | 0.0672679 | 0.134616 |
| 2.27918 | 0.154025 | 0.154209 | 0.0737906 | 0.0209839 | 118774272 | 144703488 | 0.211347 | correlated_s22 | 0.0675793 | 0.135239 |
| 2.16439 | 0.14404 | 0.143562 | 0.0730267 | 0.0218572 | 118774272 | 144703488 | 0.205506 | correlated_s33 | 0.0665499 | 0.132879 |

Combined A+B monitoring/training ratio: 0.13325638792116318; excessive-cost flag (>1): False. This threshold was frozen before analysis and is a pilot budget signal, not a deployment standard. GPU peak allocated/reserved and incremental allocated probe memory are measured with PyTorch counters; process RSS before/after and Windows process peak working set are logged. Process peak working set is cumulative and cannot isolate transient CPU probe allocations. GPU allocator reserved memory differs from actual device-wide usage, and WDDM does not give reliable per-process device totals. No claim of exact system-wide incremental memory is made. Initialization monitoring is separately identifiable by step=0; warm-up was excluded from feature summaries.

## Reproducibility, failed setup attempts and limitations

Python `3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]`; device `cuda` / `NVIDIA GeForce RTX 5060 Ti`; CUDA runtime `12.8`. Exact versions, OS, RAM, driver/cuDNN/build, process cache paths and source hashes are in provenance.json and requirements-lock.txt. All project assets/caches/temp files are under the requested root; the selected existing bundled interpreter is external read-only. The requested 3.11 standalone installation was blocked by Windows Application Control (_socket/DLL and build interpreter), preserved under `.venv_python311_blocked`. Automatic approval review rejected the signed python.org installer command as blocked by policy. Existing bundled Python 3.12 was used as the compatible fallback; no security setting was changed and no global package environment modified. Installer failure logs are preserved in root runs/. The official SpuCo checkout contains case-colliding demo notebook paths on Windows; dataset source is unchanged.

Deterministic algorithms strict, cudnn deterministic true/benchmark false, TF32/reduced precision disabled, no AMP, float32 training, CPU float64 analytical checks, four CPU threads, num_workers=0 equivalent explicit batch indexing. Training permutation is independent of monitoring RNG. Checkpoints preserve model, Adam, step, RNG, config and code hashes. Resume checks compatibility and starts from the last committed optimizer update, using deterministic epoch ordering. Bitwise reproducibility across different hardware/software is not promised. Three identical probes assess deterministic numerical behavior on this build, not hardware portability. Official validation/test outcomes untouched.

## Decision rules and next actions

- Correctness failure: repair before further training; preserve failed history.
- Identical-probe instability: resolve numerical/stochastic behavior first.
- Reliability improves at larger probes: adopt and document the measured revised probe budget; remeasure saved checkpoints rather than retrain.
- Sampling dominates even at 256/class: reconsider measurement feasibility.
- Reliable levels but flat/unstable slopes: temporal feasibility unresolved.
- Apparent head/bias dominance: require a targeted control before interpreting redistribution.
- Reliable level and slope variation surviving diagnostics: proceed to a larger completed-training outcome pilot, without treating this as prediction evidence.
- Weak/absent background-dependence contrast: investigate construction/manipulation before interpreting a weak setting effect.
- Excessive monitoring cost: revise checkpoint/probe budget and revalidate reliability.

For this attempt the machine-readable decision above applies, with independent cost, manipulation and structural flags retained in decision.json. Nothing in this experiment measures terminal WGA, outcome diversity or audit gains. A larger outcome pilot needs completed training, additional independent setting families and appropriately separated outcomes. See `../../instructions/STEP_BY_STEP.txt` for exact commands and expected artifacts.

## Primary implementation references

- [Official SpuCo repository](https://github.com/BigML-CS-UCLA/SpuCo) - pinned implementation inspected locally.
- [SpuCo dataset guide](https://spuco.readthedocs.io/en/latest/user-guide/spuco_datasets.html).
- [PyTorch local installation](https://pytorch.org/get-started/locally/) and [documented 2.7.1 wheel pair](https://pytorch.org/get-started/previous-versions/).
- [PyTorch randomness guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).


## Final engineering qualification and revised-budget audit

**The temporal result is marginal and probe-budget dependent.** At 10%, deep slope reaches 3.0501, only 0.0501 above the prespecified screen of 3. The shallow slope still fails. Slopes at 64 and 128/class fail the screen; increasing the probe budget does not improve every feature monotonically. Strong level ratios do not turn this into robust evidence for all eight features. At 256/class, five of eight 10% features pass; three still fail. The exact frozen screen is reported without changing it or adding a post-hoc success threshold. The appropriate conclusion is **qualified measurement feasibility, requiring structural controls and further temporal reliability work**.

| probe_per_class | feature | noise | spread | screen_ratio | passes_screen |
| --- | --- | --- | --- | --- | --- |
| 64 | shallow_slope | 0.543561 | 0.931703 | 1.71407 | False |
| 64 | deep_slope | 0.605149 | 1.38763 | 2.29304 | False |
| 128 | shallow_slope | 0.546237 | 0.948805 | 1.73698 | False |
| 128 | deep_slope | 0.784646 | 1.3325 | 1.69822 | False |
| 256 | shallow_slope | 0.46267 | 0.948509 | 2.05008 | False |
| 256 | deep_slope | 0.451292 | 1.3765 | 3.05014 | True |

This pilot retains all six prefixes, but all three independent-background runs miss 80% source-health accuracy. Consequently the observed between-condition variation also reflects markedly different ordinary learning progress. It is not evidence of useful variation among source-normal candidate classifiers. Three paired seeds per condition cannot establish a generally reproducible between-run distribution. Even within this pilot, A/B panel changes remain substantial for several features.

At the revised 256/class budget and across both panels, the head is 27.3% to 50.0% of deep sensitivity; bias sensitivity is 0.92% to 3.00% of total weight sensitivity. Selected-budget head flag=False, bias flag=False. Thus the flagged concern is classifier-head aggregation; the bias magnitude screen does not flag these endpoints. This does not rule out bias effects on trajectories. Inspect the separate learned layers, convolution-only shares and per-weight means below before attributing differences to depth-specific redistribution.

![256/class primary profiles](figures/probe_256/primary_shares.png)
![256/class separate layers](figures/probe_256/four_layer_totals.png)
![256/class convolution-only profiles](figures/probe_256/convolution_profiles.png)
![256/class means per weight](figures/probe_256/per_weight_means.png)
![256/class bias diagnostics](figures/probe_256/bias_diagnostics.png)

The main timing table describes the original 64/class online pilot. The revised budget must also be costed. Combined A+B component monitoring/training ratios by budget: {64: 0.1333, 128: 0.2433, 256: 0.3934}. These ratios compare seven repeated frozen-checkpoint measurements, including initialization, to the original training component time. The larger-budget timing measurements were offline, not interleaved training; device warm-up/caching context can differ. They are prospective monitoring cost estimates, not a measured retrained system. Caller-side indexing, process startup/data loading and all allocation overhead are not fully covered by the component timers. Peak CUDA incremental allocations are allocator-counter measurements; process peak working set is cumulative, not an incremental CPU peak.

| probe_per_class | panel | monitor_all_points_seconds | monitor_post_initialization_seconds | max_gpu_allocated_bytes | max_incremental_probe_allocated_bytes | max_process_peak_working_set_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | A | 0.904538 | 0.7765 | 120281600 | 51571712 | 2363359232 |
| 64 | B | 0.895504 | 0.766306 | 120281600 | 51571712 | 2363359232 |
| 128 | A | 1.78406 | 1.29207 | 118779392 | 118680576 | 2184282112 |
| 128 | B | 1.50268 | 1.27901 | 118779392 | 51571712 | 2184282112 |
| 256 | A | 2.66989 | 2.29215 | 118779392 | 51571712 | 2184282112 |
| 256 | B | 2.64424 | 2.27301 | 118779392 | 51571712 | 2184282112 |

![Probe cost and memory](figures/cost_memory_by_budget.png)

All diagnostic profiles are generated separately for 128 and 256/class under figures/probe_128 and figures/probe_256. Full per-panel structural endpoints, convolution-only/head/bias prefix diagnostics, and timing/memory for every budget are separate CSVs. Forecasting exports still contain only the same eight prespecified features and identifiers.

Independent artifact checks confirm all six Adam states reached exactly 652 updates, initial weights match across conditions for each seed, raw original IDs/labels/reservations/nested ranks match, checkpoints and current code match the pretraining source snapshot, and 67 installed SpuCo Python sources equal the pinned official checkout. Independent feature recalculation using explicit arrays and numpy.polyfit has maximum discrepancy 7.55e-15. The observed 4.44e-16 identical-summary difference includes CSV serialization roundoff; direct repeated-measurement correctness tests had exact zero difference on CPU/CUDA.

No execution step remains pending and no training process is left running. The next research work is a targeted head/aggregation control, additional independent settings/probe reservations, and reconsideration of source-health coverage before a completed-training outcome pilot. Those are follow-up studies, not missing execution steps in this test. The proposal was left unchanged. Setup failures and all inconclusive feature results remain preserved.
