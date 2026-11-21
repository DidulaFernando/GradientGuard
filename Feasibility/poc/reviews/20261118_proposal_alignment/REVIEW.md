# Read-only proposal alignment review — 18 November 2026

## Conclusion

The inspected core implementation agrees with the proposal's CNN A measurement,
eight-feature representation and primary audit comparison. The completed experiment
was an explicitly amended exploratory proof of concept, not an implementation of
every study, comparator or confirmatory prerequisite in proposal v5. No new core
measurement error was identified by this review. This is not an independent numerical
replication or a guarantee that the software contains no defects.

## Sources inspected

- Proposal: C:/Users/rangi/Documents/ChatGPT/Explanations/GradientGuard_Research_Proposal_Didula_Fernando_v5.pdf;
  complete relevant text on pages 3, 8–11, 13–14 and 16, plus previously extracted
  proposal sections on scope/outcomes and audit evaluation.
- src/model.py and src/measurement.py.
- poc/study.py and poc/prediction.py.
- poc/outputs/20261116_poc01/reports/validation_audit.json.
- poc/final_integration_tests.log and poc/ENGINEERING_HISTORY.md.
- Frozen source, predictor, prediction, protected-artifact and final artifact records.

## Verified core correspondence

1. CNN A: widths 16/32/64, 3x3 padded biased convolutions, ReLU, max pooling after
   the first two convolutions, global average pooling, 64-to-10 biased head, no
   batch normalisation/dropout.
2. Connection sensitivity: sample-count-weighted accumulation of signed mean-loss
   gradients, followed by absolute weight-times-gradient sums. Kernel/head weights
   enter primary blocks; biases are separately logged. Deep includes conv3 and head.
3. Shares use total weight sensitivity plus epsilon. The middle block contributes
   to the denominator. Shallow/deep last, mean, population SD and OLS slope produce
   exactly eight retained layer features; initialization is excluded and actual
   horizon fractions define slopes.
4. The logistic E/B/L2/L6/L8 ladder uses the same selected conventional inputs for
   augmented models and grouped development tuning. Family/seed/generator/group and
   terminal outcome identifiers are excluded from early predictor inputs. Final
   source loss/accuracy are restricted to the explicitly separate final comparator.
5. Prediction at 10%, source-normal eligibility, full-horizon training, terminal
   WGA <0.75, floor(0.25N) audit capacity and paired whole-family analysis match the
   core decision. Predictor and prediction freezes precede held-out outcome access.

## Evidence and its limits

Historical integration record: 21 checks passed. Historical completion audit: all
96 models reached 19,560 updates; final states matched full-horizon recovery states;
finite required observations and 100 outcome groups per model were checked.

Today's standard-library hash audit found zero mismatches among 582 protected
prior artifacts (including the proposal) and 2,259 recorded PoC artifacts. Current
run.py/study.py match classifier freezes; prediction.py matches predictor-freeze
source. This verifies artifact identity, not fresh execution correctness.

## Differences from the broader proposal

- The later authorised protocol freezes 60 epochs, 24 Cartesian setting families,
  96 CNN A runs and minimum support conventions. These are exploratory amendments,
  not a precision-validated confirmatory allocation.
- The earlier partition-stability, aggregation and all-seed source-health gates did
  not jointly pass. This PoC was explicitly authorised despite those unresolved gates.
- Prediction depth, some fuller telemetry/change signals, checkpoint-order and
  layer-identity permutation controls, CNN B transfer, 20% timing, matched-run and
  recolouring-association analyses were not all implemented in this PoC. Conditional
  Waterbirds/cancellation extensions were outside its authorised scope.
- Histogram boosting used min_samples_leaf=20, restricting capacity in small grouped
  training folds. The final comparator uses a limited set of final source metrics.
- Confirmatory family-interval precision/coverage planning and full operational cost
  accounting specified in the proposal were not established. Intervals are descriptive.

## Current execution limitation

On 18 November, .venv/Scripts/python.exe failed because pyvenv.cfg points to the
previous Codex runtime's removed Python path (version 26-1112). The installed app
now supplies runtime 26-1117. No environment configuration or packages were changed
in this review. Current bundled Python was used only for PDF extraction and
standard-library hashing, not training or numerical revalidation. Historical test
passes must not be described as newly rerun today. The environment needs a controlled
repair and version comparison before another numerical execution.

## Interpretation

The available evidence supports describing the completed run as a tested implementation
of the amended exploratory protocol and a core proposal-aligned PoC. It does not
support claiming a complete execution or decisive rejection of the entire proposal.
The observed primary discovery tie and lack of added slope benefit remain the
reported findings. No outcomes, models, predictions, thresholds or frozen reports
were modified during this review.
