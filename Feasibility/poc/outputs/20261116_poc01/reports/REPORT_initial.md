# GradientGuard exploratory outcome PoC

No incremental audit value observed in this pilot.

{
  "verdict": "No incremental audit value observed in this pilot.",
  "completed_classifiers": 96,
  "planned": 96,
  "source_normal_support": {
    "cohort": "eligible",
    "prefix": 0.1,
    "N": 36,
    "families": 9,
    "failures": 12,
    "nonfailures": 24,
    "evaluable": true
  },
  "primary_contrasts": [
    {
      "cohort": "eligible",
      "prefix": 0.1,
      "panel": "A",
      "refit_B": false,
      "control": "E",
      "Delta20": 0.0,
      "interval_low": -1.1111111111111112,
      "interval_high": 4.444444444444445,
      "invalid_resamples": 58,
      "family_count": 10
    },
    {
      "cohort": "eligible",
      "prefix": 0.1,
      "panel": "A",
      "refit_B": false,
      "control": "logistic_B",
      "Delta20": 0.0,
      "interval_low": -1.1111111111111112,
      "interval_high": 4.444444444444445,
      "invalid_resamples": 58,
      "family_count": 10
    },
    {
      "cohort": "eligible",
      "prefix": 0.1,
      "panel": "A",
      "refit_B": false,
      "control": "logistic_L6",
      "Delta20": 0.0,
      "interval_low": 0.0,
      "interval_high": 0.0,
      "invalid_resamples": 58,
      "family_count": 10
    },
    {
      "cohort": "eligible",
      "prefix": 0.1,
      "panel": "A",
      "refit_B": false,
      "control": "logistic_B",
      "Delta20": 0.0,
      "interval_low": -1.1111111111111112,
      "interval_high": 4.444444444444445,
      "invalid_resamples": 58,
      "family_count": 10
    }
  ],
  "limitations": [
    "Exploratory; only 10 held-out setting families",
    "Shared MNIST images and outcome panels",
    "Partition-sensitive temporal measurements and unresolved head/denominator interpretation",
    "Prediction-depth comparator deferred",
    "Family bootstrap conditional on selected forecasters; excludes development selection uncertainty"
  ],
  "legacy_preserved": true
}

See metrics.csv, contrasts.csv, development_support.csv (attempt root), evaluation_support.csv, secondary_cutoffs.csv, group_accuracy_uncertainty.csv and audit_selections.csv. Models were selected only on development validation outcomes; held-out test outcomes were unsealed after predictor and prediction freezes. The 60-epoch horizon is an explicitly user-authorized exploratory amendment, not a passed calibration gate.
