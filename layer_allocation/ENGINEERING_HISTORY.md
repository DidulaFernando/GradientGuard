# First structural-control implementation

21 November 2026. All work is namespaced here; old numerical packages and scientific evidence are preserved.

1. The venv launcher pointed to a removed Codex runtime. Backed up `pyvenv.cfg` and repaired only home/executable paths to the available matching Python 3.12.14. Exact 80-line dependency lock matches; no packages downloaded or installed.
2. First pytest collection failed because Windows application control blocked a scikit-learn native component. The execution environment's permission mechanism allowed the retry. No security settings or native binaries were altered.
3. The retry passed 15 checks and failed an RNG-isolation assertion that compared pickle bytes. Equivalent tensor storage can serialize differently. Replaced it with explicit equality of Python/NumPy/PyTorch/CUDA states, preserving numerical tolerances and the intended state check. All 16 tests then passed. Both failed logs remain under the attempt.
4. The scientific protocol froze source/tests/configuration, seeds, measurements, controls and interpretation rules before new measurements or fitting. Reused PoC evaluation outcomes were already known; no claim of untouched confirmation is possible.
5. Stage one intentionally uses saved classifiers. New forecaster fits are exploratory, development-only, and separately stored. It does not start a prospective training study, modify manuscripts or run skipped legacy stages.

The frozen scientific source is under `runs/20261121_structural01/source_snapshot`. The root completed-output audit was added as a validation tool and records its own hash; it does not change the frozen numerical implementation or selection rules.
