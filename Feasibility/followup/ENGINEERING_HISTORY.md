# Follow-up engineering history

- The legacy directories and all original artifacts are read-only. Legacy baseline fingerprints include 294 files plus the unchanged proposal.
- Initial preflight detected differences between the old provenance's importlib.metadata enumeration and current enumeration. The complete pip freeze --all lock is byte-for-line identical after ordering, and numerical package versions, GPU/CUDA/cuDNN, legacy code and installed generator source match. The old enumeration includes setuptools-vendored distributions. Both the failed initial comparison and adjudication were retained; no dependencies were changed.
- Attempt 20261116_02: protocol frozen before new measurements. Tests: 18 passed, 3 failed because Windows hash-map keys used backslashes while the resume guard requested POSIX keys. No scientific repartitioning/calibration/new-realisation measurement was run. Source snapshot, frozen record and failure logs preserved under outputs/20261116_02.
- Attempt 20261116_03: fixed hash-key serialization to portable POSIX form and started a new identifiable attempt. No scientific settings, seed, horizon candidate, tolerance or decision threshold changed.
