# Repository workflow and artifact policy

Origin: <https://github.com/DidulaFernando/GradientGuard>

## Initial import

The existing work is imported as separately reviewable commits covering repository setup, the first feasibility harness, first-study evidence, the targeted follow-up implementation and evidence, the outcome PoC implementation and evidence, and post-hoc diagnostics. All import commits use their actual creation date; they do not reconstruct an invented chronology.

Source snapshots, frozen configurations, exact dependency records, manifests, CSV measurements, figures, reports, decisions and engineering/failure logs are retained. Compact frozen forecaster `.joblib` files are retained with their compatibility records; load only trusted project files under the recorded compatible environment.

## Local-only artifacts

`.gitignore` excludes virtual environments (including the blocked old environment), caches, temporary files, downloaded interpreters/tools, raw MNIST downloads, numerical dataset arrays, classifier weight/recovery checkpoints and SQLite caches. These files are not deleted or relocated. No existing result or dataset is overwritten for this import.

Generated dataset arrays and model weights are omitted to avoid turning source history into a benchmark/model binary store. Compact original-ID manifests, background assignments, seed records and verification reports remain tracked. Exact recovery from a clone requires the matching software and the corresponding dataset/training procedures. Download access, runtime, hardware compatibility and deterministic limitations still apply; this is not a claim that omitted binaries are already hosted elsewhere.

## Automatic commits in future work

The user requested automatic local commits after completed implementation work. The repository's `AGENTS.md` records this preference for future coding sessions. Review and stage only the relevant files, run appropriate checks and create a meaningful commit. Separate implementation from substantial scientific result additions where useful. Preserve negative results.

This does not install a background watcher or auto-commit partially written experiments. Future pushes require publication authorisation in the active task; do not rewrite shared history.

## Useful PowerShell commands

From `E:\Final Year Project\Implementation`, with Git available:

```powershell
git status --short --branch
git log --oneline --decorate -12
git remote -v
git diff --stat
git diff --cached --stat
git add -- <specific completed files>
git commit -m "Describe the completed change"
git push origin main
```

Use the project's verified Python directly for study commands rather than depending on activation. Research instructions under `Feasibility/instructions` define the historical environment and safe stage boundaries.

Git uses repository-local identity settings and disables line-ending conversion. `.gitattributes` preserves historical byte content because scientific fingerprint/resume checks depend on it. Do not normalise frozen files merely for cosmetic consistency.

The import audit utility under `repository_tools` can snapshot or verify the tracked/importable scientific artifacts, check staged-file size and high-confidence credential patterns, and confirm stored blobs match working-tree bytes. Its audit outputs are kept under ignored `.repository-local`.
