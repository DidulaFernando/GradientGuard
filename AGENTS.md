# Repository collaboration instructions

## Scope and preservation

- This repository is rooted at `E:\Final Year Project\Implementation` and its origin is `https://github.com/DidulaFernando/GradientGuard.git`.
- Preserve completed scientific attempts, source snapshots, protocol freezes, outcome files and recorded conclusions. New studies belong in separate namespaced attempts.
- Legacy source/configuration directories are fingerprinted by resume guards. Do not edit them merely to add repository housekeeping.
- Keep environments, downloaded datasets, dependency caches, temporary files and training checkpoints local, as configured in `.gitignore`. Track compact manifests, exact dependency records, source, tests, reports, tables and relevant frozen forecaster pipelines.
- Never stage credentials or force-push/rewrite published history unless the user explicitly authorises that action.

## Standing user preference: automatic commits

The user explicitly requested automatic commits whenever new implementation work is completed (21 November 2026).

- After finishing a coherent implementation change and its appropriate checks, create a descriptive local commit automatically. Do not ask for repeated conversational confirmation.
- Inspect the diff and stage only files relevant to that completed work. Preserve unrelated changes; do not use a blind `git add .`.
- Record relevant validation in the commit body or accompanying documentation. Never describe unrun checks as passed or incomplete experiments as completed.
- Commit new scientific results separately from their implementation when that makes the evidence easier to review. Keep failed and inconclusive outcomes visible.
- Do not commit after every generated checkpoint or transient log update. A commit marks a completed, reviewable unit of work.
- Push when the active task authorises publication; the initial repository import explicitly authorises a push to origin. The standing automatic-commit preference alone does not authorise future history rewrites or unrelated publication.
- This is an instruction for future active coding sessions, not an unattended background watcher.
