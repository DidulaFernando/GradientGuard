# Preserved engineering attempts

- 2026-11-16: project initially empty; no applicable repository instructions. Proposal read as reference, never edited.
- Downloaded project-local uv-managed Python 3.11.18. Windows Application Control blocked _socket.pyd and isolated build Python. Logs: runs/install_dependencies.log and runs/install_dependencies_retry.log. Environment retained as .venv_python311_blocked.
- The python.org 3.11 installer command was rejected by automatic approval review as blocked by policy; it did not run. Security controls were not changed. Used existing bundled Python 3.12.14 to create .venv. This is a documented interpreter exception to the preferred 3.11 setting.
- Initial harness invocation failed before dataset construction/tests: YAML parsed unquoted attempt ID 20261116_01 as an integer. Quoted the identifier, with no changes to scientific settings. Original preparation log retained. No tolerance was changed.
- Official checkout reports colliding case variants of a demo notebook on Windows; dataset Python files unaffected. Commit pinned in provenance.
