# Harness build review

## Outcome

**READY FOR CLAUDE CODE IMPLEMENTATION.** This ZIP is a development harness and includes all needed reference assets, but it is deliberately not a finished Streamlit application. Implementing and executing the application is Claude Code's assigned task. The placeholders under `scripts/build_index.py` and `scripts/smoke_app.py` intentionally fail until Claude replaces them with real implementations.

## Actual validation performed on the harness

1. Source assets: 17 bundled files checked against SHA-256 values by `python scripts/preflight.py`.
2. Dataset schemas checked using Python CSV reader; required columns present.
3. Catalog PDF opened and verified to contain 429 pages.
4. CCEE 566 course entry found on PDF pages 344 and 347. Extracted structured records disagree on prerequisites, so the conflict demo has authentic source evidence.
5. Archive passed `ZipFile.testzip()` and unpacked to an isolated temporary directory.
6. Preflight also passed after that fresh extraction.
7. All shipped Python files parse successfully; `bash -n RUN_CLAUDE.sh` passed.
8. Both Claude project instructions (`CLAUDE.md`) and optional `.claude/skills` are included.
9. Windows `.bat` launcher is included, but was not executed on a Windows machine.

## Things not claimed

- Claude Code has not been invoked by this harness build.
- The target Streamlit UI has not been implemented or run at the time of packaging.
- Future `pytest` application tests are supplied but will fail until the app is built.
- No human-verified independent student-question accuracy has been established.
- Package compatibility on the user's specific Python/Windows installation remains to be checked by the implementer.

## Risks and controls

- **Joblib version compatibility:** `inputs/classifier/requirements_v31.txt` includes the model's original package versions; use compatible Python and dependencies.
- **No source fabrication:** all application answers must trace back to catalog passages/structured rows and page citations.
- **Low-confidence/unanswerable questions:** the dataset is not a comprehensive evaluation benchmark; fallback and clarification are first-class behavior.
- **Clutter:** technical pipeline visualization belongs in optional inspector, not the main chat bubble.
- **Overengineering:** no cloud LLM/embeddings, heavy frontend framework, account system or voice module in this phase.

## Acceptance decision

The harness is complete for its purpose: it provides concrete assets, grounded domain rules, an enforceable interface contract, a staged implementation process, tests, and a self-auditing workflow for Claude Code. Once Claude produces the app, rely on its own `FINAL_IMPLEMENTATION_REPORT.md` and executed test outputs for any claim that the chatbot is functional.
