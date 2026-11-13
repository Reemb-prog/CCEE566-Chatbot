# Acceptance checklist for Claude's completed app

Every requirement below is testable and suited to a short capstone prototype. If tests uncover a source conflict, prefer showing it over faking a single answer.

## Functional

- [ ] `python scripts/preflight.py` passes on unmodified sources.
- [ ] Full project runs on a clean local virtual environment using documented `pip install -r requirements.txt`.
- [ ] `python scripts/build_index.py` builds/reloads index from bundled source tables without network calls; subsequent launches are fast.
- [ ] `python -m pytest -q` passes all supplied contract tests. Do not weaken or delete tests to create false success.
- [ ] `python scripts/smoke_app.py` starts the Streamlit server and verifies HTTP health (if dependency installed).
- [ ] `python -m streamlit run app.py` shows a usable chat and inspector, not just static text.

## Grounding and domain cases

- [ ] `How many credits is CCEE 566?` yields 3 credits with at least one real source citation.
- [ ] `What are the prerequisites for CCEE 566?` shows different prerequisite entries with pages **344 and 347**, scoped to their source program where feasible. Does not claim one universal prerequisite.
- [ ] `What is my current GPA?` declines personal-data request; no invented GPA or irrelevant citation.
- [ ] `What is the weather in Beirut?` declines as outside this static catalog.
- [ ] A mistyped or code-less/ambiguous course question either finds relevant evidence or asks for clarification; no arbitrary wrong course.
- [ ] At least two non-course catalog policy questions show relevant, **verbatim** sourced excerpt or reasonable fallback; never invent policy rules.
- [ ] All citations open the real catalog PDF page and include an actual excerpt or source field, not a merely plausible page number.
- [ ] Wrong classifier categories do not prevent correct full-catalog retrieval.

## Inspection / pedagogy

- [ ] Per-turn visualization shows actual tokens, extracted entities, raw/reconciled classification labels, top retrieved evidence and scores, evidence decision and timings.
- [ ] Raw TF-IDF cosine and boosted rank score are differentiated. No fake '94% sure' confidence numbers.
- [ ] Chat history persists within the session. Clear chat works, and latest-turn trace is visible with per-turn selection or per-answer expandable inspector.
- [ ] Inspector contains no user-visible secrets and does not run arbitrary HTML from source text.

## Packaging / reporting

- [ ] `README_RUN.md` documents full setup and usage on Windows and macOS/Linux, including Python/model dependency compatibility.
- [ ] `FINAL_IMPLEMENTATION_REPORT.md` lists actual commands executed and observed results (including failures/changes), sources used, unresolved cases and demo script.
- [ ] No added cloud service, large new model, database server or login flow.
- [ ] No modifications to `inputs/` (except user explicitly replaces the catalog); changes reside in application/scripts/docs only.
