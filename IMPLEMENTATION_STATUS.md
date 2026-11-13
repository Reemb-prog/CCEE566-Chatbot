# Implementation status — RHU Catalog Explorer

| # | Phase (docs/IMPLEMENTATION_PLAN.md) | Status |
|---|---|---|
| 0 | Inventory: preflight, schemas, sample rows, classifier API | done. Preflight PASS (17 assets); real column names read; CCEE 566 rows inspected |
| 1 | Data/index: `data.py`, `retrieval.py`, persistent TF-IDF cache in `artifacts/` | done. 2,502 passages × 42,739 features; hash-invalidated cache, reused on reload |
| 2 | Exact lookup: course codes from original question, course rows, study-plan rows | done. Code spacing/hyphen variants, unknown-prefix suggestion, title fuzzy match, program + semester resolution |
| 3 | Grounded answering: `qa.py` structured answers, extractive excerpts, conflicts, fallback | done. Conflicting credits/prereqs listed with separate citations; page-break continuation; IDF-coverage gate for excerpts |
| 4 | Pipeline trace: `pipeline.py` wraps V3.1 API + retrieval + grounding, real timings | done. JSON-safe trace, `time.perf_counter` per stage, no confidence/probability fields |
| 5 | UI: `app.py` chat, process strip, inspector, evidence cards | done. Checked in a browser at 1400×900 and 375×812 |
| 6 | Verify: pytest, smoke_app, browser check, final report | done. 23/23 tests, smoke PASS, see FINAL_IMPLEMENTATION_REPORT.md |
