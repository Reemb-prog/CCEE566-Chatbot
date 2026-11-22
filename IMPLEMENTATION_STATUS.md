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
| 7 | Informal wording (2026-11-19): `normalize.py` shorthand, title acronyms, catalog-vocabulary spelling | done. 33/33 tests, smoke PASS, seed answers unchanged, browser checked |

## Feedback / ABSA module (docs/FEEDBACK_ABSA_PLAN.md)

| Phase | Work | Status |
|---|---|---|
| F0 | Sources copied to `inputs/feedback/raw/` with checksums; scope updated | done (raw S1 + EduRABSA originals + licences still missing) |
| F1 | `scripts/prepare_feedback_data.py` + `tests/test_feedback_data.py` | done. 26/26 data tests, 59/59 total; deterministic; see `inputs/feedback/processed/data_card.json` |
| F1b | Two-annotator re-label of the 40 human sentences | pending (team) |
| F2 | `scripts/train_feedback_models.py` + `tests/test_feedback_models.py` | done. Teammate Task A reproduced exactly (0.6187); M2 word+char chosen by train CV; metrics in `models/feedback_v1_metrics.json` |
| F3 | `scripts/eval_feedback.py` → `eval/feedback_results.md` | done. Human-40 pair F1 0.488 (95% CI 0.35–0.63) vs 0.164 without generated data; D1 beats the teammate mapping on the annotated-polarity ruler (0.579 vs 0.515) |
| F3b | Fixes after review of human-test errors (team chose to **unseal** the 40) | done. Spelling toward aspect words, 83-sentence generated patch, M3 generated-phrase weight 3 (train CV). Human pair F1 0.488 → 0.512 (now a dev number); sealed result kept in `eval/feedback_results_sealed_v1.md`. 71/71 tests |
| F4 | `src/rhu_chat/feedback.py` | done. Analyzer + `feedback_route()`; detector scores in trace |
| F5 | Feedback view, chat routing, trace, model card | done. Opinions (wording cue required) get a "not a catalog question" reply + "Open in Feedback analyser"; 92/92 tests, smoke PASS, browser checked |
| F6 | Report + run instructions | done. `FINAL_IMPLEMENTATION_REPORT.md` (2026-11-21 section, 3 real examples) and `README_RUN.md` updated |
