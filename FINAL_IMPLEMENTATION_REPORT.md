# Final implementation report — RHU Catalog Explorer

Date: 2026-11-13 · Environment: Windows 11, Python 3.12.6, project venv `.venv` with scikit-learn 1.8.0, numpy 2.3.5, pandas 2.2.3, joblib 1.5.3, streamlit 1.67.0, pytest 9.2.0.

## Outcome

A working local Streamlit app answers catalog questions using exact structured lookups or verbatim cited excerpts. It declines personal, live and off-catalog questions without attaching citations. An inspector shows the real computation for each turn. All acceptance-criteria commands were run and passed (details below). `inputs/` was not modified: preflight hashes still pass after all work.

## Files created or replaced

| File | Role |
|---|---|
| `app.py` | Streamlit UI: header, 3 example chips, chat history, status badges, source cards, Clear chat, "Show NLP steps" inspector (6-stage strip + 6 tabs, per-turn selector, Altair score chart). |
| `src/__init__.py`, `src/rhu_chat/__init__.py` | Package markers (lazy `RHUChatPipeline` export). |
| `src/rhu_chat/data.py` | Loads the 4 knowledge CSVs read-only; whitespace-normalised page text; citation URL validation (`#page=N` must match the page); hashes. |
| `src/rhu_chat/retrieval.py` | Word TF-IDF (1–2-grams, English stop words, sublinear tf) over section + subsection + passage text; cache in `artifacts/` invalidated by CSV hashes; cosine + explicit boosts (category hint +0.03, typed course code +0.10). |
| `src/rhu_chat/nlp.py` | Rules on the original question: course-code regex (`CCEE566`, `CCEE-566`, `ccee 566`), field detection, scope flags (personal / live / off-topic / instruction override), program and semester clues, fuzzy course-title match (difflib ≥ 0.88). |
| `src/rhu_chat/qa.py` | Grounded answers: credits and prerequisite/corequisite/equivalent from course rows (with conflict listing and page-text verification), page-break continuation, study-plan rows, IDF-coverage-gated extractive excerpts, fallbacks. |
| `src/rhu_chat/pipeline.py` | `RHUChatPipeline.from_project_root(Path)` / `ask(question) -> dict`, per `docs/PIPELINE_CONTRACT.md`. Imports the unmodified `rhu_classifier_api.py` by path and uses `predict_one` + `make_qa_request`. |
| `src/rhu_chat/ui.py` | Markdown/HTML escaping for untrusted text, CSS, process-strip HTML (fixed labels and numbers only). |
| `scripts/build_index.py`, `scripts/smoke_app.py` | Replaced the intentional stubs with real implementations. |
| `scripts/run_seed_eval.py` | Writes `eval/seed_run_results.csv` (what the system did; correctness columns left blank for a human). |
| `tests/test_grounding_details.py` | 16 extra tests. The supplied `tests/test_pipeline_contract.py` is unchanged. |
| `requirements.txt`, `pytest.ini`, `.streamlit/config.toml`, `.claude/launch.json` | Pinned deps, test path, navy/teal/off-white theme with telemetry off, local preview config. |
| `README_RUN.md`, `IMPLEMENTATION_STATUS.md`, this report | Docs. |

## Commands run and observed results

| Command | Result |
|---|---|
| `python scripts/preflight.py` (system Python, before any work) | PASS, 17 assets, schemas OK. |
| `python -c "import sklearn…"` (system Python) | scikit-learn **1.7.2** ≠ 1.8.0 used for the joblib, so I created `.venv` with the exact pins instead. |
| `python -m venv .venv` + `pip install` pinned versions | OK (sklearn 1.8.0, numpy 2.3.5, pandas 2.2.3, joblib 1.5.3, streamlit 1.67.0, pytest 9.2.0). |
| `.venv/Scripts/python scripts/build_index.py --force` | `Index rebuilt: 2502 passages x 42739 TF-IDF features in 527 ms`; self-check answered, cites [155, 344, 347]. |
| `.venv/Scripts/python scripts/build_index.py` (second run) | `Index reused … in 223 ms`. |
| `.venv/Scripts/python -m pytest -q` | **23 passed** (7 supplied contract tests + 16 new). |
| `.venv/Scripts/python scripts/smoke_app.py` | `AppTest: script ran, chat turn answered=True`; `HTTP: /_stcore/health -> ok; / -> 200`; **PASS**. |
| `.venv/Scripts/python scripts/run_seed_eval.py` | 25 seed questions run; results in `eval/seed_run_results.csv` (summary below). |
| Browser check (Claude desktop browser pane, `streamlit run app.py` on :8501) | Checked at 1400×900 and 375×812: example chips, typed questions, 3-turn history, conflict answer with 4 source cards and working PDF links, inspector tabs, bar chart, turn selector, Clear chat. The server log showed only Windows `ConnectionResetError 10054` asyncio noise from page reloads. No screenshots were saved to the repo. |

### Failures found during development, and fixes

1. **Off-target policy excerpts.** "How do I appeal a grade?" first returned a GPA sentence. Root cause: any single shared word was enough evidence. Fix: the evidence passage must cover ≥ 60% of the question's IDF-weighted content words. Section and subsection headings are now indexed too, so the real *Appeal of Course Grade* passage (p. 59) is found.
2. **Coverage was 0.00 everywhere** right after that change. Root cause: a shell-escaping mistake put literal backspace characters into the word regex. I fixed it in source and confirmed the coverage values in the trace.
3. **Filler words** ("catalog", "say", "tell") and instruction-override words diluted coverage and caused false fallbacks. They are now excluded from the coverage measure only; retrieval is unchanged.
4. **Hidden third CCEE 566 variant.** The p. 155 Data Analytics record is cut at a page break, and its prerequisite is on p. 156 (passage RHU-P00842: "Prerequisite: CCEE 214 or COSC 214."). I added a narrow continuation rule: the description has no final period, and the first passage of the same section on the next page does not start with a course code. It is cited separately.
5. **Code-less course questions** ("prerequisites for the NLP course") showed only one of the three variants. When the evidence passage is itself a course entry, the pipeline now switches to the conflict-aware structured lookup and says so.
6. **UI:** chart bars were alphabetical with skipped labels (now an Altair chart ordered by rank); "Clear chat" was disabled after the first turn (fixed); fuzzy title matching cost ~200 ms (now ~20–50 ms with a quick-ratio pre-filter); deprecated `use_container_width` was replaced.

## Three real example answers (exact output)

**1. `How many credits is CCEE 566?`** → `answered` / `exact_course_match`
> CCEE 566 (Natural Language Processing) is a 3-credit course (listed as 3(3,0)). All 3 catalog records agree (p. 155, p. 344, p. 347).

Citations: RHU-C0243 [p. 155](https://www.rhu.edu.lb/Library/Assets/Gallery/Files/RHU_Undegraduate_Catalog_2026-2027.pdf#page=155) (Minor in Data Analytics) · RHU-C0587 [p. 344](https://www.rhu.edu.lb/Library/Assets/Gallery/Files/RHU_Undegraduate_Catalog_2026-2027.pdf#page=344) (Computer and Communications Engineering Program) · RHU-C0596 [p. 347](https://www.rhu.edu.lb/Library/Assets/Gallery/Files/RHU_Undegraduate_Catalog_2026-2027.pdf#page=347) (Minor in Artificial Intelligence). Each excerpt is the extracted page text starting "CCEE 566 Natural Language Processing 3(3,0) …". V3.1: courses / course_credits, routing ok. Measured total 13.2 ms.

**2. `What are the prerequisites for CCEE 566?`** → `answered` / `conflicting_records`
> The catalog contains different prerequisite entries for CCEE 566 (Natural Language Processing), depending on the section:
> • Minor in Data Analytics entry (p. 155, continued on p. 156): CCEE 214 or COSC 214
> • Computer and Communications Engineering Program entry (p. 344): CCEE 214, COSC 214, or Equivalent
> • Minor in Artificial Intelligence entry (p. 347): CCEE 214
> I'm not choosing one automatically — please confirm which program applies to you with your advisor.

Citations: RHU-C0243 p. 155 · RHU-P00842 p. 156 ("…for the above NLP applications. Prerequisite: CCEE 214 or COSC 214.") · RHU-C0587 p. 344 ("Prerequisite: CCEE 214, COSC 214, or Equivalent.") · RHU-C0596 p. 347 ("Prerequisite: CCEE 214"). Each prerequisite value was found in the extracted text of its page (or the next page), so `verified_in_page_text` is true for all.

**3. `How do I appeal a grade?`** → `answered` / `excerpt_supported`
> From the catalog (Academic Regulations › Appeal of Course Grade, p. 59): "If a student feels that the grade s/he has attained in a course was unfair, s/he should discuss the matter with the instructor within five days of the grade being posted by the Registrar's Office. If the student's concern is legitimate, the Dean shall form a committee …"

Citation: RHU-P00238 [p. 59](https://www.rhu.edu.lb/Library/Assets/Gallery/Files/RHU_Undegraduate_Catalog_2026-2027.pdf#page=59). V3.1 predicted raw intent `unanswerable_current_grade` and reconciled it to `gpa` (`reconciled_from_category`). The full-catalog retrieval still found the right passage, which shows the classifier working only as a soft hint.

## Seed-question run (25 questions; not a benchmark)

- **Answered with citations (16):** credits, prerequisite conflict (4 phrasings, including `CCEE566` and `CCEE-566`), "without CCEE 214", Describe CCEE 566, COSC 214, CS semester 1 (p. 143), EE "semester four" → Year 2 Spring (p. 308), grade appeal (p. 59), probation (p. 67), attendance (p. 57), admission (p. 30), withdrawal (p. 58), CS total 93 credit hours (p. 143), "what is a minor" (p. 259), and the prompt-injection attendance question (p. 57, with a "real pages only" note).
- **Declined without citations (8):** CCEE 999 (no substitution), GPA, transcript, advisor appointment, weather, football, next week's deadline.
- **Invalid input (1):** empty question.

The `citation_correct` / `answer_correct` columns in `eval/seed_run_results.csv` are blank on purpose. They need a human check against the PDF. I checked citations against the bundled extracted page text (`RHU_CATALOG_FULL_TEXT_BY_PAGE.csv`), not by visually rendering PDF pages.

## Acceptance criteria status

All functional, grounding, inspection and packaging items in `docs/ACCEPTANCE_CRITERIA.md` were exercised. Clean-venv install: done with pinned pip installs (the same versions as `requirements.txt`) in a new `.venv`, but not by literally running `pip install -r requirements.txt` from scratch on a second machine. macOS/Linux commands are documented but were not executed (Windows only here).

## Known limitations

- **Weak-but-cited excerpts are possible.** "What is a minor?" returns a verbatim line about declaring the Business Administration minor (p. 259) rather than the general *Minors* rules (p. 65). "GPA required to graduate" returns a graduation-award sentence. Both are cited and labelled with their section, but they are not ideal answers. The IDF-coverage gate reduces off-topic answers but is lexical, not semantic.
- **Extraction artefacts** in the source tables carry through: table rows sometimes form long run-on "sentences" (e.g. fee tables), and 67 course records are flagged `needs_manual_review`. These are surfaced as notes, not fixed.
- **The continuation rule** (p. 155 → p. 156) is a heuristic and is labelled as one. It only fires when a description lacks a final period and the next page's first passage in the same section starts without a course code.
- **Scope rules** (personal, live, off-topic) are keyword regexes. Unusual phrasings can slip through to retrieval, where the coverage gate is the remaining safeguard.
- **Study-plan semester ordinals** ("semester four") count regular (non-summer) semesters in catalog order. That is a stated interpretation, not a catalog definition.
- **V3.1 is used unmodified.** Its category/intent hints are often wrong on unseen phrasings (see example 3). By design they only add +0.03 to ranking.
- **No independent accuracy figure.** No human-labelled student question set was available, and the historical QA test was not used for tuning. The thresholds (cosine 0.10, coverage 0.6, title ratio 0.88) were set by hand from a few catalog questions.
- Voice input and ABSA are **not built** in this phase.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_index.py
python -m pytest -q
python -m streamlit run app.py
```
