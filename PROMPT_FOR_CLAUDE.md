# Task for Claude Code: Implement the finished RHU Catalog Chatbot

Build and run a **fully working** local Streamlit application using the exact assets already in `inputs/`.

**User story.** A student types a question about the RHU 2026–2027 Undergraduate Catalog. The app responds with a short correct answer supported by catalog evidence and a clickable page citation, or it explicitly says that the evidence is insufficient. The student can open an inspector to observe the real NLP stages from raw question to answer: tokenization, course-code detection, category/intent prediction using V3.1, TF-IDF retrieval, final ranking, evidence selection and answer construction.

### Product brief

- Friendly, uncluttered, modern **RHU Catalog Explorer** interface. Great for a live class demo. Use Streamlit chat UI, 2-column layout at desktop widths, consistent restrained palette (navy, teal, off-white), responsive when narrow.
- Main content: header + academic-year subtitle, three tappable example questions, chat bubbles with brief answers and tidy citations. Clear chat button. Keep 10+ messages usable.
- Visual trace: accessible by default as a compact 5- or 6-stage progress strip / side inspector; rich details in expandable tabs. Show actual tokens and matched entities, classifier raw/reconciled labels and review flag, top five retrieved passage scores with a small bar chart, the selected evidence text and page URL, and step durations measured using `time.perf_counter`.
- Explain limitations without burying the UI. Never show internal V3.1 diagnostic vocabulary to students unless in the inspector.
- The app must function offline *except clicking official external PDF links*; a local PDF download link is optional. Do not fetch the live RHU website to answer.
- Retrieval: word TF-IDF and cosine similarity over existing 2,502 catalog passages, supplemented by exact code-based lookup for 748 course records and program-semester lookup for 573 study-plan rows. Category hints only soft-boost; never hard-filter all passages by prediction. No embeddings or APIs.
- Grounding: for course credits/prerequisites return fields only from matching catalog rows and validate page/evidence where feasible; for policy questions return faithful, short extractive excerpts with citation. If low relevance, invalid input, personal account-specific data, or no matching evidence, explicitly decline or ask to clarify. Never turn weak retrieval into a confident fact.
- Evidence collision: `CCEE 566` has multiple source sections and *nonidentical prerequisites* on pages 344 and 347. Treat as intended conflict demo, NOT as a bug to patch by picking one.
- Course code extraction must use original question, not V3.1 classifier-normalized input.
- Keep UI and domain logic separated, with a reusable `RHUChatPipeline.from_project_root(Path)` and `ask(question)->dict` that follows `docs/PIPELINE_CONTRACT.md`.

### Required workflow, do not stop at planning

1. Inventory inputs; run preflight. Read actual column headers and source PDF rather than assuming names.
2. Implement a local index builder, retrieval and structured course/study-plan lookup. Store a reusable local artifact cache under `artifacts/` and invalidate it when input files change.
3. Implement grounded extraction, page citations, contradictory-source handling and fallback.
4. Integrate supplied V3.1 classifier through its API, without modifying model weights. Add an honest trace schema.
5. Implement Streamlit app with polished defaults and process inspector.
6. Run `python scripts/build_index.py`, `python -m pytest -q`, `python scripts/smoke_app.py`, and manually/visually check several questions when browser tooling is available. Fix and rerun until clean. Save screenshots only if supported; don't invent them.
7. Provide `README_RUN.md` with Windows and macOS/Linux commands and `FINAL_IMPLEMENTATION_REPORT.md` with results.

**Quality gate:** Follow `docs/ACCEPTANCE_CRITERIA.md`; do not invent sources or claim the UI works when it has not been launched. Implement first, report last.
