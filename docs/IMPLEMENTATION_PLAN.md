# Implementation plan: keep the demo small

| Stage | Implementation | Validation / decision |
|---|---|---|
| 0. Inventory | Run preflight; inspect schemas and 5+ example rows; check trained classifier API | No missing sources; report each artifact and applicable catalog year |
| 1. Data/index | `src/rhu_chat/data.py` parses CSVs; `retrieval.py` builds a persistent TF-IDF index with stable passage IDs, pages and URLs | Correct 2,502 passages; index reload; no network or PDF extraction per question |
| 2. Exact lookup | Extract original course codes; lookup course records; optional study plan matching by explicit program + semester | CCEE 566 credits/prereq and program rows handled; no accidental course-code loss |
| 3. Grounded answering | `qa.py` selects structured answer or short evidence excerpt; multi-source conflicts; fallback | Every nonfallback answer cites a real source and excerpt; conflict shows two pages |
| 4. Pipeline trace | `pipeline.py` wraps V3.1 API, retrieval, grounding; real tokens, scores, timings, statuses | No fake probabilities or fabricated visualizations; trace is JSON-safe |
| 5. UI | `app.py` with Streamlit chat/history, compact process strip, expandable inspector, evidence cards | Tested with 3 example questions; responsive, visually coherent |
| 6. Verify | pytest contract tests + manual eval candidates + app startup smoke | Tests pass; no errors; final report states unsupported cases |

## Suggested small code tree to build

```
app.py
src/rhu_chat/__init__.py
src/rhu_chat/data.py
src/rhu_chat/retrieval.py
src/rhu_chat/qa.py
src/rhu_chat/pipeline.py
src/rhu_chat/ui.py            # only if necessary
scripts/build_index.py        # replace guidance stub
scripts/smoke_app.py          # replace guidance stub
requirements.txt
README_RUN.md
IMPLEMENTATION_STATUS.md
FINAL_IMPLEMENTATION_REPORT.md
```

Do not create 20 microservices/classes, a REST backend, a full RAG framework, authentication, stores of student data, or plugin infrastructure. Use deterministic local code. Python type hints and short docstrings are welcome. Avoid extra UI dependencies where built-in Streamlit charts are adequate.

## Cache and reproducibility

`artifacts/` is gitignored. Build with TF-IDF using `TfidfVectorizer` and store the vectorizer, sparse matrix, stable passage metadata and input file hashes. On start, load once with `st.cache_resource` or equivalent. Use a fresh index only when source hashes differ. Model `.joblib` comes from trusted project input; never download serialized models automatically. Keep `requirements.txt` compatible with the pinned classifier dependencies in `inputs/classifier/requirements_v31.txt`.

## Iteration budgets

One simple tested retrieval path and good answer presentation beats adding three retrieval architectures. Favor a reliable structured answer for course questions, a brief quote for policies, and honest fallback. Stop when acceptance tests are green and the UI demo is clear.
