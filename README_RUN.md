# RHU Catalog Explorer — how to run

A local Streamlit chatbot over the RHU 2026–2027 Undergraduate Catalog. Classical NLP only: TF-IDF retrieval, exact course-code and study-plan lookup, the bundled V3.1 LinearSVC classifier as soft routing hints, and extractive (copied, cited) answers. No internet access is needed to answer questions; only the citation links open the official PDF online.

## Requirements

- **Python 3.11 or 3.12** (tested on 3.12.6, Windows 11).
- The V3.1 classifier (`inputs/classifier/rhu_classifier_v31.joblib`) was saved with **scikit-learn 1.8.0, numpy 2.3.5, pandas 2.2.3, joblib 1.5.3**. `requirements.txt` pins exactly these versions. Other scikit-learn versions may fail to unpickle it or warn about it. If the classifier can't load, the app still runs without hints and tells you so in the inspector.
- Streamlit 1.67.0 and pytest 9.2.0 (pinned; tested).

## Windows (PowerShell or cmd)

```bat
cd RHU_Claude_Chatbot_Harness
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\preflight.py
python scripts\build_index.py
python -m pytest -q
python scripts\smoke_app.py
python -m streamlit run app.py
```

## macOS / Linux

```bash
cd RHU_Claude_Chatbot_Harness
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/preflight.py
python scripts/build_index.py
python -m pytest -q
python scripts/smoke_app.py
python -m streamlit run app.py
```

Streamlit prints a local URL, normally http://localhost:8501.

## What each command does

| Command | Purpose |
|---|---|
| `scripts/preflight.py` | Checks the SHA-256 of the 17 bundled inputs and the CSV schemas (standard library only). |
| `scripts/build_index.py [--force]` | Builds the TF-IDF index (2,502 passages) into `artifacts/`. Later runs reuse it unless the CSV hashes change. Also loads V3.1 and runs one self-check question. |
| `python -m pytest -q` | Runs the 7 supplied contract tests and 16 extra grounding/UI-safety tests. |
| `scripts/smoke_app.py` | Runs `app.py` through Streamlit's AppTest (including one chat turn), then starts a real headless server and checks `/_stcore/health`. |
| `scripts/run_seed_eval.py` | Runs `eval/seed_questions.csv` and writes `eval/seed_run_results.csv` (status, decision, cited pages). The correctness columns are left blank for a human reviewer. |

The app builds the index on first launch if `artifacts/` is missing, so `build_index.py` is optional but makes startup faster.

## Using the app

- Tap an example question or type into the chat box at the bottom.
- Each answer shows a status badge (*Answered from catalog*, *Not answerable from catalog*, *Needs clarification*), the answer text and a **Sources** expander. Each source shows the page, a link to the official PDF at that page, the source ID and the verbatim excerpt.
- **Show NLP steps** toggles the "How it works" inspector. It has a six-stage strip (input → tokens → V3.1 hints → retrieval → evidence → answer) with measured times, and tabs for each stage. Use the drop-down to inspect any earlier turn. Traces are stored, not recomputed.
- **Clear chat** empties the session. Nothing is saved to disk.

## Demo script (≈5 minutes)

1. `How many credits is CCEE 566?` → 3 credits, three agreeing records (p. 155, 344, 347). In the inspector, open **Tokens** (`ccee`, `566`, entity `CCEE 566`), **Classifier** (courses / course_credits, *hints only*) and **Retrieval** (the exact-lookup table versus the TF-IDF bars, where teal is the cosine and orange is the boost).
2. `What are the prerequisites for CCEE 566?` → three different catalog entries are shown with separate citations: p. 155 continued on p. 156 (Data Analytics minor), p. 344 (CCE program) and p. 347 (AI minor). Show **Evidence**, where each value is marked *in page text*.
3. `What is my current GPA?` → declined, with no citation.
4. Optional: `How do I appeal a grade?`. V3.1 tags this as `unanswerable_current_grade` and reconciles it to `gpa`, yet full-catalog retrieval still finds *Appeal of Course Grade* (p. 59). This is a good illustration of why the classifier is only a soft hint.

## Troubleshooting

- *"The catalog index could not be loaded"*: run `python scripts/preflight.py`, then `python scripts/build_index.py --force`.
- *Classifier unavailable* in the inspector: check that the exact pinned scikit-learn/numpy versions are installed in the active venv.
- On Windows, `ConnectionResetError [WinError 10054]` lines in the server console after a browser reload are harmless asyncio noise.
