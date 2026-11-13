# RHU Catalog Chatbot | Claude Code implementation harness

**This ZIP is an executable development brief and source bundle, not a prebuilt chatbot.** The objective is for Claude Code to implement a simple, polished Streamlit chatbot for the RHU 2026–2027 Undergraduate Catalog, then test it against the included inputs.

## Included: no additional source files needed for the first prototype

- Full catalog PDF at `inputs/catalog/` (reference and official page citations).
- Extracted catalog passages, course descriptions, study plans, page text, taxonomy at `inputs/knowledge/`.
- Final V3.1 notebook, **trained Joblib model**, and actual inference API at `inputs/classifier/`. You do **not** need to retrain classification.
- Original dataset release ZIP for provenance, EDA manifest, final classification handoff report.
- Claude instructions (`CLAUDE.md`, `PROMPT_FOR_CLAUDE.md`), implementation plan, QA evaluation seed, scripts and contract tests.

## Launch Claude (recommended)

1. Extract the ZIP into a folder. Open a terminal **in the extracted `RHU_Claude_Chatbot_Harness` folder**.
2. Check Python: `python --version` (use Python 3.11 or 3.12, recommended), and run `python scripts/preflight.py`. On Windows, `py -3 scripts/preflight.py` also works.
3. Ensure Claude Code is installed and authenticated, then run `claude`. In the session, paste:

   **Read `CLAUDE.md` and `PROMPT_FOR_CLAUDE.md`. Implement the entire Streamlit RHU catalog chatbot in this folder, following the phased plan. Do not stop after creating a plan: implement, test, fix, then demonstrate.**

   Alternatively use `RUN_CLAUDE.bat` on Windows, or `bash RUN_CLAUDE.sh` on macOS/Linux, to open Claude with that instruction.
4. Review Claude's terminal permission prompts. Do not run it with permission-bypass flags.
5. After implementation, install the project's finalized dependencies, build index, run tests, then launch Streamlit. Typical commands (Claude should document the exact final ones):

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   python scripts/build_index.py
   python -m pytest -q
   python -m streamlit run app.py
   ```

## Show this during the demo

1. Ask: `How many credits is CCEE 566?`
2. Open **How it works** and show tokens, exact course-code entity, category/intent hints, ranking and actual catalog evidence.
3. Ask: `What are the prerequisites for CCEE 566?` and show that conflicting entries on pages 344 and 347 are not silently collapsed.
4. Ask: `What is my current GPA?` and show a clean unsupported/personal-data fallback.

## What success means

A student can type a question and see a factual, cited extract or a clear fallback. The side panel visualizes the **actual computation performed** (tokenization, classification, retrieval candidates and grounded answer), not invented model reasoning. UI looks finished, but complexity remains suitable for a **one-month NLP capstone**.

The harness source data are educational and should not be committed to a public repository without checking rights. Model `.joblib` files must only be loaded from this trusted supplied source.
