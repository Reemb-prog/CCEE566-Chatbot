# UI / visual trace: simple, engaging, honest

## Screen layout

- Heading: **RHU Catalog Explorer**. Subtitle: `Undergraduate Catalog 2026–2027 · NLP capstone demo`.
- Main column ~65%: chat bubbles, 3 sample question chips/buttons, answer, citations, optional evidence snippet, `Clear chat`.
- Inspector ~35%: stages for currently selected/latest turn, toggled via `Show NLP steps` checkbox or side panel. On narrow layouts use stacked cards instead of squashed columns.
- Use Streamlit built-ins (`st.chat_message`, `st.chat_input`, `st.columns`, `st.expander`, `st.bar_chart`/`st.progress`, `st.tabs`). Minimal CSS only for spacing and header; ensure text contrast and simple keyboard use.

## Per-message trace: EXACT stages and facts

1. **Input.** Original question; length; valid/invalid; actual elapsed time. Avoid storing personal profiles.
2. **Tokenization and extraction.** Display actual tokens from `TfidfVectorizer.build_analyzer()` for retrieval (and, if helpful, original regex tokens). Show extracted course codes or program/semester clues. Do **not** call these transformer tokens. If empty, say no usable tokens.
3. **Classification.** V3.1 raw and reconciled `category`, `intent`, `routing_status`, `review_needed`. Show 'routing hints only'. No confidence percent from SVM margins.
4. **Retrieval.** List top 5 ranked candidate passage cards with `passage_id`, `source_page`, excerpt; small bar chart for raw cosine similarities (0-1). Show boosts and total score separately when used. Exact structured lookup should also be visible as a distinct result with exact-match tag, NOT a fake cosine score.
5. **Evidence check.** State the actual reason for selecting evidence (`exact_course_match`, `excerpt_supported`, `conflicting_records`, `insufficient_evidence`, `needs_clarification` etc). Show what fields/sentences support each claim. Highlight conflicts, low evidence and uncertainties.
6. **Answer.** Show answered/fallback status, citations and whether source text is a direct excerpt or structured result; stage duration.

Top-level process strip should be readable in seconds, with short stage names, not a busy chart showing dozens of metrics. Keep technical details collapsible. Render trace for each message from stored results, don't recompute when selected. Show total measured latency, not a guessed value. Avoid animated fake processing.

## Example user-facing flow

**User:** What are the prerequisites for CCEE 566?

**Answer:** The catalog contains different prerequisites for CCEE 566 across sections. In the Computer and Communications Engineering Program entry (p. 344) it lists `CCEE 214, COSC 214, or Equivalent`; the Minor in Artificial Intelligence entry (p. 347) lists `CCEE 214`. Please confirm which program applies. [page citations]

**Inspector:**

- Tokens: `what`, `are`, `the`, `prerequisites`, `for`, `ccee`, `566` (show actual analyzer output, which may differ).
- Entity: `CCEE 566`.
- Classification: predicted hints from V3.1, with routing status.
- Retrieval: exact course lookup; actual text candidates and TF-IDF ranks.
- Evidence: conflicting prerequisite strings across pages, both cited.

## Usability / polish

- Answers normally 1-4 sentences, except plan listing and conflicts.
- Source card includes title/section, page, clickable source URL, and a short excerpt. Never show a citation link if page is unknown.
- Start with no generated conversation and 3 suggested questions, not a giant long tutorial.
- Set page icon and responsive wide layout; show polite failure message if initial index cannot be built rather than exposing a stack trace to students.
- The application should be polished but **small**; no dashboards with meaningless decorative metrics or fake reasoning graphs.
