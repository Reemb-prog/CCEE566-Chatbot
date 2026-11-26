# RHU Catalog Explorer — how the code and the process work

This document explains the whole project: what it does, how a question becomes a cited answer, every model it
uses, how they were trained and evaluated, and what each file is for. To run every step and see the outputs, use
the companion notebook [`notebooks/RHU_Project_Walkthrough.ipynb`](../notebooks/RHU_Project_Walkthrough.ipynb).
Both follow the structure of *CCEE 566 NLP Lab 3 (Naive Bayes classification)*.

All numbers below are from the current code and data (scikit-learn TF-IDF + linear models, run on 2026-11-26).
Scores are agreement with labelled data. **Margins and cosine similarities are never probabilities.**

---

## 1. What the project is

A small local Streamlit app with two views:

1. **Catalog chat** answers questions about the *RHU 2026–2027 Undergraduate Catalog*. It answers only from that
   catalog, quotes the text word for word, and cites the PDF page. If the catalog doesn't cover a question, it says so.
2. **Feedback analyser** reads a student opinion and finds the aspects mentioned and the sentiment of each
   (aspect-based sentiment analysis, ABSA). Example: *"Registration crashed but my advisor was great"* →
   registration: negative, advising: positive.

Everything is classical NLP: regular expressions, TF-IDF, LinearSVC and logistic regression. There are no LLMs,
embeddings, external APIs or databases. A **"How it works" inspector** next to each view shows every intermediate
step with its measured time.

### Hard rules the code enforces

| Rule | Where it is enforced |
|---|---|
| Answer only from the catalog. No uncited answers. | `pipeline.ask()` downgrades any "answered" result without citations to *insufficient evidence*. |
| Keep course codes exact. Never swap one code for another. | `nlp.extract_course_codes()` reads codes from the *original* text, and the normalizer never edits them. Unknown codes get a suggestion, never an automatic substitution. |
| When records disagree, show all of them. | `qa.answer_course()` compares all records and lists each version with its own page. |
| Check values against the page. | `CatalogData.text_on_page()` confirms that each quoted value appears in the extracted page text (`verified_in_page_text`). |
| V3.1 is a soft hint only. | Its category adds +0.03 to the ranking. Nothing is filtered on it. |
| Catalog text is data. Block HTML injection. | `ui.md_escape()` / `strip_html()`. Only static CSS uses `unsafe_allow_html`. |
| Public repository. | Catalog CSVs, models, feedback data and per-sentence survey results are git-ignored. |

---

## 2. Architecture at a glance

```
                ┌────────────────────────── app.py (Streamlit UI) ──────────────────────────┐
question ──►  RHUChatPipeline.ask()                                        feedback text ──► FeedbackAnalyzer.analyze()
                │                                                                              │
   1 input check ─ 2 normalize + tokens + codes ─ 3 V3.1 hints ─ 3b feedback?─┐                 ├ clean + spelling
                │                                                            └─ yes ──► "open in Feedback analyser"
   4 TF-IDF retrieval over all 2,502 passages                                                  ├ sentences → clauses
   5 grounding: scope rules │ exact course lookup │ title match │ study plan │ passage excerpt ├ M2 aspect detectors
   6 answer + citations + trace (timings)                                                      ├ M3 clause polarity + abstain
                                                                                               └ M1 overall sentiment
```

---

## 3. Data

### 3.1 Catalog (the only knowledge source)

Tables under `inputs/knowledge/`. They were extracted from the PDF once, so the app never re-reads the 429 PDF pages.

| File | Rows | Used for |
|---|---|---|
| `RETRIEVAL_PASSAGES.csv` | 2,502 | TF-IDF retrieval and verbatim excerpts. Columns: section, subsection, category, page, URL. |
| `RHU_COURSE_DESCRIPTIONS.csv` | 748 | Exact course lookups: credits, prerequisites, corequisites, description. |
| `RHU_PROGRAM_STUDY_PLAN_ROWS.csv` | 573 | Study-plan answers (program × year × term). |
| `RHU_CATALOG_FULL_TEXT_BY_PAGE.csv` | 429 | Checking that a quoted value is really on the cited page. |

The passages are very unbalanced: 52.6 % are program pages, 13.1 % minors, 10.7 % college information, and so on.
That is one reason retrieval is never forced into one category.

`scripts/preflight.py` checks the SHA-256 of all 17 bundled inputs, plus the CSV schemas, before anything runs.

### 3.2 Feedback data (separate module, git-ignored)

Built by `scripts/prepare_feedback_data.py` from `inputs/feedback/raw/` into `inputs/feedback/processed/`:

| Table | Rows | Source | Role |
|---|---|---|---|
| `review_sentiment.csv` | 6,420 (5,438 train / 982 test) | public course/teacher reviews (S1). Labels come from star ratings: 3 = neutral (decision D1). | M1 |
| `absa_sentences.csv` | 3,557 (2,848 train / 660 EduRABSA test / 49 synthetic dev) | EduRABSA annotated sentences (S2) + generated RHU sentences (S3) | M2 |
| `absa_opinions.csv` | 4,658 | one row per opinion phrase | M3 |
| `rhu_human_test.csv` | 39 rows (the "40 human sentences" of the plan) | S4. **Never** used for training. | evaluation |
| `data_card.json` | — | input/output hashes, counts, decisions D1–D4 | provenance |

Near-duplicate texts share a group id (`near_dup_group` / `group_id`), so a group never ends up on both sides of a
split. The notebook checks this (Step 17: 0 groups cross a split). The 40 human sentences were unsealed on
2026-11-21 so their errors could be fixed. Their scores are therefore development numbers. The last sealed result
is in `eval/feedback_results_sealed_v1.md`.

---

## 4. Catalog pipeline, stage by stage (`src/rhu_chat/pipeline.py`)

### Stage 1 — Input check
Rejects empty input, input with no letters, and input over 500 characters, and asks the user to rephrase. Nothing
else runs for invalid input.

### Stage 2 — Normalisation, tokens and extraction (`normalize.py`, `nlp.py`)
- **Normalizer** (`normalize.Normalizer`). All of its knowledge comes from the catalog itself. It does no training.
  - *Student shorthand* from a short visible list: `prereqs → prerequisites`, `creds → credits`, `whats → what is`.
  - *Course acronyms* built from the initials of catalog titles: `NLP → natural language processing`. A token
    counts as an acronym only if its lower-case form never appears as an ordinary word in the catalog. If an
    acronym matches several titles, the app asks which one, never guesses.
  - *Spelling*: a word not in the catalog vocabulary is replaced by the closest catalog word (difflib, same first
    letter): `descripton → description`.
  - Every change is returned and shown in the *Tokens* tab.
- **Rules** (`nlp.py`, all regex and keywords, each reported by name when it fires):
  - `extract_course_codes`: "CCEE 566", "ccee566" and "CCEE-566" all become `CCEE 566`. Only known prefixes are
    accepted.
  - `requested_field`: credits / prerequisite / corequisite / equivalent / description.
  - `detect_program`, `detect_semester`: program aliases, year/term/ordinal semester.
  - `scope_flags`: personal record, live or time-sensitive, off-topic, instruction-override attempt.
  - `match_course_title`: fuzzy title match (ratio ≥ 0.88) when no code is typed.
- **Tokens** come from the same scikit-learn analyzer the index uses: lower-case, `\b\w\w+\b`, English stop words
  removed.

### Stage 3 — V3.1 classifier hints (frozen, `inputs/classifier/`)
The bundled `rhu_classifier_api.py` + `rhu_classifier_v31.joblib` is used **as delivered** and never retrained.

- Two pipelines, each `TfidfVectorizer → LinearSVC(C=1, class_weight='balanced')`:
  - **category**: 17 classes, 4,484 features, unigrams + bigrams.
  - **intent**: 92 classes, 991 features.
- Before vectorizing, regex steps replace course codes with `coursecode` and digits with `num`. The model therefore
  learns question *types*, not specific courses.
- When category and intent disagree, the `category_governs` policy picks the best intent inside the predicted
  category (`routing_status = reconciled_from_category`).
- `make_qa_request()` returns *hints only*.

Reported historical-test performance (V3.1 hand-off): category accuracy 0.818, macro-F1 0.480; intent accuracy
0.750, macro-F1 0.398 (0.808 on intents seen in training). Accuracy is fair, but macro-F1 is low and every input
gets *some* label. For example, *"Hold on for dear life."* gets `academic_regulations`. So the hint can only nudge
the ranking.

### Stage 3b — Feedback routing
If the feedback models are loaded, the question also goes through the feedback analyser and `feedback_route()`.
An opinion with no course code and no catalog field (*"My advisor is super helpful"*) is not searched in the
catalog. The user instead gets an **Open in Feedback analyser** button. Instruction-override wording always stays
on the catalog path.

### Stage 4 — Retrieval (`retrieval.py`)
- **Index:** `TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True)` over
  *section + subsection + passage text*. The matrix is 2,502 × 42,739, about 0.15 % non-zero. It is cached in
  `artifacts/` and rebuilt only when the CSV hashes or the index version change.
- **Score:** `rank_score = cosine + boost`.
  - The boost is +0.03 when the passage category matches the V3.1 hint, and +0.10 when the passage contains a
    typed course code.
  - Boosts never revive a passage with zero word overlap.
  - The score is a relative ranking, not a confidence.
- Retrieval always searches the **full** index. Example: for *"How do I appeal a grade?"*, *Appeal of Course
  Grade* (p. 59) ranks first on cosine alone.

### Stage 5 — Grounding (`qa.py` + `pipeline._ground`)
The first rule that applies wins:

1. **Scope rules:** personal records, live data and off-topic questions → polite decline with **no** citation.
2. **Unknown code with a close known code:** ask ("Did you mean …?"). Never substitute.
3. **Course code typed:** exact lookup in the course table (`answer_course`).
   - All records for that code are compared.
   - If they agree: one answer that cites every page (CCEE 566 credits → 3, pp. 155, 344, 347).
   - If they disagree: `conflicting_records`, with each version listed and cited separately (CCEE 566
     prerequisites → p. 155 continued on 156, p. 344, p. 347). Nothing is picked automatically.
   - A prerequisite that continues onto the next page is followed there.
4. **Title typed without a code:** fuzzy title match. Ambiguous titles or acronyms → ask.
5. **Study-plan question:** rows for the program and the year/term from the study-plan table.
6. **Otherwise, extractive passage answer** (`answer_from_passages`):
   - The best passage must have cosine ≥ **0.10**.
   - Among the top candidates, pick the one that covers ≥ **60 %** of the question's *IDF-weighted* content words.
     Words are stemmed, and filler words such as "catalog" or "tell" are ignored.
   - Copy its best-matching sentence(s) **verbatim**, at most 600 characters.
   - The excerpt is checked against the page text.
   - If no passage qualifies → *not answerable from the catalog*, with related passages shown separately as
     suggestions, never as evidence.

### Stage 6 — Answer and trace
- **Status:** `answered`, `insufficient_evidence`, `needs_clarification` or `invalid_input`.
- **Citations:** source id, page, official PDF URL with `#page=`, and the verbatim excerpt.
- **Trace:** every stage's inputs and outputs, plus `timings_ms` measured with `time.perf_counter()`. The UI shows
  stored traces and never recomputes them.
- An instruction-override attempt still gets real catalog text only, prefixed with a note that sources cannot be
  invented.

---

## 5. Feedback sentiment / ABSA module (`src/rhu_chat/feedback.py`)

### 5.1 Models (trained by `scripts/train_feedback_models.py`, training split only)

| Model | Task | Features → classifier | Train CV macro-F1 |
|---|---|---|---|
| **M1** | overall sentiment of a whole text | word 1–2 + char_wb 2–5 TF-IDF → LinearSVC C=0.1, balanced | 0.541 |
| **M2** | aspect detection: 12 aspects, one binary model each | word (+char) TF-IDF → LogisticRegression C=16, balanced. *word+char* was chosen by mean CV F1, 0.738 vs 0.693. | per aspect, 0.53–0.94 |
| **M3** | polarity of a clause (shared by all aspects) | word 1–2 TF-IDF → LogisticRegression C=2, balanced. Weight on generated phrases chosen by CV. | 0.648 |

The configurations come from a teammate's earlier CV selection. As a sanity check, the training script refits the
teammate's exact Task A setup and compares it with their reported score.

**Abstain rule (decision D2).**
- For M1 and M3, the margin is the top-1 minus top-2 decision value. The threshold is the smallest margin at which
  polar predictions reach 85 % precision on *out-of-fold* training predictions, and it abstains on at most 35 % of
  them. M1's threshold is 0.136, and it abstains on 8 % of polar calls.
- A polar call below the threshold is shown as **neutral (unclear)**.

**Aspects with no real data.** Registration, advising, admissions, library, IT systems, careers and transportation
appear only in generated RHU sentences (`trained_on_generated_only`). The app flags them.

### 5.2 Inference for one text
1. `clean_text`: Unicode and spacing cleanup. `correct_spelling` fixes misspellings **only toward aspect words**
   (`registraton → registration`).
2. Split into sentences, then into clauses at *but / however / although / though / whereas / while / yet / ;*.
3. M2 runs on the whole sentence. Each detected aspect is attached to the clause where its own detector scores
   highest.
4. M3 labels each clause and the abstain rule is applied. If the result is neutral or unclear, an unambiguous cue
   word (*crashed, overpriced, not helpful*) decides instead.
5. Aggregation: positive and negative for the same aspect → neutral (mixed). The overall sentiment comes from the
   aspects/clauses when there are any, and otherwise from M1.
6. The analyser also flags questions (`is_question`) so that catalog questions are pointed back to the chat.

### 5.3 Evaluation (`scripts/eval_feedback.py` → `eval/feedback_results.md`)
Thresholds and configurations were fixed on training data before any test set was scored.

| Evaluation | Metric | Value |
|---|---|---|
| A. M1 on S1 test, 982 reviews, with abstain rule | macro-F1 | 0.570 (accuracy 0.789) |
| Baseline: always the most frequent class | macro-F1 | 0.275 (accuracy 0.703) |
| Baseline: uniform random | macro-F1 | 0.277 |
| MultinomialNB, α = 0.1 by 5-fold group CV | macro-F1 | 0.505 |
| ComplementNB, α = 0.1 | macro-F1 | 0.561 |
| B. M1 vs human-annotated polarity (EduRABSA test) | macro-F1 | 0.579 |
| C. ABSA end to end, EduRABSA test | pair F1 | 0.578 |
| C. ABSA end to end, human RHU sentences (dev) | pair F1 | 0.621, 95 % bootstrap [0.481, 0.756] |
| C. aspects only, human RHU sentences | aspect F1 | 0.828 |
| Ablation without generated RHU data, human sentences | pair F1 | 0.189 |
| D. M3 given the gold opinion phrase, EduRABSA test | macro-F1 | 0.647 |

What the numbers show:
- **Neutral is the hard class.** M1's most-confused pairs are neutral → positive (51.7 % of neutral reviews) and
  negative → positive (28.5 %). Star-rating labels are weak: a 3-star review often reads as positive.
- **ComplementNB matches the LinearSVC.** The shipped model was kept because it was chosen before the test set was
  looked at.
- **The generated RHU sentences are essential.** Without them, pair F1 on the human sentences drops from 0.62 to
  0.19.
- **Shortcut check:** only 3 of 5,438 training reviews mention a star rating in their text, so label leakage
  through the text is negligible.

---

## 6. The Streamlit app (`app.py`, `src/rhu_chat/ui.py`)

- **Catalog chat.**
  - Example buttons and a chat box.
  - Each answer shows a status badge, the answer, and a *Sources* expander with the page, PDF link, source id and
    excerpt.
  - **Show NLP steps** opens the inspector: a six-stage strip (input → tokens → V3.1 hints → retrieval → evidence
    → answer) with times, and one tab per stage. Any earlier turn can be inspected. *Clear chat* empties the
    session; nothing is saved to disk.
- **Feedback analyser.**
  - Shows the overall sentiment, shaded clauses, an aspect table with notes (mixed, unclear, trained on generated
    examples) and spelling changes.
  - Its *How it works* panel has tabs for Clean, Aspects (detector decision values), Polarity (clause margins, cue
    rule and the **top cue words** M3 learned), Overall, and a Model card.
- `@st.cache_resource` loads the pipeline once. If the index can't load, the app shows a polite message instead of
  a stack trace. If the classifier or feedback models are missing, the app still runs and says so.

---

## 7. File map

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: both views and the inspectors. |
| `src/rhu_chat/data.py` | Loads the catalog CSVs, citation URLs, page-text verification. |
| `src/rhu_chat/normalize.py` | Shorthand, course-acronym and spelling normalizer. |
| `src/rhu_chat/nlp.py` | Regex rules: codes, field, program, semester, scope flags, title match. |
| `src/rhu_chat/retrieval.py` | TF-IDF passage index, cache, cosine + boost search. |
| `src/rhu_chat/qa.py` | Grounded answer strategies, thresholds, fallbacks. |
| `src/rhu_chat/pipeline.py` | `RHUChatPipeline`: runs the stages, V3.1 hints, feedback routing, trace. |
| `src/rhu_chat/feedback.py` | `FeedbackAnalyzer`: cleaning, clauses, M1–M3 inference, routing rule. |
| `src/rhu_chat/ui.py` | Escaping, badges, CSS, clause shading. |
| `scripts/preflight.py` | Input hashes and schemas. |
| `scripts/build_index.py` | Builds or reuses the TF-IDF index and runs a V3.1 self-check. |
| `scripts/run_seed_eval.py` | Runs the 25-question seed set → `eval/seed_run_results.csv`. |
| `scripts/smoke_app.py` | Streamlit AppTest plus a real headless server health check. |
| `scripts/prepare_feedback_data.py` | Builds the feedback training/test tables and the data card. |
| `scripts/train_feedback_models.py` | Trains M1–M3 (`--variant full / edurabsa_only / labels_v1`). |
| `scripts/eval_feedback.py` | Held-out evaluation, baselines, ablations → `eval/feedback_results.*`. |
| `tests/` | 105 pytest tests: pipeline contract, grounding, normalization, feedback data and models. |
| `notebooks/RHU_Project_Walkthrough.ipynb` | Runs every step above, Lab-3 style. |
| `docs/*.md` | Original specification: plan, UI/trace spec, grounding rules, acceptance criteria, feedback plan. |

---

## 8. The development process

1. **Specification first.** `CLAUDE.md`, `PROMPT_FOR_CLAUDE.md` and `docs/` fixed the scope (local, classical,
   cited) and the acceptance criteria. `scripts/preflight.py` proved the inputs were the expected files.
2. **Data loader and retrieval**, tested with focused tests. Then **extractive grounded answers**, including
   conflicting-record handling and page verification. Then the **trace and UI**, with an AppTest smoke check. The
   checklist is in `IMPLEMENTATION_STATUS.md` and the outcome in `FINAL_IMPLEMENTATION_REPORT.md`.
3. **Informal wording** (commit *Understand informal wording*): normalizer for shorthand, acronyms and spelling.
4. **Feedback module** in phases F1–F3 (`docs/FEEDBACK_ABSA_PLAN.md`):
   - F1: prepare the data with fixed decisions D1–D4.
   - F2: train on the train split with CV-chosen thresholds.
   - F3: evaluate on held-out sets, including ablations.
   - The human sentences were then unsealed to fix label rules R1–R6 (`docs/FEEDBACK_LABEL_REVIEW.md`).
5. **Lab 3 additions**: dummy and tuned Naive Bayes baselines, most-confused pairs, a star-rating shortcut check,
   and the top-cue-words panel.

Throughout, every failing test was fixed at the root cause and tests were never relaxed. No historical test data
was used for tuning.

---

## 9. Limitations

- Catalog answers copy text. They do not interpret policy, and rows extracted from tables may be imperfect. The
  page link is there to check against.
- Retrieval is lexical. A question worded very differently from the catalog may fall below the thresholds and be
  declined, which is the intended safe failure.
- V3.1 has low macro-F1 and labels anything, even nonsense. It is a hint only.
- The feedback models are trained on public reviews and generated RHU sentences. Neutral and several RHU aspects
  are weak, and the human evaluation set is small (40 sentences) and unsealed.
- The 25-question seed set is a manual smoke test, not proof of accuracy on real students.

## 10. Quick start

```powershell
python -m pip install -r requirements.txt
python scripts/preflight.py
python scripts/build_index.py
python -m pytest -q
python -m streamlit run app.py
jupyter notebook notebooks/RHU_Project_Walkthrough.ipynb
```
