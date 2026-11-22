# Feedback sentiment + ABSA module: data, training and integration plan

Status: PLAN (nothing below is implemented yet). Date: 2026-11-21.

This adds a **separate feedback module** to RHU Catalog Explorer. It analyses student opinions and never feeds catalog answers.
A chat message that looks like feedback is routed to it. Feedback text is never treated as catalog evidence, and catalog
evidence is never scored for sentiment.

## 0. Decisions already taken (team, 2026-11-21)

| # | Decision |
|---|---|
| D1 | Review-level labels from star ratings are re-mapped so that **only rating 3 is neutral** (details in 2.1). |
| D2 | **Neutral is kept as a safety class.** At inference, a prediction whose top score is not clearly ahead of the runner-up is reported as *neutral / unclear* instead of being forced to positive or negative. |
| D3 | The Claude-generated RHU set is used for **diagnosis**, and also as **training data for the RHU aspects that EduRABSA does not cover**. |
| D4 | The 40 human survey sentences are the **only real RHU data**. They are never used as training data. *Update 2026-11-21:* after F3 the team chose to fix the errors found on them, so they are now a development set. The sealed result is kept in `eval/feedback_results_sealed_v1.md`, and an independent RHU score needs new human sentences. |

## 1. Source inventory

| Code | File (current location) | What it is | Use |
|---|---|---|---|
| S1 | `asba_package.zip: data/clean/combined_ASBA_clean.csv` | 6,420 real reviews (teacher, course, university). Weak labels from ratings. | Review-level sentiment training (Task A) |
| S1-raw | `data/raw/combined_ASBA.csv` (**missing from zip, get from teammate**) | Uncleaned S1 | Provenance and reproducibility |
| S2 | `asba_package.zip: data/clean/edurabsa_clean.jsonl` | 1,043 human-annotated reviews, 4,264 opinions with spans | ABSA for courses, instructors, facilities, student services |
| S2-raw | EduRABSA `train/val/test.jsonl` + conversion script + licence (**missing, get from teammate**) | All 1,320 annotated reviews | Recover the 277 dropped reviews (mostly university) and check the licence |
| S3 | `Downloads/RHU_ABSA_Annotations_Consolidated_Clean.csv` | 599 aspect mentions / 465 texts: 311 `synthetic_new` (human style), 246 `synthetic_legacy` (templated), 42 human rows | RHU aspects with no EduRABSA coverage, plus diagnosis |
| S4 | `Downloads/human_final.csv` | The 40 human sentences (11 respondents) with gold pairs and label notes | RHU test (sealed for F3, unsealed 2026-11-21; see D4) |
| S5 | `Downloads/RHU_Sentiment_Augmented_Sentences.csv`, `RHU_Sentiment_ABSA_Full_EDA.ipynb` | Sentence-level view of S3 + S4, and the teammate's EDA | Cross-check only |

Checks already run (scratch analysis, 2026-11-21):

- **Leakage.** No synthetic sentence is a near copy of any human sentence. The closest match (char 3–5-gram cosine) has a median of 0.28 and a maximum of 0.50; none reaches 0.6.
- **Synthetic coverage.** `synthetic_new` has 23–27 sentences per RHU aspect across 12 aspects, with polarity roughly balanced (about 6–13 per class per aspect). 47 of its 264 texts are multi-aspect.
- **`synthetic_legacy` quality.** These rows are templated "A; B; and C" lists with one label per row, and some labels do not match their clause. They are **excluded from training** and kept for diagnosis only.
- **Human labels.** In the teammate's files, 25 of the 42 human rows sit in *train*. This must change (D4).
- **Human label provenance.** 13 human labels were revised by Claude (`label_note`), 5 respondents' grids were unusable, 1 sentence has no aspect, and 1 is a question rather than feedback (HF017).

## 2. Data preparation (`scripts/prepare_feedback_data.py`)

There is one deterministic script. It reads only `inputs/feedback/raw/` and writes `inputs/feedback/processed/` plus `data_card.json`
(row counts per step, label distribution and SHA-256 of every input and output). No manual editing happens after this step.

### 2.1 Review-level sentiment (from S1), applying D1

| Review type | Negative | Neutral | Positive |
|---|---|---|---|
| teacher (1–5, half steps) | ≤ 2.5 | **= 3.0** | ≥ 3.5 |
| university (fractional averages) | < 2.75 | **2.75–3.25** (2.8, 3.0, 3.2) | > 3.25 |
| course (thumbs 0/1) | 0 | none | 1 |

Effect on the 6,420 cleaned rows: neutral falls from 822 to **198** (174 teacher + 24 university), positive becomes 4,498 and negative 1,724.
This removes the 3.5 and 4-star reviews that the EduRABSA check showed were mostly positive text labelled neutral.
Neutral becomes very small (3.1%), which is the reason for D2.

- Keep the teammate's **test membership** (same near-duplicate groups and seed) so results stay comparable. Only the labels change.
- Store both `sentiment_v1` (the teammate's mapping) and `sentiment` (D1) so the change can be reported.
- 3.5 → positive is a choice, not a fact. It is one constant (`NEUTRAL_RATINGS`) in the script.

### 2.2 Unified aspect taxonomy (12 aspects)

`courses, instructors, registration, advising, admissions, tuition_fees, student_services, facilities, library, it_systems, career_services, transportation`

The display names come from S3, but courses and instructors stay **separate**, because EduRABSA can learn them separately.

| Source label | → unified |
|---|---|
| EduRABSA `COURSE#*` | courses |
| EduRABSA `STAFF#*` | instructors |
| EduRABSA `UNIVERSITY#CAMPUS & FACILITIES` | facilities |
| EduRABSA `UNIVERSITY#INFORMATION & SERVICES`, `#SOCIAL ENGAGEMENT & ACTIVITIES` | student_services |
| EduRABSA `UNIVERSITY#COST` | tuition_fees |
| Other EduRABSA `UNIVERSITY#*` categories (found in S2-raw) | mapped if they clearly fit, otherwise `other` (not trained) |
| S3/S4 `original_aspect` snake_case values | same name (already the unified set) |

### 2.3 ABSA training table: sentence level (`absa_sentences.csv`)

The chatbot receives short sentences, so every source is converted to **sentences**. Each row has the columns
`text, aspects (multi-label), pairs (aspect:sentiment), spans, source, is_synthetic, group_id, split`.

- **EduRABSA (S2):** split reviews into sentences. Each opinion goes to the sentence that holds its span; sentences with no opinion are kept as "no aspect" examples. `group_id` is the review, so sentences from one review never land on both sides of a split.
- **Synthetic (S3, `synthetic_new` only):** one row per text. Pairs and evidence spans come from the annotation rows. `group_id` is the text.
- **Conflicting labels for one aspect in one sentence** (positive and negative): keep both mentions in the span table. For the sentence-level label use `mixed → neutral` (consistent with D2).

### 2.4 Clause-polarity table (`absa_opinions.csv`)

One row per opinion: `opinion_text` (EduRABSA span or synthetic `evidence_span`), `sentiment`, `aspect`, `source`.
This trains **one shared polarity model for all aspects**. Synthetic data gives only about 8 examples per polarity per RHU aspect, which is too few for separate per-aspect sentiment models.

### 2.5 Splits

| Set | Content | Rule |
|---|---|---|
| `rhu_human_test` | all 40 S4 sentences | Never trained on. Sealed for the first F3 run, then unsealed by the team (see D4); scores are now development numbers. |
| `edurabsa_test` | official EduRABSA test reviews (sentence-split) | Sealed, so results can be compared with the teammate's |
| `synthetic_dev` | 20% of `synthetic_new` texts, stratified by aspect × polarity | Diagnosis only. Its scores will be optimistic and are labelled as such. |
| `train` | everything else | Model selection by grouped 5-fold CV inside train only |

### 2.6 Validation tests on the prepared data (`tests/test_feedback_data.py`)

- No S4 text, and no text with cosine ≥ 0.8 to an S4 text, appears in train.
- Every aspect label is in the 12-aspect set and every sentiment is in {positive, negative, neutral}.
- Every span lies inside its text, and no `group_id` crosses splits.
- The counts match `data_card.json`.

### 2.7 Human test label check (team task, about 1 hour)

Two teammates independently re-label the 40 sentences (aspect and polarity) without seeing the current gold labels. Report Cohen's kappa and settle disagreements. That becomes the final gold set; the Claude-revised labels are not used as is.

## 3. Training (`scripts/train_feedback_models.py`)

Run inside the project's pinned environment (scikit-learn 1.8.0). Output: `models/feedback_v1.joblib` and `models/feedback_v1_metrics.json`.
No hyperparameter search beyond the teammate's grids. Their chosen configurations are the defaults.

| Model | Input | Config (start point) | Notes |
|---|---|---|---|
| M1 review sentiment | S1 `text_clean` → D1 labels | word(1–2) + char_wb(2–5) TF-IDF, LinearSVC C=0.1, balanced | Teammate's best Task A config |
| M2 aspect detectors | `absa_sentences` | one-vs-rest LR per aspect, word TF-IDF, balanced | 12 binary models. An aspect with fewer than 15 train sentences is not trained and is reported as "not covered". |
| M3 clause polarity | `absa_opinions` | word TF-IDF LR, balanced (teammate C=2) | Shared across aspects |
| M4 abstain threshold (D2) | CV out-of-fold scores of M1 and M3 | margin between top and second class | Chosen on train CV only, to get ≥ 0.85 precision on the positive/negative outputs that are not abstained. Reported as a margin, not a probability. |

**Inference** (also the trace shown in the UI):

1. Normalise the text with the existing `normalize.py`.
2. Split into clauses on `but, however, although, though, while, ;`.
3. For each clause, run M2 to get aspects, then M3 to get polarity, then M4: if the margin is low, output `neutral (unclear)`.
4. Run M1 on the whole text for overall sentiment, with the same abstain rule.
5. If the text matches RHU aspect keywords but no detector fired, report "aspect not recognised" instead of guessing.

**Ablations to report** (these justify D3):

- (a) EduRABSA only vs (b) EduRABSA + synthetic, both scored on `rhu_human_test`.
- (c) The teammate's neutral mapping vs D1.

## 4. Evaluation (`scripts/eval_feedback.py` → `eval/feedback_results.md`)

| Ruler | Metrics | Meaning |
|---|---|---|
| S1 test (D1 labels) | macro-F1, per-class P/R, confusion matrix, abstain rate | Agreement with ratings (weak labels) |
| EduRABSA test | aspect micro-F1, pair F1, polarity macro-F1 | Real annotated text, North-American domain |
| synthetic_dev | same + by phenomenon (mixed, implicit, multi-aspect) | Diagnosis only, optimistic |
| **rhu_human_test (n=40)** | pair P/R/F1 with counts, a list of every error, abstain rate | The only real RHU evidence. Too small for per-aspect claims, so report counts, not percentages per aspect. |

The legacy synthetic rows and the Claude phenomena tags are used here as stress tests only.

## 5. App integration

- **`src/rhu_chat/feedback.py`** (the 8th module) provides `FeedbackAnalyzer.load(path)` and `.analyze(text) -> dict`. The dict holds overall sentiment, per-aspect pairs, clause trace, abstain flags, uncovered aspects and timings.
- **Routing in `pipeline.py`:** if a message has no question form and contains opinion words or a feedback cue (for example "I feel", "is terrible"), the chat replies with a note and a link to the Feedback tab, and does **not** answer from the catalog. Questions keep the current catalog path. HF017 (a question about changing major) must stay on the catalog path, and this is a test.
- **`app.py`, new "Feedback analyser" tab:** text box, overall label, an aspect table, clause highlights (escaped text only, no raw HTML), an expandable trace, and a model card built from `feedback_v1_metrics.json`. The card shows the sources, the synthetic share per aspect, the human-test results and the limits.
- **Display rules:** margins are shown as "score margin", never as %. Aspects backed only by synthetic data carry a "trained on generated examples" badge.
- **Tests:** `tests/test_feedback.py` covers model loading, output schema, abstain behaviour, routing (feedback vs question), and HTML escaping.

## 6. Phases and done-criteria

| Phase | Work | Done when |
|---|---|---|
| F0 | Collect S1-raw/S2-raw/licence from the teammate; copy S1–S4 into `inputs/feedback/raw/`; update `CLAUDE.md` scope (ABSA now in scope as a separate module) | Files present with checksums; licence recorded in `inputs/feedback/README.md` |
| F1 | `prepare_feedback_data.py` + data tests | Data tests pass; `data_card.json` written |
| F1b | Re-label the 40 human sentences with 2 annotators (team) | Kappa reported, gold set frozen |
| F2 | `train_feedback_models.py` | M1 reproduces the teammate's Task A within about ±0.03 under the teammate's mapping (sanity check); all models saved with metrics |
| F3 | `eval_feedback.py` + ablations | `eval/feedback_results.md` written, with the human-test error list |
| F4 | `feedback.py` + tests | Unit tests pass |
| F5 | Feedback tab + routing + trace + model card | Streamlit smoke check passes; catalog QA tests still pass |
| F6 | Update `FINAL_IMPLEMENTATION_REPORT.md` and `README_RUN.md` | 3 real feedback examples with traces; limits stated |

## 7. Known limits to state in the report

- Review-level labels are star ratings (weak labels). Under D1, neutral is rare and comes from a single rating value.
- RHU aspects other than courses and instructors are learned mostly from **generated** sentences (about 25 per aspect). Real performance on them is measured only on the 40 human sentences.
- The human test comes from 11 respondents and 40 sentences, and some labels were revised. It shows behaviour, not accuracy.
- The EduRABSA and S1 reviews are North-American. Cross-domain macro-F1 in the teammate's study was 0.37–0.47.
