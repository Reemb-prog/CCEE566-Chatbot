# RHU Question Classification V3.1
## Capstone acceptance review and handoff to grounded question answering

**Decision:** ACCEPT as the final *classification component* for a one-month CCEE 566 academic prototype. This does not certify a deployed university assistant, factual answer correctness, or natural-student generalization. Do not spend the remaining capstone time tuning this historical test set.

## What is implemented and validated

1. Original group-disjoint train, validation, and historical-test partitions are unchanged; 53 provenance/review-style records remain excluded from classification.
2. V2/V3 preprocessing ablation and comparison of Bag-of-Words / TF-IDF, Naive Bayes, Logistic Regression, and LinearSVC are preserved. V3.1 reused the previously executed validation grid **only after verifying identical SHA-256 hashes for all three input CSVs**, and independently refitted the selected classifiers. A fresh run without the cache environment flag runs the entire search.
3. Category and intent models are both LinearSVC. A `category_governs` policy was selected using validation only. It ranks intents among those mapped to the predicted category, using uncalibrated class scores as rankings, **not probabilities**. This is category-conditioned intent reconciliation, **not** a separately trained hierarchical classifier.
4. The trained intent-to-category dictionary is now explicitly exported in `documented_trained_intent_taxonomy.csv` and compared to the original `RHU_LABEL_TAXONOMY.csv`. All 92 trained/validation-supported intents agree with the source taxonomy. The full source taxonomy contains one **multi-parent** descriptive intent label (`catalog_section_explanation`), which is not among the trained classifier labels; therefore the source taxonomy is not globally one-to-one.
5. An importable `rhu_classifier_api.py` and serialized `rhu_classifier_v31.joblib` are verified after independent process reload. `make_qa_request()` returns **soft category/intent hints**, original question, evidence/citation requirements and fallback instructions. A blank input does not trigger retrieval. Inputs with arbitrary text can still be misclassified; the model is not an out-of-scope filter.
6. All **15 code cells** in V3.1 completed. The available historical scores remain unchanged, as intended. QA handoff smoke tests are unscored and are **not** an independent student benchmark.

## Historical classification metrics (1,693 test rows after exclusion)

| Metric | Raw | Category-conditioned routing |
|:---|---:|---:|
| Category accuracy | 81.75% | 81.75% |
| Category macro F1, gold-present test labels | 0.480 | 0.480 |
| Intent accuracy | 75.37% | 74.96% |
| Intent macro F1, all 25 gold-present test labels | 0.420 | 0.398 |
| Intent macro F1, gold labels supported by training + validation | 0.840 | 0.808 |
| Inconsistent resolved category/intent pairs | 206 | 0 |

**Interpretation:** Removing contradictions slightly lowers measured intent accuracy. The routing improvement is an invariant, not a performance gain. 324 of 1,693 historical test questions have intents absent from train + validation (12 test-only intents). The historical test was consulted during earlier development and is not a pristine final holdout. Model accuracy is concentrated in `courses` and `study_plans`; unseen and irregular questions may be misclassified. The external student-question benchmark was not run because no human-labeled independent dataset was supplied.

## What was deliberately not changed

* No test records were moved into training or validation.
* No unsupported intent labels were invented or merged; no synthetic gold labels were introduced.
* No new algorithm was selected by looking at the test results.
* No arbitrary SVM margin rejection threshold was added.
* No strict category filter was imposed on retrieval, because it could eliminate the correct catalog passage when the classifier is wrong.

## Interface contract to implement in grounded QA

```python
from rhu_classifier_api import load_bundle, predict_one, make_qa_request
classifier = load_bundle('rhu_classifier_v31.joblib')
classification = predict_one(classifier, 'What are the prerequisites for COSC 351?')
request = make_qa_request(classification)
# Pass request['original_question'] to the retriever without masking COSC 351.
# category_hint and intent_hint may be applied as SOFT ranking features only.
# Do not use the classification result itself as the answer.
```

The returned request includes `retrieve_from_full_catalog`, `category_hard_filter=False`, `intent_hard_filter=False`, `requires_verified_evidence=True`, `requires_source_citation=True`, and `fallback_if_no_evidence=True`. It does **not** mean the question is answerable from the catalog.

## Recommended next phase: grounded QA

**Data already available:** The EDA ZIP supplies `clean_data/retrieval_passages.csv` with **2,502 passages** and columns `passage_id`, `passage_text`, `section`, `category`, `subsection`, `source_page_start`, `source_page_end`, `source_url`, `academic_year`. It also supplies QA examples and a **7-question** human-reviewed gold test. Passage IDs/page references have not been certified as question-to-passage relevance gold labels.

1. Build a TF-IDF passage index on `passage_text`; keep course codes, degree names, numbers and section headings intact. Search the **whole** index using the original question.
2. Retrieve top five candidate passages using cosine similarity. Optionally compare word 1–2-grams with character n-grams for typos. Check whether category boosts help; always allow evidence outside the predicted category.
3. Add a simple, transparent answer policy: return a short, evidence-supported extract or a reviewed answer **only if** it is verified against the selected passage; otherwise show the relevant catalog excerpt and citation. For insufficient evidence, say that the information was not found in the catalog. Do not invent live schedules, student-specific facts or policy exceptions.
4. Record `passage_id`, catalog page and `source_url` for each answer. If a question requests current or personalized information not in the catalog, provide an explicit unavailability/verification message rather than forcing a citation.
5. Hand-label about **25–40 varied student questions**, including course details, program credits, admissions, fees, policy, typos, paraphrases and out-of-catalog cases. Mark relevant passages or answerability, not only expected answers. Evaluate Recall@3/5, MRR (if relevant passage labels exist), correct citation rate, unsupported-answer rate and human-scored answer correctness. Keep the existing seven reviewed gold questions as a small targeted check; do not treat seven as comprehensive evaluation.
6. Build a lightweight **Streamlit** or **Gradio** chatbot: text entry, message history, answer, page link, evidence excerpt, explicit fallback, and optionally a panel showing classifier hints. Integrate speech input and wake word after the text-based QA path works. Aspect-based sentiment analysis remains a distinct module, and user feedback should be routed there rather than treated as catalog evidence.

**Suggested capstone stop rule:** The text chatbot can answer a small diverse manual test set with relevant catalog citations, gracefully decline unsupported questions, and demonstrably use the classifier as optional metadata. Then spend remaining time completing the voice / ABSA demonstration and the final report, not chasing classification accuracy.

## Known remaining limitations

* 92 train+validation intents are mapped, but many have very few independent source groups; 12 held-out test intent labels are absent.
* The source taxonomy includes a descriptive multi-parent label that cannot be reduced to a globally unique intent/category mapping without contextual semantics.
* Independent, human-authored student-question evaluation remains outstanding.
* Classification routing cannot establish factual evidence, answerability, or currency of RHU policies.
* Treat the environment-pinned Joblib model as a **trusted local artifact**; Joblib is not a safe format for untrusted downloads.

**Final recommendation:** Freeze classification at V3.1 and move to grounded QA. Reopen the classification notebook only if real integration results expose a specific defect.
