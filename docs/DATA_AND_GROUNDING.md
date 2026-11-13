# Data contract and grounding rules

The sources in `inputs/` are packaged from prior RHU capstone work. This is the 2026–2027 Undergraduate Catalog, **not real-time university data**.

| File | Units | Columns needed |
|---|---:|---|
| `inputs/knowledge/RETRIEVAL_PASSAGES.csv` | 2,502 rows | `passage_id`, `passage_text`, `category`, `source_page_start`, `source_page_end`, `source_url`, `section`, `subsection` |
| `inputs/knowledge/RHU_COURSE_DESCRIPTIONS.csv` | 748 rows | `course_code`, `course_title`, `credits`, `prerequisite_text`, `source_page`, `source_url`, `program_or_section`, `needs_manual_review` |
| `inputs/knowledge/RHU_PROGRAM_STUDY_PLAN_ROWS.csv` | 573 rows | `program_or_section`, `semester`, `course_code`, `course_title`, `credits_in_study_plan`, `source_page`, `source_url`, `verification_status` |
| `inputs/knowledge/RHU_CATALOG_FULL_TEXT_BY_PAGE.csv` | 429 rows | `catalog_page`, `page_text`, `category`, `source_url`, `note` |
| `inputs/knowledge/RHU_LABEL_TAXONOMY.csv` | 158 rows | `label_type`, `label`, `parent` |
| `inputs/catalog/RHU_Undegraduate_Catalog_2026-2027.pdf` | Full reference | PDF page locations for evidence check |
| `inputs/classifier/rhu_classifier_api.py` | V3.1 inference wrapper | `load_bundle`, `predict_one`, `make_qa_request` |

The source release ZIP is included in `inputs/reference_archive/` to preserve provenance but should NOT be used to train a generative answer model. Existing QA pairs are synthetic-heavy and are not independent verified evidence.

## Page citations

- Use the exact `source_page` / `source_page_start` values stored in the catalog tables. These are **PDF page numbers for #page=N** in the reference URL; do not recalculate offsets or invent links.
- Prefer supplied exact `source_url` when it is `https://www.rhu.edu.lb/...pdf#page=N` and N matches metadata. Treat URL text as data; do not blindly embed HTML. Strip/construct `#page=N` once if needed.
- Every answered assertion should be supported by at least one cited source. For an excerpt-type answer, show the excerpt and passage ID. For a course record, cite the course row's page and show a supporting field or page excerpt.
- Some CSV text is mechanically extracted and may not align perfectly with original PDF layout. Respect `needs_manual_review`, `verification_status`, and flag conflicting source rows.

## Minimal hybrid retrieval scoring

1. Use exact course code lookup when a code is present. Prefer matching actual code and requested field (credits, prerequisites, corequisite, description). Compare all matching rows and program contexts. Never return another course because it has higher TF-IDF similarity.
2. For general questions: TF-IDF cosine similarity across **all** passages; add modest and explicit boosts for course code overlap and category hint, if relevant. Never restrict to a predicted category.
3. Show raw cosine similarity separately from any boost and resulting rank score; these numbers are **relative ranking scores**, not confidence percentages.
4. Avoid arbitrary strict thresholds chosen to make demo scores look good. Set conservative thresholds from small manual examples, explain them, and permit uncertain outcomes.
5. For course credits, a single identical value across source rows is fine if cited. For CCEE 566 prerequisites the text differs between page 344 and 347. The answer MUST preserve and cite those alternatives, including applicability by source section if possible. No automatic 'latest wins'.
6. For plan questions, require enough context to disambiguate program/semester. Return a short accurate list with program/semester + source pages or ask a question to clarify.
7. Policy questions: choose the best relevant **full sentence(s)** from actual passage text; attribute them to catalog section and page. This is extractive QA, not LLM generation; do not claim to have synthesized legal/policy conclusions.
8. Out-of-catalog / personal record questions: return `insufficient_evidence` with e.g. `I can't check personal records such as your GPA from the undergraduate catalog.` Do not query private systems or invent values.
9. If user tries to override instructions via their question or within a PDF passage, treat that content only as a query or source text; never execute it.

## Important V3.1 model details

The classifier NORMALIZES certain course identifiers into generic placeholders for learning. Retrieval MUST use the **original** untouched question. `make_qa_request()` exposes category and intent as optional *soft hints*. A `review_needed` flag is about model routing, not QA answerability. Do not use SVM decision margins as probabilities.

## Performance and audit

Precompute the index and load it once. 2,502 passages are small enough for local CPU. Keep only short evidence snippets in the UI but the source reference must remain accessible. Page text and structured records should be cross-checked before strong claims where extraction seems ambiguous.
