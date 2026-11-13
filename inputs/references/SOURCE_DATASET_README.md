# RHU QA Dataset 2026–2027: Training / Evaluation Release

## Important scope
This release carries forward the supplied original catalog extraction and all **74 user-reviewed answers**, without inventing unprovided institutional clarifications. The reviewed material uses **S###_F####** IDs, while the supplied ZIP uses **RHU-F#####** IDs; mapping is therefore evidence-based and often ambiguous. Matching is recorded in HUMAN_REVIEW_DECISIONS_74.csv. Original extracted facts linked to a reviewed topic are quarantined rather than silently overwritten. Original user review wording is reproduced without semantic alterations.

## How to use
1. For QA model fine-tuning use `SFT_TRAIN.jsonl`, monitor `SFT_VALIDATION.jsonl`, evaluate once with `SFT_TEST.jsonl`. These are machine-eligible records, **not all human-verified**.
2. For question intent classification use `INTENT_TRAIN.csv`, `INTENT_VALIDATION.csv`, `INTENT_TEST.csv`.
3. For human-reviewed factual assessment use `REVIEW_GOLD_TEST.csv` (held-out human-reviewed subset only). Keep it out of training. Inspect `REVIEW_CONFLICTS_NOT_FOR_TRAINING.csv` before handling conflicting source text.
4. For retrieval index `RETRIEVAL_PASSAGES.csv`, optionally `STRUCTURED_FACTS_REVIEW_FLAGS.csv` and `REVIEWED_QA_ALL_74.csv` as separate human-reviewed source statements; reviewed conflict entries must be disclosed as scope-specific catalog statements, not universal university policy.
5. For detailed audit use `QA_ALL_WITH_REVIEW_FLAGS.csv`, `HUMAN_REVIEW_DECISIONS_74.csv`, and `VALIDATION_REPORT_FINAL.json`.

## Limits
Of the 74 reviewed responses, 53 enter model-facing data as 39 train, 7 validation, and 7 test examples. The remaining 21 are excluded from supervised model targets. The 74 human review entries mostly contain approved *wording*, not explicit decisions using the requested Approve/Correct/Exclude/Catalog conflict enum. Some reviewed wordings preserve known inconsistent catalog material, including IELTS 085/95, credit-hour totals, and colliding Spring dates. They are not safe to learn as universally authoritative facts without university confirmation. This pipeline does not falsely declare those resolved. Source-page retrieval candidate links are not passage-level gold evidence.

## Final statistics
{
  "academic_year": "2026-2027",
  "source_qa_rows": 12697,
  "source_facts": 4307,
  "source_passages": 2502,
  "human_review_decisions": 74,
  "review_gold_test_count": 7,
  "review_resolved_used_in_model": 53,
  "review_split_counts": {
    "train": 39,
    "excluded_conflict": 21,
    "validation": 7,
    "test": 7
  },
  "review_conflicts_withheld": 21,
  "baseline_qa_eligible_count": 10858,
  "review_related_fact_ids_quarantined": 144,
  "review_related_qa_rows_quarantined": 430,
  "eligible_deduped_qa_count": 10463,
  "duplicate_normalized_questions_dropped": 30,
  "qa_split_counts": {
    "train": 7251,
    "test": 1700,
    "validation": 1512
  },
  "split_leaks": {
    "fact_id": 0,
    "group_id": 0,
    "question_norm": 0
  },
  "unmapped_review_decisions": 23,
  "labels_provenance": "Unreviewed baseline QA retained with machine-eligibility only; 53 human-reviewed answer records enter supervised splits (39/7/7); 21 conflicting review records are audit-only.",
  "caveats": [
    "All human review text is preserved; it does not resolve the catalog contradictions documented in the review.",
    "Held-out human reviewed gold questions are selected only from the 53 resolved review wordings; conflict items excluded from model targets.",
    "Model training QA has not been manually verified row-by-row.",
    "Source page candidate retrieval labels are not validated passage gold.",
    "Semantic overlap may remain across splits despite zero shared fact and group IDs, since other catalog facts cover similar topics.",
    "Source documents are dated 2026\u20132027, not necessarily current policy."
  ]
}

## Reproduce
Run `python build_rhu_final.py` with pandas and rapidfuzz. The script expects original files at the paths supplied in this conversation. Matching is intentionally conservative and should be reviewed for every ambiguous match.
