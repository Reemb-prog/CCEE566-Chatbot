# Integration contract between app, classifier, retrieval and evaluation

Create `src/rhu_chat/pipeline.py` with:

```python
from pathlib import Path

class RHUChatPipeline:
    @classmethod
    def from_project_root(cls, root: Path) -> "RHUChatPipeline":
        """Load trusted V3.1 classifier and local catalog index once."""

    def ask(self, question: str) -> dict:
        """Return the JSON-serializable contract below."""
```

Minimum `ask()` response (may add fields, never omit required ones):

```json
{
  "question": "What are the prerequisites for CCEE 566?",
  "status": "answered",
  "answer": "...",
  "citations": [
    {"source_page": 344, "source_url": "https://...pdf#page=344", "source_id": "RHU-...", "excerpt": "..."}
  ],
  "trace": {
    "input": {"valid": true},
    "tokenization": {"retrieval_tokens": ["prerequisites", "ccee"], "course_codes": ["CCEE 566"]},
    "classification": {"category_hint": "...", "intent_hint": "...", "routing_status": "...", "review_needed": false},
    "retrieval": {"candidates": [{"id": "RHU-P...", "source_page": 344, "tfidf_similarity": 0.35, "boost": 0.1, "rank_score": 0.45}]},
    "grounding": {"decision": "conflicting_records", "evidence_ids": ["..."]},
    "timings_ms": {"input": 0.1, "tokenization": 0.5, "classification": 4.2, "retrieval": 12.0, "grounding": 0.8, "total": 17.6}
  }
}
```

- `status` is one of `answered`, `insufficient_evidence`, `needs_clarification`, `invalid_input`.
- For `answered`, require at least one citation, with source page and real source URL. If no supported citation can be supplied, return `insufficient_evidence` instead.
- For `insufficient_evidence`/`invalid_input`, do not attach an irrelevant citation to justify a fabricated answer. Related passages can be shown separately as suggestions.
- `source_id` may be a course ID, passage ID or stable plan ID; never invent IDs. `source_page` must be a positive integer present in source CSV data.
- `trace.classification` may contain `None` values for unsupported inputs; never suppress issues. Place `make_qa_request()` output in this trace or pass it to retrieval.
- Scores must be real numeric measurements, not confidence levels. If an exact structured lookup has no TF-IDF similarity, use an explicit separate key/strategy instead of filling in 1.0.
- Plain dicts, Python strings, numbers and lists only, so UI/tests can serialize with `json.dumps()`.

**Do not expose unreviewed QA training answers as gold knowledge.** All responses should be grounded in catalog-derived passages or the structured source tables.
