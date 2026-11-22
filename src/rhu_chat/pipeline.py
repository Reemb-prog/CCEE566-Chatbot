"""End-to-end pipeline: input check -> tokenization/extraction -> V3.1 hints ->
TF-IDF retrieval -> grounding -> answer, with a JSON-safe trace and real timings.
"""
from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path

from . import nlp, qa
from .data import CatalogData
from .normalize import Normalizer, interpretation_note, nice_title
from .retrieval import CATEGORY_BOOST, CODE_BOOST, PassageIndex

MAX_QUESTION_CHARS = 500
CLASSIFIER_DIR = Path("inputs") / "classifier"


def _load_classifier_api(root: Path):
    """Import the bundled, unmodified V3.1 API module by file path."""
    path = root / CLASSIFIER_DIR / "rhu_classifier_api.py"
    spec = importlib.util.spec_from_file_location("rhu_classifier_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


class RHUChatPipeline:
    def __init__(self, root: Path, data: CatalogData, index: PassageIndex, index_meta: dict,
                 clf_api, clf_bundle, clf_error: str | None = None, feedback=None,
                 feedback_error: str | None = None):
        self.root, self.data, self.index, self.index_meta = root, data, index, index_meta
        self.clf_api, self.clf_bundle, self.clf_error = clf_api, clf_bundle, clf_error
        self.feedback, self.feedback_error = feedback, feedback_error  # optional separate module
        self.known_codes = set(data.courses["course_code"]) | set(data.plans["course_code"])
        titles: dict[str, set[str]] = {}
        for code, title in zip(data.courses["course_code"], data.courses["course_title"]):
            t = re.sub(r"[^a-z& ]", " ", str(title).lower())
            t = re.sub(r"\s+", " ", t).strip()
            if t and t != "nan":
                titles.setdefault(t, set()).add(code)
        self.titles = titles
        texts = (list(data.passages["passage_text"]) + list(data.courses["course_title"].fillna(""))
                 + list(data.courses["description"].fillna("")))
        self.normalizer = Normalizer(texts, titles, data.course_prefixes)

    @classmethod
    def from_project_root(cls, root: Path, rebuild_index: bool = False) -> "RHUChatPipeline":
        """Load trusted V3.1 classifier and local catalog index once."""
        root = Path(root).resolve()
        data = CatalogData.load(root)
        index, meta = PassageIndex.load_or_build(data, root / "artifacts", force=rebuild_index)
        api = bundle = None
        err = None
        try:
            api = _load_classifier_api(root)
            bundle = api.load_bundle(str(root / CLASSIFIER_DIR / "rhu_classifier_v31.joblib"))
        except Exception as exc:  # keep the app usable; retrieval never depends on the classifier
            err = f"{type(exc).__name__}: {exc}"
        fb = fb_err = None
        try:  # the feedback module is optional: catalog QA works without it
            from .feedback import DEFAULT_MODEL, FeedbackAnalyzer
            fb = FeedbackAnalyzer.load(root / DEFAULT_MODEL)
        except Exception as exc:
            fb_err = f"{type(exc).__name__}: {exc}"
        return cls(root, data, index, meta, api, bundle, err, fb, fb_err)

    # ------------------------------------------------------------------
    def classify(self, question: str) -> dict:
        if self.clf_bundle is None:
            return {"category_hint": None, "intent_hint": None, "raw_category": None, "raw_intent": None,
                    "routing_status": "classifier_unavailable", "review_needed": True,
                    "error": self.clf_error, "note": "Routing hints only (not answers, not probabilities)."}
        pred = self.clf_api.predict_one(self.clf_bundle, question)
        req = self.clf_api.make_qa_request(pred)
        return {"category_hint": req["category_hint"], "intent_hint": req["intent_hint"],
                "raw_category": pred["raw_category"], "raw_intent": pred["raw_intent"],
                "routing_status": req["routing_status"], "review_needed": req["routing_review_needed"],
                "qa_request": req, "used_as": "soft ranking boost only (category_hard_filter=False)",
                "note": "Routing hints only (not answers, not probabilities)."}

    def ask(self, question: str) -> dict:
        t_total = time.perf_counter()
        timings: dict[str, float] = {}
        question = question if isinstance(question, str) else ""
        trace: dict = {}

        # 1. input --------------------------------------------------------
        t = time.perf_counter()
        stripped = question.strip()
        problem = None
        if not stripped:
            problem = "empty"
        elif not re.search(r"[A-Za-z]", stripped):
            problem = "no_letters"
        elif len(stripped) > MAX_QUESTION_CHARS:
            problem = f"too_long (>{MAX_QUESTION_CHARS} chars)"
        trace["input"] = {"question": question, "length": len(question), "valid": problem is None,
                          "problem": problem}
        timings["input"] = _ms(t)
        if problem:
            for k in ("tokenization", "classification", "retrieval", "grounding", "answer"):
                timings[k] = 0.0
            trace.update({
                "tokenization": {"retrieval_tokens": [], "raw_tokens": [], "course_codes": [],
                                 "unknown_codes": [], "requested_field": None, "program": None,
                                 "semester_clues": {}, "scope_flags": [], "skipped": True,
                                 "normalization": {"original": question, "normalized": question,
                                                   "changed": False, "changes": [],
                                                   "ambiguous_acronyms": []}},
                "classification": {"category_hint": None, "intent_hint": None, "raw_category": None,
                                   "raw_intent": None, "routing_status": "invalid_input",
                                   "review_needed": True, "skipped": True},
                "retrieval": {"candidates": [], "structured_matches": [], "skipped": True},
                "grounding": {"decision": "invalid_input", "evidence_ids": [], "support": [], "notes": []},
            })
            msg = ("Please type a question about the RHU 2026–2027 Undergraduate Catalog, "
                   "e.g. “How many credits is CCEE 566?”" if problem != "too_long (>500 chars)" else
                   f"Please shorten your question to under {MAX_QUESTION_CHARS} characters.")
            timings["total"] = _ms(t_total)
            trace["timings_ms"] = timings
            return {"question": question, "status": "invalid_input", "answer": msg, "citations": [],
                    "suggestions": [], "trace": trace}

        # 2. normalization, tokenization & extraction -----------------------
        # Course codes come from the ORIGINAL question (the normalizer never edits them);
        # scope checks run on both forms; everything else uses the normalized text.
        t = time.perf_counter()
        norm = self.normalizer.normalize(stripped)
        q = norm["normalized"]
        codes, unknown = nlp.extract_course_codes(stripped, self.data.course_prefixes)
        field = nlp.requested_field(q)
        flags = list(dict.fromkeys(nlp.scope_flags(stripped) + nlp.scope_flags(q)))
        retrieval_tokens = self.index.tokens(q)
        title_match = None
        if not codes and not unknown and field in {"credits", "prerequisite", "corequisite", "description"}:
            mt, mcodes, ratio = nlp.match_course_title(q, self.titles)
            if mt:
                title_match = {"title": mt, "codes": sorted(mcodes), "similarity_ratio": round(ratio, 3)}
        tok = {"normalization": norm,
               "retrieval_tokens": retrieval_tokens, "raw_tokens": self.index.raw_tokens(q),
               "course_codes": codes, "unknown_codes": unknown, "requested_field": field,
               "program": nlp.detect_program(q), "semester_clues": nlp.detect_semester(q),
               "scope_flags": flags, "title_match": title_match,
               "tokenizer": "sklearn TfidfVectorizer analyzer on the normalized question "
                            "(word unigrams shown; stop words removed)"}
        trace["tokenization"] = tok
        timings["tokenization"] = _ms(t)

        # 3. V3.1 classification (soft hints) ------------------------------
        t = time.perf_counter()
        cls_info = self.classify(q)
        cls_info["input_text"] = q
        trace["classification"] = cls_info
        timings["classification"] = _ms(t)

        # 3b. feedback routing: an opinion is not a catalog question ----------
        if self.feedback is not None:
            t = time.perf_counter()
            from .feedback import feedback_route
            fb = self.feedback.analyze(stripped)
            route = feedback_route(stripped, fb, codes, field)
            if route["is_feedback"] and "instruction_override_attempt" in flags:
                route = {"is_feedback": False, "reasons": ["instruction-override wording keeps the catalog path"]}
            trace["feedback_route"] = {**route, "overall": fb["overall"], "aspects": fb["aspects"],
                                       "note": "Feedback is analysed by the separate feedback module and is "
                                               "never used as catalog evidence."}
            timings["feedback_check"] = _ms(t)
            if route["is_feedback"]:
                return self._feedback_reply(question, trace, timings, t_total, fb)

        # 4. retrieval over the FULL passage index -------------------------
        t = time.perf_counter()
        candidates = self.index.search(q, cls_info.get("category_hint"), codes, k=8)
        timings["retrieval"] = _ms(t)

        # 5. grounding ------------------------------------------------------
        t = time.perf_counter()
        res = self._ground(q, codes, unknown, field, flags, title_match, candidates, retrieval_tokens,
                           norm["ambiguous_acronyms"])
        timings["grounding"] = _ms(t)

        trace["retrieval"] = {
            "candidates": [{k: v for k, v in c.items() if k != "text"} | {"excerpt": c["text"][:220]}
                           for c in candidates],
            "structured_matches": res["structured_matches"],
            "strategy": res["strategy"],
            "scoring": f"rank_score = tfidf_similarity (cosine, 0-1) + boost (category hint +{CATEGORY_BOOST}, "
                       f"course code +{CODE_BOOST}); relative ranking, not confidence",
            "index": {"n_passages": self.index_meta.get("n_passages"), "n_features": self.index_meta.get("n_features"),
                      "cache": self.index_meta.get("cache")},
        }
        trace["grounding"] = {"decision": res["decision"], "evidence_ids": res["evidence_ids"],
                              "support": res["support"], "notes": res["notes"],
                              "thresholds": {"min_tfidf_similarity": qa.MIN_SIMILARITY,
                                             "min_idf_weighted_coverage": qa.MIN_COVERAGE}}

        # 6. answer ---------------------------------------------------------
        t = time.perf_counter()
        answer = res["answer"]
        if "instruction_override_attempt" in flags:
            answer = ("Note: I only use real catalog text and page numbers; I can't invent or change sources. " + answer)
        status = res["status"]
        citations = res["citations"]
        if status == "answered" and not citations:      # contract: no uncited answers
            status, answer = "insufficient_evidence", qa.FALLBACK["low_relevance"]
        if status != "answered":
            citations = []
        answer = interpretation_note(norm) + answer
        timings["answer"] = _ms(t)
        timings["total"] = _ms(t_total)
        trace["answer"] = {"status": status, "n_citations": len(citations),
                           "answer_type": res["strategy"]}
        trace["timings_ms"] = timings
        return {"question": question, "status": status, "answer": answer, "citations": citations,
                "suggestions": res["suggestions"], "trace": trace}

    def _feedback_reply(self, question: str, trace: dict, timings: dict, t_total: float, fb: dict) -> dict:
        """Opinion detected: do not search the catalog; point to the Feedback analyser instead."""
        timings["retrieval"] = timings["grounding"] = 0.0
        trace["retrieval"] = {"candidates": [], "structured_matches": [], "skipped": True,
                              "skip_reason": "feedback, not a catalog question",
                              "index": {"n_passages": self.index_meta.get("n_passages"),
                                        "n_features": self.index_meta.get("n_features"),
                                        "cache": self.index_meta.get("cache")}}
        trace["grounding"] = {"decision": "feedback_detected", "evidence_ids": [], "support": [],
                              "notes": trace["feedback_route"]["reasons"]}
        t = time.perf_counter()
        about = "; ".join(f"{v['label']}: {v['sentiment']}" for v in fb["aspects"].values())
        answer = ("This reads like feedback (an opinion" + (f" about {about}" if about else "") + "), not a question "
                  "about the catalog, so I did not search the catalog for it. Open it in the Feedback analyser to see "
                  "the aspect and sentiment breakdown. If you meant a catalog question, please phrase it as a question.")
        timings["answer"] = _ms(t)
        timings["total"] = _ms(t_total)
        trace["answer"] = {"status": "needs_clarification", "n_citations": 0, "answer_type": "feedback_routing"}
        trace["timings_ms"] = timings
        return {"question": question, "status": "needs_clarification", "answer": answer, "citations": [],
                "suggestions": [], "routed_to": "feedback", "trace": trace}

    # ------------------------------------------------------------------
    def _ground(self, q, codes, unknown, field, flags, title_match, candidates, tokens,
                ambiguous_acronyms=()) -> dict:
        # Scope rules first: they decline rather than attach unrelated citations.
        for reason in ("personal_record_request", "outside_catalog_topic", "live_or_time_sensitive"):
            if reason in flags:
                return qa.fallback(reason, candidates, suggest=(reason == "live_or_time_sensitive"))

        if unknown and not codes:
            sugg = sorted({s for u in unknown for s in nlp.suggest_prefix(u, self.data.course_prefixes, self.known_codes)})
            if sugg:
                return qa._result("needs_clarification",
                                  f"I don't recognise {', '.join(unknown)} as a catalog course code. "
                                  f"Did you mean {' or '.join(sugg)}? Please retype the code.",
                                  "needs_clarification", notes=["Close code suggested, not substituted."],
                                  strategy="course_code_check")

        if codes:
            primary = [c for c in codes if c in self.known_codes or self.data.code_in_passages(c)]
            missing = [c for c in codes if c not in primary]
            if not primary:
                return qa._result("insufficient_evidence",
                                  f"{', '.join(missing)} does not appear in the 2026–2027 catalog tables or passages, "
                                  "so I can't answer. I won't substitute a different course; please check the code.",
                                  "unknown_course_code", notes=["Exact code lookup returned no rows."],
                                  strategy="course_code_check")
            code = primary[0]
            others = [c for c in primary[1:]]
            if field in ("prerequisite",) and others:
                res = qa.answer_course(self.data, code, field, others)
            else:
                res = qa.answer_course(self.data, code, field, [])
                if res and others and res["status"] == "answered":
                    for extra in others[:2]:
                        r2 = qa.answer_course(self.data, extra, field, [])
                        if r2 and r2["status"] == "answered":
                            res["answer"] += "\n\n" + r2["answer"]
                            res["citations"] += r2["citations"]
                            res["evidence_ids"] += r2["evidence_ids"]
                            res["support"] += r2["support"]
                            res["structured_matches"] += r2["structured_matches"]
                            res["notes"] += r2["notes"]
                            if r2["decision"] == "conflicting_records":
                                res["decision"] = "conflicting_records"
            if res is None:
                res = qa.answer_code_from_passages(self.data, code, candidates)
            if res is None:
                res = qa._result("insufficient_evidence", f"I found no catalog evidence for {code}.",
                                 "unknown_course_code", strategy="course_code_check")
            if missing:
                res["notes"].append(f"Not found in the catalog: {', '.join(missing)}.")
            return res

        if title_match:
            if len(title_match["codes"]) == 1:
                res = qa.answer_course(self.data, title_match["codes"][0], field, [])
                if res:
                    res["notes"].insert(0, f"No course code typed; matched the title “{title_match['title']}” "
                                           f"(string similarity {title_match['similarity_ratio']}) to "
                                           f"{title_match['codes'][0]}.")
                    res["answer"] = f"(Matched your wording to {title_match['codes'][0]}.) " + res["answer"]
                    return res
            else:
                return qa._result("needs_clarification",
                                  f"Several courses are titled “{nice_title(title_match['title'])}”: "
                                  f"{', '.join(title_match['codes'])}. Which course code do you mean?",
                                  "needs_clarification", strategy="course_title_match")

        # An acronym that fits several catalog course titles: ask, never guess.
        if (ambiguous_acronyms and field in {"credits", "prerequisite", "corequisite", "description"}
                and not re.search(r"\b(program|minor|major|degree|department|college|faculty)\b", q, re.I)):
            amb = ambiguous_acronyms[0]
            opts = "; ".join(f"{nice_title(o['title'])} ({', '.join(o['codes'])})" for o in amb["options"][:6])
            return qa._result("needs_clarification",
                              f"“{amb['acronym']}” matches the initials of several catalog courses: {opts}. "
                              "Which one do you mean? You can type the course code.",
                              "needs_clarification", notes=["Acronym matched several course titles; none chosen."],
                              strategy="course_acronym_check")

        if nlp.PLAN_RE.search(q) and not re.search(r"\bcredits? (?:are )?(?:needed|required) to graduate|\bgraduat", q, re.I):
            res = qa.answer_study_plan(self.data, q)
            if res:
                return res

        q_cov = nlp.OVERRIDE_RE.sub(" ", q)           # instruction-override words are not search content
        content = [t for t in self.index.tokens(q_cov) if not t.isdigit()]
        res = qa.answer_from_passages(q, candidates, content, self.index.idf, self.data.text_on_page,
                                      qa.FIELD_SENTENCE.get(field))
        # A course field was asked without a code and the evidence passage IS a course entry:
        # switch to the structured lookup for that code so conflicting entries are not hidden.
        if res["status"] == "answered" and field in {"credits", "prerequisite", "corequisite"}:
            ev = next((c for c in candidates if c["id"] in res["evidence_ids"]), None)
            m = re.match(r"([A-Z]{3,4} \d{3}[A-Z]?)\b", ev["text"]) if ev else None
            if m and self.data.course_rows(m.group(1)):
                structured = qa.answer_course(self.data, m.group(1), field, [])
                if structured and structured["status"] == "answered":
                    structured["notes"].insert(0, f"No course code typed; the best-matching passage {ev['id']} "
                                                  f"(p. {ev['source_page']}) is the {m.group(1)} entry, so all "
                                                  f"{m.group(1)} records were compared.")
                    structured["answer"] = f"(Best-matching catalog entry: {m.group(1)}.) " + structured["answer"]
                    structured["evidence_ids"].insert(0, ev["id"])
                    return structured
        return res
