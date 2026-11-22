"""Feedback sentiment + aspect-based sentiment (ABSA) for student opinions.

A separate module from catalog QA: feedback text is never used as catalog evidence. The models are trained by
scripts/train_feedback_models.py (docs/FEEDBACK_ABSA_PLAN.md).

Inference, per sentence:
  1. Detect aspects on the whole sentence (one binary model per aspect, trained at sentence level).
  2. Split the sentence into clauses (but / however / although / ; ...). Each detected aspect is attached to the
     clause where its own detector scores highest.
  3. Clause polarity comes from one shared model. A positive/negative call whose margin is below the threshold
     chosen on training data becomes "neutral" with unclear=True (decision D2).
Overall sentiment comes from the review-level model with the same abstain rule. Same-aspect positive and negative
across sentences -> neutral (mixed). Scores are decision-function margins, never probabilities.
"""
from __future__ import annotations

import difflib
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

SENTIMENTS = ("positive", "negative", "neutral")
ASPECT_LABELS = {
    "courses": "Courses", "instructors": "Instructors", "registration": "Registration",
    "advising": "Academic advising", "admissions": "Admissions", "tuition_fees": "Tuition and fees",
    "student_services": "Student services", "facilities": "Campus facilities", "library": "Library",
    "it_systems": "IT systems (CampusVue, Wi-Fi)", "career_services": "Co-op and careers",
    "transportation": "Transportation",
}
DEFAULT_MODEL = Path("models") / "feedback_v1.joblib"
MAX_FEEDBACK_CHARS = 2000

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‛": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-"})
_ABBREV = {"dr", "prof", "mr", "mrs", "ms", "st", "vs", "etc", "e.g", "i.e", "approx", "dept", "no"}

# Misspellings are corrected only toward these aspect words, and only when the written word is not a known word
# from the training vocabulary. That keeps ordinary informal wording untouched.
SPELLING_TARGETS = (
    "registration", "register", "registered", "registering", "tuition", "tuitions", "advising", "advisor", "advisors",
    "admission", "admissions", "instructor", "instructors", "professor", "professors", "doctor", "doctors",
    "course", "courses", "schedule", "library", "scholarship", "financial", "transportation", "services",
    "facilities", "cafeteria", "internship", "career", "semester", "portal", "elective", "electives",
)
SPELLING_CUTOFF = 0.8
_QUESTION = re.compile(r"^\s*(where|what|how|when|who|which|why|can|could|is|are|do|does|should)\b", re.I)


def correct_spelling(text: str, vocabulary: set[str]) -> tuple[str, list[dict]]:
    """Fix misspelled aspect words ("Registeration", "Tution", "rwgister"); return the text and a change log."""
    changes = []

    def fix(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        if low in vocabulary or low in SPELLING_TARGETS:
            return word
        hit = difflib.get_close_matches(low, SPELLING_TARGETS, n=1, cutoff=SPELLING_CUTOFF)
        if not hit:
            return word
        new = hit[0].capitalize() if word[0].isupper() else hit[0]
        changes.append({"from": word, "to": new})
        return new

    return re.sub(r"\b[A-Za-z]{5,}\b", fix, text), changes


def is_question(text: str) -> bool:
    return bool(_QUESTION.match(text) or text.rstrip().endswith("?"))


# Opinion cues for routing chat messages. Strong words count on their own. Mild adjectives ("good", "easy") count only
# after a linking verb ("was easy", "are really good"), so catalog phrases such as "good standing" stay catalog questions.
_STRONG_CUE = re.compile(
    r"\b(i (?:feel|felt|think|love|loved|hate|hated|like|liked|enjoy(?:ed)?|wish)|frustrat\w*|annoy\w*|terrible|"
    r"awful|horrible|worst|nightmare|useless|stressful|rude|unfair|disappoint\w*|crash(?:es|ed)?|"
    r"(?:needs?|has) to (?:be )?(?:improv|chang|fix)\w*|should (?:be )?(?:improv|chang|fix)\w*)\b", re.I)
_MILD_CUE = re.compile(
    r"\b(?:is|are|was|were|been|be|seems?|felt|feels?|gets?|got)\s+(?:(?:so|very|really|too|super|not|quite|"
    r"extremely|pretty|always|never|kind of|a bit|mostly|sometimes|usually|honestly|totally|getting)\s+)*(?:good|great|bad|poor|easy|hard|difficult|helpful|unhelpful|"
    r"friendly|mean|boring|smooth|simple|slow|fast|expensive|cheap|amazing|excellent|fine|ok|okay|confusing|"
    r"clear|messy|organized|disorganized|supportive|available|unavailable|high|low|dead|quiet|loud|crowded|late)\b", re.I)


# Unambiguous polarity words. Used (a) as routing cues and (b) as a transparent fallback when the clause model says
# neutral although the clause contains one of them ("Registration crashed again"). A negator just before flips it.
CUE_NEGATIVE = re.compile(
    r"\b(crash(?:es|ed|ing)?|nightmare|terrible|awful|horrible|worst|useless|rude|stressful|frustrat\w*|annoy\w*|"
    r"broken|outdated|overpriced|expensive|slow|disconnect\w*|unfair|disappoint\w*|hate[sd]?|dirty|crowded|"
    r"confusing|unhelpful|unavailable|unacceptable|mess(?:y)?|dead)\b", re.I)
CUE_POSITIVE = re.compile(
    r"\b(helpful|friendly|amazing|excellent|great|love[sd]?|smooth(?:ly)?|easy|clear|supportive|quiet|clean|"
    r"fast|organi[sz]ed|welcoming|enjoy(?:ed)?|recommend(?:ed)?)\b", re.I)
_NEGATOR = re.compile(r"\b(not|never|no|hardly|isn't|wasn't|aren't|weren't|don't|doesn't|didn't)\s+(?:\w+\s+)?$", re.I)
MIN_WORDS_MODEL_ROUTE = 4  # below this, a message is treated as a search phrase ("library hours")
_SENTENCE_HINT = re.compile(r"\b(is|are|was|were|keeps?|kept|never|always|again|still|did|does|didn't|doesn't|"
                            r"won't|can't|couldn't|has|have|had|i|we|my|our)\b", re.I)


def cue_polarity(text: str) -> tuple[str | None, str | None]:
    """('negative'|'positive', matched phrase) from the cue lists, with simple negation; (None, None) if no cue."""
    hits = []
    for rx, pol in ((CUE_NEGATIVE, "negative"), (CUE_POSITIVE, "positive")):
        for m in rx.finditer(text):
            negated = bool(_NEGATOR.search(text[:m.start()]))
            flip = {"negative": "positive", "positive": "negative"}[pol] if negated else pol
            hits.append((m.start(), flip, ("not " if negated else "") + m.group(0)))
    pols = {h[1] for h in hits}
    if len(pols) != 1:  # none, or both directions -> no rule decision
        return None, None
    return hits[0][1], hits[0][2]


def feedback_route(text: str, analysis: dict | None, course_codes=(), requested_field=None) -> dict:
    """Decide whether a chat message is feedback (an opinion) rather than a catalog question.
    Questions and course codes keep the catalog path; a requested field (e.g. "credit") only when there is no
    opinion wording. A complaint phrased as a question keeps the catalog path but also offers the analyser."""
    cues = []
    for rx, why in ((_STRONG_CUE, "opinion word"), (_MILD_CUE, "linking verb + evaluative word"),
                    (CUE_NEGATIVE, "complaint word")):
        m = rx.search(text)
        if m:
            cues.append(f"{why}: “{m.group(0)}”")
    if is_question(text):
        strong = _STRONG_CUE.search(text) or CUE_NEGATIVE.search(text)
        return {"is_feedback": False, "also_feedback": bool(strong),
                "reasons": ["question form"] + (["also contains a complaint word, so the analyser is offered"]
                                                if strong else [])}
    if course_codes:
        return {"is_feedback": False, "reasons": ["course code present: catalog lookup"]}
    if requested_field and not cues:
        return {"is_feedback": False, "reasons": [f"catalog field requested ({requested_field}) and no opinion wording"]}
    reasons = list(cues)
    # Model-only routing needs a full sentence (a verb or frequency word), so search phrases such as
    # "late withdrawal from a course" stay catalog questions.
    if not reasons and analysis and len(text.split()) >= MIN_WORDS_MODEL_ROUTE and _SENTENCE_HINT.search(text):
        polar = [a for a, v in analysis["aspects"].items() if v["sentiment"] != "neutral" and not v["unclear"]]
        if polar:
            reasons.append("feedback model: clear sentiment about " + ", ".join(polar) + " in a full sentence")
    return {"is_feedback": bool(reasons), "reasons": reasons or ["no opinion cue"]}


def clean_text(s: str) -> str:
    """Unicode/whitespace cleanup only; wording and spelling are kept (informal text is signal)."""
    s = unicodedata.normalize("NFKC", str(s)).translate(_QUOTES)
    return re.sub(r"\s+", " ", s).strip()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences: split after . ! ? (runs) followed by whitespace,
    but not after initials ("J.") or common abbreviations ("Dr.", "e.g.")."""
    spans, start = [], 0
    for m in re.finditer(r"[.!?]+(?=\s)|\n+", text):
        if m.group().startswith(".") and len(m.group()) == 1:
            prev = re.search(r"([A-Za-z.]+)$", text[start:m.start()])
            word = prev.group(1).lower() if prev else ""
            if (len(word) == 1 and word.isalpha()) or word in _ABBREV:
                continue
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    trimmed = []
    for s, e in spans:
        seg = text[s:e]
        s2 = s + (len(seg) - len(seg.lstrip()))
        trimmed.append((s2, s + len(seg.rstrip())))
    return trimmed


def aggregate_pairs(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], list[str]]:
    """One sentiment per aspect. Positive and negative on the same aspect -> neutral (mixed, decision D2).
    Otherwise majority; a polar label wins a tie with neutral."""
    by_aspect: dict[str, Counter] = {}
    for a, s in pairs:
        by_aspect.setdefault(a, Counter())[s] += 1
    out, mixed = {}, []
    for a, c in by_aspect.items():
        if c["positive"] and c["negative"]:
            out[a] = "neutral"
            mixed.append(a)
        else:
            polar = "positive" if c["positive"] else ("negative" if c["negative"] else "neutral")
            out[a] = "neutral" if c["neutral"] > c[polar] else polar
    return out, mixed


def _top(model, texts: list[str]) -> tuple[list[str], np.ndarray]:
    scores = np.atleast_2d(model.decision_function(texts))
    order = np.sort(scores, axis=1)
    labels = np.asarray(model.classes_)[scores.argmax(1)]
    return list(labels), order[:, -1] - order[:, -2]


class FeedbackAnalyzer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.meta = bundle["meta"]
        self.review_model = bundle["review_model"]
        self.clause_model = bundle["clause_model"]
        self.aspect_models = bundle["aspect_models"]
        self.review_margin = float(self.meta["abstain_margins"]["review"])
        self.clause_margin = float(self.meta["abstain_margins"]["clause"])
        self.clause_re = re.compile(self.meta["clause_split_pattern"], re.I)
        self.vocabulary = set(self.meta.get("vocabulary", []))

    @classmethod
    def load(cls, path: Path | str) -> "FeedbackAnalyzer":
        import joblib
        return cls(joblib.load(path))

    def split_clauses(self, sentence: str) -> list[str]:
        parts = [p.strip(" ,") for p in self.clause_re.split(sentence)]
        return [p for p in parts if p] or [sentence]

    def _polarity(self, texts: list[str]) -> list[dict]:
        labels, margin = _top(self.clause_model, texts)
        out = []
        for text, lab, m in zip(texts, labels, margin):
            unclear = lab != "neutral" and m < self.clause_margin
            row = {"model_label": lab, "sentiment": "neutral" if unclear else lab,
                   "margin": round(float(m), 3), "unclear": bool(unclear), "rule": None}
            if row["sentiment"] == "neutral":  # model neutral or unclear: an unambiguous cue word decides
                pol, phrase = cue_polarity(text)
                if pol:
                    row.update(sentiment=pol, unclear=False, rule=f"cue word “{phrase}”")
            out.append(row)
        return out

    def analyze_sentence(self, sentence: str) -> dict:
        detected, all_scores = {}, {}
        for a, m in self.aspect_models.items():
            score = float(m.decision_function([sentence])[0])
            all_scores[a] = round(score, 3)
            if score > 0:  # identical to LogisticRegression.predict() == 1 for a binary model
                detected[a] = round(score, 3)
        clauses = self.split_clauses(sentence)
        pol = self._polarity(clauses)
        clause_rows = [{"text": c, **p, "aspects": []} for c, p in zip(clauses, pol)]
        pairs = []
        for a in detected:
            if len(clauses) == 1:
                idx = 0
            else:
                idx = int(np.argmax(self.aspect_models[a].decision_function(clauses)))
            clause_rows[idx]["aspects"].append(a)
            pairs.append({"aspect": a, "sentiment": clause_rows[idx]["sentiment"],
                          "unclear": clause_rows[idx]["unclear"], "clause": idx,
                          "detector_score": detected[a]})
        return {"sentence": sentence, "clauses": clause_rows, "pairs": pairs, "detector_scores": all_scores}

    def analyze(self, text: str) -> dict:
        t0 = time.perf_counter()
        original = clean_text(text)[:MAX_FEEDBACK_CHARS]
        if not original:
            return {"text": "", "overall": None, "aspects": {}, "sentences": [], "timings_ms": {}}
        text, spelling = correct_spelling(original, self.vocabulary) if self.vocabulary else (original, [])
        sentences = [text[s:e] for s, e in sentence_spans(text)] or [text]
        t1 = time.perf_counter()
        sent_out = [self.analyze_sentence(s) for s in sentences]
        t2 = time.perf_counter()
        lab, margin = _top(self.review_model, [text])
        unclear = lab[0] != "neutral" and margin[0] < self.review_margin
        overall = {"model_label": lab[0], "sentiment": "neutral" if unclear else lab[0],
                   "margin": round(float(margin[0]), 3), "unclear": bool(unclear)}
        all_pairs = [(p["aspect"], p["sentiment"]) for s in sent_out for p in s["pairs"]]
        agg, mixed = aggregate_pairs(all_pairs)
        aspects = {a: {"label": ASPECT_LABELS.get(a, a), "sentiment": s, "mixed": a in mixed,
                       "unclear": all(p["unclear"] for so in sent_out for p in so["pairs"] if p["aspect"] == a),
                       "trained_on_generated_only": a in self.meta.get("generated_only_aspects", [])}
                   for a, s in sorted(agg.items())}
        # The shown overall follows the aspect results when there are any polar ones: the review model was trained on
        # long public reviews and can contradict clear per-aspect findings on short RHU sentences.
        polar = {v["sentiment"] for v in aspects.values() if v["sentiment"] != "neutral"}
        overall["review_model_sentiment"] = overall["sentiment"]
        if polar:
            overall.update(sentiment=polar.pop() if len(polar) == 1 else "neutral", unclear=False,
                           mixed=len({v["sentiment"] for v in aspects.values()} - {"neutral"}) > 1,
                           source="aspects")
        else:
            clause_pol = {c["sentiment"] for s in sent_out for c in s["clauses"]} - {"neutral"}
            if len(clause_pol) == 1:  # no aspect found, but the clauses agree on a clear polarity
                overall.update(sentiment=clause_pol.pop(), unclear=False, mixed=False, source="clauses")
            else:
                overall.update(mixed=False, source="review model")
        t3 = time.perf_counter()
        return {
            "text": text, "original_text": original, "spelling_changes": spelling,
            "is_question": is_question(original),
            "overall": overall, "aspects": aspects, "sentences": sent_out,
            "uncovered_aspects": [a for a in ASPECT_LABELS if a not in self.aspect_models],
            "thresholds": {"review_margin": self.review_margin, "clause_margin": self.clause_margin},
            "timings_ms": {"split": round((t1 - t0) * 1000, 2), "aspects_and_clauses": round((t2 - t1) * 1000, 2),
                           "overall": round((t3 - t2) * 1000, 2), "total": round((t3 - t0) * 1000, 2)},
        }

    def predict_pairs(self, text: str) -> dict[str, str]:
        """{aspect: sentiment} for evaluation: the same path as analyze(), without the trace."""
        return {a: v["sentiment"] for a, v in self.analyze(text)["aspects"].items()}
