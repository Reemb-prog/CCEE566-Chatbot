"""Informal-language normalizer: student shorthand, course acronyms, spelling.

Classical and data-driven, no training. Everything it knows comes from the bundled
catalog tables or the short hand-written shorthand list below:

* **Shorthand**: a small, visible dictionary ("prereqs" -> "prerequisites",
  "intro" -> "introduction", "db" -> "database management systems").
* **Course acronyms**: built from catalog course titles by taking word initials
  ("Natural Language Processing" -> NLP, "Machine Learning and Data Mining" -> ML
  from its first part). A token is only read as an acronym if its lower-case form
  never appears as an ordinary lower-case word in the catalog (so "is", "art", "it"
  are never treated as acronyms). Ambiguous acronyms are reported, never guessed.
* **Spelling**: a word that does not occur in the catalog vocabulary is replaced by
  the closest catalog word (difflib ratio, same first letter, ties by catalog
  frequency). Course codes are never touched.

Every change is returned so the trace can show what was rewritten and why.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .nlp import CODE_RE

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Hand-written student shorthand. Applied as whole-word, case-insensitive rules, in order.
SHORTHAND: list[tuple[str, str]] = [
    (r"pre-?\s?reqs", "prerequisites"), (r"pre-?\s?req", "prerequisite"),
    (r"co-?\s?reqs", "corequisites"), (r"co-?\s?req", "corequisite"),
    (r"creds", "credits"), (r"cred", "credit"), (r"cr\.?\s?hrs?", "credit hours"),
    (r"hrs", "hours"), (r"intro", "introduction"),
    (r"whats", "what is"), (r"hows", "how is"), (r"wat", "what"), (r"r", "are"), (r"u", "you"), (r"ur", "your"),
    (r"pls|plz", "please"), (r"abt", "about"), (r"b4", "before"), (r"w/", "with"),
    (r"sems", "semesters"), (r"sem", "semester"), (r"yrs", "years"), (r"yr", "year"),
    (r"dept", "department"), (r"uni", "university"), (r"reqs", "requirements"), (r"req", "requirement"),
    (r"info", "information"), (r"gen\s?eds?", "general education"),
    (r"comp\.?\s?sci", "computer science"), (r"cs", "computer science"),
    (r"dbms|db", "database management systems"),
    (r"algos?", "algorithms"), (r"stats", "statistics"), (r"calc", "calculus"),
    (r"prog", "programming"), (r"econ", "economics"), (r"psych", "psychology"),
    (r"bio", "biology"), (r"chem", "chemistry"), (r"phys", "physics"),
]
_SHORTHAND_RE = [(re.compile(rf"(?<![\w/-])(?:{pat})(?![\w/-])", re.I), rep) for pat, rep in SHORTHAND]

# Upper-case strings in the catalog that are NOT course acronyms.
NOT_COURSE_ACRONYMS = {"gpa", "cgpa", "rhu", "dohe", "toefl", "ielts", "sat", "pdf", "faq", "usa", "uk",
                       "id", "ok", "hi"}
# Words a chat question uses that the catalog may not; never "corrected".
CHAT_WORDS = {"hi", "hello", "hey", "thanks", "thank", "please", "okay", "ok", "tell", "explain",
              "wanna", "gonna", "need", "take", "taking", "took", "pass", "passed", "fail", "failed",
              "course", "courses", "class", "classes", "prof", "professor", "doctor", "dr",
              "prerequisite", "prerequisites", "corequisite", "corequisites", "credit", "credits",
              # words the instruction-override rule looks for must survive unchanged
              "ignore", "disregard", "forget", "override", "bypass", "invent", "fabricate",
              "pretend", "hallucinate", "rules", "instructions", "prompt", "guidelines"}
TITLE_STOP = {"and", "of", "for", "the", "to", "in", "&", "a", "an", "on", "with"}
TITLE_LEAD = re.compile(r"^(introduction to|intro to|principles of|fundamentals of|foundations of|"
                        r"topics in|special topics in|selected topics in) ")
TITLE_TAIL = re.compile(r"( lab| laboratory| i{1,3}| iv| \d)+$")


def _initials(phrase: str) -> str:
    return "".join(w[0] for w in re.split(r"[\s-]+", phrase) if w and w not in TITLE_STOP)


class Normalizer:
    def __init__(self, texts: list[str], titles: dict[str, set[str]], course_prefixes: set[str]):
        counts: Counter = Counter()
        lower_seen: set[str] = set()
        for text in texts:
            for w in WORD_RE.findall(text):
                if "'" in w:
                    continue
                counts[w.lower()] += 1
                if w.islower():
                    lower_seen.add(w)
        for w in ENGLISH_STOP_WORDS | CHAT_WORDS:
            counts[w] += 1
            lower_seen.add(w)
        self.freq = counts
        self.lower_words = lower_seen
        self.prefixes = {p.lower() for p in course_prefixes}
        self.titles = titles
        self._by_letter: dict[str, list[str]] = {}
        for w in counts:
            if len(w) >= 3:
                self._by_letter.setdefault(w[0], []).append(w)

        # acronym -> {title: tier}; tier 0 = all title words, 1 = first part / without lead or tail
        acr: dict[str, dict[str, int]] = {}
        for title in titles:
            variants = {(title, 0)}
            core = TITLE_TAIL.sub("", TITLE_LEAD.sub("", title)).strip()
            variants.add((core, 1))
            for part in re.split(r" and | & ", core):
                variants.add((part.strip(), 1))
            for phrase, tier in variants:
                a = _initials(phrase)
                if 2 <= len(a) <= 5 and len(phrase.split()) >= 2:
                    prev = acr.setdefault(a, {}).get(title)
                    acr[a][title] = tier if prev is None else min(prev, tier)
        self.acronyms = acr

    # ------------------------------------------------------------------
    def _is_acronym_candidate(self, tok: str) -> bool:
        low = tok.lower()
        return (2 <= len(low) <= 5 and low in self.acronyms and low not in self.lower_words
                and low not in NOT_COURSE_ACRONYMS and low not in self.prefixes)

    def _resolve_acronym(self, low: str) -> list[str]:
        """Titles an acronym can stand for, keeping only the best tier and dropping lab twins."""
        cands = self.acronyms[low]
        best = min(cands.values())
        titles = sorted(t for t, tier in cands.items() if tier == best)
        no_lab = [t for t in titles if not re.search(r"\blab(oratory)?\b", t)]
        return no_lab or titles

    def _correct(self, low: str) -> str | None:
        pool = self._by_letter.get(low[0], [])
        cutoff = 0.84 if len(low) <= 5 else 0.8
        close = difflib.get_close_matches(low, pool, n=4, cutoff=cutoff)
        # "invent" vs "inventor", "ignore" vs "ignorance": a different word form, not a typo.
        close = [w for w in close if not (w.startswith(low) or low.startswith(w))]
        if not close:
            return None
        return max(close, key=lambda w: (round(difflib.SequenceMatcher(None, low, w).ratio(), 3), self.freq[w]))

    def normalize(self, question: str) -> dict:
        changes: list[dict] = []
        ambiguous: list[dict] = []

        # 1. shorthand (phrase level); a match that is a course prefix or sits before a
        #    course number (e.g. "PHYS 201") is left alone.
        text = question
        for rx, rep in _SHORTHAND_RE:
            def sub(m, rep=rep):
                tok = m.group(0)
                if (tok.lower() in self.prefixes or tok.lower() == rep
                        or re.match(r"\s?-?\s?\d{3}", m.string[m.end():])):
                    return tok
                changes.append({"from": tok, "to": rep, "rule": "student shorthand"})
                return rep
            text = rx.sub(sub, text)

        protected = [m.span() for m in CODE_RE.finditer(text)]

        def in_code(span) -> bool:
            return any(s <= span[0] < e for s, e in protected)

        # 2. acronyms and spelling (word level)
        out, last = [], 0
        for m in WORD_RE.finditer(text):
            tok, low = m.group(0), m.group(0).lower()
            new = None
            if in_code(m.span()) or "'" in tok:
                pass
            elif self._is_acronym_candidate(tok):
                titles = self._resolve_acronym(low)
                if len(titles) == 1:
                    new = titles[0]
                    codes = sorted(self.titles[new])
                    changes.append({"from": tok, "to": new, "rule": "course acronym",
                                    "detail": f"initials of the catalog title ({', '.join(codes)})"})
                else:
                    ambiguous.append({"acronym": tok.upper(),
                                      "options": [{"title": t, "codes": sorted(self.titles[t])} for t in titles]})
            elif (len(low) >= 4 and low.isalpha() and low not in self.freq and low not in self.prefixes):
                fix = self._correct(low)
                if fix and fix != low:
                    new = fix
                    changes.append({"from": tok, "to": fix, "rule": "spelling",
                                    "detail": "closest word in the catalog vocabulary"})
            out.append(text[last:m.start()])
            out.append(new if new is not None else tok)
            last = m.end()
        out.append(text[last:])
        normalized = re.sub(r"\s+", " ", "".join(out)).strip()
        return {"original": question, "normalized": normalized, "changed": normalized != question.strip(),
                "changes": changes, "ambiguous_acronyms": ambiguous}


def nice_title(title: str) -> str:
    """'machine learning and data mining' -> 'Machine Learning and Data Mining'."""
    words = title.split()
    return " ".join(w if (w in TITLE_STOP and i) else (w.upper() if re.fullmatch(r"i{1,3}|iv", w) else w.capitalize())
                    for i, w in enumerate(words))


def interpretation_note(norm: dict) -> str:
    """Short, user-facing line for acronym and spelling rewrites (empty if none).

    Plain shorthand ("prereq", "whats") is obvious and only shown in the inspector trace.
    """
    parts = []
    for c in norm["changes"]:
        if c["rule"] == "course acronym":
            parts.append(f"“{c['from']}” as “{nice_title(c['to'])}”")
        elif c["rule"] == "spelling":
            parts.append(f"“{c['from']}” as “{c['to']}”")
    if not parts:
        return ""
    return "(I read " + ", ".join(dict.fromkeys(parts)) + ".) "
