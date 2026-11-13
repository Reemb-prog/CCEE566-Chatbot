"""Deterministic, inspectable text analysis on the ORIGINAL question.

Regex/keyword rules only: course-code extraction, requested course field,
program/semester clues, and scope checks (personal records, live data, clearly
non-catalog topics, instruction-override attempts). Every rule that fires is
reported in the trace by name.
"""
from __future__ import annotations

import difflib
import re

# "CCEE 566", "CCEE566", "ccee-566", "COSC 351L"
CODE_RE = re.compile(r"\b([A-Za-z]{3,4})\s?[-‐–]?\s?(\d{3}[A-Za-z]?)\b")

FIELD_PATTERNS = [
    ("corequisite", r"\bco-?\s?req|\bcorequisite"),
    ("prerequisite", r"\bpre-?\s?req|\bprerequisite|\brequire(?:d|ment)?s?\b.*\bbefore\b|\bwithout\b|\bneed to (?:take|pass|complete)\b.*\bbefore\b"),
    ("equivalent", r"\bequivalent"),
    ("credits", r"\bcredit|\bhow many hours\b"),
    ("description", r"\bdescri|\bcover|\babout\b|\btopics?\b|\bcontent|\bsyllabus|\bwhat is\b|\bwhat's\b|\blearn"),
]

PERSONAL_RE = re.compile(
    r"\b(my|mine|i've|i have|am i)\b.*\b(gpa|cgpa|grade|grades|transcript|record|records|balance|account|"
    r"schedule|appointment|advisor|adviser|status|standing|password|login|fees? due|marks?|results?|"
    r"registration|enrollment|enrolment|id)\b"
    r"|\b(gpa|cgpa|grades?|transcript|appointment)\b.*\b(my|mine)\b", re.I)
LIVE_RE = re.compile(
    r"\b(right now|today|tonight|tomorrow|yesterday|this week|next week|last week|next month|"
    r"currently open|at the moment|latest news|live)\b|next week[’']s", re.I)
OFF_TOPIC_RE = re.compile(
    r"\b(weather|temperature|forecast|football|soccer|basketball|match score|who won|stock price|"
    r"bitcoin|movie|recipe|celebrity|president of|election|news)\b", re.I)
OVERRIDE_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(rules?|instructions?|prompt|guidelines?)\b"
    r"|\b(invent|make up|fabricate|pretend|hallucinate)\b", re.I)
PLAN_RE = re.compile(r"\bstudy plan|\bsemester|\bcourses? (?:in|for|of)\b.*\b(year|semester|program)|\byear \d|\bfirst year\b|\bplan for\b", re.I)

ORDINALS = {"first": 1, "1st": 1, "one": 1, "second": 2, "2nd": 2, "two": 2, "third": 3, "3rd": 3,
            "three": 3, "fourth": 4, "4th": 4, "four": 4, "fifth": 5, "5th": 5, "five": 5,
            "sixth": 6, "6th": 6, "six": 6, "seventh": 7, "7th": 7, "seven": 7, "eighth": 8,
            "8th": 8, "eight": 8}
_ORD = "|".join(sorted(ORDINALS, key=len, reverse=True))

PROGRAM_ALIASES = {
    "computer science": "Computer Science Program",
    "electrical engineering": "Electrical Engineering Program",
    "mechanical engineering": "Mechanical Engineering Program",
    "biomedical engineering": "Biomedical Engineering Program",
    "computer and communications engineering": "Computer and Communications Engineering Program",
    "computer & communications engineering": "Computer and Communications Engineering Program",
    "communications engineering": "Computer and Communications Engineering Program",
    "cce": "Computer and Communications Engineering Program",
    "mechatronics": "Mechatronics Engineering Program",
    "civil engineering": "Civil Engineering Program",
    "graphic design": "Graphic Design Program",
    "accounting": "Accounting Program",
    "healthcare information systems": "Healthcare Information Systems Program",
    "management": "Management Program",
    "business it management": "Business IT Management Program",
    "business it": "Business IT Management Program",
    "human resources": "Human Resources Management Program",
    "finance and banking": "Finance and Banking Program",
    "finance": "Finance and Banking Program",
    "marketing": "Marketing and Advertising Program",
    "english language": "English Language Program",
    "english": "English Language Program",
    "freshman": "Freshman Program",
}


def extract_course_codes(question: str, known_prefixes: set[str]) -> tuple[list[str], list[str]]:
    """Return (codes with a known catalog prefix, code-like strings with an unknown prefix)."""
    codes, unknown = [], []
    for prefix, num in CODE_RE.findall(question):
        code = f"{prefix.upper()} {num.upper()}"
        target = codes if prefix.upper() in known_prefixes else unknown
        if code not in target and not (prefix.lower() in {"year", "page", "room", "the", "and", "for"}):
            target.append(code)
    return codes, unknown


def suggest_prefix(code: str, known_prefixes: set[str], known_codes: set[str]) -> list[str]:
    """Close existing codes for an unknown prefix (suggestion only, never auto-substituted)."""
    prefix, num = code.split()
    close = difflib.get_close_matches(prefix, sorted(known_prefixes), n=3, cutoff=0.6)
    return [f"{p} {num}" for p in close if f"{p} {num}" in known_codes]


def requested_field(question: str) -> str | None:
    q = question.lower()
    for name, pat in FIELD_PATTERNS:
        if re.search(pat, q):
            return name
    return None


def scope_flags(question: str) -> list[str]:
    flags = []
    if PERSONAL_RE.search(question):
        flags.append("personal_record_request")
    if LIVE_RE.search(question):
        flags.append("live_or_time_sensitive")
    if OFF_TOPIC_RE.search(question):
        flags.append("outside_catalog_topic")
    if OVERRIDE_RE.search(question):
        flags.append("instruction_override_attempt")
    return flags


def detect_program(question: str) -> str | None:
    q = " " + re.sub(r"[^a-z& ]", " ", question.lower()) + " "
    for alias in sorted(PROGRAM_ALIASES, key=len, reverse=True):
        if f" {alias} " in q:
            return PROGRAM_ALIASES[alias]
    return None


def detect_semester(question: str) -> dict:
    """Pull year / term / ordinal-semester clues. Returns only clues actually present."""
    q = question.lower()
    clue: dict = {}
    m = re.search(rf"\byear\s+(\d|{_ORD})\b|\b({_ORD})\s+year\b", q)
    if m:
        tok = m.group(1) or m.group(2)
        clue["year"] = int(tok) if tok.isdigit() else ORDINALS[tok]
    if re.search(r"\bfall\b", q):
        clue["term"] = "fall"
    elif re.search(r"\bspring\b", q):
        clue["term"] = "spring"
    elif re.search(r"\bsummer\b", q):
        clue["term"] = "summer"
    m = re.search(rf"\b(?:semester|term)\s+(\d|{_ORD})\b|\b({_ORD}|\d)\s+(?:semester|term)\b", q)
    if m:
        tok = m.group(1) or m.group(2)
        clue["semester_number"] = int(tok) if tok.isdigit() else ORDINALS[tok]
    return clue


def match_course_title(question: str, titles: dict[str, set[str]]) -> tuple[str | None, set[str], float]:
    """Fuzzy-match a course title typed without its code (handles small typos).

    ``titles`` maps a lower-cased title to the set of codes that carry it.
    Returns (matched title, codes, similarity ratio) or (None, set(), best_ratio).
    """
    words = re.findall(r"[a-z&]+", question.lower())
    best = (None, set(), 0.0)
    for title, codes in titles.items():
        n = len(title.split())
        if n < 2:
            continue
        for size in {n - 1, n, n + 1} - {0}:
            for i in range(0, max(1, len(words) - size + 1)):
                window = " ".join(words[i:i + size])
                sm = difflib.SequenceMatcher(None, window, title)
                if sm.real_quick_ratio() < 0.88 or sm.quick_ratio() < 0.88:   # cheap upper bounds
                    continue
                r = sm.ratio()
                if r > best[2]:
                    best = (title, codes, r)
    if best[2] >= 0.88:
        return best
    return None, set(), best[2]
