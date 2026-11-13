"""Grounded, extractive answer construction. No generation: every answered claim
is copied from a catalog table field or a verbatim passage sentence, and cited.
"""
from __future__ import annotations

import re

from . import nlp
from .data import CatalogData, citation_url, clean, norm_space

# Conservative retrieval thresholds, chosen by hand on a few catalog questions
# (not tuned on the historical QA test). They are relative TF-IDF cosine values.
MIN_SIMILARITY = 0.10        # below this the best passage is treated as unrelated
MIN_COVERAGE = 0.6           # share of the question's IDF-weighted content words the evidence passage must contain
MAX_PLAN_ROWS = 12
# Conversational words ignored when measuring how well a passage covers the question.
QUERY_FILLER = {"catalog", "say", "says", "tell", "explain", "know", "want", "rhu", "please", "information",
                "info", "details", "does", "page", "university", "student", "students"}
FIELD_SENTENCE = {"prerequisite": r"pre-?requisite", "corequisite": r"co-?requisite", "credits": r"\bcredit"}

FALLBACK = {
    "personal_record_request": "I can't check personal records such as your GPA, grades, transcript, "
                               "registration or appointments. This assistant only reads the static "
                               "2026–2027 Undergraduate Catalog; please use the student portal or contact "
                               "the Registrar or your advisor.",
    "live_or_time_sensitive": "The catalog is a static 2026–2027 document, so I can't confirm live or "
                              "upcoming information (current deadlines, today's events). Please check the "
                              "official RHU website or the Registrar for up-to-date dates.",
    "outside_catalog_topic": "That question is outside the RHU Undergraduate Catalog, which is the only "
                             "source this assistant uses, so I can't answer it.",
    "low_relevance": "I couldn't find catalog text that clearly answers this question, so I won't guess. "
                     "Try rephrasing with specific terms (e.g. a course code like CCEE 566, a program name, "
                     "or a policy such as probation or withdrawal).",
}


def _result(status, answer, decision, citations=None, evidence_ids=None, support=None,
            notes=None, suggestions=None, strategy=None, structured=None):
    return {"status": status, "answer": answer, "decision": decision,
            "citations": citations or [], "evidence_ids": evidence_ids or [],
            "support": support or [], "notes": notes or [], "suggestions": suggestions or [],
            "strategy": strategy, "structured_matches": structured or []}


def _cite(source_id, page, url, excerpt, kind, section=None):
    return {"source_id": str(source_id), "source_page": int(page),
            "source_url": citation_url(url, page), "excerpt": excerpt,
            "excerpt_kind": kind, "section": section or ""}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(•])|\s*[•]\s*", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _stem(tok: str) -> str:
    return tok[:6]


# ---------------------------------------------------------------- courses
def _prereq_of(row: dict) -> tuple[str | None, str]:
    """Prerequisite from the structured field; else a 'Prerequisite:' clause inside the description."""
    p = clean(row.get("prerequisite_text"))
    if p:
        return p.rstrip(". "), "prerequisite_text"
    m = re.search(r"Pre-?requisites?\s*:\s*([^.]+)", norm_space(row.get("description")), re.I)
    if m:
        return m.group(1).strip(), "description"
    return None, "none"


def _continued_prereq(data: CatalogData, row: dict) -> tuple[str, dict] | None:
    """If a course description is cut at a page break, look for its continuation: the first passage
    of the same catalog section on the next page, not starting with a new course code."""
    desc = norm_space(row.get("description"))
    if not desc or desc.rstrip().endswith((".", ")")):
        return None
    page = int(row["source_page"]) + 1
    ps = data.passages
    nxt = ps[(ps["section"] == row.get("program_or_section")) & (ps["source_page_start"] == page)]
    if nxt.empty:
        return None
    first = nxt.iloc[0]
    text = first["passage_text"]
    if re.match(r"[A-Z]{3,4} \d{3}", text):
        return None
    m = re.search(r"Pre-?requisites?\s*:\s*([^.]+)", text[:400], re.I)
    if not m:
        return None
    return m.group(1).strip(), {"id": first["passage_id"], "source_page": page, "source_url": first["source_url"],
                                "section": first["section"], "excerpt": text[:300]}


def _course_excerpt(data: CatalogData, row: dict, needle: str | None) -> tuple[str, str]:
    """Verbatim excerpt: description sentence containing ``needle`` if any, else page-text window."""
    desc = norm_space(row.get("description"))
    if needle:
        for s in _sentences(desc):
            if needle.lower() in s.lower():
                return s, "course_record_description"
    window = data.page_window(row["course_code"], int(row["source_page"]))
    if window:
        return window, "page_text"
    head = f'{row["course_code"]} {row.get("course_title") or ""} {row.get("credit_notation") or ""}'.strip()
    return head, "course_record_fields"


def _structured(rows: list[dict]) -> list[dict]:
    return [{"strategy": "exact_course_code", "id": r["course_id"], "course_code": r["course_code"],
             "program_or_section": r.get("program_or_section"), "source_page": int(r["source_page"]),
             "credits": clean(r.get("credits")), "credit_notation": clean(r.get("credit_notation")),
             "prerequisite": _prereq_of(r)[0], "needs_manual_review": r.get("needs_manual_review") == "yes"}
            for r in rows]


def answer_course(data: CatalogData, code: str, field: str | None, extra_codes: list[str]) -> dict:
    rows = data.course_rows(code)
    if not rows:
        return None  # caller falls back to passage evidence for this code
    title = clean(rows[0].get("course_title")) or ""
    label = f"{code} ({title})" if title else code
    structured = _structured(rows)
    notes = []
    review = [r for r in rows if r.get("needs_manual_review") == "yes"]
    if review:
        notes.append(f"{len(review)} matching record(s) are flagged needs_manual_review "
                     f"({clean(review[0].get('review_reason')) or 'extraction uncertainty'}); verify on the PDF page.")
    ids = [r["course_id"] for r in rows]

    if field == "credits":
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(clean(r.get("credits")) or "?", []).append(r)
        support, cits = [], []
        for val, rs in groups.items():
            for r in rs:
                notation = clean(r.get("credit_notation"))
                verified = data.text_on_page(f'{code} {title} {notation or ""}'.strip(), int(r["source_page"]), False) \
                    or data.text_on_page(notation or val, int(r["source_page"]), False)
                ex, kind = _course_excerpt(data, r, None)
                cits.append(_cite(r["course_id"], r["source_page"], r["source_url"], ex, kind, r.get("program_or_section")))
                support.append({"claim": f"credits = {val}", "field": "credits / credit_notation",
                                "value": notation or val, "source_id": r["course_id"],
                                "source_page": int(r["source_page"]), "verified_in_page_text": bool(verified)})
        if len(groups) == 1:
            val = next(iter(groups))
            notation = clean(rows[0].get("credit_notation"))
            pages = ", ".join(f"p. {r['source_page']}" for r in rows)
            nt = f" (listed as {notation})" if notation else ""
            if len(rows) > 1:
                ans = f"{label} is a {val}-credit course{nt}. All {len(rows)} catalog records agree ({pages})."
            else:
                ans = f"{label} is a {val}-credit course{nt}, according to the catalog ({pages})."
            return _result("answered", ans, "exact_course_match", cits, ids, support, notes,
                           strategy="structured_course_lookup", structured=structured)
        lines = [f"• {val} credits — " + ", ".join(f"{r.get('program_or_section')} (p. {r['source_page']})" for r in rs)
                 for val, rs in groups.items()]
        ans = f"The catalog lists different credit values for {label}:\n" + "\n".join(lines) + \
              "\nPlease confirm which program applies with your advisor."
        return _result("answered", ans, "conflicting_records", cits, ids, support, notes,
                       strategy="structured_course_lookup", structured=structured)

    if field in ("prerequisite", "corequisite", "equivalent"):
        conts: dict[str, dict] = {}
        if field == "prerequisite":
            vals = []
            for r in rows:
                v, src = _prereq_of(r)
                if not v:
                    c = _continued_prereq(data, r)
                    if c:
                        v, src = c[0], "next_page_continuation"
                        conts[r["course_id"]] = c[1]
                vals.append((r, v, src))
        else:
            col = "corequisite_text" if field == "corequisite" else "equivalent_text"
            pat = r"Co-?requisites?\s*:\s*([^.]+)" if field == "corequisite" else r"Equivalent to\s+([^.]+)"
            vals = []
            for r in rows:
                v = clean(r.get(col))
                src = col if v else "none"
                if not v:   # fall back to the clause inside the record's own description text
                    m = re.search(pat, norm_space(r.get("description")), re.I)
                    if m:
                        v, src = m.group(1).strip(), "description"
                vals.append((r, v, src))
        found = [(r, v, src) for r, v, src in vals if v]
        missing = [r for r, v, _ in vals if not v]
        if conts:
            notes.append("Some entries are split across a page break; their prerequisite was read from the first "
                         "passage of the same section on the next page (continuation heuristic) and is cited separately.")
        noun = {"prerequisite": "prerequisite", "corequisite": "corequisite", "equivalent": "equivalent course"}[field]
        if not found:
            pages = ", ".join(f"p. {r['source_page']}" for r in rows)
            return _result("insufficient_evidence",
                           f"The extracted catalog records for {label} ({pages}) do not list a {noun}. "
                           f"I can't confirm one, so please check the course entry in the PDF or ask your advisor.",
                           "field_not_in_records", [], ids, [], notes, strategy="structured_course_lookup",
                           structured=structured)
        support, cits, distinct = [], [], {}
        for r, v, src in found:
            key = re.sub(r"[^a-z0-9]", "", v.lower())
            distinct.setdefault(key, []).append((r, v))
            ex, kind = _course_excerpt(data, r, v)
            verified = data.text_on_page(v, int(r["source_page"]))
            cits.append(_cite(r["course_id"], r["source_page"], r["source_url"], ex, kind, r.get("program_or_section")))
            c = conts.get(r["course_id"])
            if c:
                i = c["excerpt"].lower().find("prerequisite")
                cex = c["excerpt"][max(0, i - 80):i + 120].strip() if i >= 0 else c["excerpt"]
                cits.append(_cite(c["id"], c["source_page"], c["source_url"], cex, "passage_text", c["section"]))
                ids.append(c["id"])
            support.append({"claim": f"{noun} = {v}", "field": src, "value": v, "source_id": r["course_id"],
                            "source_page": int(r["source_page"]), "program_or_section": r.get("program_or_section"),
                            "verified_in_page_text": bool(verified)})
        if missing:
            notes.append("No " + noun + " was extracted from: " +
                         ", ".join(f"{r.get('program_or_section')} (p. {r['source_page']})" for r in missing) +
                         ". That entry may continue across a page break; check the PDF.")
        if field == "prerequisite" and extra_codes:
            notes.append("The catalog lists prerequisites only; it does not describe exceptions or waivers "
                         f"involving {', '.join(extra_codes)}. Ask your advisor about exceptions.")
        if len(distinct) == 1:
            v = found[0][1]
            pages = ", ".join(f"p. {r['source_page']}" for r, _, _ in found)
            ans = f"The {noun} for {label} is: {v} (catalog {pages})."
            if missing:
                ans += f" (Other entries for this course list no {noun}; see notes.)"
            if extra_codes and field == "prerequisite":
                ans += " The catalog does not describe exceptions to this requirement."
            return _result("answered", ans, "exact_course_match", cits, ids, support, notes,
                           strategy="structured_course_lookup", structured=structured)
        lines = []
        for _, entries in distinct.items():
            v = entries[0][1]
            where = "; ".join(f"{r.get('program_or_section')} entry (p. {r['source_page']}" +
                              (f", continued on p. {conts[r['course_id']]['source_page']})" if r["course_id"] in conts else ")")
                              for r, _ in entries)
            lines.append(f"• {where}: {v}")
        ans = (f"The catalog contains different {noun} entries for {label}, depending on the section:\n" +
               "\n".join(lines) +
               "\nI'm not choosing one automatically — please confirm which program applies to you with your advisor.")
        return _result("answered", ans, "conflicting_records", cits, ids, support, notes,
                       strategy="structured_course_lookup", structured=structured)

    # description, or no specific field -> short summary from the records
    r = next((x for x in rows if x.get("needs_manual_review") != "yes"), rows[0])
    sents = _sentences(norm_space(r.get("description")))
    desc_sents = [s for s in sents if not re.match(r"Pre-?requisite|Co-?requisite|Equivalent", s, re.I)][:2]
    credits = {clean(x.get("credits")) for x in rows}
    cits = [_cite(x["course_id"], x["source_page"], x["source_url"], _course_excerpt(data, x, None)[0],
                  _course_excerpt(data, x, None)[1], x.get("program_or_section")) for x in rows]
    if desc_sents:
        cits[rows.index(r)]["excerpt"] = " ".join(desc_sents)
        cits[rows.index(r)]["excerpt_kind"] = "course_record_description"
    head = f"{label}" + (f", {next(iter(credits))} credits" if len(credits) == 1 else "")
    body = (" Catalog description (p. " + str(r["source_page"]) + "): “" + " ".join(desc_sents) + "”") if desc_sents else \
        " The extracted record has no description text."
    ans = head + "." + body
    if len(rows) > 1:
        ans += f" The course appears in {len(rows)} catalog sections (" + \
               ", ".join(f"p. {x['source_page']}" for x in rows) + "); ask about credits or prerequisites for details."
    return _result("answered", ans, "exact_course_match", cits, ids,
                   [{"claim": "course description", "field": "description", "source_id": r["course_id"],
                     "source_page": int(r["source_page"]), "value": " ".join(desc_sents)[:200],
                     "verified_in_page_text": data.text_on_page(desc_sents[0][:80], int(r["source_page"])) if desc_sents else False}],
                   notes, strategy="structured_course_lookup", structured=structured)


def answer_code_from_passages(data: CatalogData, code: str, candidates: list[dict]) -> dict:
    """Code exists in passages/plans but not in the course-description table."""
    hits = [c for c in candidates if re.search(rf"\b{re.escape(code)}\b", c["text"])]
    if not hits:
        ids = data.code_in_passages(code)
        if ids:
            rows = data.passages[data.passages["passage_id"].isin(ids[:1])].to_dict("records")
            hits = [{"id": r["passage_id"], "source_page": int(r["source_page_start"]), "source_url": r["source_url"],
                     "section": r["section"], "text": r["passage_text"]} for r in rows]
    if not hits:
        return None
    h = hits[0]
    i = h["text"].find(code)
    ex = h["text"][max(0, i - 20): i + 200].strip()
    return _result("answered",
                   f"{code} has no full course-description record in the extracted tables, but it is mentioned in "
                   f"the {h['section']} section (p. {h['source_page']}). Catalog text: “{ex}”",
                   "code_mention_excerpt",
                   [_cite(h["id"], h["source_page"], h["source_url"], ex, "passage_text", h["section"])],
                   [h["id"]], [{"claim": f"{code} is mentioned", "field": "passage_text", "source_id": h["id"],
                                "source_page": h["source_page"], "value": ex[:120], "verified_in_page_text": True}],
                   ["Only a mention was found; details such as prerequisites may be missing."],
                   strategy="passage_code_mention")


# ---------------------------------------------------------------- study plans
def _semester_order(plans_for_program) -> list[str]:
    return list(dict.fromkeys(plans_for_program["semester"].tolist()))


def _pick_semester(semesters: list[str], clue: dict) -> list[str]:
    def year_of(s):
        m = re.search(r"Year (\d)", s)
        return int(m.group(1)) if m else None

    regular = [s for s in semesters if "summer" not in s.lower()]
    picked = semesters
    if "year" in clue:
        picked = [s for s in picked if year_of(s) == clue["year"]]
    if "term" in clue:
        t = clue["term"]
        if t == "fall":
            picked = [s for s in picked if "fall" in s.lower() or re.search(r"semester 1\b", s.lower())]
        elif t == "spring":
            picked = [s for s in picked if "spring" in s.lower() or re.search(r"semester 2\b", s.lower())]
        else:
            picked = [s for s in picked if "summer" in s.lower()]
    if "semester_number" in clue:
        n = clue["semester_number"]
        if "year" in clue:   # "year 2, semester 1"
            reg = [s for s in picked if "summer" not in s.lower()]
            picked = reg[n - 1:n] if n <= len(reg) else []
        else:                # "semester four" = 4th regular (non-summer) semester of the plan
            picked = [regular[n - 1]] if n <= len(regular) else []
    return picked if clue else []


def answer_study_plan(data: CatalogData, question: str) -> dict | None:
    program = nlp.detect_program(question)
    clue = nlp.detect_semester(question)
    if not program:
        return _result("needs_clarification",
                       "Which program do you mean? Study plans in the catalog exist for: " +
                       ", ".join(p.replace(" Program", "") for p in data.programs) + ".",
                       "needs_clarification", notes=["No program name recognised in the question."],
                       strategy="study_plan_lookup")
    plans = data.plans[data.plans["program_or_section"] == program]
    semesters = _semester_order(plans)
    picked = _pick_semester(semesters, clue)
    if not picked:
        listing = "; ".join(semesters)
        return _result("needs_clarification",
                       f"Which semester of the {program} study plan? The catalog lists: {listing}.",
                       "needs_clarification",
                       notes=[f"Semester clue {clue or 'missing'} did not identify a single semester."],
                       strategy="study_plan_lookup")
    if len(picked) > 1:
        return _result("needs_clarification",
                       f"That matches several semesters of the {program} plan: {'; '.join(picked)}. Which one?",
                       "needs_clarification", strategy="study_plan_lookup")
    sem = picked[0]
    rows = plans[plans["semester"] == sem].to_dict("records")[:MAX_PLAN_ROWS]
    lines = []
    for r in rows:
        cr = clean(r.get("credits_in_study_plan"))
        lines.append(f"• {r['course_code']} {clean(r.get('course_title')) or ''}" + (f" ({cr} cr)" if cr else ""))
    cits, seen = [], set()
    for r in rows:
        if r["source_page"] in seen:
            continue
        seen.add(r["source_page"])
        page_rows = [x for x in rows if x["source_page"] == r["source_page"]]
        ex = f"{sem}: " + "; ".join(f"{x['course_code']} {clean(x.get('course_title')) or ''}".strip() for x in page_rows)
        cits.append(_cite(r["plan_id"], r["source_page"], r["source_url"], ex, "study_plan_rows", program))
    pages = ", ".join(f"p. {p}" for p in sorted(seen))
    ans = f"{program} — {sem} ({pages}):\n" + "\n".join(lines)
    ans += "\nRows were extracted from the catalog's study-plan table (status: extracted_table); check the PDF for footnotes."
    return _result("answered", ans, "study_plan_rows", cits, [r["plan_id"] for r in rows],
                   [{"claim": f"{r['course_code']} in {sem}", "field": "study plan row", "source_id": r["plan_id"],
                     "source_page": int(r["source_page"]), "value": r["course_code"],
                     "verified_in_page_text": data.text_on_page(r["course_code"], int(r["source_page"]))} for r in rows],
                   strategy="study_plan_lookup",
                   structured=[{"strategy": "study_plan_lookup", "id": r["plan_id"], "course_code": r["course_code"],
                                "program_or_section": program, "semester": sem,
                                "source_page": int(r["source_page"])} for r in rows])


# ---------------------------------------------------------------- policy / general
_WORD = re.compile(r"(?u)\b\w\w+\b")


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in _WORD.findall(text.lower())}


def answer_from_passages(question: str, candidates: list[dict], content_tokens: list[str], idf,
                         verify=None, field_re: str | None = None) -> dict:
    """Pick the passage (among the top-ranked candidates) that covers most of the question's
    IDF-weighted content words, then copy its best-matching sentence(s) verbatim."""
    if not candidates or candidates[0]["tfidf_similarity"] < MIN_SIMILARITY:
        top = candidates[0]["tfidf_similarity"] if candidates else 0.0
        return _result("insufficient_evidence", FALLBACK["low_relevance"], "insufficient_evidence",
                       notes=[f"Best raw TF-IDF cosine {top:.3f} < threshold {MIN_SIMILARITY}."],
                       suggestions=_suggest(candidates), strategy="tfidf_passage_retrieval")
    weights: dict[str, float] = {}
    content_tokens = [t for t in content_tokens if t not in QUERY_FILLER]
    for t in content_tokens:                      # stem -> highest idf among its surface forms
        weights[_stem(t)] = max(weights.get(_stem(t), 0.0), idf(t))
    total = sum(weights.values()) or 1.0
    scored_cands = []
    for rank, cand in enumerate(candidates):
        if cand["tfidf_similarity"] < MIN_SIMILARITY:
            continue
        heading = f"{cand['section']} {cand.get('subsection') or ''}"
        present = _stems(cand["text"] + " " + heading) & set(weights)
        coverage = sum(weights[x] for x in present) / total
        scored_cands.append((coverage, -rank, cand, present))
    eligible = [x for x in scored_cands if x[0] >= MIN_COVERAGE]
    coverage_log = {c["id"]: round(cov, 3) for cov, _, c, _ in scored_cands}
    if not eligible:
        best_cov = max((x[0] for x in scored_cands), default=0.0)
        return _result("insufficient_evidence", FALLBACK["low_relevance"], "insufficient_evidence",
                       notes=[f"No top passage covers enough of the question's key words "
                              f"(best IDF-weighted coverage {best_cov:.2f} < {MIN_COVERAGE}).",
                              f"Coverage by passage: {coverage_log}"],
                       suggestions=_suggest(candidates), strategy="tfidf_passage_retrieval")
    # Highest coverage wins (rounded so near-ties fall back to retrieval rank).
    coverage, _, cand, present = max(eligible, key=lambda x: (round(x[0], 1), x[1]))
    sents = _sentences(cand["text"])
    sent_scores = [(sum(weights[x] for x in _stems(s) & set(weights)), i) for i, s in enumerate(sents)]
    best_score, bi = max(sent_scores, key=lambda x: (x[0], -x[1]))
    chosen = [bi]
    if sents[bi].rstrip().endswith(":") or len(sents[bi]) < 90:   # complete a lead-in sentence
        chosen += [j for j in (bi + 1, bi + 2) if j < len(sents)][:1 if not sents[bi].rstrip().endswith(":") else 2]
    for sc, j in sorted(sent_scores, reverse=True):
        if j not in chosen and sc >= 0.75 * best_score > 0 and sum(len(sents[k]) for k in chosen) < 350:
            chosen.append(j)
            break
    if field_re:                                  # keep the sentence that states the requested field
        hit = next((j for j, x in enumerate(sents) if re.search(field_re, x, re.I)), None)
        if hit is not None and hit not in chosen:
            chosen = [bi, hit]
    excerpt = " ".join(sents[j] for j in sorted(set(chosen)))
    if len(excerpt) > 600:
        excerpt = excerpt[:600].rsplit(" ", 1)[0] + "…"
    where = cand["section"] + (f" › {cand['subsection']}" if cand.get("subsection") else "")
    ans = f"From the catalog ({where}, p. {cand['source_page']}): “{excerpt}”"
    notes = ["Extractive answer: sentence(s) copied verbatim from a retrieved passage; not a summary or ruling.",
             f"Evidence passage chosen by IDF-weighted query-word coverage {coverage:.2f} "
             f"(min {MIN_COVERAGE}); coverage by passage: {coverage_log}"]
    return _result("answered", ans, "excerpt_supported",
                   [_cite(cand["id"], cand["source_page"], cand["source_url"], excerpt, "passage_text", where)],
                   [cand["id"]],
                   [{"claim": "verbatim excerpt", "field": "passage_text", "source_id": cand["id"],
                     "source_page": cand["source_page"], "value": excerpt[:160],
                     "matched_query_words": sorted(present), "coverage": round(coverage, 3),
                     "verified_in_page_text": bool(verify(excerpt[:80], cand["source_page"])) if verify else None}],
                   notes, suggestions=_suggest(candidates, exclude=cand["id"]), strategy="tfidf_passage_retrieval")


def _suggest(candidates: list[dict], exclude: str | None = None, n: int = 3) -> list[dict]:
    """Related passages shown separately from citations (never used to justify an answer)."""
    return [{"source_id": c["id"], "source_page": c["source_page"], "source_url": c["source_url"],
             "section": c["section"], "tfidf_similarity": c["tfidf_similarity"]}
            for c in candidates if c["id"] != exclude][:n]


def fallback(reason: str, candidates: list[dict] | None = None, suggest: bool = False) -> dict:
    return _result("insufficient_evidence", FALLBACK[reason], reason,
                   suggestions=_suggest(candidates or []) if suggest else [], strategy="scope_rule")
