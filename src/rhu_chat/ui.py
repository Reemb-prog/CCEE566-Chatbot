"""Small presentation helpers for app.py (no domain logic)."""
from __future__ import annotations

import html
import re

STATUS_BADGE = {
    "answered": ("Answered from catalog", "ok"),
    "insufficient_evidence": ("Not answerable from catalog", "warn"),
    "needs_clarification": ("Needs clarification", "info"),
    "invalid_input": ("Invalid input", "warn"),
    "feedback": ("Feedback detected, not a catalog question", "info"),
}
DECISION_LABEL = {
    "exact_course_match": "Exact course match",
    "conflicting_records": "Conflicting records shown",
    "excerpt_supported": "Verbatim excerpt",
    "study_plan_rows": "Study-plan rows",
    "code_mention_excerpt": "Code mention excerpt",
    "insufficient_evidence": "Insufficient evidence",
    "needs_clarification": "Needs clarification",
    "personal_record_request": "Personal data — declined",
    "outside_catalog_topic": "Outside catalog — declined",
    "live_or_time_sensitive": "Live data — declined",
    "unknown_course_code": "Unknown course code",
    "field_not_in_records": "Field not in records",
    "invalid_input": "Invalid input",
    "feedback_detected": "Feedback, routed to the Feedback analyser",
}
SENTIMENT_CLASS = {"positive": "pos", "negative": "neg", "neutral": "neu"}

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~$])")


def md_escape(text: object) -> str:
    """Neutralise Markdown/HTML in untrusted catalog or user text before st.markdown."""
    s = _MD_SPECIAL.sub(r"\\\1", str(text))
    s = re.sub(r"(?m)^(\s*)([-+=]|\d+\.)(\s)", r"\1\\\2\3", s)
    return s.replace("\n", "  \n")


def strip_html(text: object) -> str:
    return html.escape(str(text), quote=True)


CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 6rem; max-width: 1400px;}
.rhu-hero {background: linear-gradient(120deg, #14213d 0%, #1d3a5f 60%, #0f8b8d 100%);
  color: #fff; border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;}
.rhu-hero h1 {color:#fff; font-size: 1.7rem; margin: 0; padding: 0; font-weight: 700;}
.rhu-hero p {color:#d8eef0; margin: 4px 0 0 0; font-size: .95rem;}
.rhu-badge {display:inline-block; font-size:.75rem; font-weight:600; padding:2px 10px; border-radius:999px; margin-bottom:4px;}
.rhu-badge.ok {background:#d7f0ec; color:#0b5e57;}
.rhu-badge.warn {background:#fbe9d2; color:#8a4b08;}
.rhu-badge.info {background:#dde7f6; color:#1d3a5f;}
.rhu-strip {display:flex; gap:6px; flex-wrap:wrap; margin: 4px 0 10px 0;}
.rhu-stage {flex:1 1 30%; min-width:92px; background:#fff; border:1px solid #d5dde5; border-left:4px solid #0f8b8d;
  border-radius:8px; padding:6px 8px; font-size:.78rem; line-height:1.25;}
.rhu-stage.skip {border-left-color:#a7b1bc; opacity:.75;}
.rhu-stage.flag {border-left-color:#d98a1f;}
.rhu-stage b {display:block; color:#14213d; font-size:.8rem;}
.rhu-stage span {color:#5a6878;}
.rhu-cite {background:#fff; border:1px solid #d5dde5; border-radius:8px; padding:8px 10px; margin:6px 0; font-size:.88rem;}
.rhu-clauses {line-height:2.1; font-size:.95rem; margin:4px 0 8px 0;}
.rhu-clause {padding:3px 8px; border-radius:6px; margin-right:4px; border:1px solid transparent; color:#14213d;}
.rhu-clause.pos {background:#d7f0ec; border-color:#9fd6cc;}
.rhu-clause.neg {background:#f8dcdc; border-color:#e8a9a9;}
.rhu-clause.neu {background:#eceff3; border-color:#cfd6de;}
.rhu-clause.unclear {border-style:dashed; border-color:#8a96a3;}
.rhu-clause small {color:#5a6878; font-size:.72rem; margin-left:4px;}
.rhu-sent {display:inline-block; font-weight:700; padding:2px 12px; border-radius:999px; font-size:.95rem;}
.rhu-sent.pos {background:#d7f0ec; color:#0b5e57;}
.rhu-sent.neg {background:#f8dcdc; color:#8a1f1f;}
.rhu-sent.neu {background:#eceff3; color:#3b4652;}
</style>
"""


def sentiment_chip(sentiment: str, unclear: bool = False) -> str:
    label = f"{sentiment} (unclear)" if unclear else sentiment
    return f'<span class="rhu-sent {SENTIMENT_CLASS.get(sentiment, "neu")}">{strip_html(label)}</span>'


def clauses_html(analysis: dict) -> str:
    """Each clause shaded by its polarity. Clause text is user input, so it is always HTML-escaped."""
    parts = []
    for s in analysis.get("sentences", []):
        for c in s["clauses"]:
            cls = SENTIMENT_CLASS.get(c["sentiment"], "neu") + (" unclear" if c["unclear"] else "")
            tag = c["sentiment"] + (" ?" if c["unclear"] else "")
            if c["aspects"]:
                tag += " · " + ", ".join(c["aspects"])
            parts.append(f'<span class="rhu-clause {cls}">{strip_html(c["text"])}<small>{strip_html(tag)}</small></span>')
    return f'<div class="rhu-clauses">{" ".join(parts)}</div>'


def badge_html(status: str) -> str:
    label, cls = STATUS_BADGE.get(status, (status, "info"))
    return f'<span class="rhu-badge {cls}">{strip_html(label)}</span>'


def strip_html_stages(result: dict) -> str:
    """Six-stage process strip. Content is fixed labels + numbers/enums only, all HTML-escaped."""
    tr = result["trace"]
    t = tr["timings_ms"]
    tok = tr["tokenization"]
    cls = tr["classification"]
    ret = tr["retrieval"]
    gr = tr["grounding"]
    skipped = not tr["input"]["valid"]
    n_struct = len(ret.get("structured_matches", []))
    n_fix = len((tok.get("normalization") or {}).get("changes", []))
    stages = [
        ("1 · Input", "valid" if tr["input"]["valid"] else "invalid", t.get("input", 0), "flag" if skipped else ""),
        ("2 · Tokens", f"{len(tok['retrieval_tokens'])} tokens · {len(tok['course_codes'])} code(s)"
                       + (f" · {n_fix} rewrite(s)" if n_fix else ""),
         t.get("tokenization", 0), "skip" if skipped else ""),
        ("3 · V3.1 hints", str(cls.get("category_hint") or "—"), t.get("classification", 0),
         "skip" if skipped else ("flag" if cls.get("review_needed") else "")),
        ("4 · Retrieval", ("skipped: feedback" if ret.get("skip_reason", "").startswith("feedback") else
                           f"{len(ret['candidates'])} passages" + (f" + {n_struct} exact" if n_struct else "")),
         t.get("retrieval", 0), "skip" if skipped or ret.get("skipped") else ""),
        ("5 · Evidence", DECISION_LABEL.get(gr["decision"], gr["decision"]), t.get("grounding", 0),
         "flag" if gr["decision"] in {"conflicting_records"} or result["status"] != "answered" else ""),
        ("6 · Answer", f"{result['status']} · {len(result['citations'])} cite(s)", t.get("answer", 0), ""),
    ]
    cards = "".join(
        f'<div class="rhu-stage {c}"><b>{strip_html(name)}</b><span>{strip_html(val)}</span><br>'
        f'<span>{float(ms):.1f} ms</span></div>' for name, val, ms, c in stages)
    return f'<div class="rhu-strip">{cards}</div>'
