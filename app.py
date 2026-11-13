"""RHU Catalog Explorer — Streamlit UI. Domain logic lives in src/rhu_chat/."""
from __future__ import annotations

from pathlib import Path

import altair as alt  # ships with Streamlit
import pandas as pd
import streamlit as st

from src.rhu_chat.ui import CSS, DECISION_LABEL, badge_html, md_escape, strip_html_stages

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "How many credits is CCEE 566?",
    "What are the prerequisites for CCEE 566?",
    "What is my current GPA?",
]

st.set_page_config(page_title="RHU Catalog Explorer", page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)  # static CSS only; never source text


@st.cache_resource(show_spinner="Loading catalog index and V3.1 classifier…")
def get_pipeline():
    from src.rhu_chat.pipeline import RHUChatPipeline
    return RHUChatPipeline.from_project_root(ROOT)


try:
    pipe = get_pipeline()
except Exception as exc:  # polite message instead of a stack trace
    st.error("The catalog index could not be loaded. Run `python scripts/preflight.py` and "
             "`python scripts/build_index.py`, then restart the app.")
    st.caption(f"Technical detail: {type(exc).__name__}")
    st.stop()

ss = st.session_state
ss.setdefault("turns", [])
ss.setdefault("pending", None)

# ---------------------------------------------------------------- header
st.markdown(
    '<div class="rhu-hero"><h1>RHU Catalog Explorer</h1>'
    '<p>Undergraduate Catalog 2026–2027 · NLP capstone demo · answers come only from the catalog, with page citations</p></div>',
    unsafe_allow_html=True)

top_l, top_r = st.columns([0.65, 0.35], vertical_alignment="center")
with top_r:
    c1, c2 = st.columns(2)
    show_steps = c1.toggle("Show NLP steps", value=True)
    if c2.button("Clear chat", width="stretch"):
        ss.turns = []
        ss.pending = None
        st.rerun()
with top_l:
    ex_cols = st.columns(len(EXAMPLES))
    for col, q in zip(ex_cols, EXAMPLES):
        if col.button(q, width="stretch"):
            ss.pending = q

# ---------------------------------------------------------------- handle input
typed = st.chat_input("Ask about courses, prerequisites, study plans or catalog policies…", max_chars=500)
question = typed if typed is not None else ss.pending
ss.pending = None
if question is not None:
    with st.spinner("Searching the catalog…"):
        ss.turns.append(pipe.ask(question))

if show_steps:
    chat_col, insp_col = st.columns([0.63, 0.37], gap="large")
else:
    chat_col, insp_col = st.container(), None


def render_citations(result: dict, key: str) -> None:
    cits = result["citations"]
    if cits:
        with st.expander(f"Sources ({len(cits)})", expanded=result["trace"]["grounding"]["decision"] == "conflicting_records"):
            for c in cits:
                kind = {"passage_text": "verbatim passage", "course_record_description": "course record (description)",
                        "page_text": "extracted page text", "course_record_fields": "course record fields",
                        "study_plan_rows": "study-plan table rows"}.get(c.get("excerpt_kind"), c.get("excerpt_kind"))
                st.markdown(f"**p. {c['source_page']}** · {md_escape(c.get('section') or '')} · "
                            f"`{md_escape(c['source_id'])}` · {kind}  \n"
                            f"[Open catalog PDF at page {c['source_page']}]({c['source_url']})")
                st.caption(md_escape(c["excerpt"]))
    sugg = result.get("suggestions") or []
    if sugg and result["status"] != "answered":
        with st.expander("Possibly related catalog sections (not used as evidence)"):
            for s in sugg:
                st.markdown(f"- p. {s['source_page']} · {md_escape(s['section'])} · "
                            f"[open page]({s['source_url']})")


with chat_col:
    if not ss.turns:
        st.info("Ask a question about the RHU 2026–2027 Undergraduate Catalog, or tap an example above. "
                "This is a classical-NLP prototype (TF-IDF retrieval + exact lookups), not a live advisor: "
                "it cannot see personal records or current deadlines.")
    for i, r in enumerate(ss.turns):
        with st.chat_message("user"):
            st.markdown(md_escape(r["question"]) if r["question"].strip() else "_(empty message)_")
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(badge_html(r["status"]), unsafe_allow_html=True)
            st.markdown(md_escape(r["answer"]))
            render_citations(r, key=f"t{i}")

# ---------------------------------------------------------------- inspector
if insp_col is not None:
    with insp_col.container(height=760, border=False):   # scrolls independently of the chat
        st.subheader("How it works")
        if not ss.turns:
            st.caption("The six NLP stages of each question will appear here: input check, tokenization and "
                       "course-code extraction, V3.1 classifier hints, TF-IDF retrieval, evidence check and answer.")
        else:
            labels = [f"Turn {i + 1}: {t['question'][:42] or '(empty)'}" for i, t in enumerate(ss.turns)]
            sel = st.selectbox("Inspect turn", range(len(labels)), index=len(labels) - 1,
                               format_func=lambda i: labels[i], label_visibility="collapsed")
            r = ss.turns[sel]
            tr = r["trace"]
            st.markdown(strip_html_stages(r), unsafe_allow_html=True)
            st.caption(f"Total measured time: {tr['timings_ms']['total']:.1f} ms (time.perf_counter, this run)")
            tabs = st.tabs(["Input", "Tokens", "Classifier", "Retrieval", "Evidence", "Answer"])

            with tabs[0]:
                inp = tr["input"]
                st.markdown(f"**Question:** {md_escape(inp['question']) or '_(empty)_'}")
                st.markdown(f"Length: {inp['length']} chars · Valid: **{inp['valid']}**"
                            + (f" · Problem: `{inp['problem']}`" if inp.get("problem") else ""))
                st.caption(f"{tr['timings_ms']['input']:.2f} ms")

            with tabs[1]:
                tok = tr["tokenization"]
                if tok.get("skipped"):
                    st.caption("Skipped: invalid input.")
                else:
                    st.markdown("**Retrieval tokens** (scikit-learn TF-IDF analyzer, stop words removed — "
                                "word features, not transformer tokens):")
                    st.code(" · ".join(tok["retrieval_tokens"]) or "(no usable tokens)", language=None)
                    st.markdown("**Raw word tokens** (before stop-word removal):")
                    st.code(" · ".join(tok["raw_tokens"]) or "(none)", language=None)
                    ents = {
                        "Course codes (from original question)": ", ".join(tok["course_codes"]) or "—",
                        "Unrecognised code-like strings": ", ".join(tok["unknown_codes"]) or "—",
                        "Requested course field": tok["requested_field"] or "—",
                        "Program clue": tok["program"] or "—",
                        "Semester clues": ", ".join(f"{k}={v}" for k, v in tok["semester_clues"].items()) or "—",
                        "Scope rules fired": ", ".join(tok["scope_flags"]) or "—",
                        "Course-title match": (f"{tok['title_match']['title']} → {', '.join(tok['title_match']['codes'])} "
                                               f"(ratio {tok['title_match']['similarity_ratio']})")
                        if tok.get("title_match") else "—",
                    }
                    st.dataframe(pd.DataFrame(ents.items(), columns=["Extraction", "Value"]),
                                 hide_index=True, width="stretch")
                    st.caption(f"{tr['timings_ms']['tokenization']:.2f} ms")

            with tabs[2]:
                c = tr["classification"]
                st.caption("Separately trained V3.1 LinearSVC models. **Routing hints only** — used as a small "
                           "soft boost, never as a filter or as an answer. SVM margins are not probabilities.")
                rows = [("Category (raw)", c.get("raw_category")), ("Intent (raw)", c.get("raw_intent")),
                        ("Category (reconciled)", c.get("category_hint")), ("Intent (reconciled)", c.get("intent_hint")),
                        ("Routing status", c.get("routing_status")), ("Review needed", c.get("review_needed"))]
                st.dataframe(pd.DataFrame([(k, "—" if v is None else str(v)) for k, v in rows],
                                          columns=["Field", "Value"]), hide_index=True, width="stretch")
                if c.get("error"):
                    st.warning("Classifier unavailable; retrieval ran without hints.")
                st.caption(f"{tr['timings_ms']['classification']:.2f} ms")

            with tabs[3]:
                ret = tr["retrieval"]
                if ret.get("skipped"):
                    st.caption("Skipped: invalid input.")
                else:
                    if ret.get("structured_matches"):
                        st.markdown("**Exact structured lookup** (code / plan match — no cosine score):")
                        sm = pd.DataFrame(ret["structured_matches"])
                        keep = [k for k in ["strategy", "id", "course_code", "program_or_section", "semester",
                                            "source_page", "credits", "prerequisite", "needs_manual_review"] if k in sm]
                        st.dataframe(sm[keep], hide_index=True, width="stretch")
                    cands = ret["candidates"]
                    if cands:
                        st.markdown("**Top TF-IDF passages** searched over all 2,502 passages (chart: top 5; cosine in teal, boost in orange):")
                        rows = []
                        for i, x in enumerate(cands[:5]):
                            label = f"#{i + 1} {x['id']} (p.{x['source_page']})"
                            rows.append({"passage": label, "rank": i, "part": "1 TF-IDF cosine", "score": x["tfidf_similarity"]})
                            rows.append({"passage": label, "rank": i, "part": "2 boost", "score": x["boost"]})
                        chart = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                            y=alt.Y("passage:N", sort=alt.EncodingSortField("rank"), title=None,
                                    axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                            x=alt.X("score:Q", stack="zero", title="relative ranking score (not confidence)"),
                            color=alt.Color("part:N", title=None, legend=alt.Legend(orient="bottom"),
                                            scale=alt.Scale(range=["#0f8b8d", "#d98a1f"])),
                            order=alt.Order("part:N"),
                            tooltip=["passage", "part", alt.Tooltip("score:Q", format=".4f")],
                        ).properties(height=alt.Step(30))
                        st.altair_chart(chart, width="stretch")
                        st.dataframe(pd.DataFrame([{"rank": i + 1, "id": x["id"], "page": x["source_page"],
                                                    "category": x["category"], "cosine": x["tfidf_similarity"],
                                                    "boost": x["boost"], "rank_score": x["rank_score"],
                                                    "boost_reasons": ", ".join(x["boost_reasons"]) or "—"}
                                                   for i, x in enumerate(cands)]),
                                     hide_index=True, width="stretch")
                        st.caption(md_escape(ret["scoring"]))
                        for x in cands:
                            with st.expander(f"{x['id']} · p. {x['source_page']} · {x['section'][:40]}"):
                                st.caption(md_escape(x["excerpt"]))
                    else:
                        st.caption("No passage shares a TF-IDF term with the question.")
                    st.caption(f"Index: {ret['index']['n_passages']} passages × {ret['index']['n_features']} features "
                               f"(cache {ret['index']['cache']}) · {tr['timings_ms']['retrieval']:.2f} ms")

            with tabs[4]:
                g = tr["grounding"]
                st.markdown(f"**Decision:** `{g['decision']}` — {DECISION_LABEL.get(g['decision'], '')}")
                if g["support"]:
                    st.markdown("**Claims and supporting source fields:**")
                    st.dataframe(pd.DataFrame([{"claim": s.get("claim"), "source": s.get("source_id"),
                                                "page": s.get("source_page"), "field": s.get("field"),
                                                "in page text": s.get("verified_in_page_text")}
                                               for s in g["support"]]),
                                 hide_index=True, width="stretch")
                    st.caption("“in page text” = the value was found in the extracted text of that PDF page "
                               "(or the next page, for entries split across a page break).")
                for n in g["notes"]:
                    if g["decision"] == "conflicting_records":
                        st.warning(md_escape(n))
                    else:
                        st.caption(md_escape(n))
                if "thresholds" in g:
                    st.caption(f"Thresholds: min cosine {g['thresholds']['min_tfidf_similarity']}, "
                               f"min IDF-weighted coverage {g['thresholds']['min_idf_weighted_coverage']} · "
                               f"{tr['timings_ms']['grounding']:.2f} ms")

            with tabs[5]:
                a = tr.get("answer", {})
                st.markdown(f"Status: **{r['status']}** · Citations: {len(r['citations'])} · "
                            f"Type: `{a.get('answer_type') or 'none'}`")
                st.caption("Answers are extractive: structured catalog fields or verbatim sentences. "
                           "No text is generated by a language model.")
                st.json({k: v for k, v in tr["timings_ms"].items()}, expanded=False)
