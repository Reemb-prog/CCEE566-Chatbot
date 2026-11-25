"""RHU Catalog Explorer — Streamlit UI. Domain logic lives in src/rhu_chat/."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt  # ships with Streamlit
import pandas as pd
import streamlit as st

from src.rhu_chat.ui import (CSS, DECISION_LABEL, badge_html, clauses_html, md_escape, sentiment_chip,
                             strip_html_stages)

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "How many credits is CCEE 566?",
    "What are the prerequisites for CCEE 566?",
    "What is my current GPA?",
]
# Written for the demo; none is one of the 40 survey sentences.
FEEDBACK_EXAMPLES = [
    "Registration crashed again this morning, but my advisor helped me fix my schedule.",
    "The library is quiet and the staff are friendly.",
    "Tuition keeps going up and the bus is always late.",
]
CHAT_VIEW, FEEDBACK_VIEW = "Catalog chat", "Feedback analyser"

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
ss.setdefault("view", CHAT_VIEW)
ss.setdefault("fb_text", "")
ss.setdefault("fb_result", None)


def open_in_feedback(text: str) -> None:
    """Callback: copy a chat message into the Feedback analyser and switch view."""
    ss.view, ss.fb_text = FEEDBACK_VIEW, text
    ss.fb_result = pipe.feedback.analyze(text) if pipe.feedback else None


def set_feedback_example(text: str) -> None:
    ss.fb_text = text
    ss.fb_result = pipe.feedback.analyze(text) if pipe.feedback else None


@st.cache_data
def load_json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- header
st.markdown(
    '<div class="rhu-hero"><h1>RHU Catalog Explorer</h1>'
    '<p>Undergraduate Catalog 2026–2027 · NLP capstone demo · answers come only from the catalog, with page citations</p></div>',
    unsafe_allow_html=True)

st.segmented_control("View", [CHAT_VIEW, FEEDBACK_VIEW], key="view", label_visibility="collapsed", required=True)


def render_model_card() -> None:
    m = load_json("models/feedback_v1_metrics.json")
    if not m:
        st.caption("Model metrics file not found.")
        return
    st.markdown("**Sources.** Real public reviews (EduRABSA, human-annotated: courses, instructors, some facilities/"
                "services) + generated RHU sentences (Claude-written, for aspects EduRABSA lacks). Review-level "
                "sentiment is trained on 6,420 public reviews with star-rating labels (rating 3 = neutral).")
    rows = []
    for a, r in m["M2_aspect_detection"]["per_aspect"].items():
        src = r.get("by_source", {})
        rows.append({"aspect": a, "train sentences": r["train_positive_sentences"],
                     "real (EduRABSA)": src.get("edurabsa", 0), "generated": src.get("synthetic", 0),
                     "generated only": "yes" if r.get("trained_on_generated_only") else "",
                     "CV F1 (train)": r.get("cv_f1")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("CV F1 = 5-fold cross-validation on training data. For generated-only aspects it measures "
               "generated-to-generated transfer.")
    ev, sealed = load_json("eval/feedback_results.json"), load_json("eval/feedback_results_sealed_v1.json")
    lines = []
    if sealed:
        h = sealed["C_absa"]["full"]["rhu_human_test"]
        lines.append(f"- 40 RHU survey sentences, **first sealed run**: aspect+sentiment pair F1 {h['pair_f1']} "
                     f"(95% bootstrap {h['pair_f1_95ci_bootstrap']}); without generated data "
                     f"{sealed['C_absa']['edurabsa_only']['rhu_human_test']['pair_f1']}.")
    if ev:
        h = ev["C_absa"]["full"]["rhu_human_test"]
        lines.append(f"- Same 40 sentences after fixes made from their errors: pair F1 {h['pair_f1']}. "
                     "This is a development number, not an independent test.")
        e = ev["C_absa"]["full"]["edurabsa_test"]
        lines.append(f"- EduRABSA test sentences (real, North-American reviews): pair F1 {e['pair_f1']}, "
                     f"aspect F1 {e['aspect_f1']}.")
    st.markdown("**Results**\n" + "\n".join(lines))
    st.markdown("**Limits.** 40 human sentences from 11 respondents show behaviour, not reliable accuracy. "
                "Most RHU aspects come from about 20 generated sentences each. Star ratings are weak labels. "
                "Scores in this view are model score margins, never probabilities.")


def render_feedback() -> None:
    st.caption("A separate module from catalog QA. It reads a student opinion and finds which aspects it is "
               "about and whether each is positive, negative or neutral. Nothing here is catalog evidence.")
    if pipe.feedback is None:
        st.warning("The feedback models are not available. Run `python scripts/prepare_feedback_data.py` and "
                   "`python scripts/train_feedback_models.py`, then restart the app.")
        st.caption(f"Technical detail: {md_escape(pipe.feedback_error or 'unknown')}")
        return
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        ex = st.columns(len(FEEDBACK_EXAMPLES))
        for i, (col, text) in enumerate(zip(ex, FEEDBACK_EXAMPLES)):
            col.button(text, key=f"fbex{i}", width="stretch", on_click=set_feedback_example, args=(text,))
        with st.form("fb_form", border=False):
            st.text_area("Student feedback", key="fb_text", height=110, max_chars=2000,
                         placeholder="e.g. Registration was stressful but the advisors were helpful.")
            go = st.form_submit_button("Analyse feedback", type="primary")
        if go:
            ss.fb_result = pipe.feedback.analyze(ss.fb_text) if ss.fb_text.strip() else None
        res = ss.fb_result
        if not res:
            st.info("Type or pick a piece of feedback and press **Analyse feedback**.")
            return
        if res.get("is_question"):
            if any(v["sentiment"] != "neutral" for v in res["aspects"].values()):
                st.caption("Phrased as a question, analysed here as a complaint or opinion.")
            else:
                st.warning("This looks like a question. Catalog questions belong in the Catalog chat.")
        ov = res["overall"]
        st.markdown("**Overall sentiment** " + sentiment_chip(ov["sentiment"], ov["unclear"]), unsafe_allow_html=True)
        if ov.get("source") in ("aspects", "clauses"):
            st.caption(("Taken from the aspect results" if ov["source"] == "aspects" else "Taken from the clause sentiment (no aspect found)") + (" (positive and negative aspects, so mixed)" if ov.get("mixed") else "")
                       + f". The whole-text review model said {ov['review_model_sentiment']} (score margin {ov['margin']}); "
                       "it was trained on long public reviews, so the aspect results are preferred.")
        else:
            st.caption(f"Review model says {ov['model_label']} with score margin {ov['margin']} "
                       f"(threshold {res['thresholds']['review_margin']}; below it a positive/negative call is "
                       "shown as neutral/unclear). Margins are not probabilities.")
        st.markdown("**Clauses**")
        st.markdown(clauses_html(res), unsafe_allow_html=True)
        if res["aspects"]:
            st.markdown("**Aspects found**")
            st.dataframe(pd.DataFrame([{
                "aspect": v["label"], "sentiment": v["sentiment"],
                "notes": ", ".join(n for n, on in (("mixed (positive and negative)", v["mixed"]),
                                                    ("unclear: low margin", v["unclear"]),
                                                    ("trained on generated examples", v["trained_on_generated_only"]))
                                   if on) or "—"} for v in res["aspects"].values()]),
                hide_index=True, width="stretch")
        else:
            st.caption("No aspect detector fired. The text may be about something outside the 12 aspects, "
                       "or worded in a way the models have not seen.")
        if res["spelling_changes"]:
            st.caption("Spelling read as: " + ", ".join(f"{md_escape(c['from'])} → {md_escape(c['to'])}"
                                                         for c in res["spelling_changes"]))
    with right.container(height=700, border=False):
        st.subheader("How it works")
        res = ss.fb_result
        if res:
            tabs = st.tabs(["1 Clean", "2 Aspects", "3 Polarity", "4 Overall", "Model card"])
            with tabs[0]:
                st.markdown("Unicode and spacing cleanup, then misspelled **aspect words only** are corrected "
                            "(e.g. Tution → Tuition). Other wording is kept.")
                st.code(res["original_text"], language=None)
                if res["text"] != res["original_text"]:
                    st.markdown("Analysed as:")
                    st.code(res["text"], language=None)
                st.caption(f"{len(res['sentences'])} sentence(s); clauses split at but / however / although / "
                           f"though / while / yet / ; · {res['timings_ms']['split']:.1f} ms")
            with tabs[1]:
                st.markdown("One logistic-regression detector per aspect (word + character TF-IDF). A decision "
                            "value above 0 means the aspect is detected. It is not a probability.")
                for i, s in enumerate(res["sentences"]):
                    df = pd.DataFrame(sorted(s["detector_scores"].items(), key=lambda kv: -kv[1]),
                                      columns=["aspect", "decision value"])
                    df["detected"] = df["decision value"].map(lambda v: "yes" if v > 0 else "")
                    st.caption(f"Sentence {i + 1}")
                    st.dataframe(df, hide_index=True, width="stretch", height=250)
                st.caption("Each detected aspect is attached to the clause where its own detector scores highest. "
                           f"{res['timings_ms']['aspects_and_clauses']:.1f} ms (aspects + polarity)")
            with tabs[2]:
                st.markdown("One shared polarity model (logistic regression on opinion phrases) labels each clause. "
                            f"A positive/negative call with margin below **{res['thresholds']['clause_margin']}** "
                            "becomes neutral/unclear (chosen on training data).")
                st.dataframe(pd.DataFrame([{"clause": c["text"], "model": c["model_label"], "margin": c["margin"],
                                            "shown as": c["sentiment"] + (" (unclear)" if c["unclear"] else ""),
                                            "cue rule": c.get("rule") or "—",
                                            "aspects": ", ".join(c["aspects"]) or "—"}
                                           for s in res["sentences"] for c in s["clauses"]]),
                             hide_index=True, width="stretch")
                st.caption("If the model says neutral but the clause has an unambiguous complaint/praise word "
                           "(e.g. crashed, overpriced, not helpful), that word decides (cue rule). Positive and "
                           "negative for the same aspect in one message → neutral (mixed).")
                with st.expander("Top cue words the polarity model learned"):
                    terms = pipe.feedback.top_polarity_terms(10)
                    st.dataframe(pd.DataFrame({c: t for c, t in terms.items()}), hide_index=True, width="stretch")
                    st.caption("Highest logistic-regression weights per class (one-vs-rest). Weights, not "
                               "probabilities; a word that makes no sense here points to a shortcut in the training data.")
            with tabs[3]:
                ov = res["overall"]
                st.markdown("Review-level LinearSVC (word + character TF-IDF), trained on 6,420 public reviews "
                            "labelled from star ratings (rating 3 = neutral).")
                st.dataframe(pd.DataFrame([("model label", ov["model_label"]), ("score margin", ov["margin"]),
                                           ("threshold", res["thresholds"]["review_margin"]),
                                           ("shown as", ov["sentiment"] + (" (unclear)" if ov["unclear"] else ""))],
                                          columns=["field", "value"]).astype(str), hide_index=True, width="stretch")
                st.json(res["timings_ms"], expanded=False)
            with tabs[4]:
                render_model_card()
        else:
            st.caption("The steps will appear here: cleanup and spelling, aspect detectors, clause polarity, "
                       "overall sentiment, plus a model card.")
            with st.expander("Model card"):
                render_model_card()


if ss.view == FEEDBACK_VIEW:
    render_feedback()
    st.stop()

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
            routed = r.get("routed_to") == "feedback"
            st.markdown(badge_html("feedback" if routed else r["status"]), unsafe_allow_html=True)
            st.markdown(md_escape(r["answer"]))
            if routed or r.get("feedback_offer"):
                if not routed:
                    st.caption("This also sounds like a complaint. The Feedback analyser can break it down.")
                st.button("Open in Feedback analyser", key=f"fb{i}", on_click=open_in_feedback,
                          args=(r["question"],))
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
                fr = tr.get("feedback_route")
                if fr:
                    st.markdown("**Feedback check:** " + ("feedback → Feedback analyser" if fr["is_feedback"]
                                                          else "catalog question"))
                    st.caption(md_escape("; ".join(fr["reasons"])) + " · a wording cue is required; the model "
                               f"view is supporting only · {tr['timings_ms'].get('feedback_check', 0):.2f} ms")
                st.caption(f"{tr['timings_ms']['input']:.2f} ms")

            with tabs[1]:
                tok = tr["tokenization"]
                if tok.get("skipped"):
                    st.caption("Skipped: invalid input.")
                else:
                    norm = tok.get("normalization") or {}
                    st.markdown("**Normalized question** (student shorthand, course acronyms from catalog "
                                "titles, spelling checked against catalog words; course codes untouched):")
                    st.code(norm.get("normalized") or inp["question"], language=None)
                    if norm.get("changes"):
                        st.dataframe(pd.DataFrame(
                            [(c["from"], c["to"], c["rule"], c.get("detail", "")) for c in norm["changes"]],
                            columns=["Typed", "Read as", "Rule", "Why"]), hide_index=True, width="stretch")
                    else:
                        st.caption("No rewrites were needed.")
                    for amb in norm.get("ambiguous_acronyms", []):
                        st.caption(f"Ambiguous acronym {amb['acronym']}: "
                                   + "; ".join(o["title"] for o in amb["options"][:6]) + " (not expanded).")
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
                    st.caption(f"Skipped: {md_escape(ret.get('skip_reason', 'invalid input'))}.")
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
