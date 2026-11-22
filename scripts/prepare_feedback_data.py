"""Prepare feedback-sentiment / ABSA data (docs/FEEDBACK_ABSA_PLAN.md, phase F1).

Usage:  python scripts/prepare_feedback_data.py
Reads only inputs/feedback/raw/ and writes inputs/feedback/processed/:
  review_sentiment.csv  review-level sentiment (S1) with the rating-3 = neutral mapping (decision D1)
  absa_sentences.csv    sentence-level aspect/sentiment pairs from EduRABSA (S2) + generated RHU set (S3)
  absa_opinions.csv     one row per opinion phrase, for the shared clause-polarity model
  rhu_human_test.csv    the 40 human survey sentences (S4): never training data (unsealed 2026-11-21, dev set)
  data_card.json        input/output hashes, counts per step, decisions, warnings
Deterministic: no randomness, no network. Texts are data, never executed or rendered.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# One implementation shared with the app, so training data is cut exactly the way inference cuts text.
from src.rhu_chat.feedback import aggregate_pairs, clean_text, sentence_spans  # noqa: E402
RAW = ROOT / "inputs" / "feedback" / "raw"
OUT = ROOT / "inputs" / "feedback" / "processed"

SENTIMENTS = ("positive", "negative", "neutral")
ASPECTS = ("courses", "instructors", "registration", "advising", "admissions", "tuition_fees",
           "student_services", "facilities", "library", "it_systems", "career_services", "transportation")

# D1: only a rating of 3 is neutral. Teacher scale has half steps; university ratings are averages.
TEACHER_NEUTRAL = {3.0}
UNIVERSITY_NEUTRAL_BAND = (2.75, 3.25)  # inclusive; holds 2.8, 3.0, 3.2 in the current data

EDU_CATEGORY_MAP = {
    "COURSE": "courses",
    "STAFF": "instructors",
    "UNIVERSITY#CAMPUS & FACILITIES": "facilities",
    "UNIVERSITY#INFORMATION & SERVICES": "student_services",
    "UNIVERSITY#SOCIAL ENGAGEMENT & ACTIVITIES": "student_services",
    "UNIVERSITY#COST": "tuition_fees",
}
SYNTHETIC_DEV_SHARE = 0.20
LEAK_COSINE = 0.80
MIN_WORDS_NO_ASPECT = 2

FILES = {
    "S1": "combined_ASBA_clean.csv",
    "S1-split": "predictions_taskA_test.csv",
    "S2": "edurabsa_clean.jsonl",
    "S3": "RHU_ABSA_Annotations_Consolidated_Clean.csv",
    # Written by Claude on 2026-11-21 AFTER seeing the F3 errors on the 40 human sentences (team chose to unseal them).
    "S3-patch": "RHU_synthetic_patch_f3.csv",
    "S4": "human_final.csv",
}
_QUESTION_START = re.compile(r"^\s*(where|what|how|when|who|which|why|can|could|is|are|do|does|should)\b", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_key(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def label_from_rating(review_type: str, rating: str) -> str:
    r = float(rating)
    if review_type == "course review":
        if r not in (0.0, 1.0):
            raise ValueError(f"course rating must be 0/1, got {rating!r}")
        return "positive" if r == 1.0 else "negative"
    if review_type == "teacher review":
        if r * 2 != int(r * 2) or not 1 <= r <= 5:
            raise ValueError(f"teacher rating must be 1-5 in half steps, got {rating!r}")
        return "neutral" if r in TEACHER_NEUTRAL else ("negative" if r < 3 else "positive")
    if review_type == "university review":
        lo, hi = UNIVERSITY_NEUTRAL_BAND
        return "neutral" if lo <= r <= hi else ("negative" if r < lo else "positive")
    raise ValueError(f"unknown review type {review_type!r}")


def fmt_pairs(d: dict[str, str]) -> str:
    return "|".join(f"{a}:{d[a]}" for a in sorted(d))


def map_edu_category(cat: str) -> str | None:
    return EDU_CATEGORY_MAP.get(cat) or EDU_CATEGORY_MAP.get(cat.split("#", 1)[0])


# ---------------------------------------------------------------- S1 review sentiment
def build_review_sentiment(log: list, warn: list) -> pd.DataFrame:
    d = pd.read_csv(RAW / FILES["S1"], keep_default_na=False, dtype={"rating": str})
    test_ids = set(pd.read_csv(RAW / FILES["S1-split"], dtype={"id": str}).id)
    assert d.id.is_unique, "S1 ids must be unique"
    missing = test_ids - set(d.id)
    assert not missing, f"{len(missing)} teammate test ids not in S1"
    d["sentiment_v1"] = d.sentiment
    d["sentiment"] = [label_from_rating(t, r) for t, r in zip(d.type, d.rating)]
    d["split"] = np.where(d.id.isin(test_ids), "test", "train")
    grp_cross = set(d.loc[d.split == "train", "near_dup_group"]) & set(d.loc[d.split == "test", "near_dup_group"])
    if grp_cross:
        warn.append(f"S1: {len(grp_cross)} near-duplicate groups cross train/test in the teammate split")
    changed = (d.sentiment != d.sentiment_v1)
    log.append(("S1 rows", len(d)))
    log.append(("S1 labels changed by D1 (rating 3 = neutral)", int(changed.sum())))
    cols = ["id", "type", "source", "rating", "text_clean", "sentiment_v1", "sentiment", "n_words",
            "near_dup_group", "split"]
    return d[cols]


# ---------------------------------------------------------------- S2 EduRABSA -> sentences
def build_edurabsa(log: list, warn: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = [json.loads(line) for line in (RAW / FILES["S2"]).read_text(encoding="utf-8").splitlines() if line]
    sent_rows, op_rows = [], []
    crossing = mapping_mismatch = dropped_short = 0
    for r in reviews:
        text = r["text"]
        split = "edurabsa_test" if r["split"] == "test" else "train"  # official validation joins train (CV inside train)
        spans = sentence_spans(text)
        per_sent: dict[int, list] = {i: [] for i in range(len(spans))}
        for k, o in enumerate(r["opinions"]):
            aspect = map_edu_category(o["edu_category"])
            if aspect != o["aspect"]:
                mapping_mismatch += 1
            if aspect is None:
                continue
            idx = next((i for i, (s, e) in enumerate(spans) if s <= o["span_start"] < e), None)
            if idx is None:  # span starts in inter-sentence whitespace
                idx = min(range(len(spans)), key=lambda i: abs(spans[i][0] - o["span_start"]))
            if o["span_end"] > spans[idx][1]:
                crossing += 1
            per_sent[idx].append((aspect, o["sentiment"]))
            op_rows.append({
                "opinion_id": f"{r['id']}-O{k:02d}", "sentence_id": f"{r['id']}-S{idx:02d}", "group_id": r["id"],
                "text": clean_text(o["opinion_text"]), "aspect": aspect, "sentiment": o["sentiment"],
                "implicit": bool(o["implicit"]), "source": "edurabsa", "is_synthetic": False, "split": split,
            })
        for i, (s, e) in enumerate(spans):
            sent = clean_text(text[s:e])
            agg, mixed = aggregate_pairs(per_sent[i])
            if not agg and len(sent.split()) < MIN_WORDS_NO_ASPECT:
                dropped_short += 1
                continue
            sent_rows.append({
                "sentence_id": f"{r['id']}-S{i:02d}", "group_id": r["id"], "text": sent,
                "aspects": "|".join(sorted(agg)), "pairs": fmt_pairs(agg), "mixed_aspects": "|".join(sorted(mixed)),
                "n_opinions": len(per_sent[i]), "source": "edurabsa", "is_synthetic": False,
                "persona": r["persona"], "split": split,
            })
    log.append(("S2 reviews", len(reviews)))
    log.append(("S2 sentences kept", len(sent_rows)))
    log.append(("S2 sentences dropped (no opinion, < 2 words)", dropped_short))
    log.append(("S2 opinions", len(op_rows)))
    if crossing:
        warn.append(f"S2: {crossing} opinion spans run past their sentence end (assigned to the sentence where they start)")
    if mapping_mismatch:
        warn.append(f"S2: {mapping_mismatch} opinions where our category mapping differs from the teammate's 'aspect'")
    return pd.DataFrame(sent_rows), pd.DataFrame(op_rows)


# ---------------------------------------------------------------- S3 generated RHU set
def build_synthetic(log: list, warn: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pd.read_csv(RAW / FILES["S3"])
    log.append(("S3 rows (all provenance)", len(c)))
    log.append(("S3 synthetic_legacy rows excluded (templated, diagnosis only)", int((c.provenance == "synthetic_legacy").sum())))
    log.append(("S3 human_survey rows excluded (S4 is the source of truth)", int((c.provenance == "human_survey").sum())))
    n = c[c.provenance == "synthetic_new"].copy()
    patch = pd.read_csv(RAW / FILES["S3-patch"])
    patch = patch.assign(feedback_text_clean=patch.feedback_text.map(clean_text), source_type="synthetic_patch_f3",
                         provenance="synthetic_patch_f3")
    log.append(("S3-patch rows (generated after the F3 human-test review; always train)", len(patch)))
    n = pd.concat([n, patch], ignore_index=True)
    bad = set(n.original_aspect) - set(ASPECTS)
    assert not bad, f"unknown synthetic aspects {bad}"
    assert set(n.sentiment) <= set(SENTIMENTS)
    sent_rows, op_rows = [], []
    for fid, g in n.groupby("feedback_id", sort=True):
        agg, mixed = aggregate_pairs(list(zip(g.original_aspect, g.sentiment)))
        sent_rows.append({
            "sentence_id": fid, "group_id": fid, "text": clean_text(g.feedback_text_clean.iloc[0]),
            "aspects": "|".join(sorted(agg)), "pairs": fmt_pairs(agg), "mixed_aspects": "|".join(sorted(mixed)),
            "n_opinions": len(g), "source": "synthetic", "is_synthetic": True,
            "persona": g.source_type.iloc[0], "split": None,
        })
        for k, row in enumerate(g.itertuples()):
            op_rows.append({
                "opinion_id": f"{fid}-O{k:02d}", "sentence_id": fid, "group_id": fid,
                "text": clean_text(row.evidence_span), "aspect": row.original_aspect, "sentiment": row.sentiment,
                "implicit": False, "source": "synthetic", "is_synthetic": True, "split": None,
            })
    sents = pd.DataFrame(sent_rows)
    # synthetic_dev: ~20% per stratum (primary pair), chosen by hash of the id -> deterministic, no RNG
    sents["stratum"] = sents.pairs.str.split("|").str[0]
    sents["_h"] = sents.sentence_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    dev_ids = set()
    for _, g in sents[sents.persona != "synthetic_patch_f3"].groupby("stratum"):  # dev set identical to v1
        k = int(round(SYNTHETIC_DEV_SHARE * len(g)))
        dev_ids |= set(g.sort_values("_h").sentence_id.iloc[:k])
    sents["split"] = np.where(sents.sentence_id.isin(dev_ids) & (sents.persona != "synthetic_patch_f3"),
                              "synthetic_dev", "train")
    sents = sents.drop(columns=["stratum", "_h"])
    ops = pd.DataFrame(op_rows)
    ops["split"] = ops.sentence_id.map(dict(zip(sents.sentence_id, sents.split)))
    log.append(("S3 synthetic_new sentences", len(sents)))
    log.append(("S3 synthetic_dev sentences", int((sents.split == "synthetic_dev").sum())))
    return sents, ops


# ---------------------------------------------------------------- S4 human test
def build_human_test(log: list) -> pd.DataFrame:
    h = pd.read_csv(RAW / FILES["S4"], dtype=str, keep_default_na=False)
    rows = []
    for r in h.itertuples():
        raw_pairs = [tuple(p.split(":", 1)) for p in r.gold_pairs.split("|") if p]
        assert all(a in ASPECTS and s in SENTIMENTS for a, s in raw_pairs), f"bad gold pair in {r.sentence_id}"
        agg, mixed = aggregate_pairs(raw_pairs)
        text = clean_text(r.sentence_text)
        rows.append({
            "sentence_id": r.sentence_id, "writer_id": r.writer_id, "text": text,
            "gold_pairs_raw": r.gold_pairs, "aspects": "|".join(sorted(agg)), "pairs": fmt_pairs(agg),
            "mixed_aspects": "|".join(sorted(mixed)), "is_question": bool(_QUESTION_START.match(text) or text.endswith("?")),
            "label_revised_by_claude": r.label_note.startswith("CLAUDE"), "label_note": r.label_note,
            "source": "human_survey", "is_synthetic": False, "split": "rhu_human_test",
        })
    df = pd.DataFrame(rows)
    log.append(("S4 human sentences (sealed test)", len(df)))
    log.append(("S4 labels revised by Claude (need 2-annotator check, F1b)", int(df.label_revised_by_claude.sum())))
    return df


# ---------------------------------------------------------------- leakage guard
def drop_near_copies(train_texts: pd.Series, sealed_texts: pd.Series) -> tuple[np.ndarray, float]:
    """Boolean mask of training rows that are near copies (char 3-5-gram cosine >= LEAK_COSINE) of a sealed text."""
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(
        pd.concat([train_texts, sealed_texts]))
    S = tv.transform(sealed_texts) @ tv.transform(train_texts).T
    mx = S.max(axis=0).toarray().ravel()
    return mx >= LEAK_COSINE, float(mx.max()) if len(mx) else 0.0


def main() -> int:
    missing = [f for f in FILES.values() if not (RAW / f).exists()]
    if missing:
        print("Missing raw inputs in inputs/feedback/raw/:", ", ".join(missing))
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    log: list[tuple[str, int]] = []
    warn: list[str] = []

    review = build_review_sentiment(log, warn)
    edu_s, edu_o = build_edurabsa(log, warn)
    syn_s, syn_o = build_synthetic(log, warn)
    human = build_human_test(log)

    sentences = pd.concat([edu_s, syn_s], ignore_index=True)
    opinions = pd.concat([edu_o, syn_o], ignore_index=True)

    # Exact and near copies of a human test sentence are removed from every non-sealed set.
    human_keys = set(human.text.map(text_key))
    exact = sentences.text.map(text_key).isin(human_keys)
    near, max_cos = drop_near_copies(sentences.text, human.text)
    leak = exact.values | near
    if leak.any():
        warn.append(f"removed {int(leak.sum())} sentences that copy a human test sentence: "
                    f"{sentences.sentence_id[leak].tolist()}")
        opinions = opinions[~opinions.sentence_id.isin(set(sentences.sentence_id[leak]))]
        sentences = sentences[~leak]
    log.append(("sentences removed as copies of human test sentences", int(leak.sum())))
    review_leak = review.text_clean.map(text_key).isin(human_keys)
    if review_leak.any():
        warn.append(f"S1: removed {int(review_leak.sum())} reviews identical to a human test sentence")
        review = review[~review_leak]

    # EduRABSA test reviews must not sit in the review-sentiment training split (teammate guaranteed this; re-check).
    edu_test_keys = set(text_key(t) for t in pd.read_json(RAW / FILES["S2"], lines=True).query("split == 'test'").text)
    bad = review[(review.split == "train") & review.text_clean.map(text_key).isin(edu_test_keys)]
    if len(bad):
        warn.append(f"S1: {len(bad)} EduRABSA test reviews were in S1 train; moved to test")
        review.loc[bad.index, "split"] = "test"

    outputs = {
        "review_sentiment.csv": review,
        "absa_sentences.csv": sentences,
        "absa_opinions.csv": opinions,
        "rhu_human_test.csv": human,
    }
    for name, df in outputs.items():
        df.to_csv(OUT / name, index=False, encoding="utf-8", lineterminator="\n")

    def counts(df, *cols):
        return {" / ".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v)
                for k, v in df.groupby(list(cols)).size().items()}

    pair_series = sentences.assign(p=sentences.pairs.str.split("|")).explode("p")
    pair_series = pair_series[pair_series.p.fillna("") != ""]
    pair_series = pair_series.assign(aspect=pair_series.p.str.split(":").str[0])
    card = {
        "generated_by": "scripts/prepare_feedback_data.py",
        "plan": "docs/FEEDBACK_ABSA_PLAN.md",
        "decisions": {
            "D1_teacher_neutral_ratings": sorted(TEACHER_NEUTRAL),
            "D1_university_neutral_band": list(UNIVERSITY_NEUTRAL_BAND),
            "D2_same_aspect_pos_and_neg": "neutral (mixed)",
            "D3_synthetic_used_for": "RHU aspects absent from EduRABSA; synthetic_legacy excluded",
            "D4_sealed_sets": ["rhu_human_test", "edurabsa_test", "S1 test (review sentiment)"],
            "aspects": list(ASPECTS),
            "synthetic_dev_share": SYNTHETIC_DEV_SHARE,
            "leak_cosine_threshold": LEAK_COSINE,
        },
        "inputs": {code: {"file": f, "sha256": sha256(RAW / f)} for code, f in FILES.items()},
        "outputs": {name: {"rows": int(len(df)), "sha256": sha256(OUT / name)} for name, df in outputs.items()},
        "steps": [{"step": s, "rows": int(n)} for s, n in log],
        "review_sentiment": {
            "by_split_and_label": counts(review, "split", "sentiment"),
            "v1_vs_d1": counts(review, "sentiment_v1", "sentiment"),
        },
        "absa_sentences": {
            "by_split_and_source": counts(sentences, "split", "source"),
            "aspect_sentences_by_source": counts(pair_series, "aspect", "source"),
            "no_aspect_sentences": int((sentences.aspects.fillna("") == "").sum()),
        },
        "absa_opinions": {"by_source_and_sentiment": counts(opinions, "source", "sentiment")},
        "rhu_human_test": {
            "aspect_pairs": dict(Counter(p for ps in human.pairs for p in ps.split("|") if p)),
            "questions_not_feedback": human.loc[human.is_question, "sentence_id"].tolist(),
            "no_aspect": human.loc[human.aspects == "", "sentence_id"].tolist(),
        },
        "max_cosine_train_to_human_test": round(max_cos, 3),
        "warnings": warn,
    }
    (OUT / "data_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

    for s, n in log:
        print(f"{n:>7}  {s}")
    print(f"max char-cosine of any training sentence to a human test sentence: {max_cos:.3f}")
    for w in warn:
        print("WARNING:", w)
    print("wrote", ", ".join(outputs), "and data_card.json to", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
