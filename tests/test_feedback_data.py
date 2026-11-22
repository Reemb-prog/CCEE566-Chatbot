"""Feedback/ABSA data preparation (docs/FEEDBACK_ABSA_PLAN.md, phase F1).

Rule tests always run. Data tests need inputs/feedback/processed/ (run scripts/prepare_feedback_data.py),
and are skipped when the git-ignored feedback data is not present on this machine.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "inputs" / "feedback" / "processed"
RAW = ROOT / "inputs" / "feedback" / "raw"

_spec = importlib.util.spec_from_file_location("prepare_feedback_data", ROOT / "scripts" / "prepare_feedback_data.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

needs_data = pytest.mark.skipif(not (PROC / "data_card.json").exists(),
                                reason="feedback data not prepared on this machine")


# ------------------------------------------------------------------ rules
@pytest.mark.parametrize("rtype,rating,label", [
    ("teacher review", "3", "neutral"),
    ("teacher review", "3.5", "positive"),
    ("teacher review", "4", "positive"),
    ("teacher review", "2.5", "negative"),
    ("university review", "2.8", "neutral"),
    ("university review", "3.2", "neutral"),
    ("university review", "3.3", "positive"),
    ("university review", "2.6", "negative"),
    ("course review", "1", "positive"),
    ("course review", "0", "negative"),
])
def test_rating_three_is_the_only_neutral(rtype, rating, label):
    assert prep.label_from_rating(rtype, rating) == label


@pytest.mark.parametrize("rtype,rating", [("course review", "0.5"), ("teacher review", "3.3"), ("x", "3")])
def test_unexpected_ratings_fail_loudly(rtype, rating):
    with pytest.raises(ValueError):
        prep.label_from_rating(rtype, rating)


def test_same_aspect_positive_and_negative_becomes_neutral_mixed():
    agg, mixed = prep.aggregate_pairs([("registration", "positive"), ("registration", "negative"),
                                       ("advising", "positive")])
    assert agg == {"registration": "neutral", "advising": "positive"}
    assert mixed == ["registration"]


def test_polar_label_wins_tie_with_neutral_but_not_majority():
    assert prep.aggregate_pairs([("courses", "positive"), ("courses", "neutral")])[0] == {"courses": "positive"}
    assert prep.aggregate_pairs([("courses", "negative"), ("courses", "neutral"),
                                 ("courses", "neutral")])[0] == {"courses": "neutral"}


def test_sentence_split_keeps_initials_and_titles():
    text = "I took this with J. Dean and Dr. Smith. It was hard! Would take again."
    sents = [text[s:e] for s, e in prep.sentence_spans(text)]
    assert sents == ["I took this with J. Dean and Dr. Smith.", "It was hard!", "Would take again."]


# ------------------------------------------------------------------ prepared data
@pytest.fixture(scope="module")
def data():
    read = lambda n: pd.read_csv(PROC / n, keep_default_na=False, dtype=str)
    return {
        "card": json.loads((PROC / "data_card.json").read_text(encoding="utf-8")),
        "review": read("review_sentiment.csv"),
        "sent": read("absa_sentences.csv"),
        "op": read("absa_opinions.csv"),
        "human": read("rhu_human_test.csv"),
    }


@needs_data
def test_human_test_is_sealed_and_complete(data):
    h, s = data["human"], data["sent"]
    assert len(h) == 40 and set(h.split) == {"rhu_human_test"}
    assert not set(s.split) & {"rhu_human_test"}
    keys = set(h.text.map(prep.text_key))
    assert not s.text.map(prep.text_key).isin(keys).any()
    assert not data["review"].text_clean.map(prep.text_key).isin(keys).any()


@needs_data
def test_no_training_sentence_is_a_near_copy_of_a_human_sentence(data):
    near, max_cos = prep.drop_near_copies(data["sent"].text, data["human"].text)
    assert not near.any(), f"max cosine {max_cos:.3f}"


@needs_data
def test_labels_use_the_closed_taxonomy(data):
    for df in (data["sent"], data["human"]):
        pairs = [p for ps in df.pairs for p in ps.split("|") if p]
        assert all(p.split(":")[0] in prep.ASPECTS and p.split(":")[1] in prep.SENTIMENTS for p in pairs)
    assert set(data["op"].aspect) <= set(prep.ASPECTS)
    assert set(data["op"].sentiment) <= set(prep.SENTIMENTS)
    assert set(data["review"].sentiment) <= set(prep.SENTIMENTS)


@needs_data
def test_groups_never_cross_splits(data):
    for df in (data["sent"], data["op"]):
        assert (df.groupby("group_id").split.nunique() == 1).all()
    r = data["review"]
    assert not set(r.loc[r.split == "train", "near_dup_group"]) & set(r.loc[r.split == "test", "near_dup_group"])


@needs_data
def test_opinion_rows_belong_to_existing_sentences_and_spans_are_inside(data):
    s, op = data["sent"].set_index("sentence_id"), data["op"]
    syn = op[op.source == "synthetic"]
    assert syn.sentence_id.isin(s.index).all()
    assert all(o in s.loc[sid, "text"] for o, sid in zip(syn.text, syn.sentence_id))


@needs_data
def test_review_sentiment_neutral_only_from_rating_three(data):
    r = data["review"]
    neu = r[r.sentiment == "neutral"]
    assert set(neu.type) <= {"teacher review", "university review"}
    assert set(neu.loc[neu.type == "teacher review", "rating"].astype(float)) == {3.0}
    uni = neu.loc[neu.type == "university review", "rating"].astype(float)
    assert uni.between(*prep.UNIVERSITY_NEUTRAL_BAND).all()


@needs_data
def test_review_test_split_is_the_teammates(data):
    test_ids = set(pd.read_csv(RAW / prep.FILES["S1-split"], dtype=str).id)
    assert set(data["review"].query("split == 'test'").id) == test_ids


@needs_data
def test_legacy_synthetic_and_survey_copies_are_not_used(data):
    s = data["sent"]
    assert set(s.source) == {"edurabsa", "synthetic"}
    assert s.loc[s.source == "synthetic", "sentence_id"].str.match(r"^SY[NP]\d").all()
    patch = s[s.sentence_id.str.startswith("SYP")]
    assert len(patch) and set(patch.split) == {"train"}  # post-F3 patch never enters an evaluation set


@needs_data
def test_data_card_matches_outputs(data):
    for name, meta in data["card"]["outputs"].items():
        path = PROC / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"], name
        assert len(pd.read_csv(path, keep_default_na=False)) == meta["rows"], name
    for code, meta in data["card"]["inputs"].items():
        assert hashlib.sha256((RAW / meta["file"]).read_bytes()).hexdigest() == meta["sha256"], code


@needs_data
def test_preparation_is_deterministic(data):
    before = {n: m["sha256"] for n, m in data["card"]["outputs"].items()}
    assert prep.main() == 0
    after = json.loads((PROC / "data_card.json").read_text(encoding="utf-8"))["outputs"]
    assert {n: m["sha256"] for n, m in after.items()} == before
