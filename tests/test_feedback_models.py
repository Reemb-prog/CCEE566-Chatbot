"""Feedback model bundle (docs/FEEDBACK_ABSA_PLAN.md, phase F2).

Skipped when models/feedback_v1.joblib has not been trained on this machine
(run scripts/prepare_feedback_data.py then scripts/train_feedback_models.py).
The example sentences below are written for the tests. None comes from the human test set.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "feedback_v1.joblib"
METRICS = ROOT / "models" / "feedback_v1_metrics.json"
CARD = ROOT / "inputs" / "feedback" / "processed" / "data_card.json"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="feedback model not trained on this machine")


@pytest.fixture(scope="module")
def bundle():
    import joblib
    return joblib.load(MODEL)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_bundle_built_with_installed_sklearn_and_current_data(bundle):
    meta = bundle["meta"]
    assert meta["sklearn_version"] == sklearn.__version__
    assert meta["data_card_sha256"] == hashlib.sha256(CARD.read_bytes()).hexdigest()
    assert "NOT probabilities" in meta["score_semantics"]


def test_every_aspect_is_either_trained_or_reported_uncovered(bundle, metrics):
    per = metrics["M2_aspect_detection"]["per_aspect"]
    assert set(per) == set(bundle["meta"]["aspects"])
    for a, r in per.items():
        assert r["trained"] == (a in bundle["aspect_models"])


def test_generated_only_aspects_are_flagged(metrics):
    per = metrics["M2_aspect_detection"]["per_aspect"]
    assert per["registration"]["trained_on_generated_only"] is True
    assert per["courses"]["trained_on_generated_only"] is False


def test_abstain_margins_are_positive_and_chosen_on_train(bundle, metrics):
    for key in ("review", "clause"):
        assert bundle["meta"]["abstain_margins"][key] >= 0
    for m in ("M1_review_sentiment", "M3_clause_polarity"):
        assert metrics[m]["abstain"]["measured_on"].endswith("train split only")


def test_teammate_task_a_reproduces(metrics):
    assert metrics["sanity_teammate_task_a"]["within_tolerance"] is True


def test_models_output_known_labels(bundle):
    texts = ["The professor explained everything clearly and was always helpful.",
             "This course was a complete waste of time."]
    assert set(bundle["review_model"].predict(texts)) <= {"positive", "negative", "neutral"}
    assert set(bundle["clause_model"].predict(texts)) <= {"positive", "negative", "neutral"}
    for m in bundle["aspect_models"].values():
        assert set(m.predict(texts)) <= {0, 1}


def test_obvious_examples(bundle):
    praise = "The professor explained everything clearly and was always helpful."
    waste = "This course was a complete waste of time."
    assert bundle["aspect_models"]["instructors"].predict([praise])[0] == 1
    assert bundle["clause_model"].predict([praise])[0] == "positive"
    assert bundle["clause_model"].predict([waste])[0] == "negative"
    assert bundle["review_model"].predict([waste])[0] == "negative"


def test_scores_are_margins_not_probabilities(bundle):
    s = bundle["review_model"].decision_function(["Great class, would take again."])
    assert s.shape == (1, 3) and not np.isclose(s.sum(), 1.0)


# ------------------------------------------------------------------ analyzer (src/rhu_chat/feedback.py)
from src.rhu_chat.feedback import FeedbackAnalyzer, correct_spelling, is_question  # noqa: E402


def test_spelling_fixes_aspect_words_only():
    vocab = {"process", "needs", "changed", "studied", "honestly"}
    text, changes = correct_spelling("Registeration process needs to be changed, honestly", vocab)
    assert text.startswith("Registration process")
    assert changes == [{"from": "Registeration", "to": "Registration"}]
    same, none = correct_spelling("The professor studied hard", vocab)
    assert same == "The professor studied hard" and none == []


def test_question_flag():
    assert is_question("Where do I go to switch my major")
    assert is_question("the bus leaves at five?")
    assert not is_question("Signing up for classes went fine")


def test_analyze_trace_shape(bundle):
    out = FeedbackAnalyzer(bundle).analyze("Tution is too high, but the advisor was really helpful.")
    assert out["spelling_changes"] == [{"from": "Tution", "to": "Tuition"}]
    assert out["is_question"] is False
    assert out["overall"]["sentiment"] in {"positive", "negative", "neutral"}
    assert {"tuition_fees", "advising"} <= set(out["aspects"])
    assert len(out["sentences"][0]["clauses"]) == 2
    assert all(isinstance(v, float) for v in out["timings_ms"].values())


def test_html_in_feedback_is_kept_as_plain_text(bundle):
    out = FeedbackAnalyzer(bundle).analyze("<script>alert(1)</script> registration was bad")
    assert "<script>" in out["text"]  # stored verbatim as data; the UI must escape it (F5)


def test_top_polarity_terms_are_per_class_and_sensible(bundle):
    terms = FeedbackAnalyzer(bundle).top_polarity_terms(10)
    assert set(terms) == {"positive", "negative", "neutral"}
    assert all(len(v) == 10 for v in terms.values())
    assert {"great", "good"} & set(terms["positive"]) and {"worst", "useless"} & set(terms["negative"])
