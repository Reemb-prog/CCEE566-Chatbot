"""Train the feedback sentiment / ABSA models (docs/FEEDBACK_ABSA_PLAN.md, phase F2).

Usage:
  python scripts/train_feedback_models.py                          # variant "full" -> models/feedback_v1.joblib
  python scripts/train_feedback_models.py --variant edurabsa_only  # ablation: no generated RHU sentences
  python scripts/train_feedback_models.py --variant labels_v1      # ablation: teammate's neutral mapping for M1

Reads only inputs/feedback/processed/ (run scripts/prepare_feedback_data.py first).
Uses ONLY split == "train" rows. The evaluation sets (rhu_human_test, edurabsa_test, S1 test) and synthetic_dev
are not used for fitting or for choosing anything. The single exception is a sanity check that refits the
teammate's exact Task A setup (their labels, their split) and compares with their reported test score. That
choice was already fixed by them, so it does not tune anything here.

No hyperparameter search: configurations are the ones the teammate selected by CV (asba_package notebook 02).
Scores are SVM / logistic-regression decision values. Margins are stored as margins, never as probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "inputs" / "feedback" / "processed"
MODELS = ROOT / "models"

SENTIMENTS = ["positive", "negative", "neutral"]
ASPECTS = ["courses", "instructors", "registration", "advising", "admissions", "tuition_fees",
           "student_services", "facilities", "library", "it_systems", "career_services", "transportation"]
MIN_ASPECT_TRAIN = 15          # aspects with fewer positive training sentences are "not covered"
N_FOLDS = 5
ABSTAIN_TARGET_PRECISION = 0.85  # D2: polar outputs that survive the margin rule should be this precise (CV)
ABSTAIN_MAX_RATE = 0.35          # ...but never abstain on more than this share of polar predictions
TEAMMATE_TASK_A_TEST_MACRO_F1 = 0.6187  # asba_package notebook 02, cell 11
SANITY_TOLERANCE = 0.03
SYNTHETIC_WEIGHT_CANDIDATES = (1.0, 3.0, 5.0)  # M3 only; chosen by train CV (added after the F3 review)
CLAUSE_SPLIT_PATTERN =r"\s*(?:;|,?\s+\b(?:but|however|although|though|whereas|while|yet)\b,?)\s+"

VARIANTS = {
    "full": {"absa_sources": {"edurabsa", "synthetic"}, "review_label": "sentiment", "out": "feedback_v1"},
    "edurabsa_only": {"absa_sources": {"edurabsa"}, "review_label": "sentiment", "out": "feedback_v1_edurabsa_only"},
    "labels_v1": {"absa_sources": {"edurabsa", "synthetic"}, "review_label": "sentiment_v1", "out": "feedback_v1_labels_v1"},
}


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=150000)


def m1_pipeline() -> Pipeline:
    """Review sentiment: teammate's best Task A config (word+char TF-IDF, LinearSVC C=0.1, balanced)."""
    return Pipeline([("f", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
                     ("m", LinearSVC(C=0.1, class_weight="balanced", random_state=0))])


def m2_word() -> Pipeline:
    """Aspect detector (one per aspect): teammate's selection, word TF-IDF + LR C=16, balanced."""
    return Pipeline([("f", word_tfidf()), ("m", LogisticRegression(C=16, class_weight="balanced", max_iter=3000))])


def m2_word_char() -> Pipeline:
    """Same classifier, plus character n-grams. Declared before looking at any sealed set. Rare RHU aspects have
    about 20 training sentences, each with its own words (wifi, CampusVue, helpdesk), which word features alone miss."""
    return Pipeline([("f", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
                     ("m", LogisticRegression(C=16, class_weight="balanced", max_iter=3000))])


# Exactly two candidates, chosen by mean CV F1 over all trained aspects (train split only).
M2_CANDIDATES = {"word (teammate)": m2_word, "word+char": m2_word_char}


def m3_pipeline() -> Pipeline:
    """Clause polarity, shared across aspects: teammate's opinion-phrase model, LR C=2, unigrams+bigrams, min_df=1."""
    return Pipeline([("f", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                     ("m", LogisticRegression(C=2, class_weight="balanced", max_iter=3000))])


def margins(scores: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 decision value per row."""
    s = np.sort(scores, axis=1)
    return s[:, -1] - s[:, -2]


def oof_scores(pipe: Pipeline, X: pd.Series, y: pd.Series, groups: pd.Series,
               weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list]:
    """Out-of-fold predictions and decision values with group-aware stratified folds."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=7)
    classes = sorted(y.unique())
    scores = np.zeros((len(y), len(classes)))
    for tr, te in cv.split(X, y, groups):
        fit_kw = {} if weights is None else {"m__sample_weight": weights[tr]}
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr], **fit_kw)
        assert list(m.classes_) == classes
        scores[te] = m.decision_function(X.iloc[te])
    pred = np.array(classes)[scores.argmax(1)]
    return pred, scores, classes


def choose_abstain_margin(y: np.ndarray, pred: np.ndarray, margin: np.ndarray) -> dict:
    """Smallest margin t such that polar predictions with margin >= t reach the target precision (D2).
    Polar predictions below t are reported as 'neutral (unclear)'. Chosen on out-of-fold train scores only."""
    polar = np.isin(pred, ["positive", "negative"])
    base_prec = float((pred[polar] == y[polar]).mean())
    candidates = np.unique(np.concatenate([[0.0], np.quantile(margin[polar], np.linspace(0, ABSTAIN_MAX_RATE, 71))]))
    chosen, reached = None, False
    for t in candidates:
        kept = polar & (margin >= t)
        if kept.sum() == 0:
            continue
        prec = (pred[kept] == y[kept]).mean()
        if prec >= ABSTAIN_TARGET_PRECISION:
            chosen, reached = float(t), True
            break
    if chosen is None:
        chosen = float(np.quantile(margin[polar], ABSTAIN_MAX_RATE))
    kept = polar & (margin >= chosen)
    abstained = polar & ~kept
    final = np.where(abstained, "neutral", pred)
    return {
        "margin_threshold": round(chosen, 4),
        "target_precision": ABSTAIN_TARGET_PRECISION,
        "target_reached": reached,
        "polar_precision_before": round(base_prec, 4),
        "polar_precision_after": round(float((pred[kept] == y[kept]).mean()), 4),
        "abstain_rate_of_polar_predictions": round(float(abstained.sum() / polar.sum()), 4),
        "oof_macro_f1_before": round(float(f1_score(y, pred, average="macro")), 4),
        "oof_macro_f1_after": round(float(f1_score(y, final, average="macro")), 4),
        "measured_on": "out-of-fold predictions, train split only",
    }


def per_class(y, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=SENTIMENTS, zero_division=0)
    return {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                "support": int(n[i])} for i, c in enumerate(SENTIMENTS)}


def train_m1(review: pd.DataFrame, label_col: str) -> tuple[Pipeline, dict]:
    tr = review[review.split == "train"].reset_index(drop=True)
    X, y, g = tr.text_clean, tr[label_col], tr.near_dup_group
    pred, scores, _ = oof_scores(m1_pipeline(), X, y, g)
    abstain = choose_abstain_margin(y.values, pred, margins(scores))
    model = m1_pipeline().fit(X, y)
    return model, {
        "task": "review-level sentiment (labels from star ratings: weak labels)",
        "label_column": label_col, "config": "word(1-2)+char_wb(2-5) TF-IDF, LinearSVC C=0.1, balanced",
        "train_rows": int(len(tr)), "train_label_counts": y.value_counts().to_dict(),
        "cv_macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "cv_accuracy": round(float(accuracy_score(y, pred)), 4),
        "cv_per_class": per_class(y, pred), "abstain": abstain,
    }


def train_m2(sent: pd.DataFrame, sources: set) -> tuple[dict, dict]:
    tr = sent[(sent.split == "train") & sent.source.isin(sources)].reset_index(drop=True)
    aspects = tr.aspects.fillna("").str.split("|")
    folds = list(GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr.group_id))
    labels = {a: aspects.map(lambda xs, a=a: a in xs).astype(int) for a in ASPECTS}
    trainable = [a for a in ASPECTS if labels[a].sum() >= MIN_ASPECT_TRAIN]

    cv_by_cand: dict[str, dict] = {}
    for cname, make in M2_CANDIDATES.items():
        cv_by_cand[cname] = {}
        for a in trainable:
            y = labels[a]
            oof = np.zeros(len(tr), dtype=int)
            for f_tr, f_te in folds:
                oof[f_te] = make().fit(tr.text.iloc[f_tr], y.iloc[f_tr]).predict(tr.text.iloc[f_te])
            p, r, f, _ = precision_recall_fscore_support(y, oof, average="binary", zero_division=0)
            cv_by_cand[cname][a] = (float(p), float(r), float(f))
    mean_f1 = {c: float(np.mean([v[2] for v in d.values()])) for c, d in cv_by_cand.items()}
    best = max(mean_f1, key=mean_f1.get)
    make = M2_CANDIDATES[best]

    models, report = {}, {}
    for a in ASPECTS:
        y = labels[a]
        by_src = tr[y == 1].source.value_counts().to_dict()
        n_pos = int(y.sum())
        if a not in trainable:
            report[a] = {"trained": False, "train_positive_sentences": n_pos, "by_source": by_src,
                         "note": f"not covered (< {MIN_ASPECT_TRAIN} training sentences)"}
            continue
        p, r, f = cv_by_cand[best][a]
        models[a] = make().fit(tr.text, y)
        report[a] = {"trained": True, "train_positive_sentences": n_pos, "by_source": by_src,
                     "trained_on_generated_only": set(by_src) == {"synthetic"},
                     "cv_precision": round(p, 4), "cv_recall": round(r, 4), "cv_f1": round(f, 4),
                     "cv_f1_by_candidate": {c: round(d[a][2], 4) for c, d in cv_by_cand.items()}}
    return models, {"task": "aspect detection, one binary model per aspect (sentence level)",
                    "config": f"{best} TF-IDF, LogisticRegression C=16, balanced",
                    "candidate_mean_cv_f1": {c: round(v, 4) for c, v in mean_f1.items()}, "selected": best,
                    "train_sentences": int(len(tr)), "train_by_source": tr.source.value_counts().to_dict(),
                    "per_aspect": report,
                    "cv_note": "GroupKFold by review/text on train only. For aspects trained only on generated "
                               "sentences, CV measures generated-to-generated transfer and is optimistic."}


def train_m3(op: pd.DataFrame, sources: set) -> tuple[Pipeline, dict]:
    tr = op[(op.split == "train") & op.source.isin(sources)].reset_index(drop=True)
    X, y, g = tr.text, tr.sentiment, tr.group_id
    srcs = sorted(tr.source.unique())
    # Weight of generated (RHU-style) phrases vs EduRABSA phrases. Chosen by the mean of the per-source CV macro-F1
    # (train split only), so short RHU wording is not drowned out by about 4,000 North-American review phrases.
    cand = {}
    for w in (SYNTHETIC_WEIGHT_CANDIDATES if len(srcs) > 1 else (1.0,)):
        wts = np.where(tr.source == "synthetic", w, 1.0)
        p, s, _ = oof_scores(m3_pipeline(), X, y, g, wts)
        by = {src: float(f1_score(y[tr.source == src], p[tr.source == src], average="macro")) for src in srcs}
        cand[w] = (float(np.mean(list(by.values()))), p, s, by)
    best_w = max(cand, key=lambda w: cand[w][0])
    _, pred, scores, by = cand[best_w]
    abstain = choose_abstain_margin(y.values, pred, margins(scores))
    weights = np.where(tr.source == "synthetic", best_w, 1.0)
    model = m3_pipeline().fit(X, y, m__sample_weight=weights)
    by_src = {s: round(v, 4) for s, v in by.items()}
    return model, {
        "task": "clause/opinion polarity, shared across aspects",
        "config": f"word(1-2) TF-IDF min_df=1, LogisticRegression C=2, balanced; generated-phrase weight {best_w}",
        "synthetic_weight_candidates_mean_source_f1": {str(w): round(v[0], 4) for w, v in cand.items()},
        "synthetic_weight": best_w,
        "train_opinions": int(len(tr)), "train_by_source": tr.source.value_counts().to_dict(),
        "cv_macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "cv_macro_f1_by_source": by_src, "cv_per_class": per_class(y, pred), "abstain": abstain,
    }


def sanity_teammate_task_a(review: pd.DataFrame) -> dict:
    tr, te = review[review.split == "train"], review[review.split == "test"]
    pred = m1_pipeline().fit(tr.text_clean, tr.sentiment_v1).predict(te.text_clean)
    f1 = float(f1_score(te.sentiment_v1, pred, average="macro"))
    return {"purpose": "reproduce teammate Task A (their labels + split) in this environment; not used for any choice",
            "test_macro_f1": round(f1, 4), "teammate_reported": TEAMMATE_TASK_A_TEST_MACRO_F1,
            "within_tolerance": abs(f1 - TEAMMATE_TASK_A_TEST_MACRO_F1) <= SANITY_TOLERANCE}


def build_vocabulary(review: pd.DataFrame, sent: pd.DataFrame, op: pd.DataFrame, sources: set) -> list[str]:
    """Lower-case words (5+ letters) seen at least twice in training text. The spelling step leaves these alone."""
    texts = pd.concat([review.loc[review.split == "train", "text_clean"],
                       sent.loc[(sent.split == "train") & sent.source.isin(sources), "text"]])
    counts = texts.str.lower().str.findall(r"\b[a-z]{5,}\b").explode().value_counts()
    return sorted(counts[counts >= 2].index)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args(argv)
    cfg = VARIANTS[args.variant]
    card_path = PROC / "data_card.json"
    if not card_path.exists():
        print("Run scripts/prepare_feedback_data.py first.")
        return 1
    read = lambda n: pd.read_csv(PROC / n, keep_default_na=False)
    review, sent, op = read("review_sentiment.csv"), read("absa_sentences.csv"), read("absa_opinions.csv")

    t0 = time.perf_counter()
    m1, r1 = train_m1(review, cfg["review_label"])
    print(f"M1 review sentiment  CV macro-F1 {r1['cv_macro_f1']:.3f}  abstain margin {r1['abstain']['margin_threshold']}"
          f"  ({time.perf_counter() - t0:.0f}s)")
    m2, r2 = train_m2(sent, cfg["absa_sources"])
    for a, r in r2["per_aspect"].items():
        print(f"M2 {a:<17} " + (f"CV F1 {r['cv_f1']:.3f}  n={r['train_positive_sentences']}"
                                 + ("  [generated only]" if r["trained_on_generated_only"] else "")
                                 if r["trained"] else r["note"]))
    m3, r3 = train_m3(op, cfg["absa_sources"])
    print(f"M3 clause polarity   CV macro-F1 {r3['cv_macro_f1']:.3f}  by source {r3['cv_macro_f1_by_source']}"
          f"  abstain margin {r3['abstain']['margin_threshold']}")
    sanity = None if args.skip_sanity or args.variant != "full" else sanity_teammate_task_a(review)
    if sanity:
        print(f"Sanity: teammate Task A refit -> test macro-F1 {sanity['test_macro_f1']:.4f} "
              f"(reported {TEAMMATE_TASK_A_TEST_MACRO_F1}) within tolerance: {sanity['within_tolerance']}")

    card = json.loads(card_path.read_text(encoding="utf-8"))
    meta = {
        "name": cfg["out"], "variant": args.variant, "sklearn_version": sklearn.__version__,
        "data_card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
        "data_outputs": {k: v["sha256"] for k, v in card["outputs"].items()},
        "absa_sources": sorted(cfg["absa_sources"]), "aspects": ASPECTS, "sentiments": SENTIMENTS,
        "clause_split_pattern": CLAUSE_SPLIT_PATTERN,
        "abstain_margins": {"review": r1["abstain"]["margin_threshold"], "clause": r3["abstain"]["margin_threshold"]},
        "vocabulary": build_vocabulary(review, sent, op, cfg["absa_sources"]),
        "generated_only_aspects": sorted(a for a, r in r2["per_aspect"].items()
                                         if r.get("trained_on_generated_only")),
        "score_semantics": "decision-function margins (top-1 minus top-2); NOT probabilities",
    }
    bundle = {"meta": meta, "review_model": m1, "aspect_models": m2, "clause_model": m3}
    metrics = {"meta": meta, "M1_review_sentiment": r1, "M2_aspect_detection": r2, "M3_clause_polarity": r3,
               "sanity_teammate_task_a": sanity,
               "not_measured_here": "evaluation sets (rhu_human_test, edurabsa_test, S1 test) are scored in F3"}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS / f"{cfg['out']}.joblib", compress=3)
    (MODELS / f"{cfg['out']}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote models/{cfg['out']}.joblib and models/{cfg['out']}_metrics.json "
          f"({time.perf_counter() - t0:.0f}s total)")
    return 0 if (sanity is None or sanity["within_tolerance"]) else 2


if __name__ == "__main__":
    sys.exit(main())
