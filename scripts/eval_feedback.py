"""Score the feedback models on the held-out sets + ablations (docs/FEEDBACK_ABSA_PLAN.md, phase F3).

Usage:  python scripts/eval_feedback.py
Needs models/feedback_v1*.joblib (scripts/train_feedback_models.py --variant full|edurabsa_only|labels_v1).
Writes eval/feedback_results.md and eval/feedback_results.json.

Nothing here feeds back into training: thresholds and configurations were fixed in F2 on training data.
The markdown reproduces the 40 anonymous survey sentences (needed for the error list), but no text from the
public review datasets (those name real instructors).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rhu_chat.feedback import FeedbackAnalyzer  # noqa: E402

PROC = ROOT / "inputs" / "feedback" / "processed"
RAW = ROOT / "inputs" / "feedback" / "raw"
MODELS = ROOT / "models"
EVAL = ROOT / "eval"
PRIVATE = EVAL / "private"  # git-ignored: per-sentence survey responses
SENT = ["positive", "negative", "neutral"]
EDU_ASPECTS = {"courses", "instructors", "facilities", "student_services", "tuition_fees"}
BOOT = 2000
SEED = 7


def parse_pairs(s: str) -> dict[str, str]:
    return dict(p.split(":", 1) for p in str(s).split("|") if p)


def set_scores(gold: list[dict], pred: list[dict]) -> dict:
    """Micro P/R/F1 over (aspect, sentiment) pairs and over aspects alone, plus exact-set match."""
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    gp = [set(g.items()) for g in gold]
    pp = [set(p.items()) for p in pred]
    tp = sum(len(g & p) for g, p in zip(gp, pp)); fp = sum(len(p - g) for g, p in zip(gp, pp))
    fn = sum(len(g - p) for g, p in zip(gp, pp))
    ga = [set(g) for g in gold]; pa = [set(p) for p in pred]
    atp = sum(len(g & p) for g, p in zip(ga, pa)); afp = sum(len(p - g) for g, p in zip(ga, pa))
    afn = sum(len(g - p) for g, p in zip(ga, pa))
    pair, asp = prf(tp, fp, fn), prf(atp, afp, afn)
    found = [(g[a], p[a]) for g, p in zip(gold, pred) for a in set(g) & set(p)]
    return {"n_sentences": len(gold), "gold_pairs": tp + fn, "pred_pairs": tp + fp,
            "pair_tp": tp, "pair_precision": round(pair[0], 3), "pair_recall": round(pair[1], 3), "pair_f1": round(pair[2], 3),
            "aspect_tp": atp, "aspect_precision": round(asp[0], 3), "aspect_recall": round(asp[1], 3), "aspect_f1": round(asp[2], 3),
            "polarity_acc_on_found_aspects": round(float(np.mean([g == p for g, p in found])), 3) if found else None,
            "n_found_aspects": len(found),
            "exact_set_match": round(float(np.mean([g == p for g, p in zip(gp, pp)])), 3)}


def per_aspect(gold: list[dict], pred: list[dict]) -> dict:
    out = {}
    for a in sorted({a for g in gold for a in g} | {a for p in pred for a in p}):
        tp = sum(a in g and a in p for g, p in zip(gold, pred)); fp = sum(a in p and a not in g for g, p in zip(gold, pred))
        fn = sum(a in g and a not in p for g, p in zip(gold, pred))
        ptp = sum(a in g and g.get(a) == p.get(a) for g, p in zip(gold, pred))
        out[a] = {"gold": tp + fn, "pred": tp + fp, "detected": tp, "detected_and_polarity_right": ptp}
    return out


def boot_pair_f1(gold, pred) -> list[float]:
    rng = np.random.RandomState(SEED); n = len(gold); vals = []
    for _ in range(BOOT):
        ix = rng.randint(0, n, n)
        vals.append(set_scores([gold[i] for i in ix], [pred[i] for i in ix])["pair_f1"])
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def review_scores(y, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=SENT, zero_division=0)
    return {"n": int(len(y)), "macro_f1": round(float(f1_score(y, pred, labels=SENT, average="macro", zero_division=0)), 3),
            "accuracy": round(float(np.mean(np.asarray(y) == np.asarray(pred))), 3),
            "per_class": {c: {"p": round(float(p[i]), 3), "r": round(float(r[i]), 3), "f1": round(float(f[i]), 3),
                              "support": int(n[i])} for i, c in enumerate(SENT)},
            "confusion_rows_gold_cols_pred": confusion_matrix(y, pred, labels=SENT).tolist(),
            "top_confusions": top_confusions(y, pred)}


def top_confusions(y, pred, k: int = 2) -> list[dict]:
    """The k largest off-diagonal cells of the row-normalised confusion matrix (gold -> predicted)."""
    cm = confusion_matrix(y, pred, labels=SENT, normalize="true")
    np.fill_diagonal(cm, 0)
    idx = np.argsort(cm.ravel())[::-1][:k]
    return [{"gold": SENT[i], "pred": SENT[j], "share_of_gold": round(float(cm[i, j]), 3)}
            for i, j in zip(*np.unravel_index(idx, cm.shape)) if cm[i, j] > 0]


SHORTCUT_RE = r"(?i)\b(?:[1-5](?:\.\d)?\s*(?:/\s*5|out of 5|stars?)|rating:?\s*[1-5])\b"


def baselines(review: pd.DataFrame, label_col: str = "sentiment") -> dict:
    """Lab-3 style reference rows for the review task: two dummy classifiers and a tuned Naive Bayes.
    Fit and tuned (alpha by group-aware 5-fold CV, macro-F1) on the train split only; scored once on S1 test.
    These are comparisons only; the shipped M1 model is unchanged."""
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.pipeline import Pipeline
    tr, te = review[review.split == "train"], review[review.split == "test"]
    out = {}
    for strat in ("most_frequent", "uniform"):
        d = DummyClassifier(strategy=strat, random_state=SEED).fit(tr.text_clean, tr[label_col])
        out[f"dummy_{strat}"] = review_scores(te[label_col], d.predict(te.text_clean))
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
              .split(tr.text_clean, tr[label_col], tr.near_dup_group))
    for name, nb in (("multinomial_nb", MultinomialNB()), ("complement_nb", ComplementNB())):
        pipe = Pipeline([("f", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
                         ("m", nb)])
        g = GridSearchCV(pipe, {"m__alpha": [0.01, 0.1, 0.5, 1.0, 10.0]}, scoring="f1_macro", cv=cv, n_jobs=-1)
        g.fit(tr.text_clean, tr[label_col])
        r = review_scores(te[label_col], g.predict(te.text_clean))
        r.update(best_alpha=g.best_params_["m__alpha"], cv_macro_f1=round(float(g.best_score_), 3))
        out[name] = r
    out["shortcut_check"] = {"pattern": SHORTCUT_RE,
                             "train_texts_matching": int(tr.text_clean.str.contains(SHORTCUT_RE).sum()),
                             "train_rows": int(len(tr))}
    return out


def review_predict(an: FeedbackAnalyzer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    m = an.review_model
    scores = m.decision_function(texts)
    s = np.sort(scores, axis=1)
    raw = np.asarray(m.classes_)[scores.argmax(1)]
    margin = s[:, -1] - s[:, -2]
    final = np.where((raw != "neutral") & (margin < an.review_margin), "neutral", raw)
    return raw, final


def main() -> int:
    variants = {v: MODELS / f"feedback_v1{'' if v == 'full' else '_' + v}.joblib"
                for v in ("full", "edurabsa_only", "labels_v1")}
    missing = [str(p.relative_to(ROOT)) for p in variants.values() if not p.exists()]
    if missing:
        print("Train these first (scripts/train_feedback_models.py --variant ...):", ", ".join(missing))
        return 1
    an = {v: FeedbackAnalyzer.load(p) for v, p in variants.items()}
    read = lambda n: pd.read_csv(PROC / n, keep_default_na=False)
    review, sent, op, human = read("review_sentiment.csv"), read("absa_sentences.csv"), read("absa_opinions.csv"), read("rhu_human_test.csv")
    R: dict = {"models": {v: a.meta["data_card_sha256"][:12] for v, a in an.items()}}

    # ---- A. review sentiment on the S1 test split (each model on its own label definition)
    te = review[review.split == "test"]
    R["A_review_test"] = {}
    for v, col in (("full", "sentiment"), ("labels_v1", "sentiment_v1")):
        raw, final = review_predict(an[v], te.text_clean.tolist())
        R["A_review_test"][v] = {"labels": col, "model_only": review_scores(te[col], raw),
                                 "with_unclear_rule": review_scores(te[col], final),
                                 "unclear_rate": round(float(np.mean(raw != final)), 3)}

    # ---- B. common ruler: annotated EduRABSA test reviews, gold = dominant opinion polarity (teammate's "second ruler")
    edu = pd.read_json(RAW / "edurabsa_clean.jsonl", lines=True).query("split == 'test'")
    gold = [Counter(o["sentiment"] for o in ops).most_common(1)[0][0] for ops in edu.opinions]
    R["B_review_vs_annotated_polarity"] = {}
    for v in ("full", "labels_v1"):
        raw, final = review_predict(an[v], edu.text.tolist())
        R["B_review_vs_annotated_polarity"][v] = {"model_only": review_scores(gold, raw),
                                                  "with_unclear_rule": review_scores(gold, final)}

    # ---- C. sentence-level ABSA, end to end
    sets = {"rhu_human_test": human, "synthetic_dev": sent[sent.split == "synthetic_dev"],
            "edurabsa_test": sent[sent.split == "edurabsa_test"]}
    R["C_absa"] = {}
    preds_store = {}
    for v in ("full", "edurabsa_only"):
        R["C_absa"][v] = {}
        for name, df in sets.items():
            g = [parse_pairs(s) for s in df.pairs]
            outs = [an[v].analyze(t) for t in df.text]
            p = [{a: x["sentiment"] for a, x in o["aspects"].items()} for o in outs]
            res = set_scores(g, p)
            res["per_aspect"] = per_aspect(g, p)
            res["unclear_pairs"] = sum(x["unclear"] for o in outs for x in o["aspects"].values())
            if name == "edurabsa_test":
                res["pred_aspects_outside_edurabsa_scheme"] = sum(a not in EDU_ASPECTS for d in p for a in d)
            if name == "rhu_human_test":
                res["pair_f1_95ci_bootstrap"] = boot_pair_f1(g, p)
                preds_store[v] = outs
            R["C_absa"][v][name] = res

    # ---- D. clause polarity given the gold opinion phrase (upper bound for the polarity step)
    R["D_polarity_gold_phrase"] = {}
    for v in ("full", "edurabsa_only"):
        R["D_polarity_gold_phrase"][v] = {}
        for name in ("edurabsa_test", "synthetic_dev"):
            o = op[op.split == name]
            pr = an[v].clause_model.predict(o.text.tolist())
            R["D_polarity_gold_phrase"][v][name] = review_scores(o.sentiment, pr)

    # ---- F. baselines (dummy + tuned Naive Bayes) on the S1 review task, D1 labels
    R["F_baselines"] = baselines(review)

    EVAL.mkdir(exist_ok=True)
    (EVAL / "feedback_results.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    write_markdown(R, human, preds_store)
    c = R["C_absa"]
    print(f"A  S1 test review macro-F1 (D1 labels): model {R['A_review_test']['full']['model_only']['macro_f1']} / "
          f"with unclear rule {R['A_review_test']['full']['with_unclear_rule']['macro_f1']}")
    print(f"B  annotated-polarity ruler macro-F1: D1 {R['B_review_vs_annotated_polarity']['full']['with_unclear_rule']['macro_f1']}"
          f" vs teammate mapping {R['B_review_vs_annotated_polarity']['labels_v1']['with_unclear_rule']['macro_f1']}")
    for name in sets:
        print(f"C  {name:<15} pair F1 full {c['full'][name]['pair_f1']}  edurabsa_only {c['edurabsa_only'][name]['pair_f1']}"
              f"  | aspect F1 full {c['full'][name]['aspect_f1']}  edurabsa_only {c['edurabsa_only'][name]['aspect_f1']}")
    print(f"   human pair F1 95% CI {c['full']['rhu_human_test']['pair_f1_95ci_bootstrap']}")
    print("wrote eval/feedback_results.md and eval/feedback_results.json")
    return 0


def fmt_pairs(d: dict) -> str:
    return ", ".join(f"{a}:{s}" for a, s in sorted(d.items())) or "(none)"


def write_markdown(R: dict, human: pd.DataFrame, preds: dict) -> None:
    A, B, C, D = R["A_review_test"], R["B_review_vs_annotated_polarity"], R["C_absa"], R["D_polarity_gold_phrase"]
    L = ["# Feedback sentiment / ABSA: evaluation (phase F3)", "",
         "Generated by `scripts/eval_feedback.py`. Thresholds and model choices are set on training data only. "
         "Scores are agreement with the stated gold labels, not calibrated probabilities. Macro-F1 averages "
         "positive, negative and neutral equally.", "",
         "> **The 40 human sentences are no longer a sealed test.** On 2026-11-21 the team chose to fix the errors "
         "found on them: spelling correction toward aspect words, 83 generated patch sentences "
         "(`RHU_synthetic_patch_f3.csv`), and a generated-phrase weight in the polarity model (chosen by train CV). "
         "The human-test numbers below are therefore *development* numbers. The last independent result is in "
         "`eval/feedback_results_sealed_v1.md`. EduRABSA test, S1 test and synthetic_dev were not used for these "
         "fixes and remain held out.", ""]
    sealed = EVAL / "feedback_results_sealed_v1.json"
    if sealed.exists():
        S = json.loads(sealed.read_text(encoding="utf-8"))["C_absa"]["full"]
        L += ["## Before / after the F3 fixes (full model)", "",
              "| Set | sealed v1 pair F1 | now pair F1 | sealed v1 aspect F1 | now aspect F1 | correct pairs v1 → now |",
              "|---|---|---|---|---|---|"]
        for name in ("rhu_human_test", "synthetic_dev", "edurabsa_test"):
            a, b = S[name], C["full"][name]
            L.append(f"| {name}{' (unsealed)' if name == 'rhu_human_test' else ''} | {a['pair_f1']} | {b['pair_f1']} | "
                     f"{a['aspect_f1']} | {b['aspect_f1']} | {a['pair_tp']} → {b['pair_tp']} |")
        L += ["", "The human-test gain is within the bootstrap interval, so treat it as noise-level. It is a check "
              "that the fixes do what they were meant to, not evidence of better accuracy.", ""]
    L += [
         "## A. Review-level sentiment, S1 test split (982 reviews, star-rating labels)", "",
         "| Model (label definition) | macro-F1 model only | macro-F1 with unclear rule | unclear rate | neutral F1 (with rule) | neutral support |",
         "|---|---|---|---|---|---|"]
    for v, r in A.items():
        L.append(f"| {v} ({r['labels']}) | {r['model_only']['macro_f1']} | {r['with_unclear_rule']['macro_f1']} | "
                 f"{r['unclear_rate']} | {r['with_unclear_rule']['per_class']['neutral']['f1']} | "
                 f"{r['with_unclear_rule']['per_class']['neutral']['support']} |")
    tc = A["full"]["with_unclear_rule"]["top_confusions"]
    L += ["", "Most-confused pairs (full, with rule; share of the gold class): "
          + "; ".join(f"{c['gold']} → {c['pred']} {c['share_of_gold']}" for c in tc) + "."]
    F = R["F_baselines"]
    L += ["", "### Baselines on the same S1 test split (D1 labels)", "",
          "Fit on the train split only; Naive Bayes `alpha` chosen by group-aware 5-fold CV (macro-F1) from "
          "0.01, 0.1, 0.5, 1, 10. Reference rows only: the shipped model is the LinearSVC above.", "",
          "| Model | macro-F1 | accuracy | note |", "|---|---|---|---|",
          f"| shipped M1 LinearSVC (model only) | {A['full']['model_only']['macro_f1']} | {A['full']['model_only']['accuracy']} | word+char TF-IDF |"]
    for k in ("dummy_most_frequent", "dummy_uniform", "multinomial_nb", "complement_nb"):
        r = F[k]
        note = f"alpha {r['best_alpha']}, CV macro-F1 {r['cv_macro_f1']}" if "best_alpha" in r else "no text used"
        L.append(f"| {k} | {r['macro_f1']} | {r['accuracy']} | {note} |")
    s = F["shortcut_check"]
    L += ["", f"Shortcut check: {s['train_texts_matching']} of {s['train_rows']} training reviews mention a star "
          "rating in the text (the labels come from star ratings, so such text would leak the label)."]
    L += ["", "The two rows use different gold labels, so they cannot be compared with each other. Section B gives "
          "the common ruler.", "",
          "## B. Same models against human-annotated polarity (208 EduRABSA test reviews)", "",
          "Gold = the most frequent annotated opinion polarity in the review (the teammate's \"second ruler\").", "",
          "| M1 trained with | macro-F1 model only | macro-F1 with unclear rule | pos F1 | neg F1 | neu F1 |", "|---|---|---|---|---|---|"]
    for v, r in B.items():
        w = r["with_unclear_rule"]["per_class"]
        L.append(f"| {'rating 3 = neutral (D1)' if v == 'full' else 'teammate mapping (3-4 stars neutral)'} | "
                 f"{r['model_only']['macro_f1']} | {r['with_unclear_rule']['macro_f1']} | {w['positive']['f1']} | "
                 f"{w['negative']['f1']} | {w['neutral']['f1']} |")
    L += ["", "## C. Aspect + sentiment pairs, end to end (sentence level)", "",
          "`full` = EduRABSA + generated RHU sentences. `edurabsa_only` = ablation without generated sentences.", "",
          "| Set | Model | sentences | gold pairs | pair P | pair R | **pair F1** | aspect F1 | polarity acc. on found aspects | exact match |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("rhu_human_test", "synthetic_dev", "edurabsa_test"):
        for v in ("full", "edurabsa_only"):
            r = C[v][name]
            L.append(f"| {name} | {v} | {r['n_sentences']} | {r['gold_pairs']} | {r['pair_precision']} | {r['pair_recall']} | "
                     f"**{r['pair_f1']}** | {r['aspect_f1']} | {r['polarity_acc_on_found_aspects']} ({r['n_found_aspects']}) | {r['exact_set_match']} |")
    h = C["full"]["rhu_human_test"]
    L += ["", f"Human test pair F1 95% bootstrap interval (full): {h['pair_f1_95ci_bootstrap']} (40 sentences, "
          f"11 respondents; resampling noise only). synthetic_dev is generated text. It was written to include hard "
          "cases (sarcasm, neutral statements of fact, several aspects per sentence), so its polarity scores are "
          "lower than on the human sentences. It is a diagnostic set, not evidence of real accuracy. "
          f"On edurabsa_test, predicted aspects outside the 5 EduRABSA aspects: "
          f"{C['full']['edurabsa_test']['pred_aspects_outside_edurabsa_scheme']}. They count as errors, even where "
          "they may be right (for example bus = transportation).", "",
          f"### Per aspect on the {len(human)} human sentences (counts)", "",
          "| Aspect | gold | predicted (full) | detected (full) | detected + right polarity (full) | detected (edurabsa_only) |",
          "|---|---|---|---|---|---|"]
    pa, pe = h["per_aspect"], C["edurabsa_only"]["rhu_human_test"]["per_aspect"]
    for a in sorted(set(pa) | set(pe)):
        x = pa.get(a, {"gold": 0, "pred": 0, "detected": 0, "detected_and_polarity_right": 0})
        L.append(f"| {a} | {x['gold']} | {x['pred']} | {x['detected']} | {x['detected_and_polarity_right']} | "
                 f"{pe.get(a, {}).get('detected', 0)} |")
    L += ["", "## D. Polarity step alone, given the gold opinion phrase", "", "| Set | Model | n | macro-F1 | accuracy |", "|---|---|---|---|---|"]
    for v in D:
        for name, r in D[v].items():
            L.append(f"| {name} | {v} | {r['n']} | {r['macro_f1']} | {r['accuracy']} |")
    L += ["", "The teammate reported 0.642 macro-F1 for this step on the EduRABSA test (their split and taxonomy).", "",
          "## E. Every human test sentence", "",
          f"The per-sentence table is written to `{PRIVATE.relative_to(ROOT).as_posix()}/feedback_human_errors.md`. That "
          "folder is git-ignored, because the repository is public and the survey responses are not published.", ""]
    E = ["# Human test sentences: per-sentence predictions (full model)", "",
         "Private: anonymous RHU survey responses. Do not commit or publish.", "",
         "✓ = predicted pairs equal gold exactly. `?` marks a pair the unclear rule turned into neutral. "
         "`rev` = gold label revised by Claude (to be checked by two annotators, F1b). `Q` = a question, not feedback.", "",
         "| id | flags | sentence | gold | predicted | |", "|---|---|---|---|---|---|"]
    for row, out in zip(human.itertuples(), preds["full"]):
        pred = {a: x["sentiment"] + ("?" if x["unclear"] else "") for a, x in out["aspects"].items()}
        plain = {a: x["sentiment"] for a, x in out["aspects"].items()}
        flags = " ".join(f for f, on in (("rev", str(row.label_revised_by_claude) == "True"), ("Q", str(row.is_question) == "True")) if on)
        text = str(row.text).replace("|", "/")
        E.append(f"| {row.sentence_id} | {flags} | {text} | {fmt_pairs(parse_pairs(row.pairs))} | {fmt_pairs(pred)} | "
                 f"{'✓' if plain == parse_pairs(row.pairs) else '✗'} |")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "feedback_human_errors.md").write_text("\n".join(E) + "\n", encoding="utf-8")
    L += ["## Limits", "",
          f"- {len(human)} human sentences from 11 respondents (labels v2: team rules R1-R6, HF017 dropped): this shows behaviour, not a reliable accuracy. Per-aspect "
          "counts are 1 to 8.",
          "- Review-level labels are star ratings (weak labels). With rating 3 = neutral there are only 29 neutral test reviews.",
          "- Most RHU aspects were learned from about 20 generated sentences each.",
          "- The public review data is North-American course/teacher reviews."]
    (EVAL / "feedback_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
