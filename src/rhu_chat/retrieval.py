"""Word TF-IDF index over all 2,502 catalog passages, cached under ``artifacts/``.

Scores reported here are *relative ranking scores* (cosine similarity plus small,
explicit boosts). They are not probabilities or confidence levels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import CatalogData, citation_url

INDEX_VERSION = 2   # v2: section/subsection headings are indexed with the passage text
INDEX_FILE = "tfidf_index.joblib"
META_FILE = "tfidf_index_meta.json"

# Soft boosts. Kept small so the cosine similarity dominates the ranking.
CATEGORY_BOOST = 0.03   # passage category matches the (mapped) V3.1 category hint
CODE_BOOST = 0.10       # passage contains a course code typed in the question

# V3.1 category labels -> passage `category` values. Soft hint only; never a filter.
CATEGORY_HINT_MAP = {
    "academic_calendar": {"academic_calendar"},
    "academic_regulations": {"academic_regulations"},
    "admission": {"admission"}, "admissions": {"admission"},
    "co_op_careers": {"coop_career"}, "coop_career": {"coop_career"},
    "college_information": {"college_information"},
    "courses": {"programs", "minors"}, "study_plans": {"programs"},
    "programs": {"programs"}, "minors": {"minors"},
    "fees": {"fees"}, "tuition_fees": {"fees"},
    "freshman": {"freshman"}, "general_education": {"general_education"},
    "registrar": {"registrar"}, "registration_advising": {"registrar", "academic_regulations"},
    "student_records": {"registrar"}, "student_assistance": {"student_services"},
    "student_services": {"student_services"},
    "offices_contacts": {"university_information"},
    "university_information": {"university_information"},
}


def make_vectorizer() -> TfidfVectorizer:
    # Default word pattern keeps numbers such as "566"; bigrams keep "ccee 566" together.
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                           sublinear_tf=True, min_df=1)


def index_documents(passages) -> list[str]:
    """Text that is vectorised: catalog section + subsection heading + passage text (all source data)."""
    sec = passages["section"].fillna("")
    sub = passages["subsection"].fillna("")
    return (sec + ". " + sub + ". " + passages["passage_text"]).tolist()


class PassageIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, passages):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.passages = passages.reset_index(drop=True)
        self.analyzer = vectorizer.build_analyzer()
        self._stop = vectorizer.get_stop_words() or frozenset()

    # ----- build / cache ------------------------------------------------
    @classmethod
    def build(cls, data: CatalogData) -> "PassageIndex":
        vec = make_vectorizer()
        matrix = vec.fit_transform(index_documents(data.passages))
        return cls(vec, matrix, data.passages)

    @classmethod
    def load_or_build(cls, data: CatalogData, artifacts: Path, force: bool = False) -> tuple["PassageIndex", dict]:
        """Reuse the cached index when the source CSV hashes match; otherwise rebuild it."""
        artifacts.mkdir(parents=True, exist_ok=True)
        hashes = CatalogData.source_hashes(data.root)
        meta_path, idx_path = artifacts / META_FILE, artifacts / INDEX_FILE
        if not force and meta_path.exists() and idx_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_hashes") == hashes and meta.get("index_version") == INDEX_VERSION:
                bundle = joblib.load(idx_path)  # our own local artifact
                ids_ok = bundle["passage_ids"] == data.passages["passage_id"].tolist()
                if ids_ok:
                    return cls(bundle["vectorizer"], bundle["matrix"], data.passages), {**meta, "cache": "reused"}
        index = cls.build(data)
        joblib.dump({"vectorizer": index.vectorizer, "matrix": index.matrix,
                     "passage_ids": data.passages["passage_id"].tolist()}, idx_path)
        meta = {"index_version": INDEX_VERSION, "source_hashes": hashes,
                "n_passages": int(index.matrix.shape[0]), "n_features": int(index.matrix.shape[1])}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return index, {**meta, "cache": "rebuilt"}

    # ----- query --------------------------------------------------------
    def tokens(self, text: str) -> list[str]:
        """Unigram tokens produced by the real TF-IDF analyzer (stop words removed)."""
        return [t for t in self.analyzer(text) if " " not in t]

    def idf(self, token: str) -> float:
        """Inverse document frequency of a unigram in the passage index (max idf if unseen)."""
        j = self.vectorizer.vocabulary_.get(token)
        return float(self.vectorizer.idf_[j]) if j is not None else float(self.vectorizer.idf_.max())

    def raw_tokens(self, text: str) -> list[str]:
        """Lower-cased word tokens before stop-word removal (same regex as the vectorizer)."""
        return re.findall(r"(?u)\b\w\w+\b", text.lower())

    def search(self, query: str, category_hint: str | None = None,
               course_codes: list[str] | None = None, k: int = 5) -> list[dict]:
        qv = self.vectorizer.transform([query])
        if qv.nnz == 0:
            return []
        sims = (self.matrix @ qv.T).toarray().ravel()       # rows are L2-normalised -> cosine
        boosts = np.zeros_like(sims)
        reasons = [[] for _ in range(len(sims))]
        cats = CATEGORY_HINT_MAP.get(category_hint or "", set())
        pcats = self.passages["category"].to_numpy()
        texts = self.passages["passage_text"].to_numpy()
        for i in np.flatnonzero(sims > 0):                   # boosts never resurrect zero-overlap passages
            if pcats[i] in cats:
                boosts[i] += CATEGORY_BOOST
                reasons[i].append("category_hint")
            for code in course_codes or []:
                if re.search(rf"\b{re.escape(code)}\b", texts[i]):
                    boosts[i] += CODE_BOOST
                    reasons[i].append(f"code:{code}")
                    break
        rank = sims + boosts
        order = np.argsort(-rank, kind="stable")[:k]
        out = []
        for i in order:
            if sims[i] <= 0:
                continue
            row = self.passages.iloc[int(i)]
            page = int(row["source_page_start"])
            out.append({
                "id": row["passage_id"], "source_page": page,
                "source_url": citation_url(row["source_url"], page),
                "section": row["section"], "subsection": row.get("subsection") or "",
                "category": row["category"],
                "tfidf_similarity": round(float(sims[i]), 4),
                "boost": round(float(boosts[i]), 4), "boost_reasons": reasons[i],
                "rank_score": round(float(rank[i]), 4),
                "text": row["passage_text"],
            })
        return out
