# Five-minute capstone demonstration script

1. Introduce the system as a **2026–2027 catalog-based prototype** with classical NLP, not a live university advisor.
2. Ask a course-credit question. Show extraction of `CCEE 566` and course-credit answer, page citation.
3. Expand the inspector. Explain regex/tokenization, TF-IDF word features, **separately trained V3.1 Linear SVM** category/intent hints, cosine ranking, catalog evidence and extractive answer. Clearly state score bars are *retrieval similarity*, not language-model confidence.
4. Ask a prerequisite question; demonstrate multiple catalog entries and why the system does not silently choose one.
5. Ask an unsupported personal GPA question; show safe fallback.
6. Summarize evaluation: supported structured and excerpt queries, limitations of the synthetic QA data and lack of independent student testing; no generative hallucination claims.
7. Distinguish built module from later planned voice activation and ABSA. Do not imply those exist if not built.

Do not treat the seven reviewed QA cases or seed cases as a statistically representative benchmark.
