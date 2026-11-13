# Manual QA evaluation: candidate questions, not a gold standard

`seed_questions.csv` was authored as a convenient demo/evaluation starting point. It is **not** an independent human-labeled student benchmark and must not be reported as one. Only the user/instructor can validate answers against the original PDF.

Use columns `actual_status`, `citation_correct`, `answer_correct`, `notes` to record manual review after implementation. Do not tune retrieval based on the historical QA test or seven reviewed gold questions and then call that held out. Suggested evaluation: 25–40 **independently written** diverse student questions with relevant passage/page gold labels. Report Recall@3 and Recall@5 only if manually verified passage IDs actually exist; otherwise report exact code/ page citation rate on reviewed cases.
