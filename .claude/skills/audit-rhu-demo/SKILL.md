---
name: audit-rhu-demo
description: Inspect the completed RHU NLP chatbot against the acceptance checklist and supplied catalog sources; run tests, smoke-check app, investigate citations and report corrections.
---

# Audit RHU Catalog Chatbot

1. Read `docs/ACCEPTANCE_CRITERIA.md`, `docs/DATA_AND_GROUNDING.md`, `docs/PIPELINE_CONTRACT.md`.
2. Run preflight, index build, pytest, Streamlit health check, and manually evaluate `eval/seed_questions.csv`.
3. Check PDF citations against the bundled catalog text and file, especially CCEE 566 prerequisite differences.
4. Check actual inspector data rather than screenshots/fake constant values.
5. Fix implementable defects, rerun the full validation, and update `FINAL_IMPLEMENTATION_REPORT.md` with honest limitations.
