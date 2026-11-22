# Feedback sentiment / ABSA data

Inputs for the separate feedback module (see `docs/FEEDBACK_ABSA_PLAN.md`). This is **student opinion data, not catalog
evidence**: nothing here may be used to answer catalog questions.

`raw/` holds copies of the files exactly as received. Do not edit them. `scripts/prepare_feedback_data.py` reads only
`raw/` and regenerates `processed/` plus `processed/data_card.json`.

Both folders are git-ignored until the licences are confirmed, because the public reviews contain real instructor names.

| Code | File in `raw/` | Received from | SHA-256 |
|---|---|---|---|
| S1 | `combined_ASBA_clean.csv` | teammate `asba_package.zip` (`data/clean/`) | `2202bac3d7adac8395e01ebbe56e58de19e27efa5a1375ea5f309d51470fb051` |
| S1-log | `cleaning_log.csv` | same zip | `fd652dff59a7ca6761a06b9445bd93878b7b1dadca7b5f3430428d57efe7c1b4` |
| S1-split | `predictions_taskA_test.csv` (ids of the teammate's 982 Task A test rows) | same zip | `eaccc23d8ccde523ffc72f18e514fba63f81a6e9626d44c04301f4374859c782` |
| S2 | `edurabsa_clean.jsonl` | same zip | `34216d4460007458288641bb41487c35db6d6c54b71dbc44a3b14ad17276e8d6` |
| S3 | `RHU_ABSA_Annotations_Consolidated_Clean.csv` (Claude-generated RHU set + human rows) | team, 2026-11-17 | `2feef3292e4cbd6151e265b9d0e0842237d5747091d1253773bf7e083b7b591c` |
| S3-patch | `RHU_synthetic_patch_f3.csv` (83 generated sentences written by Claude **after** seeing the F3 errors on S4) | Claude, 2026-11-21 | see `processed/data_card.json` |
| S4 | `human_final.csv` (40 human survey sentences, 11 respondents) | team, 2026-11-17 | `846790f88d1d9a00f7240aa1172b46b8a6ff0924880fcfc9e225bb32a583116b` |
| S4-v2 | `human_final_v2.csv` (S4 with team rules R1–R3 applied, HF017 excluded by R6; used by prepare) | Claude, 2026-11-22 | `2674f53dc31011196d175b5fe6c371c23e60808c5281068ab3580934ecd6ab5b` |

## Still missing (blocks full provenance, not F1)

- `combined_ASBA.csv`, the raw S1 file the teammate's notebook 01 reads.
- The original EduRABSA `train/val/test.jsonl`, plus the script that produced `edurabsa_clean.jsonl`. That script dropped
  277 of the 1,320 annotated reviews (most of them university reviews) and moved 93 official-validation reviews into train.
- **Licences** for EduRABSA and for the three review sources (`data1_19145`, `data2_14810`, `data3_*`). Record each licence
  and citation here before any commit or public demo.

## Use rules

- S4 (the 40 human sentences) was the sealed RHU test set. On 2026-11-21 the team chose to **unseal** it and fix
  the errors found there, so its scores are now development numbers. It is still never used as training data. An
  independent RHU score needs a new set of human sentences.
- The `synthetic_legacy` rows in S3 (templated October export) are diagnosis-only and are not used for training.
- The `synthetic_new` rows in S3 are generated (not student) text. They train only the RHU aspects that EduRABSA lacks,
  and the UI labels those aspects as trained on generated examples.
