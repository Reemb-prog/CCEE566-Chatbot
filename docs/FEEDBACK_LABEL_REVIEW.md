# Feedback module: decisions and label review for the team

Checklist and decisions (2026-11-22). Survey sentences are referred to **by ID only**, because the repository is public.
The text, the gold labels and the current predictions are in `eval/private/feedback_human_errors.md`
(git-ignored, on the machine that ran `scripts/eval_feedback.py`).

## 1. Aspect rules (agreed by the team 2026-11-22, applied)

| # | Rule (as agreed) | Applied to |
|---|---|---|
| R1 | Bus fees are **both** `tuition_fees` and `transportation` | HF021, HF024 → both negative |
| R2 | A crashing registration portal is **both** `registration` and `it_systems` | HF009 → both negative |
| R3 | Rising fees count as a complaint (negative) | HF007 → `tuition_fees:negative` (was neutral) |
| R4 | "Doctors" = `instructors`; "some good, some not" = neutral (mixed) | HF016, HF023, HF032 already matched, no change |
| R5 | "Needs more services" = `student_services` only | HF033 already matched, no change |
| R6 | A question that is not feedback is dropped from feedback scoring | HF017 excluded (39 sentences scored) |

The labels are in `inputs/feedback/raw/human_final_v2.csv` (git-ignored; `human_final.csv` is kept unchanged for the
sealed v1 result). Each changed row has a `RULE-Rn` or `EXCLUDE-R6` note. `scripts/prepare_feedback_data.py` uses v2
when it exists.

## 2. Label check (done by Claude, single annotator)

The team asked Claude to do this check instead of two teammates. Claude re-read all 40 sentences against R1–R6 and found
no other label that breaks a rule. **Limits:** one annotator only, so there is no Cohen's kappa and no measured
agreement. Claude also wrote 12 of the earlier label fixes (`CLAUDE-FIXED` notes), so this is not an independent check.

## 3. New sentences

Not possible: the team cannot collect more responses. So there is **no independent score** for the feedback module.
The human-set numbers are development numbers on data that was used to fix errors. The sealed v1 result
(`eval/feedback_results_sealed_v1.md`) is the last independent number.

**Effect (retrained 2026-11-22):** human pair F1 0.512 → 0.621 (95% CI 0.48–0.76), aspect F1 0.767 → 0.828. Most of
this gain comes from the new labels, which now match what the model already predicted (R1, R2), not from a better
model. EduRABSA test barely moved (0.562 → 0.578), and synthetic_dev did not move (0.267).

## 4. Known model gaps (need data, not labels)

| Gap | Example | Fix |
|---|---|---|
| Cost of credits is not recognised as tuition | HF003 | About 10 more generated sentences on credit/course cost |
| "X is easy" sometimes scored negative by the clause model | HF020 | More short positive generated sentences per aspect |
| IT systems detector is weak (CV F1 about 0.3) | — | 15–20 more varied IT sentences (portal, Wi-Fi, email, login) |
| Review-level model disagrees with aspects on short RHU text | — | Handled in the UI: the overall label follows the aspects (section 5) |

## 5. What was already fixed (2026-11-22, no retraining)

- **Overall label follows the aspect results** when there are polar aspects (otherwise the clause sentiment, otherwise
  the review model). The review model's own view is still shown in the trace.
- **Cue-word rule:** when the clause model says neutral but the clause has an unambiguous complaint/praise word
  (crashed, overpriced, broken, not helpful, friendly…), that word decides. A negator flips it. The trace marks it
  as a cue rule.
- **Routing:**
  - "credit" with opinion wording now counts as feedback.
  - Complaint words such as broken, outdated or overpriced count as feedback cues.
  - Full sentences where the model sees clear sentiment are routed. Search phrases ("late withdrawal from a course")
    are not.
  - A complaint phrased as a question ("Why is registration always so terrible?") is answered from the catalog
    **and** offered the Feedback analyser.
