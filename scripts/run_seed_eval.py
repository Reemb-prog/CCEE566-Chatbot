"""Run eval/seed_questions.csv through the pipeline and write eval/seed_run_results.csv.

This records what the system DID (status, decision, cited pages). The correctness columns are left
blank on purpose: only a human checking the PDF can fill them. Not an accuracy benchmark.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rhu_chat.pipeline import RHUChatPipeline  # noqa: E402


def main() -> int:
    pipe = RHUChatPipeline.from_project_root(ROOT)
    src = ROOT / "eval" / "seed_questions.csv"
    out = ROOT / "eval" / "seed_run_results.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "case_type", "expected_behavior", "actual_status", "decision", "cited_pages",
                    "cited_ids", "answer_preview", "total_ms", "citation_correct", "answer_correct", "notes"])
        for r in rows:
            res = pipe.ask(r["question"])
            w.writerow([r["question"], r["case_type"], r["expected_behavior"], res["status"],
                        res["trace"]["grounding"]["decision"],
                        " ".join(str(c["source_page"]) for c in res["citations"]),
                        " ".join(c["source_id"] for c in res["citations"]),
                        res["answer"].replace("\n", " / ")[:300], res["trace"]["timings_ms"]["total"], "", "", ""])
            print(f"{res['status']:<22} {res['trace']['grounding']['decision']:<26} "
                  f"{[c['source_page'] for c in res['citations']]!s:<22} {r['question'][:60]}")
    print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
