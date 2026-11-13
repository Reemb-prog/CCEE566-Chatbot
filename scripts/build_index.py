"""Build (or reuse) the local TF-IDF passage index under artifacts/ and smoke-load the V3.1 classifier.

Usage:  python scripts/build_index.py [--force]
No network access; reads only the bundled CSVs in inputs/knowledge.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rhu_chat.data import CatalogData  # noqa: E402
from src.rhu_chat.retrieval import PassageIndex  # noqa: E402


def main() -> int:
    force = "--force" in sys.argv
    t0 = time.perf_counter()
    data = CatalogData.load(ROOT)
    print(f"Loaded tables: {len(data.passages)} passages, {len(data.courses)} course records, "
          f"{len(data.plans)} study-plan rows, {len(data.pages)} catalog pages")
    index, meta = PassageIndex.load_or_build(data, ROOT / "artifacts", force=force)
    print(f"Index {meta['cache']}: {meta['n_passages']} passages x {meta['n_features']} TF-IDF features "
          f"in {(time.perf_counter() - t0) * 1000:.0f} ms -> artifacts/")
    from src.rhu_chat.pipeline import RHUChatPipeline
    pipe = RHUChatPipeline.from_project_root(ROOT)
    if pipe.clf_error:
        print("WARNING: V3.1 classifier could not be loaded:", pipe.clf_error)
        return 1
    r = pipe.ask("How many credits is CCEE 566?")
    print(f"Self-check: status={r['status']} citations={[c['source_page'] for c in r['citations']]} "
          f"total={r['trace']['timings_ms']['total']} ms")
    return 0 if r["status"] == "answered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
