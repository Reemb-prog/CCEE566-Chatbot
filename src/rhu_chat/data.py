"""Load the bundled catalog tables (read-only) and expose small lookup helpers.

All values come from the CSVs under ``inputs/knowledge``. Nothing here fetches
from the network or re-extracts the PDF.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

KNOWLEDGE_FILES = {
    "passages": "RETRIEVAL_PASSAGES.csv",
    "courses": "RHU_COURSE_DESCRIPTIONS.csv",
    "plans": "RHU_PROGRAM_STUDY_PLAN_ROWS.csv",
    "pages": "RHU_CATALOG_FULL_TEXT_BY_PAGE.csv",
}
CATALOG_PDF_URL = "https://www.rhu.edu.lb/Library/Assets/Gallery/Files/RHU_Undegraduate_Catalog_2026-2027.pdf"
_URL_RE = re.compile(r"^https://www\.rhu\.edu\.lb/[^\s#]+\.pdf#page=(\d+)$")


def norm_space(text: object) -> str:
    """Collapse whitespace; NaN/None become an empty string."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean(value: object) -> str | None:
    """Return a stripped string, or None for empty / NaN cells."""
    s = norm_space(value)
    return s or None


def citation_url(source_url: object, page: int) -> str:
    """Use the supplied URL when it is the official PDF and its #page matches; else build it once."""
    s = norm_space(source_url)
    m = _URL_RE.match(s)
    if m and int(m.group(1)) == int(page):
        return s
    return f"{CATALOG_PDF_URL}#page={int(page)}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class CatalogData:
    root: Path
    passages: pd.DataFrame
    courses: pd.DataFrame
    plans: pd.DataFrame
    pages: pd.DataFrame
    page_text: dict = field(default_factory=dict)   # page -> whitespace-normalised text
    course_prefixes: set = field(default_factory=set)
    programs: list = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "CatalogData":
        kdir = Path(root) / "inputs" / "knowledge"
        read = lambda name: pd.read_csv(kdir / KNOWLEDGE_FILES[name], encoding="utf-8-sig", dtype=str)
        passages, courses, plans, pages = (read(n) for n in ("passages", "courses", "plans", "pages"))
        for df, col in ((passages, "source_page_start"), (passages, "source_page_end"),
                        (courses, "source_page"), (plans, "source_page"), (pages, "pdf_page")):
            df[col] = df[col].astype(int)
        passages["passage_text"] = passages["passage_text"].map(norm_space)
        courses["course_code"] = courses["course_code"].map(norm_space).str.upper()
        plans["course_code"] = plans["course_code"].map(norm_space).str.upper()
        data = cls(root=Path(root), passages=passages, courses=courses, plans=plans, pages=pages)
        data.page_text = {int(p): norm_space(t) for p, t in zip(pages["pdf_page"], pages["page_text"])}
        data.course_prefixes = {c.split()[0] for c in courses["course_code"]} | {
            c.split()[0] for c in plans["course_code"] if " " in c}
        data.programs = sorted(plans["program_or_section"].dropna().unique().tolist())
        return data

    @staticmethod
    def source_hashes(root: Path) -> dict:
        kdir = Path(root) / "inputs" / "knowledge"
        return {name: file_sha256(kdir / fname) for name, fname in KNOWLEDGE_FILES.items()}

    # ----- lookups -------------------------------------------------------
    def course_rows(self, code: str) -> list[dict]:
        rows = self.courses[self.courses["course_code"] == code]
        return rows.to_dict("records")

    def code_in_passages(self, code: str) -> list[str]:
        pattern = re.escape(code).replace(r"\ ", r"\s?")
        mask = self.passages["passage_text"].str.contains(rf"\b{pattern}\b", regex=True)
        return self.passages.loc[mask, "passage_id"].tolist()

    def text_on_page(self, needle: str, page: int, also_next: bool = True) -> bool:
        """Case-insensitive, whitespace-normalised containment check against extracted page text."""
        n = norm_space(needle).lower()
        if not n:
            return False
        pages = [page, page + 1] if also_next else [page]
        return any(n in self.page_text.get(p, "").lower() for p in pages)

    def page_window(self, needle: str, page: int, width: int = 220) -> str | None:
        """Return a verbatim window of extracted page text starting at ``needle`` (if present)."""
        text = self.page_text.get(page, "")
        i = text.lower().find(norm_space(needle).lower())
        if i < 0:
            return None
        snippet = text[i:i + width]
        return snippet + ("…" if i + width < len(text) else "")
