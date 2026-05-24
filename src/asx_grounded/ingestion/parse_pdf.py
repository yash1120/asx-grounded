"""PDF → clean text extraction for ASX announcement PDFs.

ASX PDFs are noisy: ASX-template cover page, page-numbered footers, multi-column
tables. We extract per-page text and trim boilerplate. Image-only PDFs are
flagged for an OCR pass (not implemented in v1 — listed in the eval blog as a
known limitation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import structlog

log = structlog.get_logger()


_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*ASX (?:Market )?Announcement\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*page \d+ of \d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # bare page numbers
    re.compile(r"^\s*©.*all rights reserved.*$", re.IGNORECASE | re.MULTILINE),
]


@dataclass(slots=True)
class ParsedPage:
    page_num: int
    text: str


@dataclass(slots=True)
class ParsedPdf:
    ann_id: str
    pages: list[ParsedPage]
    total_chars: int
    image_only: bool

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _scrub(text: str) -> str:
    out = text
    for pat in _BOILERPLATE_PATTERNS:
        out = pat.sub("", out)
    # collapse 3+ blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def parse_pdf(pdf_path: Path, ann_id: str) -> ParsedPdf:
    pages: list[ParsedPage] = []
    total_chars = 0
    image_only = True

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _scrub(raw)
            if cleaned:
                image_only = False
            pages.append(ParsedPage(page_num=i, text=cleaned))
            total_chars += len(cleaned)

    if image_only:
        log.warning("parse.image_only", ann_id=ann_id, path=str(pdf_path))

    return ParsedPdf(
        ann_id=ann_id,
        pages=pages,
        total_chars=total_chars,
        image_only=image_only,
    )
