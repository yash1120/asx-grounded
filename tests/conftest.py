from __future__ import annotations

from asx_grounded.ingestion.parse_pdf import ParsedPage, ParsedPdf
from asx_grounded.models import Chunk, RetrievedChunk


def make_parsed_pdf(ann_id: str = "TEST_001", pages: list[str] | None = None) -> ParsedPdf:
    pages = pages or [
        "Commonwealth Bank of Australia announces a fully-franked dividend of $2.40 per share. "
        "The record date is 14 February 2026. The payment date is 28 March 2026.",
        "This is page two with additional context about the dividend policy and historical payments.",
    ]
    parsed_pages = [ParsedPage(page_num=i + 1, text=p) for i, p in enumerate(pages)]
    return ParsedPdf(
        ann_id=ann_id,
        pages=parsed_pages,
        total_chars=sum(len(p) for p in pages),
        image_only=False,
    )


def make_retrieved(chunk_id: str, ann_id: str, text: str, code: str = "CBA") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            ann_id=ann_id,
            asx_code=code,
            chunk_idx=int(chunk_id.split(":")[-1]),
            page_num=1,
            text=text,
        ),
        score=1.0,
        rerank_score=1.0,
    )
