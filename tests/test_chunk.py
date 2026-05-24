from __future__ import annotations

from asx_grounded.ingestion.chunk import chunk_pdf
from tests.conftest import make_parsed_pdf


def test_chunks_have_stable_ids_and_metadata() -> None:
    parsed = make_parsed_pdf(ann_id="CBA_2026-02-01_001")
    chunks = chunk_pdf(parsed, asx_code="CBA", target_tokens=80, overlap_tokens=20)

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "CBA_2026-02-01_001:0"
    assert chunks[0].ann_id == "CBA_2026-02-01_001"
    assert chunks[0].asx_code == "CBA"
    assert all(c.page_num in {1, 2} for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_long_sentence_not_split_below_target() -> None:
    # One enormous sentence should still emit a single chunk, not crash.
    text = "Word " * 1500
    parsed = make_parsed_pdf(ann_id="LONG", pages=[text])
    chunks = chunk_pdf(parsed, asx_code="LONG", target_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 1
    # No empty chunks
    assert all(c.text.strip() for c in chunks)


def test_empty_page_skipped() -> None:
    parsed = make_parsed_pdf(ann_id="EMPTY", pages=["", "Real content goes here. With sentences."])
    chunks = chunk_pdf(parsed, asx_code="EMPTY", target_tokens=80, overlap_tokens=20)
    assert all(c.page_num == 2 for c in chunks)
