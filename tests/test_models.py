from __future__ import annotations

from datetime import datetime

from asx_grounded.models import (
    Announcement,
    AnnouncementType,
    Chunk,
    Citation,
    QueryResponse,
    RetrievedChunk,
)

# --------------------------------------------------------------------------- #
# AnnouncementType StrEnum
# --------------------------------------------------------------------------- #


def test_announcement_type_values() -> None:
    assert AnnouncementType.PRICE_SENSITIVE == "price_sensitive"
    assert AnnouncementType.PERIODIC_REPORT == "periodic_report"
    assert AnnouncementType.DIVIDEND == "dividend"
    assert AnnouncementType.SUBSTANTIAL_HOLDER == "substantial_holder"
    assert AnnouncementType.GENERAL == "general"
    assert AnnouncementType.OTHER == "other"


def test_announcement_type_is_str() -> None:
    # StrEnum members behave as plain strings (usable in f-strings, JSON, etc.).
    assert isinstance(AnnouncementType.DIVIDEND, str)
    assert f"{AnnouncementType.DIVIDEND}" == "dividend"


def test_announcement_type_constructs_from_value() -> None:
    assert AnnouncementType("price_sensitive") is AnnouncementType.PRICE_SENSITIVE


def test_announcement_type_round_trips_through_string() -> None:
    for member in AnnouncementType:
        assert AnnouncementType(str(member)) is member


# --------------------------------------------------------------------------- #
# Model defaults
# --------------------------------------------------------------------------- #


def test_chunk_defaults() -> None:
    chunk = Chunk(chunk_id="CBA:0", ann_id="CBA_2026", asx_code="CBA", chunk_idx=0, text="hello")
    assert chunk.page_num is None
    assert chunk.token_count == 0


def test_retrieved_chunk_default_rerank_score() -> None:
    chunk = Chunk(chunk_id="CBA:0", ann_id="CBA_2026", asx_code="CBA", chunk_idx=0, text="hello")
    rc = RetrievedChunk(chunk=chunk, score=0.5)
    assert rc.rerank_score is None


def test_query_response_defaults() -> None:
    resp = QueryResponse(query="q", answer="a")
    assert resp.citations == []
    assert resp.refused is False
    assert resp.refusal_reason == ""
    assert resp.retrieval_debug == {}
    assert resp.latency_ms == 0
    assert resp.model == ""


def test_citation_defaults() -> None:
    cit = Citation(chunk_id="CBA:0", ann_id="CBA_2026", asx_page_url="https://asx/CBA")
    assert cit.verified is False
    assert cit.verification_note == ""


def test_announcement_defaults() -> None:
    ann = Announcement(
        ann_id="CBA_2026-02-01_001",
        asx_code="CBA",
        company_name="Commonwealth Bank",
        headline="Dividend",
        released_at=datetime(2026, 2, 1, 9, 30),
        pdf_url="https://asx/pdf",
        asx_page_url="https://asx/page",
    )
    assert ann.announcement_type is AnnouncementType.OTHER
    assert ann.is_price_sensitive is False
    assert ann.pages == 0


# --------------------------------------------------------------------------- #
# Round-trip serialisation via model_dump_json / model_validate_json
# --------------------------------------------------------------------------- #


def test_chunk_json_round_trip() -> None:
    chunk = Chunk(
        chunk_id="CBA:3",
        ann_id="CBA_2026-02-01",
        asx_code="CBA",
        chunk_idx=3,
        page_num=2,
        text="Fully-franked dividend of $2.40.",
        token_count=7,
    )
    restored = Chunk.model_validate_json(chunk.model_dump_json())
    assert restored == chunk


def test_retrieved_chunk_json_round_trip_preserves_nested_chunk() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(
            chunk_id="BHP:1",
            ann_id="BHP_2026-03-10",
            asx_code="BHP",
            chunk_idx=1,
            page_num=None,
            text="Production update.",
            token_count=3,
        ),
        score=0.873,
        rerank_score=0.91,
    )
    restored = RetrievedChunk.model_validate_json(rc.model_dump_json())
    assert restored == rc
    assert restored.chunk.chunk_id == "BHP:1"
    assert restored.rerank_score == 0.91


def test_query_response_json_round_trip_with_citations() -> None:
    resp = QueryResponse(
        query="What was CBA's dividend?",
        answer="CBA declared a $2.40 dividend [CBA:0].",
        citations=[
            Citation(
                chunk_id="CBA:0",
                ann_id="CBA_2026-02-01",
                asx_page_url="https://asx.com.au/CBA",
                verified=True,
                verification_note="exact match",
            )
        ],
        refused=False,
        retrieval_debug={"bm25_hits": 5, "vector_hits": 5},
        latency_ms=1234,
        model="claude-sonnet",
    )
    restored = QueryResponse.model_validate_json(resp.model_dump_json())
    assert restored == resp
    assert len(restored.citations) == 1
    assert restored.citations[0].verified is True
    assert restored.retrieval_debug == {"bm25_hits": 5, "vector_hits": 5}


def test_query_response_refusal_round_trip() -> None:
    resp = QueryResponse(
        query="What did CBA announce in 2099?",
        answer="",
        refused=True,
        refusal_reason="No supporting chunks in corpus.",
    )
    restored = QueryResponse.model_validate_json(resp.model_dump_json())
    assert restored.refused is True
    assert restored.refusal_reason == "No supporting chunks in corpus."
    assert restored.citations == []
