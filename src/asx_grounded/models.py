from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AnnouncementType(str, Enum):
    PRICE_SENSITIVE = "price_sensitive"
    PERIODIC_REPORT = "periodic_report"
    DIVIDEND = "dividend"
    SUBSTANTIAL_HOLDER = "substantial_holder"
    GENERAL = "general"
    OTHER = "other"


class Announcement(BaseModel):
    """A single ASX announcement (one PDF document)."""

    ann_id: str = Field(description="Stable ASX announcement identifier")
    asx_code: str = Field(description="3-letter ASX ticker, e.g. CBA")
    company_name: str
    headline: str
    released_at: datetime
    announcement_type: AnnouncementType = AnnouncementType.OTHER
    is_price_sensitive: bool = False
    pdf_url: str
    asx_page_url: str = Field(description="Canonical ASX page for the announcement (for citation deep-links)")
    pages: int = 0


class Chunk(BaseModel):
    """A retrievable chunk of text from one announcement."""

    chunk_id: str = Field(description="Stable identifier: {ann_id}:{chunk_idx}")
    ann_id: str
    asx_code: str
    chunk_idx: int
    page_num: int | None = None
    text: str
    token_count: int = 0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rerank_score: float | None = None


class Citation(BaseModel):
    """A citation emitted by the agent and verified against retrieved chunks."""

    chunk_id: str
    ann_id: str
    asx_page_url: str
    verified: bool = False
    verification_note: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    retrieval_debug: dict = Field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""


class EvalQuestion(BaseModel):
    qid: str
    category: str  # answerable | unanswerable | time_bounded | comparative | adversarial
    question: str
    expected_refusal: bool = False
    expected_citations: list[str] = Field(default_factory=list)
    notes: str = ""


class JudgeVerdict(BaseModel):
    qid: str
    factually_correct: bool
    citation_accuracy: float  # 0..1 — of cited chunks, % that support the claim
    citation_recall: float    # 0..1 — of expected citations, % present
    refusal_correct: bool
    format_compliant: bool
    hallucination: bool
    reasoning: str
