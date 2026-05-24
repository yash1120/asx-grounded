"""FastAPI app — single source of truth for the live demo.

Endpoints:
    GET  /healthz
    POST /query              { "question": "..." }  -> QueryResponse
    GET  /eval/scoreboard    -> latest published scoreboard JSON
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from asx_grounded.agent.generate import generate
from asx_grounded.agent.verify_citations import verify_citations
from asx_grounded.config import get_settings
from asx_grounded.models import QueryResponse
from asx_grounded.retrieval.filters import extract_filter
from asx_grounded.retrieval.hybrid import HybridRetriever
from asx_grounded.retrieval.rerank import CrossEncoderReranker

log = structlog.get_logger()

BM25_CORPUS = Path("data/processed/bm25_corpus.jsonl")
SCOREBOARD_PATH = Path("data/processed/scoreboard.json")


class _State:
    retriever: HybridRetriever | None = None
    reranker: CrossEncoderReranker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:  # type: ignore[override]
    log.info("api.startup")
    _State.retriever = HybridRetriever(BM25_CORPUS)
    _State.reranker = CrossEncoderReranker()
    yield
    log.info("api.shutdown")


app = FastAPI(title="asx-grounded", version="0.1.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    verify_with_llm: bool = False


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "model": settings.generator_model}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    if _State.retriever is None or _State.reranker is None:
        raise HTTPException(status_code=503, detail="retriever not ready")

    started = time.perf_counter()
    mfilter = extract_filter(req.question)
    candidates = _State.retriever.retrieve(req.question, mfilter=mfilter)
    top = _State.reranker.rerank(req.question, candidates, top_k=req.top_k)

    if not top:
        return QueryResponse(
            query=req.question,
            answer="No relevant ASX announcements found in the indexed corpus.",
            refused=True,
            refusal_reason="empty retrieval",
            retrieval_debug={
                "filter": {
                    "asx_codes": sorted(mfilter.asx_codes),
                    "released_after": mfilter.released_after.isoformat() if mfilter.released_after else None,
                    "released_before": mfilter.released_before.isoformat() if mfilter.released_before else None,
                },
                "candidates": len(candidates),
            },
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=settings.generator_model,
        )

    gen = generate(req.question, top)
    verified = verify_citations(gen.text, top, verify_with_llm=req.verify_with_llm)

    # Backfill the asx_page_url on each citation from the retriever's metadata.
    for c in verified.citations:
        c.asx_page_url = _State.retriever.page_url(c.chunk_id)

    return QueryResponse(
        query=req.question,
        answer=verified.answer_text,
        citations=verified.citations,
        refused=gen.refused,
        refusal_reason=gen.refusal_reason,
        retrieval_debug={
            "filter": {
                "asx_codes": sorted(mfilter.asx_codes),
                "released_after": mfilter.released_after.isoformat() if mfilter.released_after else None,
                "released_before": mfilter.released_before.isoformat() if mfilter.released_before else None,
            },
            "candidates": len(candidates),
            "reranked": [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "asx_code": c.chunk.asx_code,
                    "score": c.score,
                    "rerank_score": c.rerank_score,
                }
                for c in top
            ],
            "fabricated_citations": verified.fabricated_ids,
            "unsupported_claims": verified.unsupported_claims,
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
        },
        latency_ms=int((time.perf_counter() - started) * 1000),
        model=gen.model,
    )


@app.get("/eval/scoreboard")
def scoreboard() -> dict:
    if not SCOREBOARD_PATH.exists():
        return {"status": "no_eval_runs_yet"}
    with SCOREBOARD_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
